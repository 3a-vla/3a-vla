"""In-process open-p2p transport for a single Windows machine.

The published open-p2p deployment separates Windows game I/O from a WSL
inference server reached over ``/tmp/uds.recap``. Unix domain sockets do not
exist on Windows, so that split requires WSL.

This module removes the split for single-machine setups by loading the
open-p2p policy directly in the GameEval process and satisfying the same
:class:`~gameeval.agents.open_p2p_agent.OpenP2PTransport` protocol. The action
mapping, artifact writer, judge, and metrics paths are therefore identical to
the sidecar deployment; only the transport changes.

Two constraints of the upstream checkout are handled explicitly:

``elefant_rust``
    Ships Linux shared objects only and is imported at module scope by the
    training data pipeline. It is unused on the inference path, so a guard stub
    is installed that raises if that ever stops being true.

Model placement
    Flex-attention block masks and the rolling KV cache are created during model
    construction rather than registered as buffers, so the model must be built
    inside ``Trainer.init_module()`` on the target device. Building on CPU and
    moving afterwards strands those tensors.

Like the published UDS path this transport is visual-only: the policy receives
pixels and no per-episode instruction, so results must be reported as a
pixel-to-action baseline.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MAX_VIRTUAL_STEPS = 20 * 60 * 60


def install_elefant_rust_stub() -> None:
    """Stub the Linux-only Rust extension so inference imports work on Windows."""
    if "elefant_rust" in sys.modules and getattr(
        sys.modules["elefant_rust"], "_gameeval_stub", False
    ):
        return

    def _unavailable(*_args: Any, **_kwargs: Any):
        raise RuntimeError(
            "elefant_rust is unavailable on Windows. It is only used by the "
            "open-p2p training data pipeline and must not be reached from the "
            "inference path."
        )

    stub = types.ModuleType("elefant_rust")
    stub._gameeval_stub = True
    stub.resize_image = _unavailable

    zmq_stub = types.ModuleType("elefant_rust.zmq_queue")
    zmq_stub.ZMQQueueServer = _unavailable
    zmq_stub.ZMQQueueClient = _unavailable

    dataset_stub = types.ModuleType("elefant_rust.video_proto_dataset")
    dataset_stub.ShuffleThread = _unavailable

    stub.zmq_queue = zmq_stub
    stub.video_proto_dataset = dataset_stub

    sys.modules["elefant_rust"] = stub
    sys.modules["elefant_rust.zmq_queue"] = zmq_stub
    sys.modules["elefant_rust.video_proto_dataset"] = dataset_stub


class InProcessOpenP2PTransport:
    """Run the open-p2p policy inside the GameEval process.

    Parameters
    ----------
    open_p2p_root:
        The open-p2p checkout. Its model code is imported, never vendored.
    config_path, checkpoint_path:
        Defaults resolve to the ``150M`` checkpoint inside the checkout.
    device:
        CUDA device index or name. open-p2p's inference path requires CUDA.
    max_virtual_steps:
        RoPE virtual-index budget before the position counter wraps.
    """

    def __init__(
        self,
        *,
        open_p2p_root: str | Path,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | int = 0,
        max_virtual_steps: int = DEFAULT_MAX_VIRTUAL_STEPS,
        compile_model: bool = False,
        model_size: str = "150M",
    ) -> None:
        self.root = Path(open_p2p_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"open-p2p checkout not found: {self.root}")

        self.config_path = Path(
            config_path or self.root / f"checkpoints/{model_size}/model_config.yaml"
        ).expanduser()
        self.checkpoint_path = Path(
            checkpoint_path
            or self.root / f"checkpoints/{model_size}/checkpoint-step=00500000.ckpt"
        ).expanduser()
        for path in (self.config_path, self.checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(f"Required open-p2p file is missing: {path}")

        self.model_size = str(model_size)
        self.max_virtual_steps = int(max_virtual_steps)
        self.compile_model = bool(compile_model)
        self._device_request = device

        self._state: Any = None
        self._config: Any = None
        self._torch: Any = None
        self.load_seconds = 0.0
        self._load()

    # ---- transport protocol -------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "ok": self._state is not None,
            "backend": "open-p2p-in-process",
            "model_size": self.model_size,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "frame_size": [self.frame_width, self.frame_height],
            "instruction_forwarded": False,
            "load_seconds": round(self.load_seconds, 2),
        }

    def reset(self, context: dict[str, Any]) -> dict[str, Any]:
        """Clear the policy's temporal state so a new episode starts clean."""
        self._state.reset()
        return {
            "ok": True,
            "task_id": context.get("task_id"),
            "instruction_forwarded": False,
            "note": (
                "open-p2p consumes pixels only; the task instruction is recorded "
                "by GameEval but not supplied to the policy."
            ),
        }

    def act(self, frame: np.ndarray, frame_id: int) -> dict[str, Any]:
        """Map one RGB frame to an open-p2p action response."""
        torch = self._torch
        array = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("open-p2p expects an RGB frame shaped (H, W, 3)")
        if array.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError(
                f"open-p2p expects {self.frame_width}x{self.frame_height} frames, "
                f"received {array.shape[1]}x{array.shape[0]}"
            )

        # HWC uint8 -> CHW uint8 on the model device, matching the UDS server.
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            action = self._state.get_action(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        return {
            "frame_id": int(frame_id),
            "keys": [str(key) for key in action.keys],
            "mouse_buttons": [str(button) for button in action.mouse_buttons],
            "mouse_dx": int(action.mouse_delta_x),
            "mouse_dy": int(action.mouse_delta_y),
            "inference_ms": inference_ms,
        }

    # ---- properties ---------------------------------------------------------

    @property
    def frame_width(self) -> int:
        return int(self._config.shared.frame_width)

    @property
    def frame_height(self) -> int:
        return int(self._config.shared.frame_height)

    # ---- internals ----------------------------------------------------------

    def _resolve_device(self, torch: Any):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "open-p2p inference requires a CUDA GPU; none is available."
            )
        request = self._device_request
        if isinstance(request, int):
            return torch.device(f"cuda:{request}")
        text = str(request)
        if text.isdigit():
            return torch.device(f"cuda:{text}")
        device = torch.device(text)
        if device.type != "cuda":
            raise RuntimeError(f"open-p2p requires a CUDA device, got '{device}'")
        return device

    def _load(self) -> None:
        install_elefant_rust_stub()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        import lightning as pl
        import torch

        from elefant.config import load_config
        from elefant.data.action_mapping import UniversalAutoregressiveActionMapping
        from elefant.policy_model.config import LightningPolicyConfig
        from elefant.policy_model.stage3_finetune import Stage3LabelledBCLightning
        from elefant.torch import pytorch_setup

        import elefant.policy_model.inference as p2p_inference

        self._torch = torch
        self.device = self._resolve_device(torch)
        self._config = load_config(str(self.config_path), LightningPolicyConfig)

        pytorch_setup()
        if not self.compile_model:
            torch.compiler.set_stance("force_eager")

        started = time.perf_counter()
        trainer = pl.Trainer(
            precision=self._config.shared.precision,
            accelerator="gpu",
            devices=[self.device.index or 0],
            logger=False,
            enable_checkpointing=False,
        )
        with trainer.init_module():
            model = Stage3LabelledBCLightning.load_from_checkpoint(
                str(self.checkpoint_path),
                config=self._config,
                inference_mode=True,
            )
        model.eval()
        self.load_seconds = time.perf_counter() - started

        # The published UDS request carries a frame and frame id only. Keeping the
        # tokenizer unbuilt makes the visual-only contract explicit and avoids
        # downloading Gemma weights that this baseline never uses.
        original_factory = p2p_inference.get_text_tokenizer
        p2p_inference.get_text_tokenizer = lambda *_a, **_k: None
        try:
            self._state = p2p_inference.KVCacheInferenceState(
                self._config,
                UniversalAutoregressiveActionMapping(
                    config=self._config.shared.action_mapping
                ),
                model,
                max_virtual_steps=self.max_virtual_steps,
            )
        finally:
            p2p_inference.get_text_tokenizer = original_factory

        if self._state.text_tokenizer_model is not None:
            raise RuntimeError(
                "Expected a visual-only open-p2p baseline, but a text tokenizer "
                "was constructed."
            )


__all__ = ["InProcessOpenP2PTransport", "install_elefant_rust_stub"]
