# Security and Responsible Use

This document is the security policy and living responsible-use addendum for
the current 3A-VLA public release. It applies to the `main` branch and will be
updated as release artifacts or threat models change.

## Private reporting

Report suspected vulnerabilities, privacy failures, unsafe automation paths,
or platform-security risks privately to **3avla123@gmail.com** with a subject
beginning `[3A-VLA Security]`.

Do not open a public issue for an unmitigated vulnerability. Do not attach raw
frames, replay files, action logs, account identifiers, chat or voice data,
credentials, commercial game assets, or other sensitive material. Describe the
minimum necessary reproduction conditions and wait for a secure transfer method
if supporting material is required.

A useful report includes:

- the affected commit, version, adapter, or release artifact;
- a threat model and expected security or privacy impact;
- controlled reproduction conditions and a minimal proof of concept;
- whether a game platform, account, or third party may be affected; and
- suggested mitigations or defensive signatures, when available.

We will record the report, affected versions, reproduction conditions, impact,
and candidate mitigations; triage it privately; coordinate with affected
platform or trust-and-safety teams when appropriate; and disclose details only
after reasonable mitigation coordination.

## Security scope

Reports are in scope when they concern:

- code execution, credential exposure, unsafe input injection, or artifact
  leakage in GameEval;
- failure of the evaluator-only state boundary or unintended exposure of
  privileged game state to an agent;
- release of personal identifiers, private communications, proprietary assets,
  or data without documented rights;
- paths that materially enable automated aiming, resource farming, economic
  manipulation, bulk-account control, anti-cheat evasion, or social
  impersonation; or
- adaptations that create a credible real-world targeting, tracking,
  surveillance, or weaponization risk.

Good-faith research must use dedicated test servers, research accounts, and
informed participants. Do not test against public matchmaking, production game
economies or back ends, uninformed users, physical systems, or weapon
interfaces. Follow the [3A-VLA Research Use License](LICENSE).

## Release boundaries

The public release does not include source replays, unprocessed frames, raw
action logs, account identifiers, chat or voice data, commercial game assets,
or text-chat, voice, and emote modules. Released tools do not bundle commercial
game data or assets, and reproducers must obtain lawful access independently.

No fine-tuned checkpoint is currently released. Any future checkpoint release
will be delayed for at least 90 days after publication and will require privacy,
intellectual-property, and security review. No checkpoint will be released
until this reporting process and the applicable responsible-use controls are
operational.

During that delay, verified platform trust-and-safety teams may request
defensive signatures, benchmarks, or controlled testing. Raw data and
unrestricted access will not be provided.

## Out of scope

Requests for unrestricted checkpoint access, offensive testing against live
services, public exploit demonstrations, evasion assistance, and reports based
on unlawful access or non-consensual data collection are outside this process.
Ordinary bugs without a security, privacy, or safety impact may be reported
through the repository's public issue tracker.
