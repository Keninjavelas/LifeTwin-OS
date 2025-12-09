# Long-term Encryption Roadmap

This document outlines a pragmatic, staged plan to add client-side encryption, key management, and server-side handling for LifeTwin OS summaries.

Goals
- Protect user summaries in transit and at rest (server-side) using per-device encryption where server stores ciphertext only.
- Provide an operational mode where server can *optionally* decrypt when given proper wrapped keys for cloud tasks (e.g., model training), but default is end-to-end encrypted storage.
- Keep UX simple: automatic key provisioning, rotation, and recovery options.

Threat model (high level)
- Adversary: remote attacker with access to server storage (data-at-rest), or eavesdropper on network. Not in-scope: physical compromise of user's unlocked device.
- Goal of adversary: read user summaries, tamper with exports, or link exports to users.

High-level design (stages)
1. Stage 1 — Transport & metadata
   - Enforce HTTPS/TLS for all endpoints (already required).
   - Add HMAC signatures for exports to detect tampering.
2. Stage 2 — Server-side ciphertext (simple)
   - Devices encrypt summaries with a per-device symmetric key derived from a device secret stored in platform keystore.
   - Backend stores ciphertext + metadata (device id, algorithm, key version). Server does not have plaintext.
   - Add an export endpoint that returns ciphertext blobs.
3. Stage 3 — Envelope encryption and key wrapping
   - Use an envelope scheme: device generates a random DEK (data encryption key), encrypts payload with DEK, then wraps DEK with a per-device KEK stored in device keystore or derived from user secret.
   - Server stores wrapped DEKs and ciphertext. For cloud processing, an admin process can request unwrapping via an unlocking API or HSM (optional).
4. Stage 4 — Recovery and rotation
   - Implement key rotation: devices re-wrap DEKs with new KEK versions and upload rotation metadata.
   - Provide optional server-side key escrow (encrypted with a service key) for account recovery.

Key management options
- Device-keystore-only: simplest, highest privacy. If device lost, no recovery.
- Server-escrow (wrapped KEK): allow recovery but requires protecting escrow keys (e.g., KMS/HSM).

Implementation plan (next concrete steps)
1. Add server-side schema to store ciphertext blobs and metadata (cipher algorithm, key id, wrapped DEK).
2. Implement a Python helper for envelope encryption using `cryptography` (POC below).
3. Implement Android-side POC to use Android Keystore to protect KEK and wrap DEKs.
4. Add admin tooling for optional unwrap using a KMS-protected service key.

Compliance & privacy notes
- Clearly document retention and provide user controls.
- If enabling server-side decryption for analytics, require user consent.

References
- NIST SP 800-57 (key management), RFC 7518 (JWE), Android Keystore docs.
