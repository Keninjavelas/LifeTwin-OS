# Security & Encryption Notes (Scaffolding)

- At this stage, summaries are stored in-memory without encryption.
- Future steps:
  - Use per-user/device keys and encrypt summaries before persistence.
  - Derive encryption keys from user secrets and store only ciphertext server-side.
  - Use HTTPS/TLS for all transport.
  - Add signature / MAC fields to detect tampering.
