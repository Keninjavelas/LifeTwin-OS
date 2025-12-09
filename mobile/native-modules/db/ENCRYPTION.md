Encrypted DB (optional)

This project supports an optional encrypted Room database using SQLCipher if the SQLCipher runtime dependency is added to the Android build.

How it works

- `AppDatabase.getInstance(context, passphrase)` accepts an optional `passphrase` string. If `passphrase` is non-empty and SQLCipher's `SupportFactory` is available on the classpath at runtime, the database will be opened with SQLCipher.
- The code uses reflection to avoid a hard compile-time dependency; if SQLCipher isn't included the DB will fall back to an unencrypted Room DB.
- Call `DBHelper.initializeEncrypted(context, passphrase)` early (for example in `Application.onCreate`) to attempt an encrypted DB initialization.

Enabling SQLCipher (gradle)

To enable SQLCipher, add a dependency to your Android app module (example using SQLCipher Android 4.x):

```
implementation "net.zetetic:android-database-sqlcipher:4.5.0"
```

and ensure you initialize SQLCipher per their docs if required for your version.

Security notes

- Do NOT hardcode long-term passphrases in source code. Use Android Keystore, user credentials, or a Key Derivation flow.
- When using device-bound keys (Keystore), consider deriving a database passphrase from an asymmetric key or wrapping/unwrapping a randomly generated DB key.
- This repository provides optional wiring; a full secure key management design is required before shipping.
