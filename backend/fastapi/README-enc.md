Encryption development / optional dependencies

To run the encryption POC and the SQLAlchemy-backed encrypted store tests, install the development requirements:

```bash
pip install -r requirements-dev.txt
```

If you plan to run the `backend/fastapi/security/crypto.py` helpers you will also need `cryptography`:

```bash
pip install cryptography
```
