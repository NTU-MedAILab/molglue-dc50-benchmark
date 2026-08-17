# Historical test snapshot

These tests are preserved as development provenance. They expect the original
project directory layout and, for strict OOD tests, the restricted analysis
table and historical result directories. They are not used as the public
release acceptance gate.

The clean-room gate is:

```bash
python scripts/reproduce_release_v1.py verify
```

It uses only the Python standard library and is independent of pytest.

