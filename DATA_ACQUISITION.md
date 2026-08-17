# Third-party data acquisition

Exact reconstruction starts from the owner-hosted routes in
`data_builder/config/source_manifest.json`. As checked on 2026-08-16, the
required frozen artifacts were anonymously retrievable and matched their
declared SHA-256 values, including the MolGlueDB archive available from its
official download page. The manifest, rather than this narrative, is the
machine-readable source of exact URLs, request methods, filenames, sizes,
checksums and citations.

Run the complete acquisition and build only in a private ignored workspace:

```bash
python data_builder/scripts/reproduce.py all \
  --raw-dir data/raw \
  --output-dir data/processed \
  --acknowledge-third-party-notice
```

Access to a public download is not interpreted as permission to redistribute
its contents. Do not upload downloaded files, the rebuilt table or prediction
files. A checksum mismatch stops exact reconstruction; record it as a new
temporal source snapshot instead of replacing the frozen hash silently.

The authors sent redistribution-permission requests but received no response.
No permission claim is made, and the v2 route does not depend on a reply.

