# Third-party data notice

## What this repository may contain

The public repository may contain author-written source code, source URLs,
database citations, snapshot dates, filenames, cryptographic checksums,
schemas, aggregate row counts and reproducibility reports that contain no
row-level source content.

## What this repository excludes

Raw exports from MGTbind, MGDB, MolGlueDB and TPDdb and the harmonized
row-level CSV derivatives are excluded pending database-content redistribution
permission. The `.gitignore` rules enforce this boundary for the default
locations, but contributors remain responsible for checking every commit.

## Source-specific status (checked 2026-08-16)

| Source | Official acquisition route | Database-content redistribution status |
|---|---|---|
| MGTbind | <https://mgtbind.pkumdl.cn/download> | No separate affirmative database licence identified |
| MGDB | <http://mgdb.idruglab.cn/download> | Site states “All Rights Reserved”; no separate affirmative database licence identified |
| MolGlueDB | <https://www.molgluedb.com/download> | Official export is reproducible, but the site states “All rights reserved” and no separate affirmative database licence was identified |
| TPDdb | <https://tpddb.idrblab.net/download> | No separate affirmative database licence identified |

The open-access licence of a database article does not automatically license a
separately hosted database export. This is a conservative research-data
workflow, not legal advice.

## Reproduction without redistribution

Each researcher downloads the snapshots from the database owners' endpoints
into a local ignored directory. The software then verifies hashes and creates
local derivatives. If a URL or checksum changes, open an issue containing only
the new URL, access date, response metadata and checksum; do not attach the
database export.

“Available from the corresponding author on reasonable request” must not be
promised unless the authors actually have authority to redistribute the
requested rows. A safer manuscript statement is that the public code and
source manifest permit reconstruction from owner-hosted files, subject to the
owners' terms, and that access questions may be directed to the corresponding
author.

## Required database citations

- Zhu J et al. *MGTbind: a comprehensive database of molecular glue ternary
  interactome*. Nucleic Acids Research 54, D1500–D1509 (2026).
  <https://doi.org/10.1093/nar/gkaf1075>
- Li C et al. *MGDB: a curated database for molecular glues*. Nucleic Acids
  Research 54, D1488–D1499 (2026).
  <https://doi.org/10.1093/nar/gkaf1131>
- Wang X et al. *MolGlueDB: an online database of molecular glues*. Nucleic
  Acids Research 54, D1510–D1518 (2026).
  <https://doi.org/10.1093/nar/gkaf811>
- Qin X et al. *TPDdb: the comprehensive database of targeted protein
  degrader*. Nucleic Acids Research 54, D1683–D1691 (2026).
  <https://doi.org/10.1093/nar/gkaf996>

