# Data and Code Availability

## Manuscript-ready English statement

The study dataset was reconstructed from four third-party molecular-glue
resources: MGTbind, MGDB, MolGlueDB and TPDdb. The database versions or
snapshot identities, owner-hosted download pages, exact acquisition endpoints,
access information, citations, filenames, byte sizes and SHA-256 checksums are
provided in the public source manifest accompanying the code. Because the
authors did not receive permission to redistribute the database exports, the
raw records, the harmonized 1,560-record analysis table and row-level
prediction files are not included in the repository. The public software
downloads the source files from the database owners, verifies the frozen
checksums, and deterministically reproduces the local analysis tables, quality
control, splits, CPU analyses, statistical summaries and figures, subject to
the database owners' access conditions. A checksum mismatch is treated as a
new temporal source snapshot rather than as an exact reconstruction. Publicly
redistributable author-generated aggregate source data, protocols, software
environment specifications and figure files are included in the repository.
The final repository URL, immutable release tag and archival DOI will be added
before publication.

Source database records are not offered “on reasonable request” because the
authors do not claim authority to redistribute them. Questions about the
reconstruction workflow may be directed to the corresponding author, Li Wang
(wangli@ntu.edu.cn).

## 中文核对说明（不直接提交）

四库原始导出、1,560 行清洗主表和逐样本预测不公开，也不承诺通过通讯
作者邮件转发。GitHub 公开的是来源/版本/链接/校验值、清洗与分析代码、
环境、聚合证据和图片。第三方使用者从数据库官网自行下载，在各数据库
访问条款允许的范围内本地重建。若官网文件发生变化，则作为新的时间
快照分析，不能声称精确复现本文冻结结果。投稿前只需补 GitHub URL、固定
版本 tag 和归档 DOI，并完成作者自有代码及聚合内容的许可证选择。

