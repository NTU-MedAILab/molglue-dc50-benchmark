"""Deterministic context standardization separated from model fitting."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


CONTEXT_COLUMNS = [
    "source_database",
    "recruiting_protein",
    "target_protein",
    "cell_line",
    "assay_method",
    "activity_time",
    "mode_of_action",
]


def clean_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a", "null"}:
        return "NA"
    return re.sub(r"\s+", " ", text)


def unique_join(parts: list[str], sep: str = "+") -> str:
    cleaned = [part for part in parts if part and part != "NA"]
    if not cleaned:
        return "NA"
    return sep.join(sorted(dict.fromkeys(cleaned)))


def normalize_source(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "NA"
    parts = [part.strip() for part in re.split(r"[|+/]", text) if part.strip()]
    mapped: list[str] = []
    for part in parts:
        low = part.lower()
        if "mgtbind" in low:
            mapped.append("MGTbind")
        elif "molgluedb" in low:
            mapped.append("MolGlueDB")
        elif "mgdb" in low:
            mapped.append("MGDB")
        elif "tpddb" in low:
            mapped.append("TPDdb")
        else:
            mapped.append(part)
    return unique_join(mapped)


def normalize_recruiter(value: Any) -> str:
    text = clean_token(value)
    upper = text.upper().replace("-", "_")
    if text == "NA":
        return "NA"
    if "CRBN" in upper:
        return "CRBN"
    if "VHL" in upper:
        return "VHL"
    if "DCAF16" in upper:
        return "DCAF16"
    if "DCAF15" in upper:
        return "DCAF15"
    if "DCAF11" in upper:
        return "DCAF11"
    if "RNF126" in upper:
        return "RNF126"
    if "RNF127" in upper:
        return "RNF127"
    if "MDM2" in upper:
        return "MDM2"
    if "FBXO22" in upper:
        return "FBXO22"
    if "SIAH1" in upper:
        return "SIAH1"
    if "DDB1" == upper or upper.endswith("_DDB1"):
        return "DDB1"
    return text


def normalize_target(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "NA"
    upper = text.upper()
    upper = upper.replace("α", "ALPHA").replace("Α", "ALPHA")
    upper = upper.replace("（", "(").replace("）", ")")
    upper = upper.replace("_", "-")

    parts: list[str] = []
    for token in ["IKZF1", "IKZF2", "IKZF3", "IKZF4"]:
        if re.search(rf"\b{token}\b", upper):
            parts.append(token)
    if "HELIOS" in upper:
        parts.append("IKZF2")
    if "CK1" in upper or "CSNK1A1" in upper:
        parts.append("CSNK1A1")
    if "CDK12" in upper or "CCNK" in upper or "CYCLIN K" in upper:
        if "CDK12" in upper:
            parts.append("CDK12")
        if "CCNK" in upper or "CYCLIN K" in upper:
            parts.append("CCNK")
    if "GSPT1" in upper or "ERF3A" in upper:
        parts.append("GSPT1")
    if "GSPT2" in upper:
        parts.append("GSPT2")
    if "WIZ" in upper:
        parts.append("WIZ")
    if "PLZF" in upper or "ZBTB16" in upper:
        parts.append("ZBTB16")
    if "BCL6" in upper:
        parts.append("BCL6")
    if "BCL-XL" in upper or "BCLXL" in upper:
        parts.append("BCL2L1")

    direct_targets = [
        "VAV1", "CDK2", "RIOK2", "IRAK4", "WEE1", "NEK7", "BRD4", "BRD2",
        "BRD9", "BRDT", "GEMIN3", "SMARCA2", "SMARCA4", "CDO1", "SALL4",
        "RBM39", "LIN28A", "LIN28B", "MDM2", "BTK", "IDO1", "CDK4", "CDK6",
        "KRAS", "TP53", "ALK", "ER", "AR",
    ]
    for token in direct_targets:
        if re.search(rf"\b{token}\b", upper):
            parts.append(token)
    if not parts and "LIN28" in upper:
        parts.append("LIN28")
    if parts:
        return unique_join(parts)
    return re.sub(r"\s+", " ", text).strip()


def normalize_cell_line(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "unspecified"
    low = text.lower()
    compact = re.sub(r"[^a-z0-9+;]", "", low)
    if "unspecified" in low:
        return "unspecified"
    if "jurkat" in low:
        return "Jurkat"
    if "hek293t" in compact or "293t" in compact or "hek-293t" in low:
        return "HEK293T"
    if "hek293" in compact:
        return "HEK293"
    if "ht1080" in compact or "ht-1080" in low:
        return "HT1080"
    if "molt4" in compact or "molt-4" in low or "motl4" in compact:
        return "MOLT4"
    if "rs4" in compact:
        return "RS4-11"
    if "mv4" in compact:
        return "MV4-11"
    if "u937" in compact:
        return "U937"
    if "nb4" in compact or "nb-4" in low:
        return "NB4"
    if "ocily10" in compact or "oci-ly10" in low:
        return "OCI-Ly10"
    if "ocily3" in compact or "oci-ly3" in low:
        return "OCI-Ly3"
    if "a549" in compact:
        return "A549"
    if "k562" in compact:
        return "K562"
    if "hct116" in compact:
        return "HCT116"
    if "tmd8" in compact or "tmd-8" in low:
        return "TMD8"
    if "be2c" in compact:
        return "BE(2)-C"
    if "wholeblood" in compact:
        return "whole_blood"
    if "pbmc" in compact:
        return "PBMC"
    if "monocyte" in low:
        return "monocytes"
    if "treg" in low or "regulatory t" in low:
        return "Treg"
    if "cd8" in low:
        return "CD8_T_cells"
    stripped = re.sub(r"\b(cells?|cell lines?|stable|assays?)\b", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip(" -")
    return stripped or "unspecified"


def normalize_assay(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "NA"
    parts: list[str] = []
    for raw in text.split("|"):
        low = raw.lower()
        if "mgtbind" in low:
            parts.append("dc50_assay")
        elif "primarytargetdeginfo" in low or "secondarytargetdeginfo" in low:
            parts.append("target_degradation_text")
        elif "tpddb" in low:
            parts.append("activity_table")
        elif "nanobret" in low:
            parts.append("NanoBRET")
        elif "hibit" in low:
            parts.append("HiBiT")
        elif "western" in low or "immunoblot" in low:
            parts.append("western_blot")
        elif "msd" in low or "mesoscale" in low:
            parts.append("MSD")
        elif "celltiter" in low or "ctg" in low or "viability" in low:
            parts.append("viability")
        elif "cellular" in low or "engineered" in low:
            parts.append("cellular_degradation")
        else:
            parts.append(clean_token(raw))
    return unique_join(parts)


def normalize_time(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "NA"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour)", text.lower())
    if not match:
        return text
    number = float(match.group(1))
    if abs(number - round(number)) < 1e-6:
        return f"{int(round(number))}h"
    return f"{number:g}h"


def normalize_mode(value: Any) -> str:
    text = clean_token(value)
    if text == "NA":
        return "NA"
    parts: list[str] = []
    for raw in text.split("|"):
        low = raw.strip().lower()
        if low == "degrader":
            parts.append("degrader")
        elif "heterodimerization" in low:
            parts.append("heterodimerization-degradative")
        elif "non-covalent" in low:
            parts.append("non-covalent")
        elif "covalent" in low:
            parts.append("covalent")
        elif "polymerizer" in low:
            parts.append("polymerizer")
        else:
            parts.append(clean_token(raw))
    return unique_join(parts, sep="|")


def standardize_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = rows.copy()
    for column in CONTEXT_COLUMNS:
        if column in out.columns:
            out[f"raw_{column}"] = out[column]
    out["source_database"] = out["source_database"].map(normalize_source)
    out["recruiting_protein"] = out["recruiting_protein"].map(normalize_recruiter)
    out["target_protein"] = out["target_protein"].map(normalize_target)
    out["cell_line"] = out["cell_line"].map(normalize_cell_line)
    out["assay_method"] = out["assay_method"].map(normalize_assay)
    out["activity_time"] = out["activity_time"].map(normalize_time)
    out["mode_of_action"] = out["mode_of_action"].map(normalize_mode)
    if "species" not in out.columns:
        out["species"] = "unspecified"
    summary: dict[str, Any] = {}
    for column in CONTEXT_COLUMNS:
        raw_column = f"raw_{column}"
        if raw_column in out.columns:
            summary[column] = {
                "raw_unique": int(rows[column].fillna("NA").astype(str).nunique()),
                "standardized_unique": int(out[column].fillna("NA").astype(str).nunique()),
                "changed_rows": int(
                    (rows[column].fillna("NA").astype(str) != out[column].fillna("NA").astype(str)).sum()
                ),
            }
    return out, summary

