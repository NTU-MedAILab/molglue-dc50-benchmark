"""Parse, harmonize, quality-control, split, and standardize four source databases."""

from __future__ import annotations

import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupShuffleSplit

from .context import standardize_rows
from .io import sha256_file


RDLogger.DisableLog("rdApp.*")

RANDOM_STATE = 260531
OUTPUT_FILENAMES = {
    "parsed": "all_molglue_dc50_parsed_records.csv",
    "strict": "all_molglue_dc50_strict_exact_records.csv",
    "qc": "all_molglue_dc50_qc_train_test.csv",
    "standardized": "all_molglue_dc50_qc_train_test_standardized_context.csv",
    "train": "all_molglue_dc50_train.csv",
    "test": "all_molglue_dc50_test.csv",
}
RAW_FILENAMES = {
    "molgluedb": "MolGlueDB_full.csv",
    "mgtbind_compounds": "mgtbind_compounds_260531.csv",
    "mgtbind_complexes": "mgtbind_complexes_260531.csv",
    "mgtbind_citations": "mgtbind_citations_260531.csv",
    "tpddb_main": "tpddb_MG_main_table_260531.txt",
    "tpddb_activity": "tpddb_MG_activity_260531.txt",
    "mgdb_compounds": "mgdb_MG_Compound_260531.csv",
    "mgdb_activity": "mgdb_Activity_Data_260531.csv",
}

DC50_PATTERN = re.compile(
    r"(?i)DC\s*50(?:\s*/\s*Dmax)?[^0-9<>]{0,100}"
    r"(?P<op><=|>=|<|>|\u2264|\u2265|~|\u2248)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:\+/-|\u00b1)?\s*\d*(?:\.\d+)?\s*"
    r"(?P<unit>pM|nM|uM|\u03bcM|\u00b5M|mM|M)"
)
ACTIVITY_VALUE_PATTERN = re.compile(
    r"(?P<op><=|>=|<|>|=|\u2264|\u2265|~|\u2248)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>pM|nM|uM|\u03bcM|\u00b5M|mM|M)?",
    re.IGNORECASE,
)
RANGE_PATTERN = re.compile(
    r"(?P<lo>\d+(?:\.\d+)?)\s*(?P<unit1>pM|nM|uM|\u03bcM|\u00b5M|mM|M)?\s*"
    r"(?:-|to|<\s*x\s*(?:<=|<)|\u2264\s*x\s*(?:\u2264|<))\s*"
    r"(?P<hi>\d+(?:\.\d+)?)\s*(?P<unit2>pM|nM|uM|\u03bcM|\u00b5M|mM|M)?",
    re.IGNORECASE,
)
TARGET_SPLIT_PATTERN = re.compile(r"\s*[;,/]\s*")
REJECT_CONTEXT_PREFIXES = {"assay", "cell", "dc50", "dmax", "western", "protein", "level", "levels"}
PLACEHOLDER_PATTERN = re.compile(r"(?i)^(|\.|na|nan|n/a|null|none|unknown|unspecified)$")
RECRUITER_KEYWORDS = re.compile(
    r"(?i)\b("
    r"CRBN|CEREBLON|DDB1|DCAF\d*|VHL|MDM2|SIAH\d*|RNF\d*|BTRC|BTRCP|"
    r"UBE2D\d*|ZFP91|CUL4|FBXO\d*|KLHDC\d*|TRIM\d*|PARKIN|E3|LIGASE|"
    r"VON HIPPEL|DDB1-AND-CUL4|DDB1 AND CUL4"
    r")\b"
)
KNOWN_RECRUITERS = [
    "CRBN", "DDB1", "DCAF15", "DCAF16", "VHL", "MDM2", "SIAH1", "SIAH2",
    "RNF114", "RNF126", "BTRC", "BTRCP", "UBE2D1", "UBE2D2", "ZFP91", "CUL4",
]


def raw_files(raw_dir: Path) -> dict[str, Path]:
    return {key: raw_dir / filename for key, filename in RAW_FILENAMES.items()}


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def is_placeholder(value: object) -> bool:
    return bool(PLACEHOLDER_PATTERN.fullmatch(normalize_text(value)))


def normalize_category(value: object) -> str:
    text = re.sub(r"\s+", " ", normalize_text(value)).strip(" ;,")
    return "unspecified" if is_placeholder(text) else text


def simple_key(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value).upper())


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = normalize_text(value)
        if text and not is_placeholder(text):
            return text
    return ""


def join_nonempty(values: Iterable[object]) -> str:
    return "|".join(sorted({normalize_text(value) for value in values if normalize_text(value)}))


def doi_to_url(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    urls: list[str] = []
    for token in re.split(r"\s*[;|]\s*", text):
        token = token.strip()
        if not token:
            continue
        if token.startswith(("http://", "https://")):
            urls.append(token)
        elif token.startswith("10."):
            urls.append(f"https://doi.org/{token}")
        else:
            urls.append(token)
    return "|".join(urls)


def unit_to_nm(value: float, unit: str) -> float:
    unit_norm = unit.strip().replace("\u03bc", "u").replace("\u00b5", "u").lower()
    factors = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "mm": 1e6, "m": 1e9}
    if unit_norm not in factors:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    return value * factors[unit_norm]


def canonicalize_smiles(smiles: object) -> tuple[str | None, str | None]:
    text = normalize_text(smiles)
    if not text:
        return None, "missing_smiles"
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None, "invalid_smiles"
    return Chem.MolToSmiles(molecule, isomericSmiles=True), ""


def parse_activity_value(
    value: object, unit_hint: object = "", operator_hint: object = ""
) -> tuple[str, float | None, str, float | None, str]:
    text = normalize_text(value)
    unit_hint_text = normalize_text(unit_hint)
    operator_text = normalize_text(operator_hint)
    if not text:
        return "", None, unit_hint_text, None, "missing_value"
    if "%" in text or "%" in unit_hint_text:
        return "", None, unit_hint_text, None, "percent_value"
    normalized = (
        text.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2248", "~")
        .replace("\u03bc", "u").replace("\u00b5", "u").replace(",", "")
    )
    normalized = re.sub(r"^>>+", ">", normalized)
    range_match = RANGE_PATTERN.search(normalized)
    if range_match:
        unit = range_match.group("unit2") or range_match.group("unit1") or unit_hint_text
        if not unit:
            return "range", None, "", None, "range_missing_unit"
        low = unit_to_nm(float(range_match.group("lo")), range_match.group("unit1") or unit)
        high = unit_to_nm(float(range_match.group("hi")), unit)
        midpoint = (low + high) / 2.0
        return "range", midpoint, "nM", midpoint, "range_midpoint_excluded"
    match = ACTIVITY_VALUE_PATTERN.search(normalized)
    if not match:
        return "", None, unit_hint_text, None, "unparsed_value"
    unit = match.group("unit") or unit_hint_text
    if not unit:
        return "", None, "", None, "missing_unit"
    raw_op = (match.group("op") or operator_text or "=").replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2248", "~")
    relation = "=" if raw_op in {"", "="} else raw_op
    if relation.startswith("=") and ">" in relation:
        relation = ">"
    elif relation.startswith("="):
        relation = "="
    if relation not in {"=", "<", ">", "<=", ">=", "~"}:
        relation = "=" if "=" in relation and "<" not in relation and ">" not in relation else relation
    raw_value = float(match.group("value"))
    return relation, raw_value, unit, unit_to_nm(raw_value, unit), ""


def pdc50(dc50_nm: float | None) -> float:
    if dc50_nm is None or dc50_nm <= 0:
        return float("nan")
    return 9.0 - math.log10(dc50_nm)


def extract_cell_line_from_text(text: str, match_start: int) -> str:
    segment = re.split(r"[\n,]", text[:match_start])[-1].strip()
    cell_match = re.search(r"(?i)\bCell\s*:\s*([^;\n,]+)", segment)
    if cell_match:
        return cell_match.group(1).strip()
    if ";" in segment:
        first_token = segment.split(";")[0].strip()
        if first_token and ":" not in first_token and len(first_token) <= 60:
            return first_token
    return "unspecified"


def field_targets(target_value: object) -> list[str]:
    text = normalize_text(target_value)
    return [item.strip() for item in TARGET_SPLIT_PATTERN.split(text) if item.strip()] if text else []


def extract_context_target(text: str, match_start: int, default_target: object) -> tuple[str, str]:
    targets = field_targets(default_target)
    fallback = ";".join(targets) if targets else ""
    segment = re.split(r"[\n,]", text[:match_start])[-1]
    after_semicolon = segment.split(";")[-1].strip(" :()[]")
    if not after_semicolon:
        return fallback, "field"
    first_word = after_semicolon.split()[0].strip(" :()[]")
    if first_word.lower() in REJECT_CONTEXT_PREFIXES:
        return fallback, "field"
    if targets and first_word in targets:
        return first_word, "context_matched_field"
    if re.match(r"^(?=.*[A-Za-z])[A-Za-z0-9._()+-]{2,30}$", first_word):
        return first_word, "context"
    return fallback, "field"


def base_record(
    *, record_id: str, source_database: str, source_id: object, molecular_glue_name: object,
    smiles: object, recruiting_protein: object, target_protein: object, cell_line: object,
    dc50_relation: str, dc50_raw_value: float | None, dc50_raw_unit: object,
    dc50_nM: float | None, raw_activity_text: object, source_url: object = "",
    target_uniprot: object = "", recruiting_protein_uniprot: object = "",
    target_assignment: object = "", activity_time: object = "", assay_method: object = "",
    mode_of_action: object = "", source_note: object = "",
) -> dict[str, object]:
    canonical_smiles, structure_error = canonicalize_smiles(smiles)
    return {
        "record_id": record_id, "source_database": source_database,
        "source_id": normalize_text(source_id), "molecular_glue_name": normalize_text(molecular_glue_name),
        "smiles": normalize_text(smiles), "canonical_smiles": canonical_smiles,
        "structure_qc_error": structure_error, "recruiting_protein": normalize_text(recruiting_protein),
        "recruiting_protein_uniprot": normalize_text(recruiting_protein_uniprot),
        "target_protein": normalize_text(target_protein), "target_uniprot": normalize_text(target_uniprot),
        "target_assignment": normalize_text(target_assignment),
        "cell_line": normalize_text(cell_line) or "unspecified", "activity_time": normalize_text(activity_time),
        "dc50_relation": dc50_relation, "dc50_raw_value": dc50_raw_value,
        "dc50_raw_unit": normalize_text(dc50_raw_unit), "dc50_nM": dc50_nM, "pDC50": pdc50(dc50_nM),
        "assay_method": normalize_text(assay_method), "mode_of_action": normalize_text(mode_of_action),
        "raw_activity_text": normalize_text(raw_activity_text), "source_url": normalize_text(source_url),
        "source_note": normalize_text(source_note),
    }


def parse_molgluedb(files: dict[str, Path]) -> pd.DataFrame:
    raw = pd.read_csv(files["molgluedb"])
    records: list[dict[str, object]] = []
    source_columns = [
        ("PrimaryTargetDegInfo", "PrimaryTarget", "primary"),
        ("SecondaryTargetDegInfo", "SecondaryTarget", "secondary"),
    ]
    for _, row in raw.iterrows():
        for info_column, target_column, target_role in source_columns:
            info_text = normalize_text(row.get(info_column))
            if not info_text:
                continue
            for match_index, match in enumerate(DC50_PATTERN.finditer(info_text), start=1):
                relation = (match.group("op") or "=").replace("\u2264", "<=").replace("\u2265", ">=")
                relation = "=" if relation in {"=", "~", "\u2248"} else relation
                raw_value = float(match.group("value"))
                unit = match.group("unit")
                target_protein, target_assignment = extract_context_target(info_text, match.start(), row.get(target_column))
                records.append(base_record(
                    record_id=f"MolGlueDB_{int(row['DATAID']):05d}_{info_column}_{match_index}",
                    source_database="MolGlueDB", source_id=row.get("DATAID"), molecular_glue_name=row.get("Name"),
                    smiles=row.get("SMILES"), recruiting_protein=row.get("RecruitingProtein"),
                    recruiting_protein_uniprot=row.get("RecruitingProtein_UniProtID"), target_protein=target_protein,
                    cell_line=extract_cell_line_from_text(info_text, match.start()), dc50_relation=relation,
                    dc50_raw_value=raw_value, dc50_raw_unit=unit, dc50_nM=unit_to_nm(raw_value, unit),
                    raw_activity_text=info_text, source_url=row.get("SourceAddress_Website"),
                    target_assignment=f"{target_role}:{target_assignment}", assay_method=info_column,
                    mode_of_action=row.get("ModeOfAction"),
                ))
    return pd.DataFrame(records)


def citation_urls(citations: pd.DataFrame) -> dict[str, str]:
    if citations.empty or "id" not in citations:
        return {}
    mapping: dict[str, str] = {}
    for _, row in citations.iterrows():
        ref_id = normalize_text(row.get("id"))
        if not ref_id:
            continue
        url = doi_to_url(row.get("doi"))
        if not url:
            pubmed = normalize_text(row.get("pubmed_id"))
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed}/" if pubmed else ""
        mapping[ref_id] = url
    return mapping


def mgtbind_ref_urls(ref_value: object, ref_map: dict[str, str]) -> str:
    urls = [ref_map[token] for token in re.split(r"[_;|,]\s*", normalize_text(ref_value)) if token.strip() and ref_map.get(token.strip())]
    return "|".join(sorted(set(urls)))


def split_underscore_values(value: object) -> list[str]:
    text = normalize_text(value)
    return [part.strip() for part in text.split("_") if part.strip()] if text else []


def mgtbind_target_recruiter(row: pd.Series) -> tuple[str, str, str]:
    protein_a, protein_b = normalize_text(row.get("protein_a_name")), normalize_text(row.get("protein_b_name"))
    label = normalize_text(row.get("target_protein")).upper()
    if label == "A":
        return protein_a, protein_b, "target_label_A"
    if label == "B":
        return protein_b, protein_a, "target_label_B"
    target = normalize_text(row.get("target_protein"))
    if not target or is_placeholder(target):
        if RECRUITER_KEYWORDS.search(protein_a) and not RECRUITER_KEYWORDS.search(protein_b):
            return protein_b, protein_a, "recruiter_keyword_A"
        if RECRUITER_KEYWORDS.search(protein_b) and not RECRUITER_KEYWORDS.search(protein_a):
            return protein_a, protein_b, "recruiter_keyword_B"
        return protein_b, protein_a, "fallback_B_target"
    if simple_key(target) == simple_key(protein_a):
        return protein_a, protein_b, "target_name_A"
    if simple_key(target) == simple_key(protein_b):
        return protein_b, protein_a, "target_name_B"
    recruiter = protein_a if RECRUITER_KEYWORDS.search(protein_a) else protein_b
    return target, recruiter, "target_field"


def parse_mgtbind(files: dict[str, Path]) -> pd.DataFrame:
    compounds, complexes = pd.read_csv(files["mgtbind_compounds"]), pd.read_csv(files["mgtbind_complexes"])
    citations = pd.read_csv(files["mgtbind_citations"])
    ref_map = citation_urls(citations)
    merged = complexes.merge(
        compounds[["id", "name", "canonical_smiles", "inchi_key"]], left_on="compound_id", right_on="id",
        how="left", suffixes=("", "_compound"),
    )
    records: list[dict[str, object]] = []
    for _, row in merged[merged["dc50"].notna()].iterrows():
        target, recruiter, assignment = mgtbind_target_recruiter(row)
        values = split_underscore_values(row.get("dc50")) or [normalize_text(row.get("dc50"))]
        operators, cell_lines, errors = (split_underscore_values(row.get(key)) for key in ["dc50_operator", "dc50_cell_line", "dc50_error"])
        for index, value in enumerate(values, start=1):
            operator = operators[index - 1] if index <= len(operators) else first_nonempty(operators)
            cell_line = cell_lines[index - 1] if index <= len(cell_lines) else row.get("dc50_cell_line")
            relation, raw_value, raw_unit, dc50_nm, note = parse_activity_value(value, unit_hint="nM", operator_hint=operator)
            target_is_a = assignment.endswith("_A") or assignment in {"target_label_A", "target_name_A"}
            records.append(base_record(
                record_id=f"MGTbind_{int(row['id']):05d}_{index}", source_database="MGTbind",
                source_id=row.get("id"), molecular_glue_name=row.get("name"), smiles=row.get("canonical_smiles"),
                recruiting_protein=recruiter,
                recruiting_protein_uniprot=row.get("protein_b_uniprot_id") if target_is_a else row.get("protein_a_uniprot_id"),
                target_protein=target, target_uniprot=row.get("protein_a_uniprot_id") if target_is_a else row.get("protein_b_uniprot_id"),
                cell_line=cell_line, dc50_relation=relation, dc50_raw_value=raw_value, dc50_raw_unit=raw_unit,
                dc50_nM=dc50_nm, raw_activity_text=f"dc50={normalize_text(row.get('dc50'))}",
                source_url=mgtbind_ref_urls(row.get("ref_id"), ref_map), target_assignment=assignment,
                assay_method="MGTbind dc50", mode_of_action=row.get("moa_type"),
                source_note=note or (errors[index - 1] if index <= len(errors) else ""),
            ))
    return pd.DataFrame(records)


def parse_tpddb(files: dict[str, Path]) -> pd.DataFrame:
    main, activity = pd.read_csv(files["tpddb_main"], sep="\t"), pd.read_csv(files["tpddb_activity"], sep="\t")
    dc50 = activity[activity["Activity Type"].astype(str).str.upper().eq("DC50")].copy()
    merged = dc50.merge(main, on="TPD ID", how="left", suffixes=("_activity", "_main"))
    records: list[dict[str, object]] = []
    for index, row in merged.iterrows():
        relation, raw_value, raw_unit, dc50_nm, note = parse_activity_value(row.get("Activity"))
        records.append(base_record(
            record_id=f"TPDdb_{normalize_text(row.get('TPD ID'))}_{index + 1}", source_database="TPDdb",
            source_id=row.get("TPD ID"), molecular_glue_name=row.get("TPD NAME"), smiles=row.get("SMILES"),
            recruiting_protein=row.get("Ligase"), target_protein=first_nonempty([row.get("Target Symbols"), row.get("Target Symbol")]),
            target_uniprot=first_nonempty([row.get("Target IDs"), row.get("Target ID")]), cell_line=row.get("Cell Line"),
            dc50_relation=relation, dc50_raw_value=raw_value, dc50_raw_unit=raw_unit, dc50_nM=dc50_nm,
            raw_activity_text=row.get("Activity"), source_url=doi_to_url(row.get("Source")),
            target_assignment="activity_target_symbols", assay_method="TPDdb Activity", mode_of_action=row.get("Subtype"),
            source_note=note,
        ))
    return pd.DataFrame(records)


def parse_mgdb_compounds(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1, low_memory=False)


def choose_mgdb_recruiter(row: pd.Series, target: str) -> tuple[str, str]:
    text_blob = " ".join(normalize_text(row.get(column)) for column in ["Assay", "Description", "Target", "Target name", "Gene Name", "E3 ligase", "E3 Gene Name"])
    target_key = simple_key(target)
    candidates = [row.get("E3 Gene Name"), row.get("Gene Name"), row.get("E3 ligase"), row.get("Target name")]
    for candidate in candidates:
        candidate_text = normalize_text(candidate)
        if candidate_text and not is_placeholder(candidate_text) and simple_key(candidate_text) != target_key and RECRUITER_KEYWORDS.search(candidate_text):
            return candidate_text, "known_recruiter_column"
    for symbol in KNOWN_RECRUITERS:
        if re.search(rf"(?i)\b{re.escape(symbol)}\b", text_blob) and simple_key(symbol) != target_key:
            return symbol, "known_recruiter_text"
    for candidate in candidates:
        candidate_text = normalize_text(candidate)
        if candidate_text and not is_placeholder(candidate_text) and simple_key(candidate_text) != target_key:
            return candidate_text, "fallback_column"
    return "", "missing_recruiter"


def clean_mgdb_target(target: object, assay_text: object) -> str:
    text, assay = normalize_text(target), normalize_text(assay_text)
    if not text:
        return ""
    slash_match = re.search(r"(?i)\b(?:CRBN|DDB1|DCAF\d*|VHL|MDM2|SIAH\d*|RNF\d*)\s*/\s*([A-Za-z0-9_-]{2,30})", assay)
    if slash_match and RECRUITER_KEYWORDS.search(text):
        return slash_match.group(1)
    parenthetical = re.search(r"\(([^)A-Za-z]*[A-Z0-9]{2,12}[^)]*)\)", text)
    if parenthetical:
        return parenthetical.group(1).strip()
    return text.split(" - ", 1)[0].strip() if " - " in text else text


def parse_mgdb(files: dict[str, Path]) -> pd.DataFrame:
    compounds = parse_mgdb_compounds(files["mgdb_compounds"])
    activity = pd.read_csv(files["mgdb_activity"], low_memory=False)
    dc50 = activity[activity["Efficacy Data"].astype(str).str.upper().eq("DC50")].copy()
    compound_columns = ["ID", "Name", "Function", "Type", "Smiles", "Target name", "Gene Name", "Uniprot", "E3 ligase", "E3 Gene Name", "Effector Uniprot"]
    merged = dc50.merge(compounds[[column for column in compound_columns if column in compounds.columns]], on="ID", how="left", suffixes=("_activity", "_compound"))
    records: list[dict[str, object]] = []
    for index, row in merged.iterrows():
        relation, raw_value, raw_unit, dc50_nm, note = parse_activity_value(row.get("Result"), unit_hint=row.get("Units"))
        target = clean_mgdb_target(row.get("Target"), row.get("Assay"))
        if not target:
            target = first_nonempty([row.get("Gene Name"), row.get("Target name")])
        recruiter, recruiter_assignment = choose_mgdb_recruiter(row, target)
        records.append(base_record(
            record_id=f"MGDB_{normalize_text(row.get('ID'))}_{index + 1}", source_database="MGDB", source_id=row.get("ID"),
            molecular_glue_name=first_nonempty([row.get("name"), row.get("Name")]), smiles=row.get("Smiles"),
            recruiting_protein=recruiter, recruiting_protein_uniprot=row.get("Effector Uniprot"), target_protein=target,
            target_uniprot=row.get("Uniprot"), cell_line=row.get("Model"), activity_time=row.get("Administration Time"),
            dc50_relation=relation, dc50_raw_value=raw_value, dc50_raw_unit=raw_unit, dc50_nM=dc50_nm,
            raw_activity_text=first_nonempty([row.get("Assay"), row.get("Description")]), source_url=doi_to_url(row.get("DOI")),
            target_assignment=f"activity_target:{recruiter_assignment}", assay_method=row.get("Assay Method"),
            mode_of_action=row.get("Function"), source_note=note,
        ))
    return pd.DataFrame(records)


def collect_all_sources(files: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    named_frames = {
        "MolGlueDB": parse_molgluedb(files), "MGTbind": parse_mgtbind(files),
        "TPDdb": parse_tpddb(files), "MGDB": parse_mgdb(files),
    }
    parser_counts = {name: int(len(frame)) for name, frame in named_frames.items()}
    parsed = pd.concat([frame for frame in named_frames.values() if not frame.empty], ignore_index=True, sort=False)
    parsed["dc50_nM"] = pd.to_numeric(parsed["dc50_nM"], errors="coerce")
    parsed["pDC50"] = pd.to_numeric(parsed["pDC50"], errors="coerce")
    return parsed, parser_counts


def same_protein(left: object, right: object) -> bool:
    left_key, right_key = simple_key(left), simple_key(right)
    return bool(left_key and right_key and left_key == right_key)


def build_qc_dataset(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    strict = parsed.copy()
    for column in ["recruiting_protein", "target_protein", "cell_line", "source_database"]:
        strict[column] = strict[column].apply(normalize_category)
    audit = {"parsed_rows": int(len(strict))}
    strict = strict[
        (strict["dc50_relation"] == "=") & strict["dc50_nM"].notna() & (strict["dc50_nM"] > 0)
        & strict["canonical_smiles"].notna()
        & ~strict["recruiting_protein"].str.fullmatch("(?i)unknown|unspecified|\\.")
        & ~strict["target_protein"].str.fullmatch("(?i)unknown|unspecified|\\.")
    ].copy()
    audit["after_exact_positive_structure_context_filter"] = int(len(strict))
    strict = strict[~(strict["source_database"].eq("MGDB") & strict["target_assignment"].str.contains("fallback_column|missing_recruiter", case=False, na=False))].copy()
    audit["after_mgdb_recruiter_filter"] = int(len(strict))
    strict = strict[~strict.apply(lambda row: same_protein(row["recruiting_protein"], row["target_protein"]), axis=1)].copy()
    audit["strict_rows"] = int(len(strict))
    group_columns = ["canonical_smiles", "recruiting_protein", "target_protein", "cell_line"]
    aggregated = strict.groupby(group_columns, dropna=False).agg(
        record_id=("record_id", join_nonempty), source_database=("source_database", join_nonempty),
        source_id=("source_id", join_nonempty), molecular_glue_name=("molecular_glue_name", join_nonempty),
        smiles=("smiles", "first"), dc50_nM=("dc50_nM", "median"), n_measurements=("dc50_nM", "size"),
        dc50_nM_min=("dc50_nM", "min"), dc50_nM_max=("dc50_nM", "max"),
        target_uniprot=("target_uniprot", join_nonempty), recruiting_protein_uniprot=("recruiting_protein_uniprot", join_nonempty),
        activity_time=("activity_time", join_nonempty), assay_method=("assay_method", join_nonempty),
        mode_of_action=("mode_of_action", join_nonempty), source_url=("source_url", join_nonempty),
    ).reset_index()
    aggregated["pDC50"] = 9.0 - np.log10(aggregated["dc50_nM"])
    aggregated.insert(0, "qc_id", [f"MGDC50_QC_{index:05d}" for index in range(1, len(aggregated) + 1)])
    if len(aggregated) >= 10 and aggregated["canonical_smiles"].nunique() >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        _, test_indices = next(splitter.split(aggregated, groups=aggregated["canonical_smiles"]))
        aggregated["split"] = "train"
        aggregated.loc[test_indices, "split"] = "test"
    else:
        aggregated["split"] = "train"
    audit["qc_rows_after_aggregation"] = int(len(aggregated))
    audit["train_rows"] = int((aggregated["split"] == "train").sum())
    audit["test_rows"] = int((aggregated["split"] == "test").sum())
    return strict, aggregated, audit


def _table_summary(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {"path": str(path), "rows": int(len(frame)), "columns": int(len(frame.columns)), "sha256": sha256_file(path)}


def build_dataset(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    files = raw_files(raw_dir)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed, parser_counts = collect_all_sources(files)
    strict, qc, exclusion_audit = build_qc_dataset(parsed)
    frames = {
        "parsed": parsed,
        "strict": strict,
        "qc": qc,
        "train": qc[qc["split"] == "train"],
        "test": qc[qc["split"] == "test"],
    }
    output_summary: dict[str, object] = {}
    for key, frame in frames.items():
        path = output_dir / OUTPUT_FILENAMES[key]
        frame.to_csv(path, index=False)
        output_summary[OUTPUT_FILENAMES[key]] = _table_summary(path, frame)

    # The historical standardization stage consumed the serialized QC table,
    # not the higher-precision in-memory frame. Preserve that contract so the
    # frozen CSV identity is byte-for-byte reproducible.
    serialized_qc = pd.read_csv(output_dir / OUTPUT_FILENAMES["qc"])
    standardized, standardization_summary = standardize_rows(serialized_qc)
    standardized_path = output_dir / OUTPUT_FILENAMES["standardized"]
    standardized.to_csv(standardized_path, index=False)
    output_summary[OUTPUT_FILENAMES["standardized"]] = _table_summary(
        standardized_path, standardized
    )
    report: dict[str, object] = {
        "builder_version": "1.0.0", "random_state": RANDOM_STATE,
        "parser_counts": parser_counts, "exclusion_audit": exclusion_audit,
        "standardization_summary": standardization_summary, "outputs": output_summary,
        "software": {
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "rdkit": rdkit.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    with (output_dir / "build_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report
