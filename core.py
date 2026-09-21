"""
BenefitIQ — Core Pipeline Utilities
Shared across all payer-specific extractors (extract_uhc.py, extract_aetna.py, etc.)
Handles: PDF text extraction, section mapping, metadata parsing, LLM wrapper,
         validation, INDICATIONS/DRUG_INFO config, record assembly, and CLI.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

from openai import OpenAI
import pymupdf as fitz  # pymupdf

# ── Config ────────────────────────────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).parent / "provider_templates"
MODEL = "gpt-4o"
MAX_TOKENS = 8192
CLIENT = OpenAI()  # reads OPENAI_API_KEY from env

def extract_text(pdf_path: str) -> tuple[str, str, str]:
    """Extract full text, first-page raw text, and compute sha256."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    full_text = "\n".join(pages)
    # First 2 pages usually contain all header metadata (policy number, effective date, etc.)
    first_pages_text = "\n".join(pages[:2]) if len(pages) >= 2 else full_text
    sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    return full_text, first_pages_text, sha256


def section_map(full_text: str, template: dict) -> dict[str, str]:
    """
    Stage 1: Split PDF text into named sections using the provider template.
    Returns dict of {canonical_section_name: section_text}.
    Skips sections flagged in template['section_map']['skip'].

    Important: when the same section heading appears more than once (e.g. "Coverage Rationale"
    appearing both as the real section on page 1 AND as a reference in the Policy History table),
    the FIRST occurrence is kept. Subsequent occurrences are appended only if they add substantial
    new content (>200 chars) not already captured.
    """
    keep_aliases = template["section_map"]["keep"]
    skip_list = [s.lower() for s in template["section_map"]["skip"]]

    # Build regex that matches any known heading on its own line
    all_headings = [alias for aliases in keep_aliases.values() for alias in aliases]
    all_headings += template["section_map"]["skip"]
    pattern = r"(?m)^(" + "|".join(re.escape(h) for h in all_headings) + r")\s*$"

    # Split on headings
    parts = re.split(pattern, full_text)
    sections = {}
    i = 1
    while i < len(parts):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        if heading.lower() in skip_list:
            continue

        # Map to canonical name
        canonical = next(
            (canon for canon, aliases in keep_aliases.items() if heading in aliases),
            heading
        )

        if canonical not in sections:
            # First occurrence — always keep
            sections[canonical] = content
        else:
            # Duplicate heading — only append if this chunk adds substantial new content
            # (guards against Policy History tables that re-reference section names)
            if len(content) > 200 and content[:100] not in sections[canonical]:
                sections[canonical] = sections[canonical] + "\n\n" + content

    return sections


# ── Stage 2: Metadata ─────────────────────────────────────────────────────────
def extract_metadata(first_pages_text: str, full_text: str, sections: dict, payer: str, lob: str, sha256: str, pdf_path: str, template: dict) -> dict:
    """
    Stage 2: Pull policy identity fields deterministically.
    Searches the raw first 2 pages of the PDF (before section splitting),
    which reliably contain the document header with policy number and effective date.
    """
    # Also include Policy History section for predecessor policy number
    search_text = first_pages_text + "\n" + sections.get("Policy History", "") + "\n" + full_text[-4000:]

    # Policy number: e.g. "Policy Number: 2026D0039W"
    policy_number = re.search(r"Policy Number[:\s]+([A-Z0-9]{6,})", search_text)

    # Effective date: e.g. "Effective Date: June 1, 2026" or "Effective Date: 06/01/2026"
    effective_date_match = re.search(
        r"Effective Date[:\s]+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{2}/\d{2}/\d{4})",
        search_text,
        re.IGNORECASE
    )
    if effective_date_match:
        raw_date = effective_date_match.group(1).strip().rstrip(",")
        try:
            from datetime import datetime
            for fmt in ("%B %d %Y", "%B %d, %Y", "%m/%d/%Y"):
                try:
                    effective_date_str = datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                effective_date_str = str(date.today())
        except Exception:
            effective_date_str = str(date.today())
    else:
        effective_date_str = str(date.today())

    # Predecessor: look in Policy History for "Archived previous policy versions XXXX"
    predecessor = re.search(
        r"(?:Archived previous policy versions?|supersedes?|replaces?)\s+([A-Z0-9]{6,})",
        search_text,
        re.IGNORECASE
    )

    return {
        "policy_number": policy_number.group(1) if policy_number else "UNKNOWN",
        "effective_date": effective_date_str,
        "retrieved_date": str(date.today()),
        "document_sha256": sha256,
        "pdf_path": pdf_path,
        "payer_name": template["payer_name"],
        "line_of_business": lob,
        "predecessor_policy_number": predecessor.group(1) if predecessor else None,
        "related_policies": []
    }


# ── LLM helper ────────────────────────────────────────────────────────────────
def llm(system: str, user: str) -> str:
    """Single LLM call. Returns text content."""
    response = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    return response.choices[0].message.content


def llm_json(system: str, user: str, retries: int = 2) -> dict | list:
    """LLM call expecting JSON back. Strips markdown fences. Retries on parse failure."""
    for attempt in range(retries + 1):
        raw = llm(system, user)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Sometimes the model wraps output in extra text before/after the JSON
        # Try to extract just the JSON object/array
        json_match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(1)
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            if attempt < retries:
                print(f"        JSON parse failed (attempt {attempt+1}/{retries+1}), retrying... ({e})")
                # Ask the model to fix it
                user = f"Your previous response was not valid JSON. Error: {e}\n\nReturn ONLY valid JSON, nothing else:\n\n{raw}"
            else:
                print(f"        JSON parse failed after {retries+1} attempts: {e}")
                print(f"        Raw response snippet: {raw[:200]}")
                raise


# ── Stage 6: Validate ─────────────────────────────────────────────────────────
REQUIRED_TOP_KEYS = ["record_id","status","policy_source","service","payer","benefit_routing","indications","exclusions","review"]

def validate(record: dict) -> list[str]:
    """
    Stage 6: Basic structural validation.
    Returns list of error strings. Empty list = pass.
    """
    errors = []

    for key in REQUIRED_TOP_KEYS:
        if key not in record:
            errors.append(f"Missing top-level key: {key}")

    for i, ind in enumerate(record.get("indications", [])):
        if "indication_id" not in ind:
            errors.append(f"indications[{i}] missing indication_id")
        for j, cs in enumerate(ind.get("criteria_sets", [])):
            if "root" not in cs:
                errors.append(f"indications[{i}].criteria_sets[{j}] missing root")
            if cs.get("tier") not in ("proven", "medically_necessary"):
                errors.append(f"indications[{i}].criteria_sets[{j}] invalid tier: {cs.get('tier')}")
            if cs.get("therapy_phase") not in ("initial", "continuation"):
                errors.append(f"indications[{i}].criteria_sets[{j}] invalid therapy_phase")

    if not record.get("service", {}).get("hcpcs_codes"):
        errors.append("service.hcpcs_codes is empty — confirm J-code with clinic billing")

    return errors


# ── Assemble Rule Record ──────────────────────────────────────────────────────
INDICATIONS = {
    "orencia": [
        ("ra",                  "Rheumatoid Arthritis"),
        ("psa",                 "Psoriatic Arthritis"),
        ("pjia",                "Polyarticular Juvenile Idiopathic Arthritis"),
        ("chronic_gvhd",        "Chronic Graft-Versus-Host Disease"),
        ("agvhd_prophylaxis",   "Acute Graft-Versus-Host Disease Prophylaxis"),
        ("checkpoint_toxicity", "Immune Checkpoint Inhibitor-Related Toxicities"),
    ],
    "cimzia": [
        ("ra",         "Rheumatoid Arthritis"),
        ("psa",        "Psoriatic Arthritis"),
        # AS and nr-axSpA share the same UHC heading/criteria — treat as one
        ("as_nraxspa", "Ankylosing Spondylitis"),
        ("cd",         "Crohn Disease"),    # "Crohn" (5 chars) matches PDF heading "Crohn's disease"
        ("pjia",       "Polyarticular Juvenile Idiopathic Arthritis"),
        ("plaque_ps",  "Plaque Psoriasis"),
    ],
    "simponi_aria": [
        ("ra",  "Rheumatoid Arthritis"),
        ("psa", "Psoriatic Arthritis"),
        ("as",  "Ankylosing Spondylitis"),
        ("pjia","Polyarticular Juvenile Idiopathic Arthritis"),
    ],
}

DRUG_INFO = {
    "orencia":     {"drug_name": "Orencia",     "generic_name": "abatacept",           "manufacturer": "Bristol Myers Squibb"},
    "cimzia":      {"drug_name": "Cimzia",       "generic_name": "certolizumab pegol",  "manufacturer": "UCB"},
    "simponi_aria":{"drug_name": "Simponi Aria", "generic_name": "golimumab",           "manufacturer": "Janssen"},
}


def build_record(
    pdf_path: str,
    payer: str,
    drug: str,
    lob: str,
    template: dict,
    extract_gate_fn=None,
    extract_criteria_fn=None,
    extract_codes_fn=None,
    extract_exclusions_fn=None,
) -> dict:
    print(f"\n[1/6] Extracting text + hashing PDF...")
    full_text, first_pages_text, sha256 = extract_text(pdf_path)

    print(f"[2/6] Mapping sections...")
    sections = section_map(full_text, template)
    print(f"      Sections found: {list(sections.keys())}")

    print(f"[3/6] Extracting metadata...")
    metadata = extract_metadata(first_pages_text, full_text, sections, payer, lob, sha256, pdf_path, template)

    print(f"[4/6] Extracting benefit routing gate...")
    benefit_routing = extract_gate_fn(sections, template)

    print(f"[5/6] Extracting indications + codes...")
    indications = []
    indication_list = INDICATIONS.get(drug, [])

    codes_result = extract_codes_fn(sections)
    exclusions = extract_exclusions_fn(sections)

    for ind_id, ind_name in indication_list:
        print(f"      → {ind_name}")
        try:
            criteria_sets = extract_criteria_fn(sections, ind_name, ind_id, drug)
        except Exception as e:
            print(f"        WARNING: criteria extraction failed for {ind_name}: {e}")
            criteria_sets = []

        icd10 = codes_result.get("icd10_by_indication", {}).get(ind_id, [])

        indications.append({
            "indication_id": ind_id,
            "indication_name": ind_name,
            "coverage_status": "covered",
            "icd10_codes": icd10,
            "prescriber_specialties_accepted": None,
            "criteria_sets": criteria_sets,
        })

    drug_info = DRUG_INFO.get(drug, {})
    record_id = f"{drug}__{payer}__{lob}"

    record = {
        "$schema": "benefitiq/rule_record/v1",
        "record_id": record_id,
        "status": "draft",
        "policy_source": metadata,
        "service": {
            **drug_info,
            "route": "iv_infusion",
            "hcpcs_codes": codes_result.get("hcpcs_codes", []),
            "admin_cpt_codes": codes_result.get("admin_cpt_codes", []),
        },
        "payer": {
            "payer_id": payer,
            "payer_name": template["payer_name"],
            "line_of_business": lob,
            "provider_template_version": template["version"],
        },
        "benefit_routing": benefit_routing,
        "indications": indications,
        "exclusions": exclusions,
        "review": {
            "reviewer": None,
            "review_date": None,
            "confidence": None,
            "notes": f"Auto-extracted by pipeline v1 from {Path(pdf_path).name}. Requires human review before activation.",
            "golden_cases_added": 0,
        }
    }

    print(f"[6/6] Validating...")
    errors = validate(record)
    if errors:
        print(f"      VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"        - {e}")
    else:
        print(f"      Validation passed.")

    return record, errors


# ── CLI ───────────────────────────────────────────────────────────────────────
def main(payer_id=None, extract_gate_fn=None, extract_criteria_fn=None, extract_codes_fn=None, extract_exclusions_fn=None):
    parser = argparse.ArgumentParser(description="BenefitIQ Phase 1 Extraction Pipeline")
    parser.add_argument("--pdf",   required=True,  help="Path to payer policy PDF")
    parser.add_argument("--payer", default=payer_id, help="Payer slug (e.g. uhc)")
    parser.add_argument("--drug",  required=True,  help="Drug slug (orencia | cimzia | simponi_aria)")
    parser.add_argument("--lob",   required=True,  help="Line of business (commercial | individual_exchange | ...)")
    parser.add_argument("--out",   default=None,   help="Output JSON path (default: <record_id>.json)")
    args = parser.parse_args()

    template_path = TEMPLATES_DIR / f"{args.payer}_template.json"
    if not template_path.exists():
        print(f"ERROR: No provider template found at {template_path}")
        sys.exit(1)

    template = json.loads(template_path.read_text())

    record, errors = build_record(
        pdf_path=args.pdf,
        payer=args.payer,
        drug=args.drug,
        lob=args.lob,
        template=template,
        extract_gate_fn=extract_gate_fn,
        extract_criteria_fn=extract_criteria_fn,
        extract_codes_fn=extract_codes_fn,
        extract_exclusions_fn=extract_exclusions_fn,
    )

    out_path = args.out or Path(__file__).parent / "rule_records" / f"{record['record_id']}.json"
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(record, indent=2))
    print(f"\nOutput written to: {out_path}")

    if errors:
        print(f"Status: DRAFT (validation errors present — review before activating)")
        sys.exit(1)
    else:
        print(f"Status: DRAFT (ready for human review)")


if __name__ == "__main__":
    main()