"""
BenefitIQ — UHC Extraction Pipeline
UHC-specific prompts and extraction logic for Stages 3–5.

Usage:
    python extract_uhc.py --pdf policies/orencia_uhc.pdf --drug orencia --lob commercial

Payer notes:
- Dual-tier structure: "proven" and "medically necessary"
- Key sections: Coverage Rationale, Applicable Codes, Policy History/Revision Information
- Template: provider_templates/uhc_template.json
"""

from core import *

PAYER_ID = "uhc"

# ── Stage 3: Benefit Routing Gate ─────────────────────────────────────────────
GATE_SYSTEM_TEMPLATE = """You are extracting structured data from a {payer_name} medical benefit drug policy.
Return ONLY valid JSON. No prose, no markdown fences.
Schema:
{{
  "benefit_type": "medical" | "pharmacy",
  "applies_to_this_request": true | false,
  "routing_notes": string | null,
  "site_of_care_policy_link": string | null,
  "medicare_advantage_precedence": string | null
}}"""

def extract_benefit_gate(sections: dict, template: dict) -> dict:
    """Stage 3: Extract benefit routing info from Coverage Rationale opening."""
    system = GATE_SYSTEM_TEMPLATE.format(payer_name=template["payer_name"])
    text = sections.get("Coverage Rationale", "")[:1500]
    return llm_json(system, f"Extract benefit routing from this policy text:\n\n{text}")


# ── Stage 4: Criteria Tree ────────────────────────────────────────────────────
CRITERIA_SYSTEM = """You are extracting prior authorization criteria from a payer medical benefit drug policy into structured JSON.

You will be given policy text for ONE indication. You MUST extract ALL criteria sets present — there can be up to 4:
  1. proven × initial
  2. proven × continuation
  3. medically_necessary × initial
  4. medically_necessary × continuation

Do NOT stop after finding the first set. Read the entire text and extract every tier+phase combination that appears.

Return ONLY valid JSON — a list of criteria set objects, no prose, no markdown fences:
[
  {
    "tier": "proven" | "medically_necessary",
    "therapy_phase": "initial" | "continuation",
    "auth_duration_months": integer | null,
    "auth_duration_unit": "months" | "doses",
    "root": <criterion_node>
  }
]

criterion_node is one of:
- Group: {"node_type":"group","operator":"ALL"|"ANY","label":string,"children":[...criterion_nodes]}
- Leaf:  {
    "node_type":"criterion",
    "criterion_id":"<indication_id>.<tier_short>.<phase_short>.<short_name>",
    "criterion_type": one of [diagnosis, step_therapy, prior_biologic_exposure, currently_on_drug, combination_exclusion, prescriber_specialty, attestation, dosing_per_fda_label, positive_clinical_response, age_requirement, screening_or_lab, procedure_requirement],
    "description": string (quote policy language as closely as possible),
    "criticality": "critical" | "supporting",
    "parameters": object | null,
    "chart_fact_key": string,
    "fhir_resource_hint": "Condition"|"MedicationStatement"|"MedicationRequest"|"MedicationAdministration"|"Observation"|"Practitioner"|"Coverage"|null,
    "documentation_examples": [string],
    "source": {"page": integer|null, "section": string, "policy": string|null}
  }

STRICT RULES — follow every one:

1. EXTRACT ALL 4 SETS. If proven×continuation or medically_necessary×initial etc. appear in the text, they MUST appear in your output. Never return only one set when the text contains more.

2. chart_fact_key is REQUIRED on every leaf — never leave it blank. Use snake_case describing what the Phase 2 chart matcher needs to look up. Examples:
   - diagnosis criterion → "diagnosis_ra_active"
   - step therapy → "prior_dmard_trial"
   - combination exclusion → "concurrent_immunomodulators"
   - currently on drug → "prior_receipt_of_{drug}_iv"
   - prescriber specialty → "prescriber_specialty"
   - positive response → "positive_clinical_response"
   - dosing → "dosing_regimen"

3. combination_exclusion parameters MUST include:
   - "excluded_drugs": list every specific drug name mentioned (both brand and generic where given) — do NOT summarize as "other systemic targeted immunomodulators"
   - "same_indication_scope": true | false

4. step_therapy parameters MUST include:
   - "min_trial_days": integer or null
   - "at_max_dose_required": true | false
   - "contraindication_excuses_trial": true | false

5. diagnosis criterion: always extract if the policy states a required diagnosis (e.g. "moderately to severely active RA"). Use criterion_type "diagnosis" and put severity qualifier in parameters as {"severity_qualifier": "..."}.

6. prescriber_specialty criterion: extract whenever the policy requires prescribing by or in consultation with a specialist. Put accepted specialties in parameters as {"accepted_specialties": [...]}.

7. dosing_per_fda_label criterion: extract whenever the policy states drug must be dosed per FDA label. parameters: null.

8. positive_clinical_response criterion: extract for continuation sets whenever the policy requires documented response to prior therapy.

9. "all of the following" → operator ALL. "one of the following" / "any of the following" → operator ANY.

10. Never invent criteria not present in the text. Return null for parameters only when no structured parameters apply.

11. COMBINED TIER ("proven and/or medically necessary"): Some indications (e.g. GVHD, checkpoint toxicities) use a single combined block with shared criteria rather than separate proven/MN blocks. When this happens, output the criteria sets using tier "proven" (for initial) and "proven" (for continuation), since the criteria are identical for both tiers. Do NOT duplicate into medically_necessary copies — only create separate tier entries when the policy text explicitly lists different criteria for each tier."""

def extract_criteria(sections: dict, indication_name: str, indication_id: str, drug: str) -> list:
    """Stage 4: Extract criteria sets for one indication."""
    rationale = sections.get("Coverage Rationale", "")

    # Split Coverage Rationale on top-level indication headings.
    # Each heading: "Orencia is proven/MN/unproven for the treatment/prophylaxis of X..."
    heading_pattern = re.compile(
        r"((?:\w+\s+)?is\s+"
        r"(?:proven(?:\s+and/or\s+medically\s+necessary)?|medically\s+necessary|unproven(?:\s+and\s+not\s+medically\s+necessary)?)"
        r"\s+for\s+(?:the\s+)?(?:treatment|prophylaxis)[^\n]{0,120})",
        re.IGNORECASE
    )
    parts = heading_pattern.split(rationale)
    # parts = [pre, heading1, body1, heading2, body2, ...]
    blocks = []
    i = 1
    while i < len(parts) - 1:
        blocks.append((parts[i].strip(), parts[i + 1].strip()))
        i += 2

    # key words from indication name (len > 4 to skip short words like "for")
    # Strip punctuation so e.g. "Crohn's" matches "crohn's disease (CD)" heading
    import string
    key_words = [w.strip(string.punctuation) for w in indication_name.lower().split() if len(w.strip(string.punctuation)) > 4]

    matched_blocks = []
    for heading, body in blocks:
        h = heading.lower()
        if all(kw in h for kw in key_words) and "unproven" not in h:
            matched_blocks.append(heading + "\n" + body)

    indication_text = "\n\n".join(matched_blocks)

    if not indication_text:
        print(f"        WARNING: No policy text found for '{indication_name}' — skipping LLM call")
        return []

    print(f"        Found {len(indication_text)} chars of policy text")

    prompt = f"""Drug: {drug}
Indication: {indication_name} (indication_id: {indication_id})

IMPORTANT: Extract criteria ONLY for "{indication_name}". Do NOT include criteria from any other indication even if their text appears below. If the text below contains criteria for multiple indications, extract ONLY the ones explicitly labeled for "{indication_name}".

Policy text scoped to this indication:
{indication_text[:6000]}

Extract all criteria sets for {indication_name} ONLY."""

    raw = llm_json(CRITERIA_SYSTEM, prompt)
    # Normalise auth_duration: LLM sometimes returns null or wrong unit for dose-limited indications
    dose_limited = {"agvhd_prophylaxis", "checkpoint_toxicity"}
    for cs in raw:
        if indication_id in dose_limited:
            cs["auth_duration_months"] = 4
            cs["auth_duration_unit"] = "doses"
        elif cs.get("auth_duration_months") is None:
            cs["auth_duration_months"] = 12
            cs["auth_duration_unit"] = "months"
    return raw


# ── Stage 5: Codes & Exclusions ───────────────────────────────────────────────
HCPCS_SYSTEM = """Extract HCPCS J-codes and administration CPT codes from this payer policy codes section.
Return ONLY valid JSON:
{
  "hcpcs_codes": [{"code": string, "description": string}],
  "admin_cpt_codes": [string]
}
HCPCS codes start with J (e.g. J0129). Admin CPT codes are 5-digit numbers like 96413, 96415.
Return empty arrays if none found."""

ICD10_SYSTEM = """Extract ICD-10 diagnosis codes from this payer policy and group them by indication.
Return ONLY valid JSON:
{
  "icd10_by_indication": {
    "<indication_id>": [{"code": string, "description": string}]
  }
}
Use these indication_ids: ra, psa, pjia, chronic_gvhd, agvhd_prophylaxis, checkpoint_toxicity.
Group by prefix: M05/M06=ra, M08=pjia, L40=psa, D89=chronic_gvhd or agvhd_prophylaxis, T45=checkpoint_toxicity.
Return only the first 20 codes per indication — do not enumerate every site-specific variant.
Return empty object if none found."""

def extract_codes(sections: dict) -> dict:
    """Stage 5a: Extract HCPCS, CPT, ICD-10 codes.

    The Applicable Codes section can be very large (25k+ chars) due to
    exhaustive ICD-10 site-specific variants. We:
    1. Extract HCPCS/CPT from the first 2000 chars (they always appear first)
    2. Extract ICD-10 from a trimmed sample (first 6000 chars) to stay within token limits
    """
    codes_text = sections.get("Applicable Codes", "")
    if not codes_text:
        return {"hcpcs_codes": [], "admin_cpt_codes": [], "icd10_by_indication": {}}

    # Step 1: HCPCS + CPT from the top of the section (always listed first)
    try:
        hcpcs_result = llm_json(HCPCS_SYSTEM, codes_text[:2000])
    except Exception as e:
        print(f"        WARNING: HCPCS extraction failed: {e}")
        hcpcs_result = {"hcpcs_codes": [], "admin_cpt_codes": []}

    # Step 2: ICD-10 from a trimmed sample
    try:
        icd10_result = llm_json(ICD10_SYSTEM, codes_text[:6000])
    except Exception as e:
        print(f"        WARNING: ICD-10 extraction failed: {e}")
        icd10_result = {"icd10_by_indication": {}}

    return {
        "hcpcs_codes": hcpcs_result.get("hcpcs_codes", []),
        "admin_cpt_codes": hcpcs_result.get("admin_cpt_codes", []),
        "icd10_by_indication": icd10_result.get("icd10_by_indication", {}),
    }


EXCLUSIONS_SYSTEM = """Extract the list of indications explicitly marked as unproven or not medically necessary.
Return ONLY valid JSON:
[{"indication_name": string, "reason": "unproven"|"not_medically_necessary"|"unproven_and_not_mn", "source": string|null}]
Return empty array if none found."""

def extract_exclusions(sections: dict) -> list:
    """Stage 5b: Extract unproven/not-MN exclusion list."""
    rationale = sections.get("Coverage Rationale", "")
    # Find the unproven block — always at the end of Coverage Rationale
    unproven_match = re.search(
        r"is\s+unproven.*",
        rationale, re.IGNORECASE | re.DOTALL
    )
    unproven_text = unproven_match.group(0) if unproven_match else rationale[-3000:]
    text = sections.get("Unproven", "") + "\n" + unproven_text
    return llm_json(EXCLUSIONS_SYSTEM, text[:4000])




if __name__ == "__main__":
    main(
        payer_id=PAYER_ID,
        extract_gate_fn=extract_benefit_gate,
        extract_criteria_fn=extract_criteria,
        extract_codes_fn=extract_codes,
        extract_exclusions_fn=extract_exclusions,
    )
