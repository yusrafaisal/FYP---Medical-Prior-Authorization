"""
BenefitIQ — Cigna Extraction Pipeline
Cigna-specific prompts and extraction logic for Stages 3-5.

Usage:
    python extract_cigna.py --pdf policies/simponi_aria_cigna.pdf --drug simponi_aria --lob commercial

Payer notes:
- Single-tier structure: "considered medically necessary when ONE of the following is met"
  (no proven/medically-necessary split like UHC)
- Every numbered indication (1., 2., 3'...) contains its own
  A) Initial Therapy  /  B) Currently Receiving (or "Already Received") split
- Key sections: Coverage Policy / POLICY STATEMENT, Conditions Not Covered,
  Coding Information, Revision Details, APPENDIX
- Some drugs (Cimzia, Orencia IV) require a separate "preferred product" /
  step-therapy policy (PSM00x) NOT contained in this PDF -- must be flagged,
  never fabricated.
- Template: provider_templates/cigna_template.json
"""

from core import *

PAYER_ID = "cigna"

INDICATIONS = {
    "simponi_aria": [("as", "Ankylosing Spondylitis"), ("pjia", "Polyarticular Juvenile Idiopathic Arthritis"),
                      ("psa", "Psoriatic Arthritis"), ("ra", "Rheumatoid Arthritis")],
    "cimzia": [("as", "Ankylosing Spondylitis"), ("cd", "Crohn's Disease"),
               ("pjia", "Juvenile Idiopathic Arthritis"), ("nraxspa", "Non-Radiographic Axial Spondyloarthritis"),
               ("plaque_pso", "Plaque Psoriasis"), ("psa", "Psoriatic Arthritis"),
               ("ra", "Rheumatoid Arthritis"), ("other_spa", "Spondyloarthritis, Other Subtypes")],
    "orencia_iv": [("agvhd_prophylaxis", "Graft-Versus-Host Disease - Prevention"),
                   ("pjia", "Juvenile Idiopathic Arthritis"), ("psa", "Psoriatic Arthritis"),
                   ("ra", "Rheumatoid Arthritis"), ("chronic_gvhd", "Chronic Graft-Versus-Host Disease - Treatment")],
}

# ── Stage 3: Benefit Routing Gate ─────────────────────────────────────────────
GATE_SYSTEM_TEMPLATE = """You are extracting structured data from a {payer_name} medical benefit "Drug Coverage Policy".
Return ONLY valid JSON. No prose, no markdown fences.
Schema:
{{
  "benefit_type": "medical" | "pharmacy",
  "applies_to_this_request": true | false,
  "routing_notes": string | null
}}
Cigna "Drug Coverage Policy" documents are administered under the medical benefit unless the
text states otherwise. If the document does not explicitly state medical vs. pharmacy benefit,
set benefit_type to "medical" and note in routing_notes that this was inferred from document
type, not stated explicitly -- flag for human confirmation."""

def extract_benefit_gate(sections: dict, template: dict) -> dict:
    """Stage 3: Extract benefit routing info from the Coverage Policy opening + header."""
    system = GATE_SYSTEM_TEMPLATE.format(payer_name=template["payer_name"])
    text = (sections.get("Header", "") + "\n" + sections.get("Coverage Policy", "")[:1200])
    return llm_json(system, f"Extract benefit routing from this policy text:\n\n{text}")


# ── Stage 3b: Step-Therapy / Preferred-Product Dependency ─────────────────────
STEP_THERAPY_SYSTEM = """Check this Cigna policy text for a NOTE requiring use of preferred products
before approval (a step-therapy dependency on a separate Preferred Specialty Management policy).
Return ONLY valid JSON:
{
  "has_dependency": true | false,
  "referenced_policies": [string],
  "note": string | null
}
Only report policies explicitly named in the text (e.g. "PSM001", "PSM017"). Never invent
policy numbers. If no such NOTE is present, return has_dependency: false and an empty list."""

def extract_step_therapy_dependency(sections: dict) -> dict:
    """Stage 3b: Detect external PSM policy dependency, unique to Cigna's format."""
    text = sections.get("Coverage Policy", "")[:2000]
    return llm_json(STEP_THERAPY_SYSTEM, text)


# ── Stage 4: Criteria Tree ────────────────────────────────────────────────────
CRITERIA_SYSTEM = """You are extracting prior authorization criteria from a Cigna medical benefit
drug policy into structured JSON.

Cigna uses a SINGLE tier only ("considered medically necessary when ONE of the following is
met") -- there is no separate "proven" tier like UHC. You will be given policy text for ONE
numbered indication. Extract BOTH phases if present:
  1. medically_necessary x initial   (labeled "A) Initial Therapy" in the source)
  2. medically_necessary x continuation  (labeled "B) Currently Receiving" / "Already Received")

Do NOT stop after finding the first phase. Read the entire block and extract both if both appear.

Return ONLY valid JSON -- a list of criteria set objects, no prose, no markdown fences:
[
  {
    "tier": "medically_necessary",
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
    "criterion_id":"<indication_id>.mn.<phase_short>.<short_name>",
    "criterion_type": one of [diagnosis, step_therapy, prior_biologic_exposure, currently_on_drug, combination_exclusion, prescriber_specialty, attestation, dosing_per_fda_label, positive_clinical_response, age_requirement, screening_or_lab, procedure_requirement],
    "description": string (quote policy language as closely as possible),
    "criticality": "critical" | "supporting",
    "parameters": object | null,
    "chart_fact_key": string,
    "fhir_resource_hint": "Condition"|"MedicationStatement"|"MedicationRequest"|"MedicationAdministration"|"Observation"|"Practitioner"|"Coverage"|null,
    "documentation_examples": [string],
    "source": {"page": integer|null, "section": string, "policy": string|null}
  }

STRICT RULES:

1. EXTRACT BOTH PHASES when both "Initial Therapy" and "Currently Receiving"/"Already Received"
   sub-blocks appear under the numbered indication.

2. chart_fact_key is REQUIRED on every leaf. Use snake_case, e.g.:
   - age gate -> "patient_age"
   - DMARD/prior-therapy trial -> "prior_therapy_trials" or "dmard_trial_months"
   - combination exclusion -> "current_medications"
   - currently on drug -> "months_on_current_drug"
   - prescriber specialty -> "prescriber_specialty"
   - objective/symptom response -> "objective_disease_activity_score" / "symptom_improvement_note"
   - dosing -> "requested_dose_and_frequency"

3. combination_exclusion parameters MUST include:
   - "excluded_drug_classes": list drugs from the policy's own APPENDIX table (do not summarize)
   - "allowed_concurrent": conventional synthetic DMARDs explicitly carved out (methotrexate,
     leflunomide, hydroxychloroquine, sulfasalazine) if the text says they are NOT excluded

4. step_therapy / prior_biologic_exposure parameters MUST include "min_months" (Cigna states
   trial length in months, not days, e.g. "for at least 3 months").

5. "ALL of the following" -> operator ALL. "ONE of the following" -> operator ANY.

6. Weight-based dosing (Cimzia JIA; Orencia IV all weight brackets): capture the full set of
   weight brackets and doses in the dosing_per_fda_label leaf's description -- do not average
   or drop brackets.

7. GVHD indications (Orencia IV only) use a different specialist set (oncologist, hematologist,
   transplant center physician) and different objective measures (liver function tests, RBC/
   platelet counts, fever/rash resolution) -- do not reuse rheumatology objective measures here.

8. "Other Uses with Supportive Evidence" indications (e.g. Cimzia's Spondyloarthritis Other
   Subtypes; Orencia's Chronic GVHD Treatment) are NOT FDA-approved indications -- still extract
   full criteria, but the calling code sets coverage_status to "covered_supportive_evidence".

9. Never invent criteria not present in the text. Use null for parameters only when no
   structured parameters apply."""

def extract_criteria(sections: dict, indication_name: str, indication_id: str, drug: str) -> list:
    """Stage 4: Extract criteria sets for one numbered indication."""
    policy_text = sections.get("Coverage Policy", "")

    # Cigna numbers indications: "1. Ankylosing Spondylitis. Approve for..."
    # Split on the numbered-heading pattern, keep the number+name as anchor.
    heading_pattern = re.compile(
        r"(\n\s*\d+\.\s+[A-Z][^\n]{0,80}\.\s+Approve for)",
    )
    parts = heading_pattern.split(policy_text)
    blocks = []
    i = 1
    while i < len(parts) - 1:
        blocks.append((parts[i].strip(), parts[i + 1].strip()))
        i += 2

    import string
    key_words = [w.strip(string.punctuation) for w in indication_name.lower().split()
                 if len(w.strip(string.punctuation)) > 4]

    matched_blocks = []
    for heading, body in blocks:
        h = heading.lower()
        if all(kw in h for kw in key_words):
            matched_blocks.append(heading + " " + body)

    indication_text = "\n\n".join(matched_blocks)

    if not indication_text:
        print(f"        WARNING: No policy text found for '{indication_name}' -- skipping LLM call")
        return []

    print(f"        Found {len(indication_text)} chars of policy text")

    prompt = f"""Drug: {drug}
Indication: {indication_name} (indication_id: {indication_id})

IMPORTANT: Extract criteria ONLY for "{indication_name}". If the text below contains criteria
for a different numbered indication, ignore it.

Policy text scoped to this indication:
{indication_text[:6000]}

Extract all criteria sets for {indication_name} ONLY."""

    raw = llm_json(CRITERIA_SYSTEM, prompt)
    for cs in raw:
        cs["tier"] = "medically_necessary"  # Cigna is always single-tier
        if cs.get("auth_duration_months") is None and cs.get("auth_duration_unit") != "doses":
            cs["auth_duration_months"] = 12 if cs.get("therapy_phase") == "continuation" else 6
            cs["auth_duration_unit"] = "months"
    return raw


# ── Stage 5: Codes & Exclusions ───────────────────────────────────────────────
HCPCS_SYSTEM = """Extract the HCPCS J-code from this Cigna "Coding Information" section.
Return ONLY valid JSON:
{
  "hcpcs_codes": [{"code": string, "description": string}],
  "admin_cpt_codes": [string]
}
Cigna policies typically list only ONE HCPCS J-code with a full description (sometimes
including a Medicare self-administration note -- keep that in the description verbatim).
Cigna source PDFs do NOT list administration CPT codes -- admin_cpt_codes should almost
always be an empty array; do not infer or fabricate CPT codes."""

def extract_codes(sections: dict) -> dict:
    """Stage 5a: Extract HCPCS code (Cigna does not publish ICD-10 tables in these PDFs)."""
    codes_text = sections.get("Coding Information", "")
    if not codes_text:
        return {"hcpcs_codes": [], "admin_cpt_codes": [], "icd10_by_indication": {}}
    try:
        hcpcs_result = llm_json(HCPCS_SYSTEM, codes_text[:1500])
    except Exception as e:
        print(f"        WARNING: HCPCS extraction failed: {e}")
        hcpcs_result = {"hcpcs_codes": [], "admin_cpt_codes": []}
    # Cigna PDFs do not include ICD-10 code tables (unlike UHC) -- always empty here.
    # ICD-10 mapping must be sourced separately (e.g. NLM Clinical Tables API) per indication.
    return {
        "hcpcs_codes": hcpcs_result.get("hcpcs_codes", []),
        "admin_cpt_codes": hcpcs_result.get("admin_cpt_codes", []),
        "icd10_by_indication": {},
    }


EXCLUSIONS_SYSTEM = """Extract the list of indications/uses explicitly listed under "Conditions
Not Covered" (or "for any other use is considered not medically necessary") -- excluding the
generic "concurrent use with another biologic" combination rule, which is handled separately
per-indication, not as a record-level exclusion.
Return ONLY valid JSON:
[{"indication_name": string, "reason": "not_medically_necessary", "note": string, "source": string|null}]
Return empty array if no named condition/indication exclusions are present (only the
combination-exclusion rule)."""

def extract_exclusions(sections: dict) -> list:
    """Stage 5b: Extract named not-medically-necessary exclusions."""
    text = sections.get("Conditions Not Covered", "")
    if not text:
        return []
    return llm_json(EXCLUSIONS_SYSTEM, text[:4000])


if __name__ == "__main__":
    main(
        payer_id=PAYER_ID,
        extract_gate_fn=extract_benefit_gate,
        extract_criteria_fn=extract_criteria,
        extract_codes_fn=extract_codes,
        extract_exclusions_fn=extract_exclusions,
    )
