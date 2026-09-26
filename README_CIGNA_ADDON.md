# Cigna Add-On — Medical Prior Authorization Project

New files only. Nothing from your existing project (FYP---Medical-Prior-Authorization-main)
was opened for writing or modified.

## What's in this zip

```
provider_templates/cigna_template.json   <- new payer template (Cigna)
extract_cigna.py                          <- new Cigna extraction script
policies/simponi_aria_cigna.pdf           <- copy of your uploaded PDF
policies/cimzia_cigna.pdf                 <- copy of your uploaded PDF
policies/orencia_cigna.pdf                <- copy of your uploaded PDF
rule_records/simponi_aria__cigna__commercial.json
rule_records/cimzia__cigna__commercial.json
rule_records/orencia__cigna__commercial.json
```

## How to merge into your existing project

1. Unzip this file.
2. Drag/copy each folder's contents into the matching folder in your existing
   `FYP---Medical-Prior-Authorization-main/` project root:
   - `provider_templates/cigna_template.json` -> your `provider_templates/` folder
   - `extract_cigna.py` -> your project root (same level as `extract_uhc.py`)
   - `policies/*.pdf` -> your `policies/` folder
   - `rule_records/*.json` -> your `rule_records/` folder
3. No existing filename collides with anything already in your zip, so nothing
   gets overwritten. `core.py` and every existing UHC/Humana file are untouched.

## What was verified before packaging

- All 3 rule_record JSON files pass your project's OWN `core.py` `validate()`
  function (imported and run directly against these files) -- 0 errors on all three.
- `document_sha256` in each `policy_source` is the REAL sha256 of the actual PDF
  you uploaded (computed with Python's hashlib), not a placeholder.
- Every criterion was built from the actual policy text (Cigna IP0668, IP0672,
  IP0664) you provided in this conversation -- nothing invented.

## Known gaps flagged INSIDE the JSON files (see each file's "review.notes" and,
where present, "step_therapy_dependency")

- `icd10_codes` is empty for every indication in all 3 files -- Cigna's PDFs do
  not publish ICD-10 tables (unlike UHC's). Map these via the NLM Clinical
  Tables API before using the records for real matching.
- `admin_cpt_codes` is empty for all 3 -- Cigna's PDFs list only the drug HCPCS
  J-code, never an administration CPT code. Confirm with clinic billing.
- `line_of_business` is set to "commercial" as a best guess with a note
  attached -- the source PDFs reference "employer plans" and separate
  Individual/Family + Legacy drug lists but never say "commercial" outright.
- Cimzia and Orencia IV each carry a `step_therapy_dependency` block: both
  require a separate Cigna "Preferred Specialty Management" (PSM) policy
  (PSM001/017/002 for Cimzia; PSM006/010/018 for Orencia) that is NOT contained
  in the drug coverage policy PDF. These records are intentionally marked
  incomplete for step-therapy purposes until those PSM policies are obtained.
- Orencia IV's `exclusions` list includes Ankylosing Spondylitis, Crohn's
  Disease, Ulcerative Colitis, and Plaque Psoriasis as explicitly
  NOT-medically-necessary for this drug (per the PDF's "Conditions Not
  Covered") -- worth knowing since Simponi Aria and Cimzia DO cover some of
  these same conditions. Good real-world example of cross-drug variation for
  your pitch.
