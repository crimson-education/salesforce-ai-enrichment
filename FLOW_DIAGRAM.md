# Lead Enrichment System - Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LEAD ENRICHMENT PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA MERGING                                                     │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐        ┌─────────────────────┐
    │ leads_positive.csv  │        │ leads_negative.csv  │
    │   (new data)        │        │   (new data)        │
    └──────────┬──────────┘        └──────────┬──────────┘
               │                              │
               ├──────────────────────────────┤
               │     merge_lead_datasets()    │
               ├──────────────────────────────┤
               │                              │
    ┌──────────▼──────────┐        ┌─────────▼──────────┐
    │leads_positive_old.csv│        │leads_negative_old.csv│
    │  (historical data)   │        │  (historical data)  │
    └──────────┬──────────┘        └──────────┬──────────┘
               │                              │
               │  Fill missing values         │
               │  from old data               │
               │                              │
    ┌──────────▼──────────┐        ┌─────────▼──────────┐
    │   df_positive       │        │   df_negative       │
    │  (merged dataset)   │        │  (merged dataset)   │
    └──────────┬──────────┘        └──────────┬──────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: CHECKPOINT LOADING                                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────┐    ┌───────────────────────────┐
    │ leads_positive_checkpoint │    │ leads_negative_checkpoint │
    │         .csv              │    │         .csv              │
    └─────────────┬─────────────┘    └─────────────┬─────────────┘
                  │                                │
                  │   Resume from last position    │
                  │   Skip already enriched IDs    │
                  │                                │
                  └────────────┬───────────────────┘
                               │
                               ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: BATCH PROCESSING (Alternates between datasets)                  │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │  ITERATION LOOP (max 100 iterations)                        │
    │                                                              │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Process POSITIVE dataset → Process NEGATIVE dataset   │ │
    │  └───────────────────────────────────────────────────────┘ │
    │                                                              │
    │  Stop when: Both datasets make no progress                  │
    └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  BATCH SIZE = 10 records       │
              │  Guardian records only         │
              └────────────────┬───────────────┘
                               │
                               ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: ENRICHMENT PER BATCH (Sequential Phases)                        │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 1: APOLLO ENRICHMENT (apollo.py)                      │
    └─────────────────────────────────────────────────────────────┘
              │
              │  Input: email, first_name, last_name, phone
              │  API: Apollo.io Bulk People Enrichment
              │  Output: job_title, company, seniority, linkedin,
              │          city, state, employment_history, education
              │
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Rate Limit Check                                             │
    │   ✓ Success → Continue to Phase 2                           │
    │   ✗ Rate Limited → Save checkpoint, switch to other dataset │
    └─────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 2: SCHOOL ENRICHMENT (school.py)                      │
    └─────────────────────────────────────────────────────────────┘
              │
              │  Input: school_name, state, lead_zip
              │
              ├─── PRIMARY: School Matcher API ────────────────┐
              │    • Deterministic matching                    │
              │    • Fuzzy matching                            │
              │    • Confidence scoring                        │
              │    • Match confidence ≥ 80%                    │
              │                                                 │
              ├─── SECONDARY: NCES Database ───────────────────┤
              │    (nces_schools.csv)                          │
              │    • Token-based fuzzy matching                │
              │    • ZIP code lookup                           │
              │    • School type identification                │
              │                                                 │
              ├─── TUITION LOOKUP ─────────────────────────────┤
              │    (school_tuition_mapping.csv)                │
              │    • Match by school_name (normalized)         │
              │    • Match by school_id (fallback)             │
              │    • Public schools = $0                       │
              │    • Private schools = mapped or NULL          │
              │                                                 │
              ├─── FALLBACK: Lead ZIP code ────────────────────┤
              │    • Use if no school match found              │
              │                                                 │
              │  Output: zip_code, school_type, tuition_cost,  │
              │          matched_school_id, matched_school_name │
              │          match_method, match_confidence         │
              │
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ PHASE 3: PROPERTY ENRICHMENT (property.py)                  │
    └─────────────────────────────────────────────────────────────┘
              │
              │  Input: zip_code (from Phase 2), state
              │  API: Census Bureau ACS 5-Year Data
              │  Output: median_property_value
              │
              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CHECKPOINT SAVE                                                  │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │ save_checkpoint_batch()                                      │
    │   • Load existing checkpoint                                 │
    │   • Remove old versions of batch IDs                         │
    │   • Append new batch results                                 │
    │   • Save to checkpoint CSV                                   │
    └─────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌───────────────────────────┐    ┌───────────────────────────┐
    │ leads_positive_checkpoint │    │ leads_negative_checkpoint │
    │         .csv              │    │         .csv              │
    │     (updated)             │    │     (updated)             │
    └───────────────────────────┘    └───────────────────────────┘
              │
              └──────────────┬────────────────┘
                             │
                    Return to STEP 3
                  (next batch/iteration)

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: COMPLETION                                                       │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │ All records enriched → Generate Summary Report               │
    │   • Total records processed                                  │
    │   • Apollo enrichment success rate                           │
    │   • School matching success rate                             │
    │   • Property enrichment success rate                         │
    └─────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════

INPUT FILES:
├── leads_positive.csv              (new positive leads)
├── leads_positive_old.csv          (historical positive leads)
├── leads_negative.csv              (new negative leads)
├── leads_negative_old.csv          (historical negative leads)
├── nces_schools.csv               (optional: public school database)
└── school_tuition_mapping.csv     (optional: tuition costs)

CHECKPOINT FILES (auto-created/updated):
├── leads_positive_checkpoint.csv   (incremental progress)
└── leads_negative_checkpoint.csv   (incremental progress)

OUTPUT FIELDS ADDED:
├── Apollo: apollo_job_title, apollo_company_name, apollo_seniority,
│           apollo_linkedin_url, apollo_city, apollo_state, apollo_country,
│           apollo_employment_history, apollo_education, apollo_phone_numbers,
│           apollo_confidence, apollo_enriched
├── School: zip_code, school_type, tuition_cost, school_quality_score,
│           matched_school_id, matched_school_name, match_method,
│           match_confidence
└── Property: median_property_value, property_quality_score


═══════════════════════════════════════════════════════════════════════════
KEY FEATURES
═══════════════════════════════════════════════════════════════════════════

• Batch-Level Completion: All 10 records in a batch complete before moving on
• Robust Checkpointing: Never deletes data, only appends/updates
• Rate Limit Handling: Alternates datasets when Apollo rate limit hit
• Resume Capability: Picks up exactly where it left off
• Multi-Source School Matching: API → NCES → Fallback (ZIP)
• Tuition Mapping: Supports both school_name and school_id matching
```
