# Data Directory

This directory contains CSV files for the enrichment pipeline.

## File Naming Convention

### Working Files
Place your actual data files here with these exact names:
- `leads_positive.csv`
- `leads_positive_old.csv`
- `leads_negative.csv`
- `leads_negative_old.csv`
- `nces_schools.csv` (optional)
- `school_tuition_mapping.csv` (optional)

### Example Files
Example files are provided with `.example` extension:
- `*.csv.example` - Example input file formats

To use the examples:
```bash
# Copy and rename example files to working files
cp leads_positive.csv.example leads_positive.csv
cp leads_positive_old.csv.example leads_positive_old.csv
cp leads_negative.csv.example leads_negative.csv
cp leads_negative_old.csv.example leads_negative_old.csv
cp school_tuition_mapping.csv.example school_tuition_mapping.csv
cp nces_schools.csv.example nces_schools.csv
```

### Checkpoint Files
These are automatically created/updated by the enrichment process:
- `leads_positive_checkpoint.csv` - Incremental progress for positive leads
- `leads_negative_checkpoint.csv` - Incremental progress for negative leads

**Do not manually edit checkpoint files.**

## File Formats

### Required Fields

**Positive Leads:**
- `OPP_ID` - Unique opportunity identifier
- `PRIMARY_GUARDIAN_EMAIL__C` - Email address
- `NAME` - Full name
- `SCHOOL_NAME__C` - School name
- `STATE_PROVINCE__C` - State code (e.g., CA, NY)
- `PERSON_TYPE__C` - Type (should contain "Guardian")

**Negative Leads:**
- `LEAD_ID` - Unique lead identifier
- `EMAIL` - Email address
- `NAME` - Full name
- `SCHOOL__C` - School name
- `STATE` - State code
- `LEAD_PRIORITY__C` - Priority/type (should contain "Guardian")

### Optional Fields

**School Tuition Mapping:**
- `school_name` - Canonical school name (normalized)
- `tuition` - Annual tuition cost in USD

**NCES Schools:**
- `SCH_NAME` - School name
- `ST` - State code
- `LZIP` - Location ZIP code
- `MZIP` - Mailing ZIP code
- `CHARTER_TEXT` - Charter status (Yes/No)

## Notes

- All `.csv` files in this directory are ignored by git (except `.example` files)
- Checkpoint files can be safely deleted to restart enrichment from scratch
- Old data files are used to fill missing values in new data during merge
