# Lead Enrichment System

Data enrichment pipeline for lead scoring using Apollo.io, Census Bureau, and school matching APIs.

📊 **[View Flow Diagram](FLOW_DIAGRAM.md)** - Visual pipeline overview  
🚀 **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes


## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy requests fuzzywuzzy python-Levenshtein python-dotenv --break-system-packages
```


### 2. Start School Matcher API

```bash
cd school-matcher
npm run dev
```


### 3. Configure Environment

Create `.env` file:

```env
APOLLO_KEY=your_apollo_api_key
CENSUS_API_KEY=your_census_api_key
SCHOOL_MATCHER_API_URL=http://localhost:4000/match
```


### 4. Prepare Data Files

Place in `data/` directory or use the provided examples:

```bash
# Option 1: Use example files for testing
cd data
cp leads_positive.csv.example leads_positive.csv
cp leads_positive_old.csv.example leads_positive_old.csv
cp leads_negative.csv.example leads_negative.csv
cp leads_negative_old.csv.example leads_negative_old.csv
cp school_tuition_mapping.csv.example school_tuition_mapping.csv
cp nces_schools.csv.example nces_schools.csv

# Option 2: Place your own CSV files in data/
# See data/README.md for format specifications
```

**Required Files:**
- `leads_positive.csv` - New positive leads
- `leads_positive_old.csv` - Historical positive leads  
- `leads_negative.csv` - New negative leads
- `leads_negative_old.csv` - Historical negative leads

**Optional Files:**
- `nces_schools.csv` - NCES public school database for fallback matching
- `school_tuition_mapping.csv` - Manual tuition cost overrides


### 5. CSV File Formats

**Positive Leads CSV:**
```csv
OPP_ID,PRIMARY_GUARDIAN_EMAIL__C,NAME,SCHOOL_NAME__C,STATE_PROVINCE__C,PERSON_TYPE__C
001,parent@email.com,John Smith,Lincoln High School,CA,Guardian
002,contact@example.com,Jane Doe,Washington Academy,NY,Guardian
```

**Negative Leads CSV:**
```csv
LEAD_ID,EMAIL,NAME,SCHOOL__C,STATE,LEAD_PRIORITY__C
L001,lead@email.com,Bob Johnson,Roosevelt Middle School,TX,Guardian
L002,prospect@email.com,Alice Brown,Jefferson Elementary,FL,Guardian
```

**NCES Schools CSV (optional):**
```csv
SCH_NAME,ST,LZIP,MZIP,CHARTER_TEXT
Lincoln High School,CA,90210,90210,No
Washington Academy,NY,10001,10001,No
```

**School Tuition Mapping CSV (optional):**
```csv
school_name,tuition
Washington Academy,45000
St. Mary's Preparatory,38500
```

Supported column names:
- **School identifier:** `school_name`, `school_id`, `matched_school_id`
- **Tuition amount:** `tuition`, `tuition_cost`, `annual_tuition`


## Usage

### Quick Start (with example data)

```bash
# 1. Setup environment
./setup.sh

# 2. Edit .env with your API keys
nano .env

# 3. Use example data files
cd data
cp *.example $(basename {} .example) 2>/dev/null || for f in *.example; do cp "$f" "${f%.example}"; done
cd ..

# 4. Start School Matcher API (in separate terminal)
cd school-matcher
npm run dev

# 5. Run enrichment
python -m src.main
```

### Run Enrichment

```bash
python -m src.main
```

The script will:
1. Merge new and old lead data
2. Enrich in batches, alternating between positive/negative datasets
3. Save checkpoints after each batch
4. Resume from checkpoints if rate-limited


### Analyze Results

```bash
python -m src.analysis.analyze_enrichment data/leads_positive_checkpoint.csv
```


## Data Structure

### Input CSVs

**Positive Leads:**
- `OPP_ID`: Opportunity ID
- `PRIMARY_GUARDIAN_EMAIL__C`: Email
- `NAME`: Full name
- `SCHOOL_NAME__C`: School name
- `STATE_PROVINCE__C`: State code
- `PERSON_TYPE__C`: Person type

**Negative Leads:**
- `LEAD_ID`: Lead ID
- `EMAIL`: Email
- `NAME`: Full name
- `SCHOOL__C`: School name
- `STATE`: State code
- `LEAD_PRIORITY__C`: Priority/type


### Output Fields

**Apollo Enrichment:**
- `apollo_job_title`
- `apollo_company_name`
- `apollo_seniority`
- `apollo_linkedin_url`
- `apollo_city`, `apollo_state`, `apollo_country`
- `apollo_employment_history`
- `apollo_education`
- `apollo_phone_numbers`
- `apollo_confidence`
- `apollo_enriched` (boolean)

**School Enrichment:**
- `zip_code`
- `school_type` (public/private/charter)
- `tuition_cost`
- `school_quality_score`
- `matched_school_id`
- `matched_school_name`
- `match_method`
- `match_confidence`

**Property Enrichment:**
- `median_property_value`
- `property_quality_score`


## Architecture

**📊 See [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) for visual pipeline overview**

### School Matching Strategy

1. **PRIMARY:** School Matcher API
   - Deterministic matching (exact, alias, token)
   - Fuzzy matching with confidence scores
   
2. **SECONDARY:** NCES local database
   - ZIP code and school type for public schools
   - Supplements API results
   
3. **FALLBACK:** Lead address ZIP code


### Checkpoint System

- Batch-level completion (all 10 records enriched before moving on)
- Never deletes data
- Resumes from last completed batch
- Alternates between datasets after rate limits


## File Structure

```
lead-enrichment/
├── src/
│   ├── enrichers/
│   │   ├── apollo.py
│   │   ├── school.py
│   │   └── property.py
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── data_merge.py
│   │   └── name_parser.py
│   ├── analysis/
│   │   └── analyze_enrichment.py
│   └── main.py
├── data/
├── config/
│   └── column_mappings.py
├── tests/
├── .env
└── README.md
```
