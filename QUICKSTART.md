# Quick Start Guide

## 1. Initial Setup (5 minutes)

```bash
# Clone or extract the repository
cd lead-enrichment

# Run setup script
./setup.sh

# Edit environment variables
nano .env
# Add your API keys:
#   APOLLO_KEY=your_apollo_key
#   CENSUS_API_KEY=your_census_key
```

## 2. Prepare Test Data (2 minutes)

```bash
# Use example CSV files for testing
cd data
for f in *.example; do cp "$f" "${f%.example}"; done
cd ..
```

## 3. Start School Matcher API (separate terminal)

```bash
cd path/to/school-matcher
npm run dev
# Should start on http://localhost:4000
```

## 4. Run Enrichment

```bash
# From lead-enrichment directory
python -m src.main
```

## 5. Monitor Progress

The script will:
- ✅ Merge new and old data
- ✅ Process batches of 10 records
- ✅ Save checkpoints after each batch
- ✅ Alternate between positive/negative datasets
- ✅ Resume automatically if rate-limited

Watch for:
```
ITERATION 1
============================================================
>>> Processing POSITIVE dataset
--- Batch 1/5 (10 records) ---
Calling Apollo API for 10 people...
Apollo API success: 8/10 people enriched
✅ API Match: Lincoln High School → Lincoln HS (95%, exact_match)
Checkpoint updated: 10 records from this batch, 10 total
```

## 6. Analyze Results

```bash
python -m src.analysis.analyze_enrichment data/leads_positive_checkpoint.csv
```

Output:
```
APOLLO ENRICHMENT ANALYSIS
==========================
  Enriched: 45 / 50 records
  Success Rate: 90.00%

SCHOOL ENRICHMENT ANALYSIS
===========================
  Matched: 48 / 50 records
  Success Rate: 96.00%
```

## Common Issues

### Rate Limited by Apollo
```
Apollo API rate limit exceeded!
Apollo rate limit hit - stopping enrichment for this dataset
```
**Solution:** Wait 1 hour, then run `python -m src.main` again. It will resume from checkpoint.

### School Matcher API Not Available
```
⚠️ School Matcher API not available at http://localhost:4000/match
Will use NCES fallback only
```
**Solution:** Start the School Matcher API in a separate terminal.

### Missing CSV Files
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/leads_positive.csv'
```
**Solution:** Copy example files: `cd data && for f in *.example; do cp "$f" "${f%.example}"; done`

## File Locations

| File | Purpose |
|------|---------|
| `data/leads_positive_checkpoint.csv` | Enriched positive leads (in progress) |
| `data/leads_negative_checkpoint.csv` | Enriched negative leads (in progress) |
| `data/*.csv.example` | Example input file formats |

## Next Steps

1. **Replace example data** with your actual CSV files
2. **Customize column mappings** in `config/column_mappings.py` if needed
3. **Add tuition data** to `data/school_tuition_mapping.csv`
4. **Run full enrichment** on production data

## Performance

- **Batch size:** 10 records
- **Apollo API:** ~2-3 seconds per batch
- **School Matcher:** ~0.4 seconds per school
- **Census API:** ~0.2 seconds per ZIP code
- **Total time:** ~50 records/minute (with all APIs)

## Need Help?

See full documentation:
- [README.md](README.md) - Complete setup and usage
- [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) - Visual pipeline overview
- [data/README.md](data/README.md) - CSV file formats
