import pandas as pd
import sys
from pathlib import Path



def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None



def analyze_apollo_enrichment(df):
    print("\n" + "="*80)
    print("APOLLO ENRICHMENT ANALYSIS")
    print("="*80)
    
    apollo_enriched = df['apollo_enriched'].sum()
    total_records = len(df)
    apollo_rate = (apollo_enriched / total_records) * 100
    
    print(f"\nOverall Apollo Enrichment:")
    print(f"  Enriched: {apollo_enriched:,} / {total_records:,} records")
    print(f"  Success Rate: {apollo_rate:.2f}%")
    
    apollo_fields = [col for col in df.columns if col.startswith('apollo_') and col != 'apollo_enriched']
    
    print(f"\nField-Level Enrichment (of {apollo_enriched} enriched records):")
    print("-" * 80)
    
    field_stats = []
    for field in apollo_fields:
        non_empty = df[field].notna() & (df[field] != '') & (df[field] != 0.0)
        count = non_empty.sum()
        pct_of_enriched = (count / apollo_enriched * 100) if apollo_enriched > 0 else 0
        pct_of_total = (count / total_records * 100)
        
        field_stats.append({
            'Field': field.replace('apollo_', ''),
            'Count': count,
            '% of Enriched': pct_of_enriched,
            '% of Total': pct_of_total
        })
    
    stats_df = pd.DataFrame(field_stats).sort_values('Count', ascending=False)
    for _, row in stats_df.iterrows():
        print(f"  {row['Field']:<25} {row['Count']:>4} ({row['% of Enriched']:>5.1f}% of enriched, {row['% of Total']:>5.1f}% of total)")
    
    return apollo_rate, stats_df



def analyze_zip_enrichment(df):
    print("\n" + "="*80)
    print("ZIP CODE & SCHOOL ENRICHMENT ANALYSIS")
    print("="*80)
    
    total_records = len(df)
    
    zip_enriched = df['zip_code'].notna() & (df['zip_code'] != '')
    zip_count = zip_enriched.sum()
    zip_rate = (zip_count / total_records) * 100
    
    print(f"\nZIP Code Enrichment:")
    print(f"  Enriched: {zip_count:,} / {total_records:,} records")
    print(f"  Success Rate: {zip_rate:.2f}%")
    
    school_matched = df['matched_school_name'].notna() & (df['matched_school_name'] != '')
    school_count = school_matched.sum()
    school_rate = (school_count / total_records) * 100
    
    print(f"\nSchool Matching:")
    print(f"  Matched: {school_count:,} / {total_records:,} records")
    print(f"  Success Rate: {school_rate:.2f}%")
    
    if school_count > 0:
        print(f"\nMatch Methods (of {school_count} matched):")
        match_methods = df[school_matched]['match_method'].value_counts()
        for method, count in match_methods.items():
            pct = (count / school_count) * 100
            print(f"  {method:<20} {count:>4} ({pct:>5.1f}%)")
        
        avg_confidence = df[school_matched]['match_confidence'].mean()
        print(f"\nAverage Match Confidence: {avg_confidence:.1f}%")
    
    zip_fields = ['school_type', 'tuition_cost', 'school_quality_score', 
                  'median_property_value', 'property_quality_score']
    
    print(f"\nOther ZIP-based Fields (of {zip_count} with ZIP codes):")
    print("-" * 80)
    
    for field in zip_fields:
        if field in df.columns:
            non_empty = df[field].notna() & (df[field] != '') & (df[field] != 0.0)
            count = non_empty.sum()
            pct_of_zip = (count / zip_count * 100) if zip_count > 0 else 0
            pct_of_total = (count / total_records * 100)
            print(f"  {field:<30} {count:>4} ({pct_of_zip:>5.1f}% of ZIP enriched, {pct_of_total:>5.1f}% of total)")
    
    return zip_rate, school_rate



def calculate_overall_enrichment(df):
    print("\n" + "="*80)
    print("OVERALL ENRICHMENT SUCCESS")
    print("="*80)
    
    total_records = len(df)
    
    apollo_enriched = df['apollo_enriched'] == True
    zip_enriched = df['zip_code'].notna() & (df['zip_code'] != '')
    
    overall_enriched = apollo_enriched | zip_enriched
    overall_count = overall_enriched.sum()
    overall_rate = (overall_count / total_records) * 100
    
    both_enriched = apollo_enriched & zip_enriched
    both_count = both_enriched.sum()
    both_rate = (both_count / total_records) * 100
    
    only_apollo = apollo_enriched & ~zip_enriched
    only_apollo_count = only_apollo.sum()
    
    only_zip = zip_enriched & ~apollo_enriched
    only_zip_count = only_zip.sum()
    
    no_enrichment = ~overall_enriched
    no_enrichment_count = no_enrichment.sum()
    
    print(f"\nEnrichment Breakdown:")
    print(f"  Both Apollo & ZIP:     {both_count:>4} ({both_rate:>5.2f}%)")
    print(f"  Only Apollo:           {only_apollo_count:>4} ({only_apollo_count/total_records*100:>5.2f}%)")
    print(f"  Only ZIP:              {only_zip_count:>4} ({only_zip_count/total_records*100:>5.2f}%)")
    print(f"  No Enrichment:         {no_enrichment_count:>4} ({no_enrichment_count/total_records*100:>5.2f}%)")
    print(f"  " + "-" * 40)
    print(f"  TOTAL ENRICHED:        {overall_count:>4} ({overall_rate:>5.2f}%)")
    print(f"  Total Records:         {total_records:>4}")
    
    return overall_rate



def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.analyze_enrichment <checkpoint_csv_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("="*80)
    print("LEAD ENRICHMENT SUCCESS ANALYSIS")
    print("="*80)
    
    df = load_data(filepath)
    if df is None:
        return
    
    apollo_rate, apollo_stats = analyze_apollo_enrichment(df)
    zip_rate, school_rate = analyze_zip_enrichment(df)
    overall_rate = calculate_overall_enrichment(df)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal Records Analyzed: {len(df):,}")
    print(f"\nEnrichment Success Rates:")
    print(f"  Overall (Apollo OR ZIP):  {overall_rate:.2f}%")
    print(f"  Apollo Only:              {apollo_rate:.2f}%")
    print(f"  ZIP Code Only:            {zip_rate:.2f}%")
    print(f"  School Matching:          {school_rate:.2f}%")
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80 + "\n")



if __name__ == "__main__":
    main()
