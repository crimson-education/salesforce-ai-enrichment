import pandas as pd
import time
import os
import logging
from typing import Dict, Tuple, List
from dotenv import load_dotenv

from .enrichers import ApolloEnricher, SchoolEnricher, PropertyEnricher
from .utils import merge_lead_datasets, load_checkpoint, save_checkpoint_batch, extract_name_from_string
from config.column_mappings import POSITIVE_COLUMNS, NEGATIVE_COLUMNS


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


APOLLO_API_KEY = os.getenv('APOLLO_KEY', '')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY', '')
SCHOOL_MATCHER_API_URL = os.getenv('SCHOOL_MATCHER_API_URL', 'http://localhost:4000/match')



def enrich_batch(df: pd.DataFrame, batch_indices: List, 
                 apollo_enricher: ApolloEnricher,
                 school_enricher: SchoolEnricher,
                 property_enricher: PropertyEnricher,
                 column_map: Dict) -> Tuple[bool, List]:
    
    enriched_ids = []
    apollo_success = True
    
    try:
        people_data = []
        valid_indices = []
        
        for idx in batch_indices:
            row = df.loc[idx]
            email = row.get(column_map['email'])
            
            if pd.isna(email) or not email:
                df.at[idx, 'apollo_enriched'] = False
                enriched_ids.append(df.at[idx, column_map['id']])
                continue
            
            first_name, last_name = None, None
            if column_map.get('name') and pd.notna(row.get(column_map['name'])):
                first_name, last_name = extract_name_from_string(row[column_map['name']])
            
            if not first_name:
                first_name, last_name = extract_name_from_string(email)
            
            person_data = {'email': str(email).strip().lower()}
            
            if first_name:
                person_data['first_name'] = first_name
            if last_name:
                person_data['last_name'] = last_name
            
            if column_map.get('phone') and pd.notna(row.get(column_map['phone'])):
                person_data['phone'] = str(row[column_map['phone']]).strip()
            
            people_data.append(person_data)
            valid_indices.append(idx)
        
        if people_data:
            results = apollo_enricher.enrich_bulk(people_data)
            
            for i, (idx, result) in enumerate(zip(valid_indices, results)):
                for col, value in result.items():
                    df.at[idx, col] = value
                enriched_ids.append(df.at[idx, column_map['id']])
        
        for idx in batch_indices:
            if pd.isna(df.at[idx, 'apollo_enriched']):
                df.at[idx, 'apollo_enriched'] = False
                if df.at[idx, column_map['id']] not in enriched_ids:
                    enriched_ids.append(df.at[idx, column_map['id']])
    
    except Exception as e:
        logger.error(f"Apollo enrichment failed for batch: {e}")
        apollo_success = False
        
        for idx in batch_indices:
            if pd.isna(df.at[idx, 'apollo_enriched']):
                df.at[idx, 'apollo_enriched'] = False
            if df.at[idx, column_map['id']] not in enriched_ids:
                enriched_ids.append(df.at[idx, column_map['id']])
    
    logger.info(f"Batch secondary enrichment: {len(batch_indices)} records")
    
    for idx in batch_indices:
        row = df.loc[idx]
        
        school_result = school_enricher.enrich(
            school_name=row.get(column_map.get('school')),
            state=row.get(column_map.get('state')),
            lead_zip=row.get(column_map.get('zip'))
        )
        df.at[idx, 'zip_code'] = school_result['zip_code']
        df.at[idx, 'school_type'] = school_result['school_type']
        df.at[idx, 'tuition_cost'] = school_result['tuition_cost']
        df.at[idx, 'school_quality_score'] = school_result['quality_score']
        df.at[idx, 'matched_school_id'] = school_result.get('matched_school_id')
        df.at[idx, 'matched_school_name'] = school_result.get('matched_school_name')
        df.at[idx, 'match_method'] = school_result.get('match_method')
        df.at[idx, 'match_confidence'] = school_result.get('match_confidence')
        
        if school_result['zip_code']:
            property_result = property_enricher.enrich(
                school_result['zip_code'],
                row.get(column_map.get('state'))
            )
            df.at[idx, 'median_property_value'] = property_result['median_property_value']
            df.at[idx, 'property_quality_score'] = property_result['quality_score']
        else:
            df.at[idx, 'property_quality_score'] = 0.0
    
    return apollo_success, enriched_ids



def enrich_dataset(df: pd.DataFrame, checkpoint_path: str, is_positive: bool,
                   column_map: Dict, nces_data_path: str = None) -> pd.DataFrame:
    
    apollo_enricher = ApolloEnricher(api_key=APOLLO_API_KEY)
    school_enricher = SchoolEnricher(
        nces_data_path=nces_data_path,
        matcher_api_url=SCHOOL_MATCHER_API_URL,
        tuition_mapping_path=os.path.join(os.path.dirname(nces_data_path) if nces_data_path else ".", "school_tuition_mapping.csv")
    )
    property_enricher = PropertyEnricher(api_key=CENSUS_API_KEY)
    
    apollo_cols = [
        'apollo_job_title', 'apollo_company_name', 'apollo_seniority',
        'apollo_linkedin_url', 'apollo_city', 'apollo_state', 'apollo_country',
        'apollo_employment_history', 'apollo_education', 'apollo_phone_numbers',
        'apollo_confidence', 'apollo_enriched'
    ]
    
    other_cols = [
        'zip_code', 'school_type', 'tuition_cost', 'school_quality_score',
        'median_property_value', 'property_quality_score',
        'matched_school_id', 'matched_school_name', 'match_method', 'match_confidence'
    ]
    
    for col in apollo_cols + other_cols:
        if col not in df.columns:
            df[col] = None
    
    checkpoint_df = load_checkpoint(checkpoint_path)
    already_enriched_ids = set()
    
    if checkpoint_df is not None:
        id_col = column_map['id']
        if id_col in checkpoint_df.columns:
            for idx, row in df.iterrows():
                record_id = row[id_col]
                checkpoint_match = checkpoint_df[checkpoint_df[id_col] == record_id]
                
                if len(checkpoint_match) > 0:
                    checkpoint_row = checkpoint_match.iloc[0]
                    for col in apollo_cols + other_cols:
                        if col in checkpoint_df.columns and pd.notna(checkpoint_row[col]):
                            df.at[idx, col] = checkpoint_row[col]
                    
                    if checkpoint_row.get('apollo_enriched') is not None:
                        already_enriched_ids.add(record_id)
            
            logger.info(f"Resumed from checkpoint: {len(already_enriched_ids)} records already enriched")
    
    person_col = column_map.get('person_type')
    if person_col and person_col in df.columns:
        guardian_mask = df[person_col].str.lower().str.contains('guardian', na=False)
    else:
        guardian_mask = pd.Series([True] * len(df), index=df.index)
    
    id_col = column_map['id']
    unchecked_mask = ~df[id_col].isin(already_enriched_ids)
    
    to_enrich = df[guardian_mask & unchecked_mask].copy()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Enriching {'POSITIVE' if is_positive else 'NEGATIVE'} dataset")
    logger.info(f"Total guardians: {guardian_mask.sum()}")
    logger.info(f"Already enriched: {len(already_enriched_ids)}")
    logger.info(f"To enrich: {len(to_enrich)}")
    logger.info(f"{'='*60}\n")
    
    if len(to_enrich) == 0:
        logger.info("All records already enriched!")
        return df
    
    batch_size = 10
    total_batches = (len(to_enrich) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(to_enrich))
        batch_indices = to_enrich.iloc[start_idx:end_idx].index.tolist()
        
        logger.info(f"\n--- Batch {batch_num + 1}/{total_batches} ({len(batch_indices)} records) ---")
        
        apollo_success, enriched_ids = enrich_batch(
            df, batch_indices,
            apollo_enricher, school_enricher, property_enricher,
            column_map
        )
        
        save_checkpoint_batch(df, checkpoint_path, enriched_ids)
        
        if not apollo_success:
            logger.warning("Apollo rate limit hit - stopping enrichment for this dataset")
            logger.info(f"\nSchool Matching Stats:")
            logger.info(f"  API Matched: {school_enricher.match_stats['api_matched']}")
            logger.info(f"  NCES Matched: {school_enricher.match_stats['nces_matched']}")
            logger.info(f"  Fallback: {school_enricher.match_stats['fallback']}")
            logger.info(f"  Failed: {school_enricher.match_stats['failed']}")
            return df
        
        if batch_num < total_batches - 1:
            time.sleep(2)
    
    logger.info(f"\nDataset enrichment complete!")
    logger.info(f"Apollo calls: {apollo_enricher.api_calls}")
    logger.info(f"Successful: {apollo_enricher.successful_enrichments}")
    logger.info(f"Failed: {apollo_enricher.failed_enrichments}")
    logger.info(f"\nSchool Matching Stats:")
    logger.info(f"  API Matched: {school_enricher.match_stats['api_matched']}")
    logger.info(f"  NCES Matched: {school_enricher.match_stats['nces_matched']}")
    logger.info(f"  Fallback: {school_enricher.match_stats['fallback']}")
    logger.info(f"  Failed: {school_enricher.match_stats['failed']}")
    
    return df



def main():
    data_path = "data"
    
    files = {
        'positive_new': os.path.join(data_path, "leads_positive.csv"),
        'positive_old': os.path.join(data_path, "leads_positive_old.csv"),
        'negative_new': os.path.join(data_path, "leads_negative.csv"),
        'negative_old': os.path.join(data_path, "leads_negative_old.csv"),
        'positive_checkpoint': os.path.join(data_path, "leads_positive_checkpoint.csv"),
        'negative_checkpoint': os.path.join(data_path, "leads_negative_checkpoint.csv"),
        'nces': os.path.join(data_path, "nces_schools.csv")
    }
    
    nces_data_path = files['nces'] if os.path.exists(files['nces']) else None
    
    logger.info("\n" + "="*60)
    logger.info("STEP 1: MERGING NEW AND OLD DATA")
    logger.info("="*60)
    
    df_positive = merge_lead_datasets(
        files['positive_new'],
        files['positive_old'],
        POSITIVE_COLUMNS['id']
    )
    
    df_negative = merge_lead_datasets(
        files['negative_new'],
        files['negative_old'],
        NEGATIVE_COLUMNS['id']
    )
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: ENRICHMENT WITH ALTERNATING BATCHES")
    logger.info("="*60)
    
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        
        logger.info("\n>>> Processing POSITIVE dataset")
        df_positive_before = len(df_positive[df_positive['apollo_enriched'] == True]) if 'apollo_enriched' in df_positive.columns else 0
        
        df_positive = enrich_dataset(
            df_positive,
            files['positive_checkpoint'],
            is_positive=True,
            column_map=POSITIVE_COLUMNS,
            nces_data_path=nces_data_path
        )
        
        df_positive_after = len(df_positive[df_positive['apollo_enriched'] == True]) if 'apollo_enriched' in df_positive.columns else 0
        positive_progress = df_positive_after - df_positive_before
        
        logger.info("\n>>> Processing NEGATIVE dataset")
        df_negative_before = len(df_negative[df_negative['apollo_enriched'] == True]) if 'apollo_enriched' in df_negative.columns else 0
        
        df_negative = enrich_dataset(
            df_negative,
            files['negative_checkpoint'],
            is_positive=False,
            column_map=NEGATIVE_COLUMNS,
            nces_data_path=nces_data_path
        )
        
        df_negative_after = len(df_negative[df_negative['apollo_enriched'] == True]) if 'apollo_enriched' in df_negative.columns else 0
        negative_progress = df_negative_after - df_negative_before
        
        if positive_progress == 0 and negative_progress == 0:
            logger.info("\n" + "="*60)
            logger.info("ALL DATASETS FULLY ENRICHED!")
            logger.info("="*60)
            break
    
    logger.info("\n" + "="*60)
    logger.info("ENRICHMENT SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nPositive dataset:")
    logger.info(f"  Total records: {len(df_positive)}")
    if 'apollo_enriched' in df_positive.columns:
        logger.info(f"  Apollo enriched: {(df_positive['apollo_enriched'] == True).sum()}")
    if 'school_quality_score' in df_positive.columns:
        logger.info(f"  School enriched: {(df_positive['school_quality_score'] > 0).sum()}")
    
    logger.info(f"\nNegative dataset:")
    logger.info(f"  Total records: {len(df_negative)}")
    if 'apollo_enriched' in df_negative.columns:
        logger.info(f"  Apollo enriched: {(df_negative['apollo_enriched'] == True).sum()}")
    if 'school_quality_score' in df_negative.columns:
        logger.info(f"  School enriched: {(df_negative['school_quality_score'] > 0).sum()}")
    
    logger.info(f"\nCheckpoint files:")
    logger.info(f"  Positive: {files['positive_checkpoint']}")
    logger.info(f"  Negative: {files['negative_checkpoint']}")
    logger.info("\nRun this script again to continue enrichment after rate limits reset!")



if __name__ == "__main__":
    main()
