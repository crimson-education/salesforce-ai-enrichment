import pandas as pd
import os
import logging


logger = logging.getLogger(__name__)



def merge_lead_datasets(new_csv: str, old_csv: str, id_column: str) -> pd.DataFrame:
    logger.info(f"Loading new data from {new_csv}")
    df_new = pd.read_csv(new_csv)
    
    if not os.path.exists(old_csv):
        logger.warning(f"Old data file not found: {old_csv}, using new data only")
        return df_new
    
    logger.info(f"Loading old data from {old_csv}")
    df_old = pd.read_csv(old_csv)
    
    logger.info(f"New data: {len(df_new)} records, Old data: {len(df_old)} records")
    
    if id_column not in df_new.columns:
        for col in df_new.columns:
            if 'id' in col.lower() and col in df_old.columns:
                id_column = col
                break
    
    logger.info(f"Using ID column: {id_column}")
    
    df_merged = df_new.copy()
    
    for idx, row in df_merged.iterrows():
        record_id = row[id_column]
        old_match = df_old[df_old[id_column] == record_id]
        
        if len(old_match) > 0:
            old_row = old_match.iloc[0]
            for col in df_old.columns:
                if col in df_merged.columns:
                    if pd.isna(df_merged.at[idx, col]) and pd.notna(old_row[col]):
                        df_merged.at[idx, col] = old_row[col]
    
    logger.info(f"Merged dataset: {len(df_merged)} records with {len(df_merged.columns)} columns")
    return df_merged
