import pandas as pd
import os
import logging
from typing import Optional, List


logger = logging.getLogger(__name__)



def load_checkpoint(checkpoint_path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        df = pd.read_csv(checkpoint_path)
        logger.info(f"Loaded {len(df)} records from checkpoint")
        return df
    return None



def save_checkpoint_batch(df: pd.DataFrame, checkpoint_path: str, batch_ids: List):
    existing_df = load_checkpoint(checkpoint_path)
    
    id_col = None
    for col in ['LEAD_ID', 'OPP_ID', 'ID', 'id']:
        if col in df.columns:
            id_col = col
            break
    
    if not id_col:
        logger.error("Cannot find ID column for checkpoint")
        return
    
    batch_df = df[df[id_col].isin(batch_ids)].copy()
    
    if existing_df is not None:
        existing_df = existing_df[~existing_df[id_col].isin(batch_ids)]
        combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
    else:
        combined_df = batch_df
    
    combined_df.to_csv(checkpoint_path, index=False)
    logger.info(f"Checkpoint updated: {len(batch_ids)} records from this batch, {len(combined_df)} total")
