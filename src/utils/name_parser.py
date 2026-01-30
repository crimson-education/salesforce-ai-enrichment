import pandas as pd
import re
from typing import Tuple



def extract_name_from_string(name_str: str) -> Tuple[str, str]:
    if pd.isna(name_str) or not name_str:
        return None, None
    
    name_str = str(name_str).strip()
    
    if '@' in name_str:
        local_part = name_str.split('@')[0]
        local_part = re.sub(r'[\d_\-\.]', ' ', local_part)
        parts = local_part.split()
        if len(parts) >= 2:
            return parts[0].capitalize(), parts[-1].capitalize()
        elif len(parts) == 1:
            return parts[0].capitalize(), None
    else:
        parts = name_str.split()
        if len(parts) >= 2:
            return parts[0].capitalize(), parts[-1].capitalize()
        elif len(parts) == 1:
            return parts[0].capitalize(), None
    
    return None, None
