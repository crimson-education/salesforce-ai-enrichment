import pandas as pd
import requests
import re
import time
import logging
from typing import Dict, Optional


logger = logging.getLogger(__name__)



class PropertyEnricher:

    STATE_FIPS = {
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
        'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
        'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
        'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
        'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
        'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
        'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
        'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
        'WI': '55', 'WY': '56', 'DC': '11', 'PR': '72'
    }
    

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = {}
        self.session = requests.Session()


    def enrich(self, zip_code: str, state: str) -> Dict:
        if not zip_code or pd.isna(zip_code):
            return self._empty_result()
        
        zip_code = self._validate_zip(zip_code)
        if not zip_code:
            return self._empty_result()
        
        if zip_code in self.cache:
            return self.cache[zip_code]
        
        result = self._query_census_zcta(zip_code)
        if result:
            self.cache[zip_code] = result
            return result
        
        result = self._empty_result()
        self.cache[zip_code] = result
        return result


    def _empty_result(self) -> Dict:
        return {'median_property_value': None, 'quality_score': 0.0}


    def _validate_zip(self, zip_code: str) -> Optional[str]:
        try:
            zip_str = re.sub(r'[^\d]', '', str(zip_code).strip())
            if len(zip_str) > 5:
                zip_str = zip_str[:5]
            if len(zip_str) < 5:
                zip_str = zip_str.zfill(5)
            return zip_str if len(zip_str) == 5 else None
        except:
            return None


    def _query_census_zcta(self, zip_code: str) -> Optional[Dict]:
        try:
            if not self.api_key:
                return None
            
            year = 2022
            url = f"https://api.census.gov/data/{year}/acs/acs5"
            params = {
                'get': 'B25077_001E,NAME',
                'for': f'zip code tabulation area:{zip_code}',
                'key': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    value_str = data[1][0]
                    if value_str and value_str not in ['-666666666', '-888888888', '-999999999', 'null']:
                        try:
                            value = float(value_str)
                            if value > 0:
                                return {
                                    'median_property_value': value,
                                    'quality_score': 1.0
                                }
                        except (ValueError, TypeError):
                            pass
            
            time.sleep(0.2)
            return None
        
        except Exception as e:
            logger.debug(f"Census query failed for {zip_code}: {e}")
            return None
