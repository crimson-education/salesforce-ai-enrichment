import pandas as pd
import requests
import re
import os
import json
import time
import logging
from typing import Dict, Optional
from fuzzywuzzy import fuzz, process


logger = logging.getLogger(__name__)



class SchoolEnricher:

    def __init__(self, nces_data_path: str = None, matcher_api_url: str = "http://localhost:4000/match", 
                 tuition_mapping_path: str = None):
        self.cache = {}
        self.session = requests.Session()
        self.matcher_api_url = matcher_api_url
        self.api_available = self._test_api_connection()
        self.match_stats = {'api_matched': 0, 'nces_matched': 0, 'fallback': 0, 'failed': 0}
        
        self.abbreviations = {
            'high school': ['hs', 'h.s.', 'high', 'secondary', 'sr'],
            'elementary': ['elem', 'el', 'elementary school', 'primary'],
            'middle school': ['ms', 'm.s.', 'middle', 'junior high', 'intermediate'],
            'academy': ['acad', 'academic'],
            'saint': ['st', 'st.'],
            'preparatory': ['prep', 'preparatory school'],
        }
        
        self.private_keywords = [
            'prep', 'preparatory', 'academy', 'collegiate', 'day school',
            'catholic', 'christian', 'jesuit', 'episcopal', 'private', 'independent'
        ]
        
        self.tuition_map = {}
        if tuition_mapping_path and os.path.exists(tuition_mapping_path):
            logger.info(f"Loading tuition mapping from {tuition_mapping_path}")
            tuition_df = pd.read_csv(tuition_mapping_path)
            
            tuition_col = None
            for col in ['tuition', 'tuition_cost', 'annual_tuition', 'Tuition']:
                if col in tuition_df.columns:
                    tuition_col = col
                    break
            
            if tuition_col:
                if 'school_name' in tuition_df.columns:
                    for _, row in tuition_df.iterrows():
                        school_name = row['school_name']
                        tuition = row[tuition_col]
                        if pd.notna(school_name) and pd.notna(tuition):
                            try:
                                normalized_name = self.normalize_school_name(str(school_name))
                                self.tuition_map[normalized_name] = float(tuition)
                            except (ValueError, TypeError):
                                pass
                elif 'school_id' in tuition_df.columns:
                    for _, row in tuition_df.iterrows():
                        school_id = row['school_id']
                        tuition = row[tuition_col]
                        if pd.notna(school_id) and pd.notna(tuition):
                            try:
                                self.tuition_map[str(school_id)] = float(tuition)
                            except (ValueError, TypeError):
                                pass
                
                logger.info(f"Loaded {len(self.tuition_map)} tuition mappings")
            else:
                logger.warning(f"Could not find tuition column in mapping")
        else:
            logger.info("No tuition mapping file provided")
        
        self.nces_df = None
        self.nces_by_state = {}
        
        if nces_data_path and os.path.exists(nces_data_path):
            logger.info(f"Loading NCES data from {nces_data_path}")
            self.nces_df = pd.read_csv(nces_data_path, dtype=str, low_memory=False)
            logger.info(f"Loaded {len(self.nces_df)} NCES schools")
            
            if 'ST' in self.nces_df.columns:
                for state_code in self.nces_df['ST'].unique():
                    if pd.notna(state_code):
                        state_data = self.nces_df[self.nces_df['ST'] == state_code].copy()
                        state_data['normalized'] = state_data['SCH_NAME'].apply(self.normalize_school_name)
                        self.nces_by_state[state_code.upper()] = state_data
                logger.info(f"Indexed {len(self.nces_by_state)} states")


    def _test_api_connection(self) -> bool:
        try:
            response = self.session.get(self.matcher_api_url.replace('/match', '/'), timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ School Matcher API available at {self.matcher_api_url}")
                
                possible_paths = [
                    os.path.join('..', 'school-matcher', 'src', 'data', 'match-logs.json'),
                    r'C:\Users\taske\Desktop\Crimson Fulltime\school-matcher\src\data\match-logs.json',
                    'match-logs.json',
                    os.path.join('school-matcher', 'src', 'data', 'match-logs.json'),
                ]
                
                log_file_found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"✅ Found match log file at: {path}")
                        log_file_found = True
                        break
                
                if not log_file_found:
                    logger.warning(f"⚠️ Could not find match-logs.json")
                
                return True
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to School Matcher API: {e}")
        
        logger.warning(f"⚠️ School Matcher API not available at {self.matcher_api_url}")
        logger.warning("Will use NCES fallback only")
        return False


    def _get_tuition_cost(self, school_id: Optional[str], school_name: Optional[str], school_type: Optional[str]) -> Optional[float]:
        if school_name:
            normalized_name = self.normalize_school_name(str(school_name))
            if normalized_name in self.tuition_map:
                return self.tuition_map[normalized_name]
        
        if school_id and str(school_id) in self.tuition_map:
            return self.tuition_map[str(school_id)]
        
        if school_type and 'public' in str(school_type).lower():
            return 0.0
        
        return None


    def normalize_school_name(self, name: str) -> str:
        if pd.isna(name):
            return ""
        name = str(name).lower().strip()
        for full, abbrevs in self.abbreviations.items():
            for abbrev in abbrevs:
                name = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, name)
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name


    def enrich(self, school_name: str, state: str, lead_zip: str = None) -> Dict:
        if pd.isna(school_name) or not school_name:
            if lead_zip and not pd.isna(lead_zip):
                self.match_stats['fallback'] += 1
                return {
                    'zip_code': str(lead_zip).strip()[:5],
                    'school_type': 'unknown',
                    'tuition_cost': None,
                    'quality_score': 0.3,
                    'matched_school_id': None,
                    'matched_school_name': None,
                    'match_method': 'lead_zip_fallback',
                    'match_confidence': 30
                }
            self.match_stats['failed'] += 1
            return self._empty_result()
        
        cache_key = f"{self.normalize_school_name(school_name)}_{str(state).upper()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        api_result = None
        if self.api_available:
            api_result = self._query_school_matcher_api(school_name, state)
            if api_result and api_result.get('match_confidence', 0) >= 80:
                tuition = self._get_tuition_cost(
                    api_result.get('matched_school_id'),
                    api_result.get('matched_school_name'),
                    api_result.get('school_type')
                )
                api_result['tuition_cost'] = tuition
                
                if not api_result.get('zip_code') and self.nces_df is not None:
                    nces_result = self._query_nces_local(school_name, state)
                    if nces_result and nces_result.get('zip_code'):
                        api_result['zip_code'] = nces_result['zip_code']
                        if not api_result.get('school_type'):
                            api_result['school_type'] = nces_result.get('school_type')
                
                if not api_result.get('zip_code') and lead_zip and not pd.isna(lead_zip):
                    api_result['zip_code'] = str(lead_zip).strip()[:5]
                
                self.match_stats['api_matched'] += 1
                self.cache[cache_key] = api_result
                return api_result
        
        if self.nces_df is not None:
            result = self._query_nces_local(school_name, state)
            if result:
                tuition = self._get_tuition_cost(
                    result.get('matched_school_id'),
                    result.get('matched_school_name'),
                    result.get('school_type')
                )
                result['tuition_cost'] = tuition
                
                self.match_stats['nces_matched'] += 1
                self.cache[cache_key] = result
                return result
        
        is_likely_private = self._is_likely_private(school_name)
        
        if lead_zip and not pd.isna(lead_zip):
            self.match_stats['fallback'] += 1
            result = {
                'zip_code': str(lead_zip).strip()[:5],
                'school_type': 'private' if is_likely_private else 'unknown',
                'tuition_cost': None,
                'quality_score': 0.4,
                'matched_school_id': None,
                'matched_school_name': None,
                'match_method': 'lead_zip_fallback',
                'match_confidence': 40
            }
            self.cache[cache_key] = result
            return result
        
        self.match_stats['failed'] += 1
        result = self._empty_result()
        self.cache[cache_key] = result
        return result


    def _query_school_matcher_api(self, school_name: str, state: str) -> Optional[Dict]:
        try:
            possible_paths = [
                os.path.join('..', 'school-matcher', 'src', 'data', 'match-logs.json'),
                r'C:\Users\taske\Desktop\Crimson Fulltime\school-matcher\src\data\match-logs.json',
                'match-logs.json',
                os.path.join('school-matcher', 'src', 'data', 'match-logs.json'),
            ]
            
            log_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    log_file = path
                    break
            
            if not log_file:
                logger.debug(f"Log file not found - API may not be available")
                return None
            
            pre_call_log_count = 0
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                    pre_call_log_count = len(existing_logs)
            except (json.JSONDecodeError, IOError, FileNotFoundError):
                pre_call_log_count = 0
            
            payload = [{
                "School__c": str(school_name).strip(),
                "state": str(state).strip() if state and not pd.isna(state) else "",
                "country": "United States of America",
                "email": f"enrichment_{int(time.time() * 1000)}@placeholder.com",
                "firstName": "",
                "lastName": ""
            }]
            
            response = self.session.post(
                self.matcher_api_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                logger.debug(f"School Matcher API error: {response.status_code}")
                return None
            
            time.sleep(0.4)
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    all_logs = json.load(f)
                
                if len(all_logs) <= pre_call_log_count:
                    logger.debug(f"No new log entry found for {school_name}")
                    return None
                
                latest_log = all_logs[-1]
                
                raw_input_from_log = latest_log.get('raw_school_input', '').lower().strip()
                our_input = school_name.lower().strip()
                
                if raw_input_from_log != our_input:
                    logger.debug(f"Latest log entry doesn't match our school")
                    return None
                
                matched_name = latest_log.get('matched_school_name')
                match_confidence = latest_log.get('match_confidence', 0)
                match_method = latest_log.get('match_method')
                
                if not matched_name or match_confidence < 80:
                    logger.debug(f"No match or low confidence for {school_name}")
                    return None
                
                school_type = self._determine_school_type_from_match(matched_name, school_name)
                matched_state = latest_log.get('matched_state', '')
                
                logger.info(f"✅ API Match: {school_name} → {matched_name} ({match_confidence}%, {match_method})")
                
                return {
                    'zip_code': None,
                    'school_type': school_type,
                    'tuition_cost': None,
                    'quality_score': match_confidence / 100.0,
                    'matched_school_id': latest_log.get('matched_school_id'),
                    'matched_school_name': matched_name,
                    'match_method': match_method,
                    'match_confidence': match_confidence,
                    'matched_state': matched_state
                }
                
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading match logs: {e}")
                return None
        
        except Exception as e:
            logger.error(f"School Matcher API error for {school_name}: {e}")
            return None


    def _determine_school_type_from_match(self, matched_name: str, original_name: str) -> str:
        if pd.isna(matched_name):
            matched_name = ""
        if pd.isna(original_name):
            original_name = ""
        
        matched_lower = str(matched_name).lower()
        original_lower = str(original_name).lower()
        
        if self._is_likely_private(matched_name) or self._is_likely_private(original_name):
            return 'private'
        
        return 'public'


    def _query_nces_local(self, school_name: str, state: str) -> Optional[Dict]:
        try:
            if not state or pd.isna(state):
                return None
            
            state_upper = str(state).upper().strip()[:2]
            state_schools = self.nces_by_state.get(state_upper)
            
            if state_schools is None or len(state_schools) == 0:
                return None
            
            normalized_name = self.normalize_school_name(school_name)
            if not normalized_name:
                return None
            
            matches = process.extract(
                normalized_name,
                state_schools['normalized'].tolist(),
                scorer=fuzz.token_sort_ratio,
                limit=1
            )
            
            if matches and matches[0][1] >= 80:
                matched_normalized = matches[0][0]
                match_idx = state_schools[state_schools['normalized'] == matched_normalized].index[0]
                school_row = state_schools.loc[match_idx]
                
                zip_code = None
                if pd.notna(school_row.get('LZIP')):
                    zip_code = str(school_row['LZIP']).strip()[:5]
                elif pd.notna(school_row.get('MZIP')):
                    zip_code = str(school_row['MZIP']).strip()[:5]
                
                school_type = 'public'
                charter_text = str(school_row.get('CHARTER_TEXT', '')).lower()
                if 'yes' in charter_text or 'charter' in charter_text:
                    school_type = 'public_charter'
                
                return {
                    'zip_code': zip_code,
                    'school_type': school_type,
                    'tuition_cost': 0.0,
                    'quality_score': matches[0][1] / 100.0,
                    'matched_school_id': None,
                    'matched_school_name': school_row['SCH_NAME'],
                    'match_method': 'nces_fuzzy',
                    'match_confidence': matches[0][1]
                }
        
        except Exception as e:
            logger.debug(f"NCES query failed for {school_name}: {e}")
        
        return None


    def _is_likely_private(self, school_name: str) -> bool:
        if pd.isna(school_name):
            return False
        name_lower = str(school_name).lower()
        return any(keyword in name_lower for keyword in self.private_keywords)


    def _empty_result(self) -> Dict:
        return {
            'zip_code': None,
            'school_type': None,
            'tuition_cost': None,
            'quality_score': 0.0,
            'matched_school_id': None,
            'matched_school_name': None,
            'match_method': None,
            'match_confidence': 0
        }
