import requests
import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)



class ApolloEnricher:

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'x-api-key': self.api_key
        })
        self.api_calls = 0
        self.successful_enrichments = 0
        self.failed_enrichments = 0
        
        if not self.api_key:
            logger.error("Apollo API key required")


    def enrich_bulk(self, people_data: List[Dict]) -> List[Dict]:
        if not self.api_key:
            logger.error("Apollo API key not configured")
            return []
        
        if len(people_data) > 10:
            logger.warning(f"Apollo bulk endpoint accepts max 10 people, got {len(people_data)}. Truncating.")
            people_data = people_data[:10]
        
        try:
            url = "https://api.apollo.io/api/v1/people/bulk_match"
            
            details = []
            for person in people_data:
                detail = {}
                if person.get('email'):
                    detail['email'] = person['email']
                if person.get('first_name'):
                    detail['first_name'] = person['first_name']
                if person.get('last_name'):
                    detail['last_name'] = person['last_name']
                if person.get('organization_name'):
                    detail['organization_name'] = person['organization_name']
                if person.get('phone'):
                    detail['phone'] = person['phone']
                
                if detail.get('email') or (detail.get('first_name') and detail.get('organization_name')):
                    details.append(detail)
            
            if not details:
                logger.warning("No valid person details to enrich")
                return []
            
            body = {"details": details}
            
            logger.info(f"Calling Apollo API for {len(details)} people...")
            response = self.session.post(url, json=body, timeout=30)
            self.api_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                self.successful_enrichments += len([m for m in matches if m])
                self.failed_enrichments += len([m for m in matches if not m])
                
                logger.info(f"Apollo API success: {len([m for m in matches if m])}/{len(matches)} people enriched")
                return self._parse_apollo_results(matches)
            
            elif response.status_code == 429:
                logger.error("Apollo API rate limit exceeded!")
                raise Exception("Apollo rate limit exceeded")
            
            else:
                logger.error(f"Apollo API error: {response.status_code}")
                raise Exception(f"Apollo API error {response.status_code}")
        
        except Exception as e:
            logger.error(f"Apollo enrichment failed: {e}")
            raise


    def _parse_apollo_results(self, matches: List) -> List[Dict]:
        results = []
        
        for match in matches:
            if not match:
                results.append(self._empty_result())
                continue
            
            result = {
                'apollo_job_title': match.get('title'),
                'apollo_company_name': match.get('organization', {}).get('name') if match.get('organization') else None,
                'apollo_seniority': match.get('seniority'),
                'apollo_linkedin_url': match.get('linkedin_url'),
                'apollo_city': match.get('city'),
                'apollo_state': match.get('state'),
                'apollo_country': match.get('country'),
                'apollo_employment_history': self._extract_employment_history(match.get('employment_history', [])),
                'apollo_education': self._extract_education(match.get('education', [])),
                'apollo_phone_numbers': self._extract_phone_numbers(match.get('phone_numbers', [])),
                'apollo_confidence': 1.0,
                'apollo_enriched': True
            }
            
            results.append(result)
        
        return results


    def _extract_employment_history(self, history: List) -> Optional[str]:
        if not history:
            return None
        positions = []
        for job in history[:3]:
            title = job.get('title', 'Unknown')
            company = job.get('organization_name', 'Unknown')
            positions.append(f"{title} at {company}")
        return " | ".join(positions) if positions else None


    def _extract_education(self, education: List) -> Optional[str]:
        if not education:
            return None
        degrees = []
        for edu in education:
            degree = edu.get('degree')
            school = edu.get('school_name')
            if degree or school:
                degrees.append(f"{degree or 'Degree'} from {school or 'Unknown'}")
        return " | ".join(degrees[:2]) if degrees else None


    def _extract_phone_numbers(self, phones: List) -> Optional[str]:
        if not phones:
            return None
        numbers = [phone['raw_number'] for phone in phones if phone.get('raw_number')]
        return " | ".join(numbers) if numbers else None


    def _empty_result(self) -> Dict:
        return {
            'apollo_job_title': None,
            'apollo_company_name': None,
            'apollo_seniority': None,
            'apollo_linkedin_url': None,
            'apollo_city': None,
            'apollo_state': None,
            'apollo_country': None,
            'apollo_employment_history': None,
            'apollo_education': None,
            'apollo_phone_numbers': None,
            'apollo_confidence': 0.0,
            'apollo_enriched': False
        }
