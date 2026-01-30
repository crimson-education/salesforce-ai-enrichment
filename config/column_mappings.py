POSITIVE_COLUMNS = {
    'id': 'OPP_ID',
    'email': 'PRIMARY_GUARDIAN_EMAIL__C',
    'name': 'NAME',
    'school': 'SCHOOL_NAME__C',
    'state': 'STATE_PROVINCE__C',
    'zip': None,
    'person_type': 'PERSON_TYPE__C',
    'phone': None
}



NEGATIVE_COLUMNS = {
    'id': 'LEAD_ID',
    'email': 'EMAIL',
    'name': 'NAME',
    'school': 'SCHOOL__C',
    'state': 'STATE',
    'zip': None,
    'person_type': 'LEAD_PRIORITY__C',
    'phone': None
}
