import json
import pandas as pd
from tqdm import tqdm

# Build a reference document for all label headers extracted from the full set of 12 FDA dumps
def build_headers_csv():
    file_numbers = [f'{i:04}' for i in range(1, 13)]
    headers = []
    for number in tqdm(file_numbers):
        f = open(f'data/drug-label-{number}-of-0012.json','r')
        data = json.load(f)
        headers = _extract_headers(data, headers)  
    headers_df = pd.DataFrame({'headers': headers})
    headers_df['headers'].value_counts().reset_index().to_csv('fda_headers.csv')
    print('FDA label headers reference saved to directory!')

def _extract_headers(data, headers_list):  # Extract all keys across entire document
    for entry in data['results']:
        for key in list(entry.keys()):
            headers_list.append(key)    
    return(headers_list)

# Headers to extract
# TODO: Load this from an external txt or csv document
HEADERS = ['brand_name',
           'application_number',
            'adverse_reactions',
            'clinical_studies',
            'indications_and_usage',
            'contraindications',
            'warnings_and_cautions',
            'warnings',
            'precautions',
            'pharmacokinetics',
            'purpose',
            'clinical_pharmacology',
            'active_ingredient',
            'stop_use',
            'boxed_warning',
            'pharmacodynamics',
            'pharmacogenomics'
            ]


def build_fda_database():
    file_numbers = [f'{i:04}' for i in range(1, 13)]
    fda_df = pd.DataFrame(columns=HEADERS)

    for number in tqdm(file_numbers):
        f = open(f'data/drug-label-{number}-of-0012.json','r')
        data = json.load(f)
        fda_df = pd.concat([fda_df,__extract_fda_data(data)],axis=0).reset_index(drop=True)
    fda_df.to_excel('openfda.xlsx')
    print('FDA labels sectioned and saved to directory!')

# Extract FDA Data
def __extract_fda_data(data):
    tdf = pd.DataFrame(columns=HEADERS)
    for entry in data['results']:
        build_dict = {}
        for header in HEADERS:
            # Only two fields with differing structure than other headers
            if header == 'brand_name':
                try:
                    build_dict[header] = entry['openfda'][header][0] #TODO: Handle broken brand names
                except:
                    build_dict[header] = None
            elif header == 'application_number': 
                try:
                    build_dict[header] = entry['openfda'][header][0] #TODO: Handle broken application numbers
                except:
                    build_dict[header] = None
            # All other headers extracted this way
            else:
                try:
                    build_dict[header] = entry[header][0]
                except:
                    build_dict[header] = None
        tdf = pd.concat([tdf,pd.DataFrame.from_dict([build_dict])], axis=0).reset_index(drop=True)

    return(tdf)