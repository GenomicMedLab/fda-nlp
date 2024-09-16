import pandas as pd
import llm
from tqdm import tqdm
import pdb




def main():

    search_space = define_search_space('20240423_trial1_total.xlsx')

    start_point = 0

    test_run = search_space[start_point:start_point+50].copy()
    test_run['outcome_raw'] = None

    results = model_loop(start_point,test_run)
    print(results)

def define_search_space(filename):
    return pd.read_excel(filename).reset_index(drop=True).drop('Unnamed: 0', axis=1)


def model_loop(start_point,test_run):
    position = start_point
    model = llm.get_model('mistral-7b-instruct-v0')
    for idx, row in tqdm(test_run.iterrows(), total=test_run.shape[0]):
        text_to_process = f'Read the following clinical studies for {row["brand_name"]}. Identify the primary treatment outcomes for each patient group. Format your answer using this JSON format: {{"result": {{"outcome": {{"metric": "string", "value": "string", "descriptors": "string"}}}}}} \n Text: {row["clinical_studies"][0:2000]}'
        response = model.prompt(text_to_process)
        test_run.at[idx, 'outcome_raw'] = response.text()
        print(response.text())
        position += 1

    return(test_run)












if __name__ == "__main__":
    main()






