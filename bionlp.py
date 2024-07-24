from transformers import pipeline
import pandas as pd

def process_text(text):    
    tdf1 = __tag_genes(text)
    tdf2 = __tag_chemicals(text)
    tdf3 = __tag_diseases(text)
    df = pd.concat([tdf1,tdf2,tdf3])
    return __drop_unknowns(df)

def batch(corpus):
    df = pd.DataFrame()
    for entry in corpus:
        tdf1 = __tag_genes(entry)
        tdf2 = __tag_chemicals(entry)
        tdf3 = __tag_diseases(entry)
        df = pd.concat([tdf1,tdf2,tdf3])
    return __drop_unknowns(df)

def __drop_unknowns(result):
    return result[result['entity_group']!='0'].reset_index(drop=True)

def __tag_genes(text):
    __pipe_gene = pipeline("token-classification", model="alvaroalon2/biobert_genetic_ner",aggregation_strategy="first")
    gene_results = __pipe_gene(text)
    df = __drop_unknowns(pd.DataFrame(gene_results))
    df['original_text'] = text
    return df

def __tag_chemicals(text):
    __pipe_chemical = pipeline("token-classification", model="alvaroalon2/biobert_chemical_ner", aggregation_strategy="first")
    chem_results = __pipe_chemical(text)
    df = __drop_unknowns(pd.DataFrame(chem_results))
    df['original_text'] = text
    return df

def __tag_diseases(text):
    __pipe_disease = pipeline("token-classification", model="alvaroalon2/biobert_diseases_ner", aggregation_strategy="first")
    disease_results = __pipe_disease(text)
    df = __drop_unknowns(pd.DataFrame(disease_results))
    df['original_text'] = text
    return df