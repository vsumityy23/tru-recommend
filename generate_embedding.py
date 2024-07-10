
import pandas as pd
import numpy as np

#load datasets
resumes_df = pd.read_csv('./datasets/resumes.csv')
projects_df = pd.read_csv('./datasets/projects.csv')


#combined text to of features
resumes_df['Combined_Text'] = resumes_df.apply(lambda row: ' '.join([row['Skills'], row['Tools'], row['Certification'], row['Past_Project_Descriptions']]), axis=1)
projects_df['Combined_Text'] = projects_df.apply(lambda row: ' '.join([row['Project_Description'], row['Skills_Required']]), axis=1)


#import generate_embedding function from utility.py
from utility import get_embeddings


#import models and tokenizers from utility.py
from utility import bert_model,bert_tokenizer,gpt2_tokenizer,gpt2_model,roberta_model,roberta_tokenizer



#generating embedding of resumes for each models
resumes_df['bert_Embeddings'] = resumes_df['Combined_Text'].apply(lambda x: get_embeddings(x,bert_tokenizer,bert_model).tolist())
resumes_df['gpt2_Embeddings'] = resumes_df['Combined_Text'].apply(lambda x: get_embeddings(x,gpt2_tokenizer,gpt2_model).tolist())
resumes_df['roberta_Embeddings'] = resumes_df['Combined_Text'].apply(lambda x: get_embeddings(x,roberta_tokenizer,roberta_model).tolist())


#generate embedding of projects for each models
projects_df['bert_Embeddings'] = projects_df['Combined_Text'].apply(lambda x: get_embeddings(x,bert_tokenizer,bert_model).tolist())
projects_df['gpt2_Embeddings'] = projects_df['Combined_Text'].apply(lambda x: get_embeddings(x,gpt2_tokenizer,gpt2_model).tolist())
projects_df['roberta_Embeddings'] = projects_df['Combined_Text'].apply(lambda x: get_embeddings(x,roberta_tokenizer,roberta_model).tolist())


#save to pickel file
resumes_df.to_pickle('./embedding_pickelfile/resumes_with_embeddings.pkl')
projects_df.to_pickle('./embedding_pickelfile/projects_with_embeddings.pkl')


print("Embedding Generated Successfully....")