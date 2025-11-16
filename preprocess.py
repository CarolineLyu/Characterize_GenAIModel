
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

TRAINING_DATA = "training_data_clean.csv"

def prep_data(data=TRAINING_DATA):
    df = pd.read_csv(data)
    print(df.dtypes)
    
    # rename
    df.columns = [
        'id', 
        'best_tasks_free', 
        'acad_tasks_rating', 
        'best_tasks_select', 
        'subopt_freq_select',  
        'subopt_tasks_select', 
        'subopt_tasks_free', 
        'evidence_freq_select', 
        'verify_freq_select', 
        'verify_method_free', 
        'target'
        ]
    
    students = data['id'].unique()
    train_df, test_df = train_test_split(students, test_size=0.3, random_state=42)

def impute_data():
    """replacing missing or inconsistent values in a dataset with estimated values 
    
    numerical data: replace with median, scaled with standard scaler
    categorical data: replace with mode 
    text data: replace with empty string
    """
    # TODO: can use KNN to find the imputation point instead, in-person imputation
    pass 
