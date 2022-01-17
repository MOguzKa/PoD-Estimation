import pandas as pd
import numpy as np
from category_encoders.one_hot import OneHotEncoder
import pickle


def preprocess(df):
    
    df = df.set_index("uuid")
    df.drop("default",axis = 1,inplace= True)
    
    
    
    with open("saved_models/one_hot_encoder.pickle","rb") as handle:
        ohe = pickle.load(handle)
        
    df = ohe.transform(df)
    
    return df

