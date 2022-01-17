import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


def generate_score(df):
    
    with open("saved_models/final_model.pickle","rb") as handle:
        clf = pickle.load(handle)
        
    score = clf.predict_proba(df)[:,1][0]
    
    
    return score