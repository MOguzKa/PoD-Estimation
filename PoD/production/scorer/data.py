import pandas as pd
import numpy as np


def generate_raw_data(uuid):
    
    
    data = pd.read_csv("dataset/dataset.csv",sep = ";")
    df = data.loc[data.uuid.isin([uuid])]
    
    
    return df 