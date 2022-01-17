import pandas as pd
import numpy as np
import json
import sys,os
from preprocessing import preprocess
from scoring import generate_score
from data import generate_raw_data
import pickle
import xgboost as xgb
from category_encoders.one_hot import OneHotEncoder

def handler(event, context):


    
    try:
        
        uuid = event["uuid"]
        df = generate_raw_data(uuid)
        
        if len(df):

                uuid_data = preprocess(df)
                score = generate_score(uuid_data)
                error = None


        else:
            score = None
            error = "Invalid uuid"

    
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        score = None

        error = "python_" + str(e) + " in line: " + str(exc_tb.tb_lineno)

    return {'statusCode': 200, "error":error, "uuid": uuid, "score": score}
