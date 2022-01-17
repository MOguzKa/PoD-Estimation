import pandas as pd
import numpy as np

def undersample_majority(X_train, y_train, sampling_rate):
	
	majority = y_train.value_counts().index[0]
    
	selected_majority = y_train[y_train == majority].sample(frac = sampling_rate, random_state = 42).index
	drop_majority = y_train[y_train == majority].index.difference(selected_majority)
	y_train = y_train.drop(drop_majority)
	X_train = X_train.drop(drop_majority)

	return X_train, y_train