import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score,make_scorer, confusion_matrix, f1_score


class CostFunctions:

    def maximize_profit(self, y_true, y_pred):

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return (tn - fp - fn*0.08 + tp*0.08)/(tn+tp*0.08+fp+fn*0.08)
    

    def tnr_tpr_optimizer(self, y_true, y_pred):

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        a = tn/(tn+fp)
        b = tp/(tp+fn)
        a*=100
        b*=100
        
        if a-b != 0:
            return ((a+b)/abs(a-b))/(200 - (a+b))
        elif a- b == 0:
            return ((a+b))/(200)
        return (a + b) / ((200 - (a+b)) + abs(a-b))
     

    def negative_precision(self, y_true, y_pred):

        return precision_score(y_true, y_pred, pos_label=0)

    def model_cost(self, y_true, y_pred):
        """
            Returns:
        1. cost by fn and fp
        2. specifity
        3. fpr
        """

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
  
        cost = np.round((cm[0,1]*100 + cm[1,0]*6)/(cm[:,1].sum()*100),3)
        fpr = np.round(cm[0,0]/cm[:,0].sum(),3) 
        specifity = np.round(cm[0,0]/cm[0,:].sum(),3)
        
        return [tn, fp, fn, tp, cost, fpr, specifity]


