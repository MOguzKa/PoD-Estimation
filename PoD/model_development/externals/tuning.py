import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score,make_scorer, confusion_matrix, f1_score
import scipy
from externals.cost_functions import CostFunctions

class ParameterTuning(CostFunctions):

  def __init__(self,X_train, y_train, X_test, y_test):
       
       self.X_train = X_train
       self.X_test = X_test
       self.y_train = y_train
       self.y_test = y_test

  def create_scorer(self):

      maximize_profit = make_scorer(self.maximize_profit, greater_is_better=True, needs_proba=False)
      negative_precision_score = make_scorer(self.negative_precision, greater_is_better=True, needs_proba=False)
      tnr_tpr_optimizer_score = make_scorer(self.tnr_tpr_optimizer, greater_is_better=True, needs_proba=False)
      scoring = {'auc': 'roc_auc', "precision" : "precision", "f1_micro":"f1_micro","recall" : "recall", 
                 "balanced_accuracy": "balanced_accuracy" ,
                 "neg_precision" : negative_precision_score, "profit" : maximize_profit, 
                "tnr_tpr_optimizer_score" : tnr_tpr_optimizer_score}
      return scoring




  def param_tuning(self,n_iter, refit_metric, n_splits = 5):

      default_count = (self.y_train == 1).sum()
      on_time_count = len(self.y_train) - default_count
      parameters = {'n_jobs':[-1], 
                    'objective':['binary:logistic'],
                    'learning_rate': [0.01, 0.15, 0.3],
                    'max_depth': [3,6,9,12],
                    'n_estimators': [100, 250, 500, 1000,1200],
                    'random_state': [42],
                   "reg_alpha" : [0,1],
                    "reg_lambda" : [0,1],
                    "gamma" : [0,1,3],
                   "min_child_weight" : [1,3,10],
                    'scale_pos_weight' : [10,20,30,80,100,on_time_count/default_count],#1,20,30
                    "max_delta_step" : [0,1]
                   }

      if "farmer_open_accounts_total_risk_cash" in self.X_train.columns.to_list():
        print("monotone constraints")
        cols = self.X_train.columns
        parameters["monotone_constraints"] = f"{tuple((cols == 'farmer_open_accounts_total_risk_cash').astype(int))*-1}"


      xgb_model = xgb.XGBClassifier()
        
      
      scoring = self.create_scorer()

        

      self.clf = RandomizedSearchCV(xgb_model, parameters, n_jobs = -1, 
                         cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=42), 
                         scoring = scoring, verbose = 2, refit = refit_metric, n_iter = n_iter)

      self.clf.fit(self.X_train, self.y_train)
      return self.clf


  def cv_results(self):

    cv_late = pd.DataFrame(self.clf.cv_results_)
    cost_types = cv_late.filter(like = "rank").columns.str[10:]

    model_dict = {}
    results = []
    param_list = []
    for ct in cost_types:
      params = cv_late.sort_values("rank_test_" + ct).head(1).params.values[0]


      tmp_clf = xgb.XGBClassifier(**params)
      tmp_clf.fit(self.X_train, self.y_train)



      
      costs = self.model_cost(self.y_test, tmp_clf.predict(self.X_test))


      res = costs + [ct]
      results.append(res)
      model_dict[ct] = tmp_clf

    results = pd.DataFrame(results)
    results.columns = ["tn", "fp", "fn", "tp", "pred_cost", "fpr", "specifity","cost_type"]
    
    results["prec"] = np.round(results.tp / (results.tp + results.fp),3)
    results["tnr"] = np.round(results.tn / (results.tn + results.fp),3)
    results["tpr"] = np.round(results.tp / (results.tp + results.fn),3)

    return results,model_dict