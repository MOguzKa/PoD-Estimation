import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.figure_factory as ff

def feature_importance_plot(clf,X_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fi = pd.DataFrame(clf.feature_importances_)
    fi.columns = ["importance"]
    fi["feature"] = X_test.columns
    fi = fi.sort_values(by = "importance", ascending = False).reset_index(drop = True)
    plt.figure(figsize = (10,8))
    sns.barplot(x = fi.importance.head(20), y = fi.feature.head(20))
    plt.title("Feature Importances", size = "medium")
    plt.xlabel("Importance", size = "medium")
    plt.ylabel("Feature Name", size = "medium")
    plt.xticks(size = "small")
    plt.yticks(size = "small")
    plt.plot()
    

def label_distribution_plot(pr):
    x1 = pr[pr.true == 0].prob
    x2 = pr[pr.true == 1].prob

    group_labels = ['on_time', 'default']

    colors = ['slategray', 'magenta']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot([x1, x2], group_labels, bin_size=.1,
                             curve_type='normal', # override default 'kde'
                             colors=colors)

    # Add title
    fig.update_layout(title_text='Score distributions of repayments')
    fig.show()

def feature_importance_shap_plot(clf,df):


    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(df)
    shap.initjs()
    plt.figure(figsize=(8, 7))
    plt.subplots_adjust(left=.62, bottom=.06, right=.97, top=.62, wspace=.20, hspace=.20)
    return shap.summary_plot(shap_values, df)



def score_distribution_plot(clf, X_test,y_test):
    prob = clf.predict_proba(X_test)
    res = pd.DataFrame(prob[:,1], columns = ["score"])
    res.index = y_test.index
    res["true"] = y_test
    res["pred"] = 1
    idx = res[res["score"] < 0.5].index
    res.loc[idx, "pred"] = 0
    
    
    
    plt.figure(figsize = (20,5))
    plt.subplot(122)

    bins = []
    tps = []
    fps = []
    allp = []
    precisions = []
    for i in range(10,20):
        try:
            tmp = res[(res["score"] > i*0.05) & (res["score"] <= (i+1)*0.05)]
            tp = len(tmp[(tmp["pred"] == 1) & (tmp["true"] == 1)])
            fp = len(tmp[(tmp["pred"] == 1) & (tmp["true"] == 0)])
            fn = len(tmp[(tmp["pred"] == 0) & (tmp["true"] == 1)])
            tn = len(tmp[(tmp["pred"] == 0) & (tmp["true"] == 0)])
            bins.append(i*0.05)
            tps.append(tp)
            fps.append(fp)
            allp.append(tp + fp)
            precisions.append(tp/ (tp + fp))
            #print(i,tp/ (tp + fp), tp, (tp + fp)) #(tp + tn)/(tp + tn + fp + fn))
        except:
            pass

    bins = list(pd.Series(bins).astype(str).str[:4].values)
    plt.bar(x = bins, height = tps, edgecolor='white', width = 1, align = "edge")
    plt.bar(x= bins, height = fps, bottom=tps, edgecolor='white', width = 1, align = "edge")
    plt.title("Precision")
    bins.append("1")
    plt.xticks(bins)
    for i in range(len(tps)):
        perc = tps[i]/(tps[i] + fps[i])*100
        plt.text(0.5 + (i*1),tps[i] + fps[i],"%" + "%.2f" % perc, ha="center", va="center")
    plt.legend(["True Posistives", "False Positives"])
    #plt.plot()


    plt.subplot(121)
    bins = []
    tns = []
    fns = []
    alln = []
    precisions = []
    for i in range(10):
        try:
            tmp = res[(res["score"] > i*0.05) & (res["score"] <= (i+1)*0.05)]
            tp = len(tmp[(tmp["pred"] == 1) & (tmp["true"] == 1)])
            fp = len(tmp[(tmp["pred"] == 1) & (tmp["true"] == 0)])
            fn = len(tmp[(tmp["pred"] == 0) & (tmp["true"] == 1)])
            tn = len(tmp[(tmp["pred"] == 0) & (tmp["true"] == 0)])
            bins.append(i*0.05)
            tns.append(tn)
            fns.append(fn)
            alln.append(tn + fn)
            precisions.append(tn/ (tn + fn))
            #print(i,tp/ (tp + fp), tp, (tp + fp)) #(tp + tn)/(tp + tn + fp + fn))
        except:
            pass

    bins = list(pd.Series(bins).astype(str).str[:4].values)
    plt.bar(x = bins, height = tns, edgecolor='white', width = 1, align = "edge")
    plt.bar(x= bins, height = fns, bottom=tns, edgecolor='white', width = 1, align = "edge")
    plt.title("Negative Predictive Value")
    bins.append("0.5")
    plt.xticks(bins)
    for i in range(len(tns)):
        perc = tns[i]/(tns[i] + fns[i])*100
        plt.text(0.5 + (i*1),tns[i] + fns[i],"%" + "%.2f" % perc, ha="center", va="center")
    plt.legend(["True Negatives", "False Negatives"])

    plt.subplots_adjust(wspace=0.05)
    plt.show()
    

def confusion_matrix_plot(y_true, y_pred, title_name, figsize=(4,4)):
    """
        Returns:
    1. cost by fn and fp
    2. specifity: batıkların kaçını bildim
    """

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax).set_title(title_name, fontsize=20)

    cm = confusion_matrix(y_true, y_pred)
    cost = np.round((cm[0,1]*100 + cm[1,0]*6)/(cm[:,1].sum()*100),3)
    fpr = np.round(cm[0,0]/cm[:,0].sum(),3) # batar dediklerimin kaçı battı
    specifity = np.round(cm[0,0]/cm[0,:].sum(),3) # batıkların kaçını bildim

    print(f'Pred Cost: {cost:.3f}, fpr: {fpr: .3f}, specifity: {specifity: .3f}')

    return cost, fpr, specifity


def probability_pdf_perf(pr, step):
    x = []
    interval = 1/step
    for i in range(step):
        tmp = pr[(pr.prob >= interval*i) & (pr.prob < interval*(i+1))]
        
        x.append([len(tmp)/len(pr), len(tmp[tmp.true == 0])/len(pr[pr.true == 0]), tmp.true.mean()])
    x = pd.DataFrame(x, index = np.arange(1,step + 1)/step)
    x.columns = ["size", "default_dist", "precision"]
    return x


def probability_cdf_perf(pr, step):
    x = []
    interval = 1/step
    for i in range(step):
        tmp = pr[(pr.prob >= interval*i)]
        
        x.append([len(tmp)/len(pr), len(tmp[tmp.true == 0])/len(pr[pr.true == 0]), tmp.true.mean()])
    x = pd.DataFrame(x, index = np.arange(1,step + 1)/step)
    x.columns = ["size", "default_dist", "precision"]
    return x




def threshold_search_plot(clf,X_test,y_test):
    tmp = pd.DataFrame(clf.predict_proba(X_test)[:,1], index = X_test.index)
    tmp.columns = ["prob"]
    tmp["true"] = y_test

    to_plot = pd.DataFrame(columns = ["tnr","tpr"],index = range(1,100,1))

    for th in range(1,100,1):


        tmp["pred"] = 0
        tmp.loc[tmp["prob"] > th/100,"pred"] = 1
        tn, fp, fn, tp = confusion_matrix(tmp.true, tmp.pred).ravel()
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)

        to_plot.loc[th,"tnr"] = tnr
        to_plot.loc[th,"tpr"] = tpr
    to_plot.index = to_plot.index/100    
    plt.plot(to_plot)
    plt.legend(["on_time","default"])
    plt.xlabel("score")
    plt.ylabel("recall")
