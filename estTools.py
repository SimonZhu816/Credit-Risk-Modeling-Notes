# 5.模型效果评估
from sklearn.metrics import roc_curve,auc
import numpy as np
import pandas as pd
def gini_ks(inds,prob,target):
    inds2 = inds[inds[prob].notnull()]
    fpr,tpr,thresholds=roc_curve(inds2[target].values,inds2[prob].values)
    ks = round(max(abs(tpr-fpr)),4)
    return inds.shape[0],inds2.shape[0],max(round(auc(fpr,tpr),4),round(1-auc(fpr,tpr),4)),ks

def calcAucKs(inds,probList,label,group):
  inds_n = inds[inds['train_test'].isin(['train','testA','testB']) if group in ['train_test','biz_date'] else inds['train_test']=='testB']
    groupValues = [col for col in list(set(inds_n[group])) if col!=None]
    description = inds_n.groupby([group])[label].agg([np.size,np.mean])
    for prob in probList:
        probAUCDict=[]
        probKSDict=[]
        for g in groupValues:
            _Set = inds_n[inds_n[group]==g]
            probAUCDict.append(gini_ks(_Set,prob,label)[2])
            probKSDict.append(gini_ks(_Set,prob,label)[3])

        probAUC=pd.DataFrame(probAUCDict,columns=[prob],index=groupValues)
        probKS=pd.DataFrame(probKSDict,columns=[prob],index=groupValues)
        if prob == probList[0]:
            _mergeAUC = probAUC
            _mergeKS  = probKS
        else:
            _mergeAUC = pd.merge(_mergeAUC,probAUC,how='outer',left_index=True,right_index=True)
            _mergeKS = pd.merge(_mergeKS,probKS,how='outer',left_index=True,right_index=True)
    _mergeKS = pd.merge(description, _mergeKS,how = 'outer',left_index = True,right_index = True)
    _mergeAUC.index.name ='group'
    _mergeKS.index.name ='group'
    _mergeAUC = _mergeAUC.reset_index().sort_values(by=['group'])
    _mergeKS = _mergeKS.reset_index().sort_values(by=['group'])

    AucKs = pd.merge(_mergeKS,_mergeAUC,how='left',left_on='group',right_on='group',suffixes=('_ks','_auc'))
    return AucKs
