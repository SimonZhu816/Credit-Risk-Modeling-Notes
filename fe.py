# 第3章 特征工程
# =============================================================================
# 第3.0节 准备工作
# 简介：调用模块和自定义函数
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 第3.1节 数据理解
# 简介：探索性数据分析（EDA）
# =============================================================================
def EDA(data,featuredict):
    data_numericalCols = data[featuredict['intervalFeatures']]
    data_categoryCols  = data[featuredict['nominalFeatures']+featuredict['ordinalFeatures']]
    # 定义字符型特征输出的最高频（低频）取值和频数，默认是5
    topN = 5
    categoryColsName = []
    for i in range(topN):
        categoryColsName.append('top'+str(i+1))
    for i in range(topN):
        categoryColsName.append('bot'+str(i+1))
    # 数据概览(全变量统计，缺失值，数值变量和类别变量分布)  
    distinct_values={}
    for col in data.columns:
        distinct_values[col] = pd.DataFrame(data[col].value_counts()).shape[0]
    eda_info = pd.concat([data.dtypes,data.count(),data.isnull().sum(),pd.Series(distinct_values)],axis=1)
    eda_info.columns  = ['dtype','nValue','nMiss','unique']
    eda_info['numObs']= eda_info['nValue']+eda_info['nMiss']
    eda_info['missingRate'] = eda_info['nMiss']/eda_info['numObs']
    eda_info['label'] = ''
    # 字符特征
    eda_categoryDist = []
    for col in data_categoryCols:
        fillup=pd.DataFrame([0]*5,columns=[col],index=['Null']*5)
        value_counts = pd.DataFrame(data[col].value_counts())
        value_counts_head=pd.concat([value_counts.head(5),fillup],axis=0)
        value_counts_head=value_counts_head.head(5)
        value_counts_tail=pd.concat([value_counts.tail(5).sort_values(by=[col],ascending=[True]),fillup],axis=0)
        value_counts_tail=value_counts_tail.head(5)
        value_counts_head[col]=value_counts_head.index.astype(str)+':'+value_counts_head[col].astype(str)
        value_counts_head=pd.DataFrame(value_counts_head.values.T,index=value_counts_head.columns)
        value_counts_tail[col]=value_counts_tail.index.astype(str)+':'+value_counts_tail[col].astype(str)
        value_counts_tail=pd.DataFrame(value_counts_tail.values.T,index=value_counts_tail.columns)
        values_counts=pd.merge(value_counts_head, value_counts_tail,left_index=True,right_index=True)
        eda_categoryDist.append(values_counts)
    eda_categoryDist=pd.concat(eda_categoryDist)
    eda_categoryDist.columns = categoryColsName
    eda_categoryDist = pd.merge(eda_categoryDist,eda_info,left_index=True,right_index=True)
    eda_categoryDist = eda_categoryDist.reset_index()
    eda_categoryDist = eda_categoryDist.rename(columns={'index':'name'})
    eda_categoryDist = eda_categoryDist[['name','label','dtype','numObs','nMiss',
                                         'missingRate','unique','top1','top2','top3',
                                         'top4','top5','bot5','bot4','bot3','bot2','bot1']]
    # 数值特征
    eda_numericalDist = data_numericalCols.describe(percentiles=[.05,.25,.50,.75,.95])
    eda_numericalDist = pd.DataFrame(eda_numericalDist.values.T,index = eda_numericalDist.columns,
                                     columns = eda_numericalDist.index)
    skew = pd.DataFrame(data_numericalCols.skew(),columns=['skew'])
    kurt = pd.DataFrame(data_numericalCols.kurt(),columns=['kurt'])
    eda_numericalDist = pd.concat([eda_numericalDist,skew,kurt],axis=1)
    eda_numericalDist = pd.merge(eda_numericalDist,eda_info,left_index=True,right_index=True)
    eda_numericalDist.drop(['nValue','count'],axis=1)
    eda_numericalDist = eda_numericalDist.reset_index()
    eda_numericalDist = eda_numericalDist.rename(columns={'index':'name'})
    eda_numericalDist = eda_numericalDist[['name','label','dtype','numObs','nMiss',
                                           'missingRate','unique','mean','min','max',
                                           'std','skew','kurt','5%','25%','50%','75%','95%']]
    return eda_numericalDist,eda_categoryDist

# =============================================================================
# 第3.2节 数据清洗
# 简介：
# =============================================================================

# =============================================================================
# 第3.3节 特征构造
# 简介：
# =============================================================================
# 3.3a 利用先验知识构造  
# 3.3b 构造统计量 
# 3.3c 数据分箱
def autoBinning_num(data,cols,method,bins):
    # 针对数值型变量，选择不同的分箱方法，自动产生分箱列表binList
    def ff(x,binList):
        if x<=binList[0]:
            return '00(-inf_'+str(binList[0])+']'
        elif x>binList[-1]:
            return ('0'+str(len(binList)))[-2:]+'('+str(binList[-1])+'_+inf]'
        else:
            for i in range(len(binList)):
                if x>binList[i] and x<=binList[i+1]:
                    return ('0'+str(i+1))[-2:]+'('+str(binList[i])+'_'+str(binList[i+1])+']'
    bin = pd.DataFrame(columns=['binList','pctList','groupList','binMethod'])
    for col in cols:
        total = data[col].shape[0]
        uniqs = data[col].nunique()
        bins_n = min(uniqs,bins,99)
        if method == 'frequency':
            # 方法1：无监督——等频分箱
            binList = [round(np.percentile(data[~data[col].isnull()][col],min(100,100/bins_n*i)),2) for i in range(bins_n+1)][1:-1]
        elif method == 'distance':
            # 方法2：无监督——等距分箱
            umax = data[~data[col].isnull()][col].max()
            umin = data[~data[col].isnull()][col].min()
            binList = [round(umin+i*((umax-umin)/bins_n),2) for i in range(bins_n+1)][1:-1]
        elif method == 'chi2':
            return 3
        elif method == 'bestks':
            return 4
        binList = sorted(list(set(binList)))
        data[col+'_bin'] = data[col].map(lambda x:ff(x,binList))
        pct = pd.DataFrame(data.groupby([col+'_bin']).size(),columns=['freq'])
        pct['pct'] = round(pct['freq']/total,4)
        pct['group'] = pct.index
        binDict = {'binList'  : str(binList),
                   'pctList'  : str(list(pct['pct'])),
                   'groupList': str(list(pct['group'])),
                   'binMethod': method}
        bin = pd.concat([bin,pd.DataFrame(binDict,index=[col])],axis=0)
    bin.index.names = ['name']
    bin = bin.reset_index()
    return bin

def mapBinning_num(data,cols,bin):
    # 根据入参分箱列表binList映射分箱
    def ff(x,binList):
        if x<=binList[0]:
            return '00(-inf_'+str(binList[0])+']'
        elif x>binList[-1]:
            return ('0'+str(len(binList)))[-2:]+'('+str(binList[-1])+'_+inf]'
        else:
            for i in range(len(binList)):
                if x>binList[i] and x<=binList[i+1]:
                    return ('0'+str(i+1))[-2:]+'('+str(binList[i])+'_'+str(binList[i+1])+']'
    for col in cols:
        binList = str(bin[bin['name']==col]['binList'].values)[3:-3]
        binList = binList.split(',')
        binList = list(map(lambda x0:float(x0),binList))
        data[col+'_bin'] = data[col].map(lambda x:ff(x,binList))
    return data

def autoBinning_cat(data,cols,method,bins):
    # 针对字符型变量，选择不同的分箱方法，自动产生分箱列表binList
    def ff(x,binList):
        if x not in binList:
            return ('0'+str(len(binList)))[-2:]+'{'+str(binList[-1])+'}'
        else:
            return ('0'+str(binList.index(x)+1))[-2:]+'{'+x+'}'
    bin = pd.DataFrame(columns=['binList','pctList','groupList','binMethod'])
    for col in cols:
        total = data[col].shape[0]
        df_gb = pd.DataFrame(data.groupby([col]).size(),columns=['freq'])
        df_gb = df_gb.sort_values(by=['freq'],ascending=False)
        df_gb.index.names = ['value']
        df_gb = df_gb.reset_index()
        binList = list(df_gb['value'].head(bins))
        binList[-1]='etc.'
        data[col+'_bin'] = data[col].apply(lambda x:ff(x,binList))
        pct = pd.DataFrame(data.groupby([col+'_bin']).size(),columns=['freq'])
        pct['pct'] = round(pct['freq']/total,4)
        pct['group'] = pct.index
        binDict = {'binList'  : str(binList),
                   'pctList'  : str(list(pct['pct'])),
                   'groupList': str(list(pct['group'])),
                   'binMethod': method}
        bin = pd.concat([bin,pd.DataFrame(binDict,index=[col])],axis=0)
    bin.index.names=['name']
    bin= bin.reset_index()
    return bin

def mapBinning_cat(data,cols,bin):
    # 根据入参分箱列表binList映射分箱
    def ff(x,binList):
        if x not in binList:
            return ('0'+str(len(binList)))[-2:]+'{'+str(binList[-1])+'}'
        else:
            return ('0'+str(binList.index(x)+1))[-2:]+'{'+x+'}'
    for col in cols:
        binList = str(bin[bin['name']==col]['binList'].values)[3:-3].replace(" ","")
        binList = binList.split(',')
        binList = list(map(lambda x0:x0[1:-1],binList))
        data[col+'_bin'] = data[col].map(lambda x:ff(x,binList))
    return data

# =============================================================================
# 第3.4节 特征筛选
# 简介：
# =============================================================================
def calc_psi(data,cols,bin):
    # 从bins中读取变量col的基准分组和分布
    psiDicts={}
    for col in cols:
        pctList = str(bin[bin['name']==col]['pctList'].values)[3:-3]
        pctList = pctList.split(',')
        pctList = list(map(lambda x0:float(x0),pctList))
        groupList = str(bin[bin['name']==col]['groupList'].values)[3:-3].replace(" ","")
        groupList = groupList.split(",")
        groupList = list(map(lambda x0:x0[1:-1],groupList))
        # 计算psi
        total = data.shape[0]
        df_base = pd.DataFrame(pctList,columns=['base_pct'],index=groupList)
        df_test = pd.DataFrame(data.groupby([col+'_bin']).size(),columns=['test_pct'])
        df_test['test_pct'] = df_test['test_pct']/total
        df = pd.merge(df_base,df_test,how='left',left_index=True,right_index=True)
        df = df.fillna(0.0001)
        df['psi'] = (df['base_pct']-df['test_pct'])*np.log(df['base_pct']/df['test_pct'])
        psiDicts[col]=df
    psi = pd.concat(psiDicts,axis=0)
    psi.index.names = ['name','group']
    psi = psi.reset_index()
    return psi

def calc_woeIV(data,cols,target):
    woeIVDicts={}
    for col in cols:
        total = data.shape[0]
        bad = data[target].sum()
        good = total - bad
        df = data.groupby([col+'_bin'])['target'].agg([np.size,np.sum])
        df.columns = ['total','bad']
        df['good'] = df['total']-df['bad']
        df['pct']  = df['total']/total
        df['badRate'] = df['bad']/df['total']
        df['badAttr'] = df['bad']/bad
        df['badAttr'] = df['badAttr'].apply(lambda x:x if x>0.0 else 0.000001)
        df['goodAttr'] = df['good']/good
        df['goodAttr'] = df['goodAttr'].apply(lambda x:x if x>0.0 else 0.000001)  
        df['woe'] = round(np.log(df['badAttr']/df['goodAttr']),6)
        df['iv']  = round((df['badAttr']-df['goodAttr'])*np.log(df['badAttr']/df['goodAttr']),4)
        woeIVDicts[col] = df
    woeIV = pd.concat(woeIVDicts,axis=0)
    woeIV.index.names=['name','group']
    woeIV = woeIV.reset_index()
    return woeIV

import matplotlib.font_manager as fm
def binPlot(data,cols):
    myfont=fm.FontProperties(fname='simhei.ttf')
    fonsize = 6
    for col in cols:
        x =range(len(data[data['name']==col]['name']))
        y1 = data[data['name']==col]['pct'].values
        y2 = data[data['name']==col]['badRate'].values
        fig= plt.figure(figsize=(8,len(cols)*4),dpi=100)
        # 主坐标轴
        ax1 = fig.add_subplot(len(cols),1,cols.index(col)+1)
        ax1.bar(x,y1,alpha=0.6,color='lightblue')
        ax1.set_ylim([0,data[data['name']==col]['pct'].max()*1.2])
        ax1.set_title("binPlot:"+col+"("+vardict[col]+")",fontproperties=myfont,fontsize="14")
        plt.xticks(x,data[data['name']==col]['group'],fontproperties=myfont,fontsize=fonsize,rotation=45)
        plt.yticks(fontsize=fonsize)
        plt.legend(['proportion'],loc=2,fontsize=fonsize)
        # 次坐标轴
        ax2 =ax1.twinx()
        ax2.plot(x,y2,'darkblue',mark=".",ms=6,lw=2)
        ax2.set_ylim([0,data[data['name']==col]['badRate'].max()*1.2])
        plt.xticks(x,data[data['name']==col]['group'],rotation=45)
        plt.yticks(fontsize=fonsize)
        plt.legend(['badRate'],loc=1,fontsize=fonsize)
        # 添加数据标签
        # for x,y,z in zip(x,y1,y2):
        #     plt.text(x,y,'%.2f%%' % (y*100),ha='center',ya='top',fontsize=fonsize,rotation=45,alpha=.6)
        #     plt.text(x,z,'%.2f%%' % (y*100),ha='center',ya='bottom',fontsize=fonsize,rotation=45,alpha=.6)
        plt.show()
