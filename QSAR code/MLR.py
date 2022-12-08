# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:30:01 2022

@author: TYJ
"""

from QSAR_package.data_split import extractData,randomSpliter
from QSAR_package.feature_preprocess import correlationSelection,RFE_ranking
from QSAR_package.data_scale import dataScale
from QSAR_package.grid_search import gridSearchPlus,gridSearchBase
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer,accuracy_score,mean_squared_error
from QSAR_package.model_evaluation import modeling
import pandas as pd
import numpy as np

spliter = randomSpliter(test_size=0.25,random_state=1024)
spliter.ExtractTotalData(data_path=r"D:\academic\01_degree_doctor\01_project\mPGES\06_QSAR\model\7_MLR\01_dataset\rdkit_200.csv",label_name='label')
spliter.SplitRegressionData()
tr_x = spliter.tr_x
tr_y = spliter.tr_y
te_y = spliter.te_y
corr = correlationSelection()
corr.PearsonXX(tr_x, tr_y)

scaler = dataScale(scale_range=(0.1, 0.9))

tr_scaled_x = scaler.FitTransform(corr.selected_tr_x.iloc[:,:])
te_scaled_x = scaler.Transform(spliter.te_x,DataSet='test')

for rankN in ['SVR','RFR']: 
    if rankN == 'Person':
        tr_ranked_x = tr_scaled_x
        te_ranked_x = te_scaled_x
        notes1 = 'Person'
    else:
        rfe = RFE_ranking(rankN,features_num=1)
        rfe.Fit(tr_scaled_x, tr_y)
        tr_ranked_x = rfe.tr_ranked_x
        te_ranked_x = te_scaled_x.loc[:,tr_ranked_x.columns]
        notes1 = rankN+'-RFE'
 
estimator= LinearRegression()
grid_scorer= make_scorer(mean_squared_error,greater_is_better=False)  
grid_dict = {'fit_intercept':[True]}
grid = gridSearchBase(grid_estimator=estimator,grid_dict =grid_dict,grid_scorer=grid_scorer,fold=5,repeat=2)
grid.FitWithFeaturesNum(tr_ranked_x, tr_y,features_range=(5,41))   # 用RFE排序
model = modeling(grid.best_estimator,params=grid.best_params)
model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
model.CrossVal(cv='LOO')

model.SaveResults(r"D:\academic\01_degree_doctor\01_project\mPGES\06_QSAR\model\7_MLR\rdkit\MLR_result_1024.csv")
