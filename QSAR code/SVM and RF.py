from QSAR_package.data_split import extractData,randomSpliter
from QSAR_package.feature_preprocess import correlationSelection,RFE_ranking
from QSAR_package.data_scale import dataScale
from QSAR_package.grid_search import gridSearchPlus,gridSearchBase
from QSAR_package.model_evaluation import modeling
import pandas as pd
import numpy as np
from IPython.display import SVG

file = r"D:\academic\01_degree_doctor\01_project\mPGES\06_QSAR\C2_cal_rdk.csv"
for randx in [8,12,42,50,65,78,105]:
    spliter = randomSpliter(test_size=0.25,random_state=randx)
    spliter.ExtractTotalData(file,label_name='label')
    spliter.SplitData()
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
            
        for estimatorName in ['SVR','RFR']:
            grid = gridSearchPlus(grid_estimatorName=estimatorName, fold=5, repeat=2,scoreThreshold=-0.6)
            grid.FitWithFeaturesNum(tr_ranked_x, tr_y,features_range=(10,41))   

            model = modeling(grid.best_estimator,params=grid.best_params)
            model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
            model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
            model.CrossVal(cv='LOO')
            tr_pred = model.tr_pred_y
        
            te_pred = model.te_pred_y

            pred_results = pd.DataFrame([tr_y.values,tr_pred,te_y.values,te_pred],
                                    index=['tr_y','tr_pred','te_y','te_pred']).T
            path = r"D:\academic\01_degree_doctor\01_project\mPGES\06_QSAR\rdk_pred_{}_{}_{}.csv".format(randx,notes1,estimatorName)
            pred_results.to_csv(path,index=False)
            with open(r"D:\academic\01_degree_doctor\01_project\mPGES\06_QSAR\rdk_des_{}_{}_{}.csv".format(notes1,randx,estimatorName),'w') as fobj:
                fobj.write('\n'.join(grid.best_features))
            model.ShowResults(show_cv=True,make_fig=True)
            model.SaveResults(file[:-4]+'_results.csv',notes='{},split_seed={},gridCV=5'.format(notes1,randx),save_model=True)

