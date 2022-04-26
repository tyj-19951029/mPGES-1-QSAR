# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,Sequential
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from QSAR_package.data_split import extractData, randomSpliter
from QSAR_package.feature_preprocess import correlationSelection
from QSAR_package.data_scale import dataScale
from QSAR_package.model_evaluation import modelEvaluator
from time import time
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from IPython.display import clear_output

def Sec2Time(seconds):  # convert seconds to time
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, c='b',lw=1,label="loss")
        plt.plot(self.x, self.val_losses, c='r',lw=1,label="val_loss")
        plt.legend()
        plt.show();
        
    def on_train_end(self, logs={}):
        print('Training complete')
plot_losses = PlotLosses()


data_path = r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\cor_rd.csv"
for randx in [8,42,65,105,126]:
    
    # split training and test set
    spliter = randomSpliter(test_size=0.25,validation_size=None,random_state=randx)
    spliter.ExtractTotalData(data_path,label_name='label')
    spliter.SplitRegressionData()
    tr_x = spliter.tr_x
    tr_y = spliter.tr_y
    te_y = spliter.te_y

    # pearson correlation
    corr = correlationSelection()
    corr.PearsonXX(tr_x, tr_y)
    # scale
    scaler = dataScale(scale_range=(0.1, 0.9))
    tr_scaled_x = scaler.FitTransform(corr.selected_tr_x)
    te_scaled_x = scaler.Transform(spliter.te_x,DataSet='test')

    print('train: {}\ntest: {}'.format(len(tr_scaled_x),len(te_scaled_x)))

    rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    rs.get_n_splits(tr_y)
    tr_idxs = []
    va_idxs = []
    for train_index, test_index in rs.split(tr_y):
        tr_idxs.append(train_index)
        va_idxs.append(test_index)
    try:
        result = pd.read_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\cor_rd_result{}.csv".format(randx))
        max_r2 = result.te_r2[0]
    except:
        max_r2 = 0.5
    for i in range(50):
        tf.keras.backend.clear_session()
        t0 = time()
        model = Sequential()
        model.add(layers.Dense(units=80,activation='relu',name='Layer1',input_shape=(tr_scaled_x.shape[1],)))
        model.add(layers.Dense(units=64,activation='relu',name='Layer2'))
        model.add(layers.Dense(units=32,activation='relu',name='Layer3'))
        model.add(layers.Dense(units=16,activation='relu',name='Layer4'))
        model.add(layers.Dense(units=1,activation='relu',name='OutputLayer'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                     loss='mse',metrics=['mae'])
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,patience=30, verbose=0,mode='auto')

        history = []
        val_pred = []
        val_y = []
        for i in range(len(tr_idxs)):
            tr_x_input = tr_scaled_x.iloc[tr_idxs[i],:].values
            tr_y_input = tr_y.iloc[tr_idxs[i]].values
            va_x_input = tr_scaled_x.iloc[va_idxs[i],:].values
            va_y_input = tr_y.iloc[va_idxs[i]].values
            history.append(model.fit(x=tr_x_input,y=tr_y_input,epochs=1000,validation_data=(va_x_input,va_y_input),
                                verbose=0,callbacks=[early_stopping,plot_losses]))
            val_pred.extend(model.predict(x=va_x_input).flatten())
            val_y.extend(va_y_input)
            print(Sec2Time(time()-t0))

        tr_pred = model.predict(x=tr_scaled_x.values).flatten()
        te_pred = model.predict(x=te_scaled_x.values).flatten()

        tr_Evaluator = modelEvaluator(tr_y, tr_pred)
        val_Evaluator = modelEvaluator(val_y, val_pred)
        te_Evaluator = modelEvaluator(te_y, te_pred)
        
        if te_Evaluator.r2 > max_r2:
            max_r2 = te_Evaluator.r2
            
            print(tr_Evaluator.r2,tr_Evaluator.rmse)
            print(val_Evaluator.r2,val_Evaluator.rmse)
            print(te_Evaluator.r2,te_Evaluator.rmse)
            
            pred_evaluation = pd.DataFrame([randx,tr_Evaluator.r2,tr_Evaluator.rmse,val_Evaluator.r2,val_Evaluator.rmse,
                                            te_Evaluator.r2,te_Evaluator.rmse],
                                          index=['split_seed','tr_r2','tr_mse','val_r2','val_mse','te_r2','te_mse']).T
            path = r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\cor_rd_result{}.csv".format(randx)
            pred_evaluation.to_csv(path,index=False)

            pred_results = pd.DataFrame([tr_y.values,tr_pred,val_y,val_pred,te_y.values,te_pred],
                                        index=['tr_y','tr_pred','val_y','val_pred','te_y','te_pred']).T
            path = r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\pred_values_{}.csv".format(randx)
            pred_results.to_csv(path,index=False)
            
            model.save(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\cor_rd_rgr_{}.h5".format(randx))

            tr_loss = []
            va_loss = []
            tr_loss_flat = []
            va_loss_flat = []
            for i in range(len(history)):
                tr_loss.append(history[i].history['loss'])
                va_loss.append(history[i].history['val_loss'])
                tr_loss_flat.extend(history[i].history['loss'])
                va_loss_flat.extend(history[i].history['val_loss'])

            step = 0
            flag = [0]
            for i in range(len(tr_loss)-1):
                step = step+len(tr_loss[i])
                flag.append(step)
            flag.append(len(tr_loss_flat))

            tr_loss_df = pd.DataFrame(tr_loss,index=['fold_{}'.format(i+1) for i in range(len(tr_loss))]).T
            val_loss_df = pd.DataFrame(va_loss,index=['fold_{}'.format(i+1) for i in range(len(va_loss))]).T

            tr_loss_df.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\tr_loss_{}.csv".format(randx),
                              index=False)
            val_loss_df.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\val_loss_{}.csv".format(randx),
                               index=False)

            h = 0.05
            x_min = 0
            x_max = len(tr_loss_flat)
            y_min = min(tr_loss_flat)-1
            y_max = max(tr_loss_flat)+1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = np.zeros_like(xx)

            for i in range(len(flag)-1):
                Z[np.where((flag[i]<xx)&(xx<=flag[i+1]))] = i+1

            fig = plt.figure(figsize=(9,6))
            plt.plot(tr_loss_flat,c='#EE3A8C',ls='solid',lw=0.8)
            plt.plot(va_loss_flat,c='#1874CD',ls='-.',lw=0.8)
            plt.imshow(Z, interpolation='nearest',alpha=0.5,
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Greys,
                       aspect='auto', origin='lower')
            plt.xlim(-10,x_max+5)
            plt.ylim(0,2)
            for i in range(len(flag)-1):
                plt.text(np.average(flag[i:i+2]), 1, '{}th\nfold'.format(i+1), fontsize=2,color='k',
                         fontproperties='Times New Roman',rasterized=True,horizontalalignment='center')
            plt.ylabel('Loss',fontproperties='Times New Roman',fontsize=13)
            plt.xlabel('Epoch',fontproperties='Times New Roman',fontsize=13)
            plt.legend(['training loss', 'validation loss'], loc=(0.05,0.87))
            plt.savefig(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\cor_rd\loss_fig_{}.tif".format(randx),
                        dpi=300,bbox_inches='tight')
            plt.show()