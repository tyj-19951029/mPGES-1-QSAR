# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import copy
import math

#calculate the active interval
data = pd.read_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\2D_des14.csv")
data_copy = copy.deepcopy(data)
act = data_copy.loc[:,['label']]

def activity_interval(data,delta):
    activity = data['label']
    std = np.std(activity,ddof=1)
    upper_limit = []
    lower_limit = []
    for i in range(len(activity)):
        act = float(activity.iloc[i])
        upper = act + (std*delta)
        lower = act - (std*delta)
        upper_limit.append(upper)
        lower_limit.append(lower)
    interval = np.array((lower_limit,upper_limit))
    return interval.T
interval = activity_interval(data_copy,1)
interval_df = pd.DataFrame(interval)
interval_df.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\interval_1.csv")

#compute the active matrix
def AM(interval,activity):
    act_matrix = pd.DataFrame(index=list(range(len(interval))),columns=list(range(len(interval))))
    for i in range(len(interval)):
        for j in range(len(activity)):
            if i == j :
                act_matrix.iat[i,j] = 0#The element on the diagonal is 0
            elif  interval[i][0] < int(activity.iloc[j]) < interval[i][1]:
                act_matrix.iat[i,j] = 0
            else:
                act_matrix.iat[i,j] = 1
    return act_matrix

act_matrix = AM(interval,act)
act_matrix.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\AM_1.csv")

#calculate the distance matrix (Euclidean distance)
def DM(data):
    distance_matrix = pd.DataFrame(index=list(range(len(data))),columns=list(range(len(data))))
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j :
                distance_matrix.iat[i,j] = 1000000
            else:
                x = data_copy.iloc[i][1:-1].tolist()
                y = data_copy.iloc[j][1:-1].tolist()
                dif_squ = list(map(lambda x,y:(x-y)**2,x,y))
                distance = math.sqrt(sum(dif_squ))
                distance_matrix.iat[i,j] = distance
    return distance_matrix

distance_matrix = DM(data)
distance_matrix.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\DM_1.csv")

#Calculate the Euclidean distance corresponding to 1,0 label in the active matrix
def one_or_zero(data,delta,DM):
    interval = activity_interval(data_copy,delta)
    activity = data.loc[:,['label']]
    one = []
    zero = []
    for i in range(len(data)):
        one_for_i = []
        zero_for_i = []
        for j in range(len(data)):
            if interval[i][0] < float(activity.iloc[j]) < interval[i][1] or i == j:
                zero_for_i.append(float(DM.iloc[i][j]))
            else:
                one_for_i.append(float(DM.iloc[i][j]))
        one_for_i.sort()
        zero_for_i.sort()
        one_for_i_df = pd.DataFrame(one_for_i)
        zero_for_i_df = pd.DataFrame(zero_for_i)
        one.append(one_for_i_df)
        zero.append(zero_for_i_df)
        one_df = pd.concat(one,axis=1)
        zero_df = pd.concat(zero,axis=1)
    return one_df,zero_df

one,zero = one_or_zero(data_copy,1,distance_matrix)
one.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\one_1.csv")
zero.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\zero_1.csv")
Dx = zero.iloc[0].tolist()
Dy = one.iloc[0].tolist()

#calculate RI values
def RI(Dx,Dy):
    ri = []
    for i in range(len(Dx)):
        dx = float(Dx[i])
        dy = float(Dy[i])
        rii = (dx-dy)/(dx+dy)
        ri.append(rii)
    RI = pd.DataFrame(ri)
    return RI

RI = RI(Dx,Dy)
RI.to_csv(r"D:\academic\degree_doctor\mPGES\05_QSAR\model\Modelability\732\corina\RI_1.csv")