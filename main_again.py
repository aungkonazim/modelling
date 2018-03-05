# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:27:17 2018

@author: aungkon
"""

from stress_marks import get_stress_marks
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from pylab import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import iqr
from scipy.stats import skew
from scipy.stats import kurtosis
from ecg import ecg_feature_computation
from outlier_calculation import Quality,compute_outlier_ecg
def get_stress_label(stress_label,start_ts,end_ts):
    for arr in stress_label:
        if start_ts >= arr[0] and end_ts <= arr[1] and start_ts <= arr[1] and end_ts >= arr[0]:
            return arr[2]
        if start_ts >= arr[0] and start_ts < arr[1] and arr[1] - start_ts > 30000:
            return arr[2]
        if start_ts < arr[0] and end_ts < arr[1] and end_ts - arr[0]  > 30000 and end_ts > arr[0]:
            return arr[2]
    return 2

window_path = "C:\\Users\\aungkon\\Desktop\\stress_wrist\\window"
participant_col = []
window_list = list(os.listdir(window_path))
for window in window_list:
    data = np.load(window_path+"\\"+window)['rr']
    participant_col.append(data[0,0,2].split('\\')[6])

participant_col = np.unique(participant_col)
stress_marks = {}
data_path = "C:\\Users\\aungkon\\Desktop\\model\\data\\"
for participant in participant_col:
    stress_marks[participant] = []
    path_to_date = data_path + '\\'+ str(participant)
    date_dir = list(filter(lambda x: x[0] == '2',os.listdir(path_to_date)))
    for date in date_dir:
        label = list(pd.read_csv(path_to_date+'\\'+date+'\\'+'label_stress.txt',header=None,sep=',').values)
        stress_marks[participant].extend(label)
#print(stress_marks)
count = []
no_of_feature = 16
feature = np.zeros((len(os.listdir(window_path)),no_of_feature+1))
participant_col = np.chararray((len(os.listdir(window_path)),1)) 
for i,window in enumerate(window_list):
    data = np.load(window_path+"\\"+window)['rr']
    participant = data[0,0,2].split('\\')[6]
    stress_label = stress_marks[participant]
    data = data[0,:,:2]
    start_ts = data[0,1][0]
    end_ts = data[0,1][-1]
#    print((end_ts-start_ts)/60000)
#    print(end_ts-start_ts)
    label = get_stress_label(stress_label,start_ts,end_ts)
    feature[i,no_of_feature] = label
    feature_window = np.zeros((np.shape(data)[0],no_of_feature))
    if label == 1 or label == 0:
        count.append(i)
    if label == 2:
        continue
    time_rr_comb = []
    rr_int_comb = []
    quality_comb = []
    quality_total = []
    for j in range(np.shape(data)[0]):
        time_rr = data[j,1]
        rr_int = data[j,0]
        rr_int = [rr_int[k] for k in range(len(rr_int)) if rr_int[k]<2000 and rr_int[k]>300]
        time_rr = [time_rr[k] for k in range(len(rr_int)) if rr_int[k]<2000 and rr_int[k]>300]
        quality_arr = compute_outlier_ecg(np.array(time_rr),np.array(rr_int)/1000)
        time_rr_comb.append(time_rr)
        rr_int_comb.append(rr_int)
        quality_comb.append(quality_arr)
        temp = [1 for k in quality_arr if k[1] == Quality.UNACCEPTABLE]
        quality_total.append(sum(temp))
#        print(len(time_rr),len(rr_int),len(quality_arr))
    min_ind = np.where(quality_total==np.min(quality_total))[0]
    for t in min_ind[:1]:
        time_rr1 = np.array(time_rr_comb)[t]
        rr_int1 = np.array(rr_int_comb)[t]
        quality_arr = np.array(quality_comb)[t]
        ind = np.array([k for k in range(len(quality_arr)) if quality_arr[k][1] == Quality.ACCEPTABLE])
        time_rr = np.array(time_rr1)[np.array(ind)]
        rr_int = np.array(rr_int1)[np.array(ind)]
        feature[i,0] = np.percentile(rr_int,15)
    feature[i,1] = np.percentile(rr_int,85)
    feature[i,14] = np.percentile(rr_int,75)
    feature[i,15] = np.percentile(rr_int,25)
    feature[i,2] = np.mean(rr_int)
    feature[i,3] = np.median(rr_int)
    feature[i,4] = np.std(rr_int)
    feature[i,5] = iqr(rr_int)
    feature[i,6] = skew(rr_int)
    feature[i,7] = kurtosis(rr_int)
    a = ecg_feature_computation(time_rr,rr_int)
    feature[i,8] = a[1]
    feature[i,9] = a[2]
    feature[i,10] = a[3]
    feature[i,11] = a[4]
    feature[i,12] = a[7]
    feature[i,13] = a[-1]
    participant_col[i,0] = participant

participant_col = participant_col[np.array(count),:]
feature = feature[np.array(count),:]

scaler = StandardScaler()
for k in range(np.shape(feature)[1]-1):
    feature[:,k] = list(scaler.fit_transform(feature[:,k].reshape(-1,1)))       

from scipy.io import savemat
savemat('feature.mat',{'feature':feature,'subject':participant_col})   
print(count)
