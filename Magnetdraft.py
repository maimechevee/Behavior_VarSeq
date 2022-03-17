# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:18:46 2022

@author: cheveemf
"""
##############################################################################
# Draft
##############################################################################
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import os
import math

#Extract data
# filename='H:\\Maxime\\Hall Sensor Data\\20220131\\HallSensor_20220131_4229.csv' #ok
# filename='H:\\Maxime\\Hall Sensor Data\\20211215\\HallSensor_20211215_4229.csv' #ok
# filename='H:\\Maxime\\Hall Sensor Data\\20211214\\HallSensor_20211214_4229.csv' #ok
# filename='H:\\Maxime\\Hall Sensor Data\\20211213\\HallSensor_20211213_4229.csv' #salvageable
# filename='H:\\Maxime\\Hall Sensor Data\\20211212\\HallSensor_20211212_4229.csv' #ok
# filename='H:\\Maxime\\Hall Sensor Data\\20211211\\HallSensor_20211211_4229.csv' #salvageable
# filename='H:\\Maxime\\Hall Sensor Data\\20211210\\HallSensor_20211210_4229.csv' #ok
# filename='H:\\Maxime\\Hall Sensor Data\\20211209\\HallSensor_20211209_4229.csv' #trash
# filename='H:\\Maxime\\Hall Sensor Data\\20211208\\HallSensor_20211208_4229.csv' #trash
# filename='H:\\Maxime\\Hall Sensor Data\\20220126\\HallSensor_20220126_4229.csv' #trash
# filename='H:\\Maxime\\Hall Sensor Data\\20220127\\HallSensor_20220127_4229.csv' #trash
# filename='H:\\Maxime\\Hall Sensor Data\\20220128\\HallSensor_20220128_4229.csv' #trash
# filename='H:\\Maxime\\Hall Sensor Data\\20220130\\HallSensor_20220130_4229.csv' #partially useable

# mouse=4229
# date='20220201'

# directory='H:/Maxime/Hall Sensor Data/CATEG'
# filename='H://Maxime//Hall Sensor Data//CATEG//20220214//HallSensor_20220214_4407.csv'
# mouse=4407
# skip=['20220208']
# Dates=os.listdir(directory)
# for date in Dates:
#     if date in skip:
#         continue
#     file=[x for x in os.listdir(os.path.join(directory,date)) if str(mouse) in x][0]
#     filename=os.path.join(directory,date, file)

#     Data=pd.read_csv(filename)
#     Data.columns=['Hall_sensor','Magnet_block','Time']


#     #smooth
#     from scipy.ndimage.filters import uniform_filter1d
#     smooth_data = uniform_filter1d(Data.Hall_sensor, size=50)
#     Data.Hall_sensor=smooth_data
    
#     # truncate data to keep only time when the magnet was on
#     beg_ind, end_ind, rough_baseline = detect_magnet_ON_OFF(Data, plot=False)
    
#     Data=Data[beg_ind:end_ind]
    
#     #Flip data if necessary:
#     if Data['Hall_sensor'].values[0] > np.mean(Data['Hall_sensor'].values):
#         Data['Hall_sensor']=Data['Hall_sensor'].values * (-1)
#         rough_baseline = rough_baseline * (-1)
#     # plt.figure()
#     # plt.plot(Data['Hall_sensor'], alpha=0.5)
#     # plt.hlines(rough_baseline,0,10000)
#     # plt.title(date)

    
    
#     #print(target, number)
# #result=
# #614 (4229/20220131)
# #638 (4229/20211215)
# #636 (4229/20211214)
# #575 (4229/20211212)
# #700 (4229/20211210)
# #check
# number_of_presses, Down_idx, Up_idx = detect_press(Data,target=700, plot=True)

# #Adjust indices to start and end at right place, not at threshold crossing
# new_Down_idx, new_Up_idx = adjust_press_idx(Data,Down_idx, Up_idx, plot=True)

# #Normalize te entire trace
# Data.Hall_sensor=sp.stats.zscore(Data.Hall_sensor.values)

# #it starts with UP and strats with Down so remove the ends accordingly
# new_Down_idx=new_Down_idx[:-1]
# new_Up_idx=new_Up_idx[1:]

# #Make a matrix with one press per row
# plt.hist(np.subtract(new_Up_idx,new_Down_idx), bins=100)
# length_of_vector=int(np.mean(np.subtract(new_Up_idx,new_Down_idx)) + np.std(np.subtract(new_Up_idx,new_Down_idx)))
# press_matrix=np.zeros((len(new_Down_idx),length_of_vector ))
# magnet=Data.Hall_sensor.values
# for i,(pressDown,pressUp) in enumerate(zip(new_Down_idx, new_Up_idx)):
#     if pressUp-pressDown>length_of_vector:
#         stop=length_of_vector
#     else:
#         stop=pressUp-pressDown
        
#     press_matrix[i,:stop]=magnet[pressDown:pressDown+stop]
    
# plt.imshow( press_matrix)
# pcorr = manual_pairwise_pearsonr(press_matrix.T, press_matrix.T)
# plt.imshow(pcorr, vmin=0, vmax=1)
# plt.colorbar()
# ##############################################################################
# # extract validated sessions
# ##############################################################################
# directory='H:/Maxime/Hall Sensor Data/CATEG'
# mouse=4407
# skip=['20220208', '20220214', '20220222', '20220218', '20220207']
# Dates=os.listdir(directory)
# test=[]
# for date in Dates:
#     print(date)
#     if date in skip:
#         continue
#     file=[x for x in os.listdir(os.path.join(directory,date)) if str(mouse) in x][0]
#     filename=os.path.join(directory,date, file)
    
#     Data=pd.read_csv(filename)
#     Data.columns=['Hall_sensor','Magnet_block','Time']
#     Data, new_Down_idx, new_Up_idx = Get_press_indices(Data, master_df, mouse, date)
        
    
    
    
#     #length_of_vector=int(np.mean(np.subtract(new_Up_idx,new_Down_idx)) + np.std(np.subtract(new_Up_idx,new_Down_idx)))
#     length_of_vector=1000
#     press_matrix=np.zeros((len(new_Down_idx),length_of_vector ))
#     magnet=Data.Hall_sensor.values
#     for i,(pressDown,pressUp) in enumerate(zip(new_Down_idx, new_Up_idx)):
#         if pressUp-pressDown>length_of_vector:
#             stop=length_of_vector
#         else:
#             stop=pressUp-pressDown
            
#         press_matrix[i,:stop]=magnet[pressDown:pressDown+stop]
    
#     temp_test=[]
#     for date in Dates:
#         print(date + '...')
#         #plt.figure()
#         if date in skip:
#             continue
#         file=[x for x in os.listdir(os.path.join(directory,date)) if str(mouse) in x][0]
#         filename=os.path.join(directory,date, file)
        
#         Data=pd.read_csv(filename) 
#         Data.columns=['Hall_sensor','Magnet_block','Time']
#         Data, new_Down_idx, new_Up_idx = Get_press_indices(Data, master_df, mouse, date)
            
#         #length_of_vector=int(np.mean(np.subtract(new_Up_idx,new_Down_idx)) + np.std(np.subtract(new_Up_idx,new_Down_idx)))
#         length_of_vector=1000
#         press_matrix1=np.zeros((len(new_Down_idx),length_of_vector ))
#         magnet=Data.Hall_sensor.values
#         for i,(pressDown,pressUp) in enumerate(zip(new_Down_idx, new_Up_idx)):
#             if pressUp-pressDown>length_of_vector:
#                 stop=length_of_vector
#             else:
#                 stop=pressUp-pressDown
                
#             press_matrix1[i,:stop]=magnet[pressDown:pressDown+stop]
            
        
#         pcorr = manual_pairwise_pearsonr(press_matrix.T, press_matrix1.T)
        
    
#         flat_pcorr=np.ravel(pcorr)
#         drop_Ones=flat_pcorr[flat_pcorr!=1]
        
#         # Cum=stats.cumfreq(drop_Ones, numbins=400, defaultreallimits=(-0.5,1.1))
#         # x= np.linspace(-0.5, Cum.binsize*Cum.cumcount.size-0.5, Cum.cumcount.size)
#         # #x= np.insert([np.log(a+1) for a in x][1:], 0,0)
#         # plt.plot(x,Cum.cumcount/np.size(drop_Ones), linestyle='dotted') #color=cmap(Cmap_index[j])
        
#         temp_test.append(drop_Ones)
#     test.append(temp_test)


# matrix=np.zeros((11,11))
# for i,each in enumerate(test):
#     for j,pcorr in enumerate(each):
#         matrix[i,j]=np.median(pcorr)
#         matrix[j,i]=np.median(pcorr)
#     plt.imshow(matrix)    
#     plt.colorbar()
#     #plt.figure()
#     plt.hist(np.ravel(pcorr ), bins=100, density=True, alpha=0.5)
#     # plt.figure()
#     # plt.imshow(pcorr, vmin=0, vmax=1)
#     # plt.colorbar()

    

# plt.plot(Data.Hall_sensor)
# plt.plot(cut_signal)    


# /Users/emma-fuze-grace/Lab/2022-02_TarVar_Categ_01/2022-02_TarVar_Categ_01_data
def Get_press_indices(Data, master_df, mouse, date):
    from scipy.ndimage.filters import uniform_filter1d
    from scipy.fftpack import rfft, irfft, fftfreq    
    W = fftfreq(len(Data.Hall_sensor), d=Data.Time.values[-1] - Data.Time.values[0])
    f_signal = rfft(Data.Hall_sensor.values)
    
    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>0.000000005) ] = 0
    
    cut_signal = irfft(cut_f_signal)

    # plt.plot(Data.Hall_sensor.values)
    # plt.plot(cut_signal)

    smooth_data = uniform_filter1d(cut_signal, size=50)
    # plt.plot(smooth_data)
    Data.Hall_sensor=smooth_data

    # truncate data to keep only time when the magnet was on
    beg_ind, end_ind, rough_baseline, Data = detect_magnet_ON_OFF(Data, plot=False)
    Data=Data[beg_ind:end_ind]
    
    #identify cutoff
    cutoff = identify_cutoff(Data, rough_baseline, master_df, mouse, date, plot=False)
    
    if math.isnan(cutoff):
        print(str(mouse) +' '+date+' is not straight forward.')
        return Data, 0,0
    
    #check
    number_of_presses, Down_idx, Up_idx = detect_press(Data,target=cutoff, plot=False)

    #Adjust indices to start and end at right place, not at threshold crossing
    new_Down_idx, new_Up_idx = adjust_press_idx(Data,Down_idx, Up_idx, plot=True)

    #Normalize te entire trace
    Data.Hall_sensor=sp.stats.zscore(Data.Hall_sensor.values)

    # Retrieve lever press time stamps
    time_stamps=Data['Time'].values
    mouse_df = master_df[master_df['Mouse'] == mouse]
    lever_presses = mouse_df[mouse_df['Date'] == date]['Lever'].values[0]
    x = [(1000 * press) + time_stamps[0] for press in lever_presses]
    y = [rough_baseline + 25 for _ in range(len(lever_presses))]
    plt.plot(x, y, '*--')
    for press_num, press_time in enumerate(x):
        plt.text(press_time, rough_baseline + 26, f'{press_num}')
    return Data, new_Down_idx, new_Up_idx
    
    
# Build matrix 
def adjust_press_idx(Data,Down_idx, Up_idx, plot=False):
    magnet=Data['Hall_sensor'].values
    time_stamps=Data['Time'].values
    new_Down_idx=np.zeros_like(Down_idx[0])
    for j,idx in enumerate(Down_idx[0]):
        while np.mean(magnet[idx:idx+10])<=magnet[idx-1]:
            idx-=1
        new_Down_idx[j]=idx
    new_Up_idx=np.zeros_like(Up_idx[0])
    for j,idx in enumerate(Up_idx[0]):
        while np.mean(magnet[idx-10:idx])<=magnet[idx+1]:
            idx+=1
        new_Up_idx[j]=idx
        
    #it starts with UP and strats with Down so remove the ends accordingly
    new_Down_idx=new_Down_idx[:-1]
    new_Up_idx=new_Up_idx[1:]
     
    if plot:
        plt.figure()
        plt.plot(Data.Time.values, Data.Hall_sensor.values, alpha=0.5)
        plt.scatter(time_stamps[new_Down_idx], magnet[new_Down_idx], c='b')
        plt.scatter(time_stamps[new_Up_idx], magnet[new_Up_idx], c='r')
        offset = (1, 1) # move text each time there's a new press in case of overlapping presses
        for [ind, a, b, v] in zip(new_Down_idx,
                            time_stamps[new_Down_idx], 
                            time_stamps[new_Up_idx], 
                            magnet[new_Down_idx]):
            plt.hlines(v,a,b)
            plt.text(a, v - (offset[0] % 2), f'{ind}', fontsize=8) # Plot predicted press num
            offset = (offset[0] + 1, offset[1])
        for [ind, a, v] in zip(new_Up_idx,
                               time_stamps[new_Up_idx], 
                               magnet[new_Up_idx]):
            plt.text(a, v - (offset[1] % 2), f'{ind}', fontsize=8)
            offset = (offset[0], offset[1] + 1)
    return new_Down_idx, new_Up_idx
    

def detect_press(Data,target, plot=False):
    magnet=Data['Hall_sensor'].values
    bool_mask = magnet<target
    crossings = np.invert( bool_mask[:-1] == bool_mask[1:] ) #not the same (ie. crossing the thrshold)
    Down_idx = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
    Up_idx = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
    Down_idx=np.where(Down_idx)
    Up_idx=np.where(Up_idx)
    
    if plot:
        plt.figure()
        plt.plot(Data.Hall_sensor.values, alpha=0.5)
        plt.hlines(target, 0,len(Data), color='r')
        plt.scatter(Down_idx, np.ones_like(Down_idx)+target, c='b')
        plt.scatter(Up_idx, np.ones_like(Up_idx)+target, c='r')
        
    
    number_of_presses= len(Down_idx[0])
    return number_of_presses, Down_idx, Up_idx
    
   

def identify_cutoff(Data, rough_baseline, master_df, mouse, date, plot=False):
    #Identify optimum threshold to detect presses
    #goal:
    mouse_df=master_df[master_df['Mouse']==mouse]
    date_df=mouse_df[mouse_df['Date']==date]
    GOAL = len(date_df['Lever'].values[0])
    Numbers=[]
    target_test_array=np.arange(rough_baseline-100, rough_baseline+100, 2)
    for target in target_test_array:
        number, D, U =detect_press(Data,target, plot=False)
        Numbers.append(number)   
        
    #find the two peaks and get the first crossings below the GOAL in between
    temp=Numbers.copy()
    Second_peak=np.max(Numbers)
    Second_peak_idx=np.argmax(Numbers)
    TRUE_Second_peak_idx=np.argmax(Numbers)
    while ( (np.argmax(temp) == Second_peak_idx) | (np.argmax(temp) == Second_peak_idx-1) | (np.argmax(temp) == Second_peak_idx-2) ):
        Second_peak=np.max(temp)
        Second_peak_idx=np.argmax(temp)
        #print(Second_peak)
        temp.remove(Second_peak)
    TRUE_First_peak_idx=np.argmax(temp)
        
    
    Index = np.where(np.array(Numbers[TRUE_First_peak_idx:TRUE_Second_peak_idx])<GOAL)
    
    #if it never detect the right amount of presses, skip this day
    try:
        cutoff_index=Index[0][0]
    except:
        cutoff=float('nan')
        return cutoff
    peak_to_peak_test_array=target_test_array[TRUE_First_peak_idx:TRUE_Second_peak_idx]
    cutoff=peak_to_peak_test_array[cutoff_index]
    if plot:
        plt.figure()
        plt.plot( peak_to_peak_test_array, Numbers[TRUE_First_peak_idx:TRUE_Second_peak_idx])
        plt.hlines(GOAL, peak_to_peak_test_array[0],peak_to_peak_test_array[-1], color='r')
        plt.vlines(cutoff, 0, GOAL, color='g')
        plt.title(date + ' ' + str(GOAL))
    return cutoff

def detect_magnet_ON_OFF(Data, plot=False):
    magnet_roll = Data.Hall_sensor.values
    roll_time = Data.Time.values
    
   # Grab data from first second of file when magnet is still off
    magnet_off_mean = np.mean(magnet_roll[roll_time < roll_time[0] + 1000]) #get mean of baseline
    magnet_off_std = np.std(magnet_roll[roll_time < roll_time[0] + 1000]) #get std of baseline
    Threshold=12*magnet_off_std #threshold 

    ii = 1    #counter   
# Identify magnet onset
    not_found = True
    beg_ind = 1
    while not_found :#loop through data until you deflect from OFF to ON
        if (abs(magnet_roll[ii] - magnet_off_mean) > Threshold) :
            beg_ind = ii
            not_found = False
        ii = ii + 1

# Grab baseline of first second of when magnet turns off 
    baseline = np.mean(magnet_roll[beg_ind:beg_ind + 500]);

# Identify magnet offset    
    not_found = True
    end_ind=beg_ind 
    while not_found :#loop through data until you deflect from ON to OFF
        if (abs(magnet_roll[ii] - magnet_off_mean) < Threshold) :
            end_ind = ii
            not_found = False
        ii = ii + 1
    
    if plot:
        plt.figure()
        plt.plot(magnet_roll, alpha=0.5)
        plt.vlines([beg_ind, end_ind], 0,max(magnet_roll), color='r')
        
    #Flip data if necessary:
    if Data['Hall_sensor'].values[0] > np.mean(Data['Hall_sensor'].values):
        Data['Hall_sensor']=Data['Hall_sensor'].values * (-1)
        baseline = baseline * (-1)
    
    return beg_ind, end_ind, baseline, Data

def manual_pairwise_pearsonr(A_array,B_array):
    # Get number of rows in either A or B
    N = B_array.shape[0]
    
    # Store columnw-wise in A and B, as they would be used at few places
    sA = A_array.sum(0)
    sB = B_array.sum(0)
    
    # Basically there are four parts in the formula. We would compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A_array,B_array)
    p2 = sA*sB[:,None]
    p3 = N*((B_array**2).sum(0)) - (sB**2)
    p4 = N*((A_array**2).sum(0)) - (sA**2)
    
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
    return pcorr

def load_excel_log(file, master_df):
    excel_log = pd.read_csv(file, delimiter=',')
    good_presses = {}
    incorrect_indices = []
    drop_sequence = ['False negative', 'Missed']
    drop = 'False positive'
    for ind, press_log in excel_log.iterrows():
        num = press_log['Medpc Press Number']
        if num != drop:
            down = press_log['Python Down Index']
            up = press_log['Python Up Index']
            if '*' in down:
                down = down[:-1]
                incorrect_indices.append(down)
            if '*' in up:
                up = up[:-1]
                incorrect_indices.append(up)
            if down.isnumeric():
                down = int(down)
            if up.isnumeric():
                up = int(up)
            good_presses[num] = down, up
    return good_presses

"""
Plots to make
5 x 5 within session
10 x 10 by day (x5 per no# in press)
                
10x10 correlations:
only presses in sequences
only presses not in sequence
"""

if __name__ == '__main__':
    # from create_medpc_master import create_medpc_master
    # mice = [i for i in range(4386, 4414)]
    # file_dir = '/Users/emma-fuze-grace/Lab/Medpc Data'
    # master_df = create_medpc_master(mice, file_dir)
    # mouse = 4407
    # date = '20220209'
    # filename = f'/Users/emma-fuze-grace/Lab/Hall Sensor Data/HallSensor_{date}_{mouse}.CSV'
    # Data = pd.read_csv(filename)
    # Data.columns = ['Hall_sensor','Magnet_block','Time']
    # Data, new_Down_idx, new_Up_idx = Get_press_indices(Data, master_df, mouse, date)
    good_presses = load_excel_log('MagnetIndexLog_20220209_4407.csv')
    
    