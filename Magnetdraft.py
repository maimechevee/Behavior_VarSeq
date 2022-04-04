# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:18:46 2022

@author: cheveemf
"""
##############################################################################
# Draft
##############################################################################
from matplotlib import pyplot as plt
from matplotlib import patches 
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import os
import math
import time
import datetime as dt

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
#     Data, new_Down_idx, new_Up_idx, rough_baseline = Get_press_indices(Data, master_df, mouse, date)
        
    
    
    
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
#         Data, new_Down_idx, new_Up_idx, rough_baseline = Get_press_indices(Data, master_df, mouse, date)
            
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
def Get_press_indices(Data, master_df, mouse, date, want_plot=[False, False, False, False]):
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
    beg_ind, end_ind, rough_baseline, Data = detect_magnet_ON_OFF(Data, want_plot[0])
    magnet_start_time = Data['Time'].values[beg_ind]
    Data=Data[beg_ind:end_ind]
    #identify cutoff
    cutoff = identify_cutoff(Data, rough_baseline, master_df, mouse, date, want_plot[1])
    # print(cutoff)
    if math.isnan(cutoff):
        print(str(mouse) +' '+date+' is not straight forward.')
        return Data, 0,0
    
    #check
    number_of_presses, Down_idx, Up_idx = detect_press(Data, cutoff, want_plot[2])
    #Adjust indices to start and end at right place, not at threshold crossing
    new_Down_idx, new_Up_idx, magnet_start_end, saved_data = adjust_press_idx(Data,Down_idx, Up_idx, want_plot[3])
    magnet_time_difference = magnet_start_end[1] - magnet_start_end[0]
    
    #Normalize the entire trace
    Data.Hall_sensor=sp.stats.zscore(Data.Hall_sensor.values)
    
    # Retrieve lever press time stamps
    
    time_stamps = Data['Time'].values
    mouse_df = master_df[master_df['Mouse'] == mouse]
    date_df=mouse_df[mouse_df['Date']==date]
    lever_presses = date_df['Lever'].values[0][1:]
    medpc_time_difference = (lever_presses[-1] - lever_presses[0]) * 1000
    ratio = magnet_time_difference / medpc_time_difference
    rewards = mouse_df[mouse_df['Date'] == date]['Reward'].values[0]
    scaled_LPs = [press * ratio * 1000 for press in lever_presses]
    zeroed_LPs = [press - scaled_LPs[0] + magnet_start_end[0] for press in scaled_LPs]
    scaled_rewards = [reward * ratio * 1000 for reward in rewards]
    # zeroed_rewards = [reward - zeroed_LPs[4] + magnet_start_end[0] for reward in scaled_rewards]
    y1 = [rough_baseline + 30 for _ in range(len(zeroed_LPs))]
    y2 = [rough_baseline + 35 for _ in range(len(scaled_rewards))]
    if want_plot[3]:
        plt.plot(zeroed_LPs, y1, '*--')
        # plt.plot(scaled_rewards, y2, 'o--', markersize=10)
        # for press_num, press_time in enumerate(zeroed_LPs):
        #     plt.text(press_time, rough_baseline + 26, f'{press_num}')
        plt.title(f'Hall Sensor data on {date} for {mouse}')
    return Data, new_Down_idx, new_Up_idx, saved_data, ratio, rough_baseline, zeroed_LPs

def medpc_press_detection(Data, master_df, mouse, date,
                          baseline_length=1000, smooth=10):
    """
    Detect press indices using similar algorithm but dividing each search
    by adjusted medpc time stamps.

    Parameters
    ----------
    Data : Original dataframe
    master_df : TYPE
        DESCRIPTION.
    baseline_length : int, optional
        How long the baseline vectors will be. The default is 1000                      
    smooth : int, optional
        How far from index to search when adjusting indices in while loop.

    """
    (Data, 
     new_Down_idx, 
     new_Up_idx, 
     saved_data, 
     ratio, 
     rough_baseline,
     zeroed_LPs) = Get_press_indices(Data, master_df, mouse, date, [False, False, False, True])
    mouse_df = master_df[master_df['Mouse'] == mouse]
    date_df = mouse_df[mouse_df['Date']==date]
    magnet= saved_data['Hall_sensor'].values
    times= saved_data['Time'].values
    baselines = np.zeros(len(zeroed_LPs) - 1)
    baselines[0] = np.median(magnet[new_Down_idx[0]-500:new_Down_idx[0]])
    final_press_ind = {ind:(0,0) for ind in range(len(zeroed_LPs))}
    
    # sort histogram data to find boundary of presses
    # first highest will be baseline, second highest will be bottom of presses
    
    histogram = np.histogram(magnet, bins=6)
    sorted_hist = sorted(zip(histogram[0], histogram[1]), reverse=True) 
    top = sorted_hist[0][1]
    bottom = sorted_hist[1][1]
    original_offset = 0.65 * (top - bottom)
    move = 0.025 * (top - bottom)
    plt.vlines(zeroed_LPs, top + 10, bottom - 5)
    print(sorted_hist)
    for press_ind, press in enumerate(zeroed_LPs[1:-1], start=1):
        search_times = times[(times > press) & (times < zeroed_LPs[press_ind + 1])]
        search_start_ind = np.where(times==search_times[0])[0][0]
        search_end_ind = np.where(times==search_times[-1])[0][0]
        search_magnet = magnet[search_start_ind:search_end_ind + 1]
        
        # Define baseline
        independent = 5000 # press will have its own baseline if its this
                        # far away from the other press
        if press - zeroed_LPs[press_ind - 1] > independent:
            curr_baseline_vector = magnet[search_start_ind-baseline_length:search_start_ind]
            curr_baseline = np.median(curr_baseline_vector)
            plt.hlines(curr_baseline, times[search_start_ind-baseline_length], times[search_start_ind],
                       linewidth=2, color='r', label='baseline')
        else:
            curr_baseline = baselines[press_ind - 1]
            end_last = np.where(times==times[times > zeroed_LPs[press_ind - 1]][0])[0][0]
            plt.hlines(curr_baseline, times[end_last], times[search_start_ind],
                       linewidth=2, linestyle='--', color='r', label='baseline')
        baselines[press_ind]
        
        ### Optimize threshold based on the session's particular trace characteristics
        offset = original_offset
        while (curr_target := curr_baseline - offset) < curr_baseline:
            plt.hlines(curr_target, times[search_start_ind], times[search_end_ind], color='k',
                       label='target', linestyle ='--')
            bool_mask = search_magnet < curr_target 
            crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
            Down_idx = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
            Up_idx = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
            if np.where(Down_idx)[0].size > 0 or np.where(Up_idx)[0].size > 0:
                break
            offset -= move
            # If can't find the press, start over at bottom of curve
            if abs(curr_baseline - curr_target) < (5 * move):
                offset = 0.9 * (top - bottom)
        baselines[press_ind] = curr_baseline
        
        ### Main Search ###
        num_down = np.where(Down_idx)[0].size
        num_up = np.where(Up_idx)[0].size
        num_detected = num_down + num_up
        prev_up = final_press_ind[press_ind-1][1]
        if num_detected == 1:
            if num_down:
                down_crossing = search_start_ind + np.where(Down_idx)[0][0]
                final_down_idx = down_search(down_crossing, magnet,
                                         baselines[press_ind], smooth)
                plt.plot(times[down_crossing], magnet[down_crossing],
                         marker='o', color='k')
                final_up_idx = up_search(search_end_ind - 50, magnet,
                                         baselines[press_ind], smooth)
            if num_up:
                up_crossing = search_start_ind + np.where(Up_idx)[0][0]
                plt.plot(times[up_crossing], magnet[up_crossing],
                         marker='o', color='k')
                if up_crossing < prev_up:
                    final_down_idx = prev_up
                    final_up_idx = up_search(prev_up, magnet, baselines[press_ind],
                                             smooth)
                elif search_start_ind < prev_up:
                    final_down_idx = prev_up
                    final_up_idx = up_search(up_crossing, magnet,
                                             baselines[press_ind], smooth)
                else: 
                    final_down_idx = down_search(search_start_ind, magnet, 
                                                 baselines[press_ind],
                                                 smooth)
                    final_up_idx = up_search(up_crossing, magnet,
                                             baselines[press_ind], smooth)
        elif num_detected == 2:
            down_crossing = search_start_ind + np.where(Down_idx)[0][0]
            up_crossing = search_start_ind + np.where(Up_idx)[0][0]
            plt.plot(times[down_crossing], magnet[down_crossing],
                     marker='o', color='k')
            plt.plot(times[up_crossing], magnet[up_crossing],
                     marker='o', color='k')
            if down_crossing < up_crossing:
                final_down_idx = down_search(down_crossing, magnet, 
                                             baselines[press_ind], smooth)
                final_up_idx = up_search(up_crossing, magnet, 
                                         baselines[press_ind], smooth)
            else: 
                if abs(final_down_idx - search_end_ind) > 25:
                    final_down_idx = down_search(down_crossing, magnet, 
                                                 baselines[press_ind], smooth)
                    final_up_idx = up_search(search_end_ind, magnet, 
                                             baselines[press_ind], smooth)
                else:
                    final_down_idx = down_search(search_start_ind, magnet, 
                                                 baselines[press_ind], smooth)
                    final_up_idx = up_search(up_crossing, magnet, 
                                             baselines[press_ind], smooth)
        elif num_detected > 2:
            down_crossing = search_start_ind + np.where(Down_idx)[0]
            up_crossing = search_start_ind + np.where(Up_idx)[0]
            plt.plot(times[down_crossing], magnet[down_crossing],
                     marker='o', color='k', linestyle='None')
            plt.plot(times[up_crossing], magnet[up_crossing],
                     marker='o', color='k', linestyle='None')
            if down_crossing[0] < prev_up:
                final_down_idx = prev_up
                final_up_idx = up_search(up_crossing[-1], magnet, 
                                         baselines[press_ind], smooth)
                plt.plot(times[up_crossing[-1]], magnet[up_crossing[-1]],
                         marker='*', color='g')
            else:
                final_down_idx = down_search(down_crossing[0], magnet, 
                                             baselines[press_ind], smooth)
                final_up_idx = up_search(up_crossing[-1], magnet, 
                                         baselines[press_ind], smooth)
                plt.plot(times[down_crossing[0]], magnet[down_crossing[0]],
                         marker='*', color='m')
                plt.plot(times[up_crossing[-1]], magnet[up_crossing[-1]],
                         marker='*', color='g')
        else:
            print(f'Error in crossings: {press_ind}, {num_detected}')
        
        ### Check for indices being stuck on "hills" during search ###
        # First need to check if far away from previous press and not too close to baseline
        hill_end = 100
        hill_start = 50
        
        # hill check for down indices
        hill_check_down = magnet[
            (final_down_idx - hill_end):(final_down_idx - hill_start)
            ]
        if abs(final_down_idx - prev_up) > 100 and abs(
                magnet[final_down_idx] - curr_baseline) > (3 * move): 
            if np.mean(hill_check_down) > magnet[final_down_idx] and magnet[final_down_idx] < curr_baseline:
                plt.plot(times[(final_down_idx - hill_end):(final_down_idx - hill_start)],
                         hill_check_down,
                         color='r')
                final_down_idx = down_search(final_down_idx, magnet, 
                                             baselines[press_ind], smooth)
            
        # hill check for up indices
        hill_check_up = magnet[
            (final_up_idx + hill_start):(final_up_idx + hill_end)
            ]
        if abs(press - zeroed_LPs[press_ind + 1]) > 200: 
            if abs(magnet[final_up_idx] - curr_baseline) > (3 * move):
                if np.mean(hill_check_up) > magnet[final_up_idx] and magnet[final_up_idx] < curr_baseline:
                    plt.plot(times[(final_up_idx + hill_start):(final_up_idx + hill_end)],
                             hill_check_up,
                             color='r')
                    final_up_idx = up_search(final_up_idx, magnet, 
                                                 baselines[press_ind], smooth)
                
        # Plot down and up idx
        if press_ind % 2:
            color = 'm'
        else:
            color = 'g'
        plt.plot(times[final_down_idx], magnet[final_down_idx],
                     marker='o', color='b', markersize=4)
        plt.plot(times[final_up_idx], magnet[final_up_idx],
                 marker='o', color='r', markersize=6)
        plt.plot(times[final_down_idx:final_up_idx], 
                 magnet[final_down_idx:final_up_idx], color=color)
        final_press_ind[press_ind] = (final_down_idx, final_up_idx)
    
    # plt.figure()
    # down = [press[0] for press in final_press_ind.values()]
    # up = [press[1] for press in final_press_ind.values()]
    # for d, u in zip(down, up):
    #     plt.plot(magnet[d:u])
    return final_press_ind


def down_search(start_ind, magnet, baseline, smooth_factor=10):
    final_down_idx = start_ind
    if (magnet[final_down_idx:final_down_idx+smooth_factor] <= magnet[final_down_idx-1]).all():
        while (
                np.mean(
                    magnet[final_down_idx:final_down_idx+smooth_factor] <= magnet[final_down_idx-1]
                    )
                and magnet[final_down_idx] < baseline
                ):
            final_down_idx -= 1 
    else:
        while (
                np.mean(
                    magnet[final_down_idx:final_down_idx+smooth_factor] > magnet[final_down_idx-1]
                    )
                and magnet[final_down_idx] < baseline
                ):
            final_down_idx -= 1 
        while (
                np.mean(
                magnet[final_down_idx:final_down_idx+smooth_factor] <= magnet[final_down_idx-1]
                )
                and magnet[final_down_idx] < baseline
                ):
            final_down_idx -= 1 
    
    # while magnet[final_down_idx] < baseline:
    #     final_down_idx -=1
    return final_down_idx


def up_search(start_ind, magnet, baseline, smooth_factor=10):
    final_up_idx = start_ind
    if (magnet[final_up_idx-smooth_factor:final_up_idx] <= magnet[final_up_idx+1]).all():
        while (
                np.mean(
                    magnet[final_up_idx-smooth_factor:final_up_idx] <= magnet[final_up_idx+1]
                    )
                and magnet[final_up_idx] < baseline
                ):
            final_up_idx += 1
    else:
        while np.mean(
                magnet[final_up_idx-smooth_factor:final_up_idx] > magnet[final_up_idx+1]
                ):
            final_up_idx += 1
        while (
                np.mean(
                    magnet[final_up_idx-smooth_factor:final_up_idx] <= magnet[final_up_idx+1]
                    )
                and magnet[final_up_idx] < baseline
                ):
            final_up_idx += 1
    return final_up_idx



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
    start_time = time_stamps[new_Down_idx[0]]
    end_time = time_stamps[new_Down_idx[-1]]
    magnet_start_end = (start_time, end_time)
    
    saved_data = Data.copy()
    if plot:
        plt.figure()
        plt.plot(Data.Time.values, Data.Hall_sensor.values, alpha=0.5)
        # plt.scatter(time_stamps[new_Down_idx], magnet[new_Down_idx], c='b')
        # plt.scatter(time_stamps[new_Up_idx], magnet[new_Up_idx], c='r')
        # for [ind, a, b, v] in zip(new_Down_idx,
        #                     time_stamps[new_Down_idx], 
        #                     time_stamps[new_Up_idx], 
        #                     magnet[new_Down_idx]):
        #     plt.hlines(v,a,b)
        #     plt.text(a, v + 1, f'{ind}', fontsize=8, color='b') # Plot predicted press num
        # for [ind, a, v] in zip(new_Up_idx,
        #                        time_stamps[new_Up_idx], 
        #                        magnet[new_Up_idx]):
        #     plt.text(a, v - 1, f'{ind}', fontsize=8, color='r')
    return new_Down_idx, new_Up_idx, magnet_start_end, saved_data
    

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
        # plt.scatter(Down_idx, np.ones_like(Down_idx)+target, c='b')
        # plt.scatter(Up_idx, np.ones_like(Up_idx)+target, c='r')
        
    
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
    
    # if it never detect the right amount of presses, skip this day
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
        plt.title(f'#{mouse} | {date} | LPs={GOAL}') # This goal will be 1 more than real # to acct for beg / end crossings)')
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
    # print(p1)
    # print(p2)
    # print(p3)
    # print(p4)
    return pcorr

def load_excel_log(file, master_df, mouse, date):
    """Read press indices from excel annotations into python to drop false + / -
    and separate into in / out of sequences.
    
    in_seq_presses: list of lists of five tuples, each tuple has a pair of 
        down and up press INDICES (each list is one sequence, each tuple
                                   is a press in that sequence)
    out_seq_presses: list of tuples of press indices 
    """
    # Find which presses are in-sequence
    mouse_df = master_df[master_df['Mouse'] == mouse]
    LPs = np.array(mouse_df[mouse_df['Date'] == date]['Lever'].values[0][1:])
    rewards = np.array(mouse_df[mouse_df['Date'] == date]['Reward'].values[0])
    in_seq_ind = []
    for rwd in rewards:
        in_seq_ind.append([ind for ind in np.where(LPs <= rwd)[0][-5:]])
    
    assert len(rewards) == len(in_seq_ind)
    
    # Read excel log annotations
    excel_log = pd.read_excel(file, usecols='A:C')
    set_in_seq_ind = set([press for seq in in_seq_ind for press in seq])
    all_ind = set(range(len(LPs)))
    out_seq_ind = all_ind - set_in_seq_ind
    
    assert len(set_in_seq_ind) == (len(in_seq_ind) * 5)
    assert len(out_seq_ind) + len(set_in_seq_ind)
    
    # Separate in and out of sequence presses
    in_seq_presses_temp = [] # list of lists with 5 tuples, each list = one sequence
    in_seq_presses = []
    out_seq_presses = [] #List of tuples
    incorrect = []
    for sequence in in_seq_ind:
        in_seq_presses_temp.append([])
        for press_ind in sequence:
            curr_press = excel_log[excel_log['medpc'] == press_ind] # This automatically filters out false positives
            down = curr_press['down'].values[0]
            up = curr_press['up'].values[0]
            if isinstance(down, str):
                if '*' in down:
                    down = int(down[:-1])
                    incorrect.append(down)
            if isinstance(up, str):
                if '*' in up:
                    up = int(up[:-1])
                    incorrect.append(up)
            in_seq_presses_temp[-1].append((down, up))
    
    for press_ind in out_seq_ind:
        curr_press = excel_log[excel_log['medpc'] == press_ind] # This automatically filters out false positives
        down = curr_press['down'].values[0]
        up = curr_press['up'].values[0]
        if isinstance(down, int) and isinstance(up, int): # Will be skipped if marked 'False negative'
            if down not in incorrect and up not in incorrect: # Skip poorly identified indices
                out_seq_presses.append((down, up))
            
    #Drop False negatives from In_seq_presses
    for sequence in in_seq_presses_temp:
        for press in sequence:
            if 'False negative' in press:
                keep = False
                break
            else:
                keep = True
        if keep:
            in_seq_presses.append(sequence)
        
    
    return in_seq_presses, out_seq_presses, excel_log

def five_by_five(magnet_file, magnet_log, master_df, mouse, date, 
                 length_of_vector=1000, save=False):
    """Median pairwise correlations between press no. 1->5 for all presses in
    a single mouse's session.
    """
    Data=pd.read_csv(magnet_file)
    Data.columns=['Hall_sensor','Magnet_block','Time']
    Data, new_Down_idx, new_Up_idx, rough_baseline = Get_press_indices(Data, master_df, mouse, date)
    magnet = Data['Hall_sensor'].values
    (in_sequence_presses, 
    out_sequence_presses, 
    incorrect_indices) = load_excel_log(magnet_log, master_df, mouse, date)
    final_matrix = np.eye(5)
    all_seq_corrs = {}
    
    press_matrix = np.zeros([5,length_of_vector], int)
    sequence = in_sequence_presses[0]
    final_matrix = [[[] for _ in range(5)] for _ in range(5)]
    for ind in range(5):
        final_matrix[ind][ind] = 1
    for seq_ind, sequence in enumerate(in_sequence_presses):
        for press_ind,press in enumerate(sequence):
            pressDown = press[0]
            pressUp = press[1]
            
            if pressUp-pressDown>length_of_vector:
                stop=length_of_vector
            else:
                stop=pressUp-pressDown
                
            press_matrix[press_ind,:stop]=magnet[pressDown:pressDown+stop]
        pcorr = manual_pairwise_pearsonr(press_matrix.T,press_matrix.T)
        all_seq_corrs[seq_ind] = pcorr
        
        for i in range(5):
            for j in range(5):
                if i != j:
                    final_matrix[i][j].append(pcorr[i, j])
                    
    for i in range(5):
        for j in range(5):
            if i != j:
                final_matrix[i][j] = np.median(final_matrix[i][j])
    
    flat_matrix = np.ravel(final_matrix)
    drop_Ones = flat_matrix[flat_matrix!=1]
    
    plt.figure()
    plt.title(f'5x5 Corr Matrix | {date} | #{mouse} | vector={length_of_vector}')
    plt.imshow(final_matrix)
    plt.colorbar()
    plt.clim(vmin=0,vmax=1)
    if save:
        plt.savefig(f'fivebyfive_{mouse}_{date}_{length_of_vector}.png', dpi=500)
    
    return final_matrix, all_seq_corrs, np.median(drop_Ones)

def in_vs_out_seq_corr(magnet_file, magnet_log, master_df, mouse, date, 
                       length_of_vector=1000, save=False):
    """Generate a correlation matrix for in and out of sequence presses."""          
    Data=pd.read_csv(magnet_file)
    Data.columns=['Hall_sensor','Magnet_block','Time']
    Data, new_Down_idx, new_Up_idx, rough_baseline = Get_press_indices(Data, master_df, mouse, date)
    magnet = Data['Hall_sensor'].values
    (in_sequence_presses, 
    out_sequence_presses, 
    incorrect_indices) = load_excel_log(magnet_log, master_df, mouse, date)
    
    all_in_seq_vectors = np.zeros((len(in_sequence_presses) * 5, length_of_vector))
    all_out_seq_vectors = np.zeros((len(out_sequence_presses), length_of_vector))
    
    # Generate list of in-sequence vectors and make a correlation matrix
    for seq_ind, sequence in enumerate(in_sequence_presses):
        for press_ind, press in enumerate(sequence):
            pressDown = press[0]
            pressUp = press[1]
            
            if pressUp-pressDown>length_of_vector:
                stop=length_of_vector
            else:
                stop=pressUp-pressDown
                
            all_in_seq_vectors[(seq_ind * 5 + press_ind),:stop]=magnet[pressDown:pressDown+stop]
    plt.figure()
    final_in_seq_matrix = manual_pairwise_pearsonr(all_in_seq_vectors.T,
                                                    all_in_seq_vectors.T)
    plt.imshow(final_in_seq_matrix)
    plt.colorbar()
    plt.clim(vmin=0, vmax=1)
    plt.show()
    plt.title(f'In Sequence LPs | {date} | #{mouse} | vector={length_of_vector}')
    if save:
        plt.savefig(f'in_seq_corr_{mouse}_{date}_{length_of_vector}.png', dpi=500)
    # Generate list of iout-of-sequence vectors and make a correlation matrix
    for press_ind, press in enumerate(out_sequence_presses):
        pressDown = press[0]
        pressUp = press[1]
        
        if pressUp-pressDown>length_of_vector:
            stop=length_of_vector
        else:
            stop=pressUp-pressDown
        all_out_seq_vectors[press_ind,:stop]=magnet[pressDown:pressDown+stop]
    
    plt.figure()
    final_out_seq_matrix = manual_pairwise_pearsonr(all_out_seq_vectors.T,
                                                   all_out_seq_vectors.T)
    plt.imshow(final_out_seq_matrix)
    plt.colorbar()
    plt.clim(vmin=0,vmax=1)
    plt.title(f'Out of Sequence LPs | {date} | #{mouse} | vector={length_of_vector}')
    plt.show()
    if save:
        plt.savefig(f'out_seq_corr_{mouse}_{date}_{length_of_vector}.png', dpi=500)


def full_session_corr(Data, in_seq_presses, out_seq_presses, mouse, date, 
                       length_of_vector=1000, show=False, save=False):
    """Generate a correlation matrix for all presses in a session."""          
    
    magnet = Data['Hall_sensor'].values
    temp = [item for item in in_seq_presses] # make a copy
    temp.append(out_seq_presses)
    all_press_ind = [ind for seq in temp for ind in seq]
    all_vectors = np.zeros((len(all_press_ind), length_of_vector))
    all_vectors.sort()
    
    # Make press matrix using length_of_vector
    for press_ind, press in enumerate(all_press_ind):
        pressDown = press[0]
        pressUp = press[1]
        
        if pressUp-pressDown>length_of_vector:
            stop=length_of_vector
        else:
            stop=pressUp-pressDown
            
        all_vectors[press_ind,:stop]=magnet[pressDown:pressDown+stop]
    if show:
        plt.figure()
        final_in_seq_matrix = manual_pairwise_pearsonr(all_vectors.T,
                                                        all_vectors.T)
        plt.imshow(final_in_seq_matrix)
        plt.colorbar()
        plt.clim(vmin=0, vmax=1)
        plt.show()
        plt.title(f'All Presses | {date} | #{mouse} | vector={length_of_vector}')
    if save:
        plt.savefig(f'all_press_corrs_{mouse}_{date}_{length_of_vector}.png', dpi=500)
    return all_vectors

def corr_by_press_no(Data, in_seq_presses, mouse, date, 
                       length_of_vector=1000, show=True, save=False):
    """Generate a correlation matrix based on press no. All press #1's
    compared against eachother, etc. 
    Will plot 5 graphs for a single session.
    """          
    magnet = Data['Hall_sensor'].values
    all_presses = [
        np.zeros((len(in_seq_presses), length_of_vector)) for _ in range(5)
        ]
    
    all_corrs = dict.fromkeys([0, 1, 2, 3, 4])
    
    # Make five press matrices each holding all first presses, second, etc.
    for seq_ind, sequence in enumerate(in_seq_presses):
        for press_ind, press in enumerate(sequence):
            pressDown = press[0]
            pressUp = press[1]
            
            if pressUp-pressDown>length_of_vector:
                stop=length_of_vector
            else:
                stop=pressUp-pressDown
            
            all_presses[press_ind][seq_ind,:stop]=magnet[pressDown:pressDown+stop]
    
    ordinals = {0: 'First', 1: 'Second', 2: 'Third', 3: 'Fourth', 4: 'Fifth'}
    if show:
        for press_ind in range(5):
            all_corrs[press_ind] = manual_pairwise_pearsonr(all_presses[press_ind].T,
                                                            all_presses[press_ind].T)
            plt.figure()
            plt.imshow(all_corrs[press_ind])
            plt.colorbar()
            plt.clim(vmin=0, vmax=1)
            plt.show()
            plt.title(f'{ordinals[press_ind]} Press | {date} | #{mouse} | vector={length_of_vector}')
            if save:
                plt.savefig(f'{ordinals[press_ind].lower()}_press_{mouse}_{date}_{length_of_vector}.png', 
                            dpi=500)
    return all_presses, all_corrs
        
if __name__ == '__main__':
    from create_medpc_master import create_medpc_master
    mice = [i for i in range(4386, 4414)]
    file_dir = '/Users/emma-fuze-grace/Lab/Medpc Data'
    master_df = create_medpc_master(mice, file_dir)
    mouse = 4407
    date = '20220216'
    dates = ['20220209', '20220216', '20220224']
    length = 1000
    lengths = [640, 1000, 1000]
    press_length_medians = [640, 524, 478]
    # press_length_25_percent = [407, 404, 364]
    # press_length_10_percent = [299, 349, 281]
    # for length, date in zip(lengths, dates):
    magnet_file = f'/Users/emma-fuze-grace/Lab/Hall Sensor Data/HallSensor_{date}_{mouse}.csv'
    Data = pd.read_csv(magnet_file)
    Data.columns = ['Hall_sensor','Magnet_block','Time']
    # (Data, new_Down_idx, new_Up_idx, saved_data, ratio, rough_baseline, scaled_LPs) = Get_press_indices(Data, master_df, mouse, date, [False, False, False, True])
    final_press_ind = medpc_press_detection(Data, master_df, mouse, date)
    # magnet_log = f'MagnetIndexLog_{date}_{mouse}.xlsx'
    # (in_seq_presses, out_seq_presses, temp)  = load_excel_log(magnet_log,master_df,mouse,date)
    # all_vectors = full_session_corr(Data, in_seq_presses, out_seq_presses, mouse, date, 
    #                     length, show=False, save=False)
    # all_presses, all_corrs = corr_by_press_no(Data, in_seq_presses, mouse, date, 
    #                         length, show=False, save=False)
    # plt.imshow(all_presses[1], aspect='auto')
    #plt.colorbar()
    ## Check that out vs in sequences are correct
    # time_stamps=Data['Time'].values
    # for sequence in in_seq_presses:
    #     x1 = time_stamps[sequence[0][0]]
    #     x2 = time_stamps[sequence[4][1]]
    #     width = x2 - x1
    #     rect = patches.Rectangle((x1, rough_baseline - 50), width, 100, 
    #                               edgecolor='r', alpha=0.1, facecolor='r')
    #     ax = plt.gca()
    #     ax.add_patch(rect)
    # for press in out_seq_presses:
    #     x1 = time_stamps[press[0]]
    #     x2 = time_stamps[press[1]]
    #     width = x2 - x1
    #     rect2 = patches.Rectangle((x1, rough_baseline - 50), width, 100, 
    #                               edgecolor='k', alpha=0.1, facecolor='k')
    #     ax = plt.gca()
    #     ax.add_patch(rect2)
    
            
    
        ### Plot five by five correlation matrices for press no. ####
        # (final_matrix, 
        # all_seq_corrs, 
        # median_corr) = five_by_five(magnet_file, magnet_log, master_df, mouse, 
        #                             date, length, save=True)

        ### Graph all preses in/out seq ###
        # plt.figure()
        # for press in out_seq_presses:
        #     plt.plot(magnet[press[0]:press[1]], 'k')
        # plt.title(f'Out of Sequence Presses | {date} | #{mouse}')
        # plt.xlim((0, 2000))
        # plt.savefig(f'in_seq_LPs_{mouse}_{date}.png', dpi=500)
        # plt.figure()
        # for sequence in in_seq_presses:
        #     for press in sequence:
        #         plt.plot(magnet[press[0]:press[1]], 'b')
        # plt.xlim((0, 2000))
        # plt.title(f'In Sequence Presses | {date} | #{mouse}')
        # plt.savefig(f'out_seq_LPs_{mouse}_{date}.png', dpi=500)
    
        ## Plot press length histograms #####
        # press_lengths = []
        # for sequence in in_seq_presses:
        #     for press in sequence:
        #         press_lengths.append(press[1] - press[0])
        # print('-----')
        # print(f'{date[4:6]}-{date[6:8]} 10th Percentile:')
        # print(np.percentile(press_lengths, 10))
        # print(f'{date[4:6]}-{date[6:8]} 25th Percentile:')
        # print(np.percentile(press_lengths, 25))
        # print(f'{date[4:6]}-{date[6:8]} Median:')
        # print(np.percentile(press_lengths, 50))
        # print('-----')
        # plt.figure()
        # plt.hist(press_lengths, bins=20, density=True)
        # plt.ylim(0, 0.003)
        # plt.title(f'Press Length Histogram | {date} | #{mouse}')
    
        # in_vs_out_seq_corr(magnet_file, magnet_log, master_df, mouse, date, length, save=True)
