#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:50:59 2022

@author: emma-fuze-grace
"""

"""Main script for analyzing hall sensor data.
I use capitals for data that should not be altered (ex. RAW_DF).

RAW_DF refers to the unedited dataframe stripped from CSV file directly.
smooth_df: after flipping, smoothing, and truncating.
"""

from create_medpc_master import create_medpc_master
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def flip(RAW_DF):
    """Flip data if necessary. Modifies RAW_DF."""
    RAW_MAGNET = RAW_DF.Hall_sensor.values
    if RAW_MAGNET[0] > np.mean(RAW_MAGNET):
        RAW_DF.Hall_sensor = RAW_DF.Hall_sensor * (-1)
    return(RAW_DF)


def smooth(RAW_DF, plot=False):
    """Smooth raw data using fourier transform. Data is not z-scored yet."""
    from scipy.ndimage.filters import uniform_filter1d
    from scipy.fftpack import rfft, irfft, fftfreq
    
    ### Copied from original file ###
    W = fftfreq(len(RAW_DF.Hall_sensor), 
                d=RAW_DF.Time.values[-1] - RAW_DF.Time.values[0])
    f_signal = rfft(RAW_DF.Hall_sensor.values)
    
    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>0.000000005)] = 0
    cut_signal = irfft(cut_f_signal)
    smooth_magnet = uniform_filter1d(cut_signal, size=50)
    
    # Save smooth data into copy of RAW_DF so we can access
    # SMOOTH_DF separately later
    smooth_df = RAW_DF.copy()
    smooth_df['Hall_sensor'] = smooth_magnet
    
    if plot:
        plt.figure()
        plt.title('Raw Data')
        plt.plot(RAW_DF.Hall_sensor.values)
        plt.plot(cut_signal)
    
    return smooth_df


def detect_magnet_ON_OFF(smooth_df, plot=False):
    """Find indices for when magnet turns on and off.
    Note: indices correspond to original, untruncated indexing scheme.
    """
    # Name arrays based on parent df to avoid confusion
    smooth_magnet = smooth_df.Hall_sensor.values
    smooth_time = smooth_df.Time.values
    
    # Grab data from first second of file when magnet is still off
    magnet_off_mean = np.mean(smooth_magnet[
        smooth_time < smooth_time[0] + 1000])
    magnet_off_std = np.std(smooth_magnet[smooth_time < smooth_time[0] + 1000])
    threshold = 12 *magnet_off_std

    # Find magnet on using np.where instead of while loops
    magnet_on_idxs = np.where(abs(smooth_magnet - magnet_off_mean) > threshold)
    beg_ind, end_ind = magnet_on_idxs[0][0], magnet_on_idxs[0][-1]
    
    # Grab baseline during OFF state using first 1000 ms 
    baseline_off = np.mean(smooth_magnet[beg_ind:beg_ind + 1000]);
    
    if plot:
        plt.figure()
        plt.title('Find beg/end indices')
        plt.plot(smooth_magnet, alpha=0.5)
        plt.vlines([beg_ind, end_ind], min(smooth_magnet), max(smooth_magnet), 
                   color='r')
        
    return beg_ind, end_ind, baseline_off


if __name__ == '__main__':
    # Load medpc master
    mice = [i for i in range(4386, 4414)] 
    file_dir = '/Users/emma-fuze-grace/Lab/Medpc Data'
    master_df = create_medpc_master(mice, file_dir)
    
    # Load magnet session
    mouse, date = 4407, '20220224'
    magnet_file = (
        '/Users/emma-fuze-grace/Lab/Hall Sensor Data/' + 
        f'HallSensor_{date}_{mouse}.csv'
        )
    
    RAW_DF = pd.read_csv(magnet_file)
    RAW_DF.columns = ['Hall_sensor','Magnet_block','Time']
    
    # Process data
    smooth_df = smooth(flip(RAW_DF))
    beg_ind, end_ind, baseline_off = detect_magnet_ON_OFF(smooth_df)
    smooth_df = smooth_df[beg_ind:end_ind]
