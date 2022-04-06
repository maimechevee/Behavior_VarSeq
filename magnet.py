#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:50:59 2022

@author: emma-fuze-grace
"""

"""Main script for analyzing hall sensor data.
I use capitals for data that should not be altered (ex. RAW_DF).

RAW_DF refers to the unedited dataframe stripped from CSV file directly. 
    RAW_DF is automatically flipped.
smooth_df: after smoothing and truncating (before z-scoring)
"""

from create_medpc_master import create_medpc_master
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Magnet():
    
    def __init__(self, magnet_file, session_df, mouse, date, process=False, 
                 keep=True, RAW_DF=None, smooth_df=None, bins=10):
        # Save hall sensor files and medpc df (from master_df)
        self.file = magnet_file
        self.session_df = session_df
        self.mouse = mouse
        self.date = date
        self.keep = keep
        self.RAW_LPs = date_df['Lever'].values[0][1:] # starts with a 0
        
        # Load hall sensor file
        self.RAW_DF = pd.read_csv(magnet_file)
        self.RAW_DF.columns = ['Hall_sensor','Magnet_block','Time']
        
        # Flip data if necessary
        RAW_MAGNET = self.RAW_DF.Hall_sensor.values
        if RAW_MAGNET[0] > np.mean(RAW_MAGNET):
            self.RAW_DF.Hall_sensor = self.RAW_DF.Hall_sensor * (-1)
        
        # Call other class functions to process data
        self.smooth()
        
        if process:
            self.detect_magnet_ON_OFF()
            self.detect_press_boundaries()
            self.detect_first_last_press()
            self.scale_LPs()
            self.create_baselines()
    

    def __repr__(self):
        return ('Magnet Session\n' +
                f'File: {self.file}\n'
                f'Mouse: {self.mouse}\n' +
                f'Date: {self.date}\n' +
                f'Keep: {self.keep}'
                f'Press Boundaries: {self.top, self.bottom}'
            )
    
    def plot(self, plot_what=[]):
        """General plot function to take advantage of saved attributes."""
        plt.figure()
        plt.title(f'#{self.mouse}, {self.date}')
        plt.plot(self.times, self.magnet, alpha=0.5)
        if 'press_boundaries' in plot_what:
            plt.hlines(self.top, self.times[0], self.times[-1], linewidth=2,
                       color='r')
            plt.hlines(self.bottom, self.times[0], self.times[-1], linewidth=2,
                       color='r')
        if 'first_last_press' in plot_what:
            plt.hlines(self.first_last_target, self.times[0], self.times[-1], 
                       linewidth=2, color='k')
            plt.plot(self.times[self.first_down_idx], 
                     self.magnet[self.first_down_idx], marker='o',
                     markersize=10,
                     color='k')
            plt.plot(self.times[self.last_down_idx], 
                     self.magnet[self.last_down_idx], marker='o', 
                     markersize=10)
        if 'scale_LPs' in plot_what:
            plt.plot(self.scaled_LPs, 
                [self.rough_baseline + 5 for _ in range(len(self.scaled_LPs))],
                linestyle='--', marker='*', markersize=5, color='m')
        if 'baselines' in plot_what:
            t, LPs = self.times, self.scaled_LPs
            for ind, press, baseline in zip(
                    range(len(self.baselines[1:])),
                    LPs[1:], self.baselines[1:]):
                end = np.where(t == t[t >= press][0])[0][0]
                end_last = np.where(t == t[t >= LPs[ind - 1]][0])[0][0]
                start = end - self.baseline_length
                if abs(press - LPs[ind - 1]) > self.independent_threshold:
                    plt.hlines(baseline, t[start], t[end],
                               linewidth=2, color='r', label='baseline')
                else:
                    plt.hlines(baseline, t[end_last], t[end],
                                linewidth=5, linestyle='--', color='r')
            
    def smooth(self, plot=False):
        """Smooth raw data using fourier transform. Data is not z-scored yet."""
        from scipy.ndimage.filters import uniform_filter1d
        from scipy.fftpack import rfft, irfft, fftfreq
        
        ### Copied from original file ###
        W = fftfreq(len(self.RAW_DF.Hall_sensor), 
                    d=self.RAW_DF.Time.values[-1] - self.RAW_DF.Time.values[0])
        f_signal = rfft(self.RAW_DF.Hall_sensor.values)
        
        # If our original signal time was in seconds, this is now in Hz    
        cut_f_signal = f_signal.copy()
        cut_f_signal[(W>0.000000005)] = 0
        cut_signal = irfft(cut_f_signal)
        smooth_magnet = uniform_filter1d(cut_signal, size=50)
        
        # Save smooth data into copy of RAW_DF so we can access
        # SMOOTH_DF separately later
        self.smooth_df = self.RAW_DF.copy()
        self.smooth_df['Hall_sensor'] = smooth_magnet
        self.magnet = smooth_magnet
        self.times = self.smooth_df['Time'].values
        
        if plot:
            plt.figure()
            plt.title('Raw Data')
            plt.plot(self.RAW_DF.Hall_sensor.values)
            plt.plot(cut_signal)
        

    def detect_magnet_ON_OFF(self, plot=False):
        """Find indices for when magnet turns on and off.
        Note: indices correspond to original, untruncated indexing scheme.
        """
        
        # Grab data from first second of file when magnet is still off
        magnet_off_mean = np.mean(self.magnet[
            self.times < self.times[0] + 1000])
        magnet_off_std = np.std(
            self.magnet[self.times < self.times[0] + 1000])
        threshold = 25 * magnet_off_std
    
        # Find magnet on indices using np.where instead of while loops
        magnet_on_idxs = np.where(abs(self.magnet - magnet_off_mean) > threshold)
        beg_ind, end_ind = magnet_on_idxs[0][0], magnet_on_idxs[0][-1]
        
        # Save to Magnet object attributes
        self.beg_ind = beg_ind
        self.end_ind = end_ind
        self.magnet = self.magnet[beg_ind:end_ind]
        self.times = self.times[beg_ind:end_ind]
        
        # Grab baseline during OFF state using first 1000 ms
        self.rough_baseline = np.mean(self.magnet[beg_ind:beg_ind + 1000]);

        if plot:
            plt.figure()
            plt.title('Find beg/end indices')
            plt.plot(self.times, self.magnet, alpha=0.5)
            plt.vlines([beg_ind, end_ind], min(self.magnet), max(self.magnet), 
                       color='r')
        
    def detect_press_boundaries(self, bins=10):
        """Find typical top and bottom boundaries of presses based on
        frequency of hall sensor values. Discard least common 
        values (these will be during magnet off), then sort by hall sensor
        values and choose top and bottom boundaries.
        """
        histogram = np.histogram(self.magnet, bins)
        sorted_hist = sorted(zip(histogram[0], histogram[1]), reverse=True)
        sorted_hist = [item for item in sorted_hist if item[0] > 1000]
        sorted_hist = sorted(sorted_hist, key=lambda x:x[1], reverse=True)
        top, bottom = sorted_hist[1][1], sorted_hist[-2][1]
        
        # Save data
        self.sorted_hist = sorted_hist
        self.top = top
        self.bottom = bottom
        
    
    def detect_first_last_press(self):
        """Detect first and last press (down) indices."""
        self.first_last_target = self.top - .25 * (self.top - self.bottom)
        bool_mask = self.magnet < self.first_last_target
        crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
        self.all_down_idxs = np.where(np.invert(bool_mask[:-1]) & crossings)
        self.all_up_idxs = np.where(bool_mask[:-1] & crossings )
        self.first_down_idx = self.all_down_idxs[0][0]
        self.last_down_idx = self.all_down_idxs[0][-2]
        
    def scale_LPs(self):
        magnet_time_difference = (
            self.times[self.last_down_idx] - self.times[self.first_down_idx]
            )
        medpc_time_difference = (self.RAW_LPs[-1] - self.RAW_LPs[0]) * 1000
        self.ratio = magnet_time_difference / medpc_time_difference
        scaled_LPs = [
            press * self.ratio * 1000 for press in self.RAW_LPs]
        self.scaled_LPs = [press - scaled_LPs[0] + self.times[self.first_down_idx] 
                           for press in scaled_LPs] # center them on first press
        
    def create_baselines(self, length=1000, independent=5000):
        """Set up baselines for presses that are more than 5000 ms apart.
        For closer presses, use previous baseline
        """
        m, t, LPs = self.magnet, self.times, self.scaled_LPs
        baselines = np.zeros(len(self.scaled_LPs))
        baselines[0] = np.median(self.magnet[
            self.first_down_idx-length:self.first_down_idx])
        
        for ind, press in enumerate(LPs[1:], start=1):
            end = np.where(t == t[t >= press][0])[0][0]
            start = end - length
            if press - LPs[ind - 1] > independent:
                curr_baseline = np.median(m[start:end])
            else:
                curr_baseline = baselines[ind - 1]
            baselines[ind] = curr_baseline
        self.independent_threshold = independent
        self.baseline_length = length
        self.baselines = baselines
        
if __name__ == '__main__':
    # Load medpc master
    mouse, date = 4404, '20220205'
    mice = [i for i in range(4386, 4414)] 
    file_dir = '/Users/emma-fuze-grace/Lab/Medpc Data'
    master_df = create_medpc_master(mice, file_dir)
    mouse_df = master_df[master_df['Mouse'] == mouse]
    date_df=mouse_df[mouse_df['Date']==date]
    
    # Load magnet session
    magnet_file = (
        '/Users/emma-fuze-grace/Lab/Hall Sensor Data/' + 
        f'HallSensor_{date}_{mouse}.csv'
        )
    
    ### Process data ###
    # beg_ind, end_ind, baseline_off = detect_magnet_ON_OFF(smooth_df)
    # smooth_df = smooth_df[beg_ind:end_ind]
    # plt.plot(smooth_df.Hall_sensor.values)
    session = Magnet(magnet_file, date_df, mouse, date, process=True)
    session.plot(['baselines', 'scale_LPs'])
    
    
