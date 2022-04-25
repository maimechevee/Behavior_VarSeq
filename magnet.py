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
from itertools import product
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import pandas as pd
import numpy as np
from scipy.signal import convolve2d
from mpl_toolkits.mplot3d import Axes3D
import scipy
import pickle

class Old_Magnet():
    
    def __init__(self, magnet_file, date_df, mouse, date, process=False, 
                 keep=True, RAW_DF=None, smooth_df=None, bins=10,
                 on_threshold=15, independent=5000,first_last_ind=(0,-2),
                 off_threshold=15, magnet_off_indices=(0,1000),
                 top_ind=1, bottom_ind=-1,weird_last_press=False,
                 normal=True, flip=False, first_threshold=0.4):
        # Save hall sensor files and medpc df (from master_df)
        self.file = magnet_file
        self.normal=normal
        self.date_df = date_df
        self.mouse = mouse
        self.flip = flip
        self.date = date
        self.keep = keep
        self.first_threshold = first_threshold
        self.weird_last_press=weird_last_press
        self.magnet_off_indices = magnet_off_indices
        self.RAW_LPs = np.array(date_df['Lever'].values[0][1:]) # starts with a 0
        self.RAW_REWARDS = np.array(date_df['Reward'].values[0])
        print(f'LPs: {len(self.RAW_LPs)}')
        print(f'Rewards: {len(self.RAW_REWARDS)}')
        print(f'Date: {date_df["Date"].values[0]}')
        in_seq_ind = []
        for rwd in self.RAW_REWARDS:
            in_seq_ind.append([ind for ind in np.where(self.RAW_LPs <= rwd)[0][-5:]])
        self.in_seq_ind = in_seq_ind
        
        # Load hall sensor file
        self.RAW_DF = pd.read_csv(magnet_file)
        self.RAW_DF.columns = ['Hall_sensor','Magnet_block','Time']
        
        # Flip data if necessary
        RAW_MAGNET = self.RAW_DF.Hall_sensor.values
        if RAW_MAGNET[0] > np.mean(RAW_MAGNET) or self.flip:
            self.RAW_DF.Hall_sensor = self.RAW_DF.Hall_sensor * (-1)
        if np.mean(RAW_MAGNET) < 1:
            self.RAW_DF.Hall_sensor = self.RAW_DF.Hall_sensor + abs(min(self.RAW_DF.Hall_sensor))
        
        self.smooth(plot=False)
        # Call other class functions to process data
        if process:
            self.detect_magnet_ON_OFF(on_threshold=on_threshold,
                                      off_threshold=off_threshold,
                                      magnet_off_indices=magnet_off_indices,
                                      plot=True)
            self.detect_press_boundaries(top_ind=top_ind,
                                         bottom_ind=bottom_ind)
            self.detect_first_last_press(first_last_ind=first_last_ind,
                                         first_threshold=first_threshold)
            self.scale_medpc()
            self.create_baselines(independent=independent)
    

    def __repr__(self):
        return ('Magnet Session\n' +
                f'File: {self.file}\n'
                f'Mouse: {self.mouse}\n' +
                f'Date: {self.date}\n' +
                f'Keep: {self.keep}\n' + 
                f'Press Boundaries: {(round(self.top), round(self.bottom))}\n' + 
                f'LPs: {len(self.RAW_LPs)}'
            )
    
    def plot(self, plot_what=[]):
        """General plot function to take advantage of saved attributes."""
        plt.figure()
        plt.title(f'#{self.mouse}, {self.date}')
        if 'raw' in plot_what:
            plt.plot(self.RAW_DF.Time.values, self.RAW_DF.Hall_sensor.values,
                     alpha=0.5)
        else:
            plt.plot(self.times, self.magnet, alpha=0.5)
        if 'press_boundaries' in plot_what:
            plt.hlines(self.top, self.times[0], self.times[-1], linewidth=2,
                       color='r')
            plt.hlines(self.bottom, self.times[0], self.times[-1], linewidth=2,
                       color='r')
        if 'hist_bins' in plot_what:
            plt.hlines([hist_data[1] for hist_data in self.sorted_hist], 
                       self.times[0], self.times[-1])
        if 'first_last_press' in plot_what:
            if self.normal:
                plt.hlines(self.first_last_target, self.times[0], self.times[-1], 
                            linewidth=2, color='k')
            else:
                plt.hlines(self.target_1, self.times[0], self.times[-1], 
                            linewidth=2, color='k')
                plt.hlines(self.target_2, self.times[0], self.times[-1], 
                            linewidth=2, color='k')
            plt.plot(self.times[self.first_down_idx], 
                     self.magnet[self.first_down_idx], marker='o',
                     markersize=10,
                     color='k')
            plt.plot(self.times[self.last_down_idx], 
                     self.magnet[self.last_down_idx], marker='o', 
                     markersize=10)
        if 'medpc' in plot_what:
            plt.plot(self.scaled_LPs, 
                [self.top + 5 for _ in range(len(self.scaled_LPs))],
                linestyle='--', marker='*', markersize=10, color='m')
            plt.plot(self.scaled_rewards,
                [self.top + 10 for _ in range(len(self.scaled_rewards))],
                linestyle='--', marker='^', markersize=10, color='c')
            for ind, press in enumerate(self.scaled_LPs):
                plt.text(press, self.top+7, f'{ind}',
                             fontsize=8)
        
        if 'lines' in plot_what:
            plt.vlines(self.scaled_LPs, self.bottom, self.top, alpha=0.75,
                        color ='k')
                
        if 'baselines' in plot_what:
            t, LPs = self.times, self.scaled_LPs
            for ind, press, baseline in zip(
                    range(len(self.baselines)),
                    LPs, self.baselines):
                if ind > 0:
                    end = np.where(t == t[t >= press][0])[0][0]
                    end_last = np.where(t == t[t >= LPs[ind - 1]][0])[0][0]
                    start = end - self.baseline_length
                    if abs(press - LPs[ind - 1]) > self.independent_threshold:
                        plt.hlines(baseline, t[start], t[end],
                                   linewidth=2, color='r', label='baseline')
                    else:
                        plt.hlines(baseline, t[end_last], t[end],
                                    linewidth=2, linestyle='--', color='r')
                else:
                    end = np.where(t == t[t >= press][0])[0][0]
                    start = end - self.baseline_length
                    plt.hlines(baseline, t[start], t[end],
                               linewidth=2, color='r', label='baseline')
        if 'presses' in plot_what:
            m, t = self.magnet, self.times
            for ind, press in self.final_press_ind.items():
                down, up = press[0], press[1]
                if down and up:
                    if ind % 2:
                        plt.plot(t[down:up],m[down:up], color='m', 
                                 linewidth=2, alpha=0.75)
                    else:
                        plt.plot(t[down:up],m[down:up], color='c', 
                                 linewidth=3, alpha=0.75)
                # plt.text(t[down], m[down]+3, f'{ind} D',
                #              fontsize=8)
                # plt.text(t[up], m[up], f'{ind} U',
                #              fontsize=8)
                    
                    
            
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
        self.magnet_orig = self.magnet.copy()
        self.times_orig = self.times.copy()
        
        if plot:
            plt.figure()
            plt.title('Raw Data')
            plt.plot(self.RAW_DF.Hall_sensor.values)
            plt.plot(cut_signal)
        

    def detect_magnet_ON_OFF(self, on_threshold=15, off_threshold=15, 
                             magnet_off_indices=(0,1000), plot=False):
        """Find indices for when magnet turns on and off.
        Note: indices correspond to original, untruncated indexing scheme.
        """
        
        # Grab data from first second of file when magnet is still off
        if self.flip:
            magnet_off_sample = self.magnet[-10_000:]
            magnet_off_median = np.median(magnet_off_sample)
            magnet_off_std = np.std(magnet_off_sample)
        else:
            magnet_off_sample = self.magnet[
                magnet_off_indices[0]:magnet_off_indices[1]]
            magnet_off_median = np.median(magnet_off_sample)
            magnet_off_std = np.std(magnet_off_sample)
        # print(magnet_off_median)
        # print(magnet_off_std)
        # Identify magnet onset
        ii = magnet_off_indices[1] # start at end of magnet_off baseline   
        not_found = True
        beg_ind = 1
        #loop through data until you deflect from OFF to ON
        if self.flip or not isinstance(on_threshold, int):
            beg_ind = 0
        else:
            while not_found :
                if (abs(self.magnet_orig[ii] - magnet_off_median) > on_threshold * magnet_off_std) :
                    beg_ind = ii
                    not_found = False
                ii = ii + 1
        # print(beg_ind)
        # Identify magnet offset 
        if isinstance(off_threshold, int):
            not_found = True
            end_ind = beg_ind 
            while not_found :#loop through data until you deflect from ON to OFF
                if (abs(self.magnet_orig[ii] - magnet_off_median) < off_threshold * magnet_off_std) :
                    end_ind = ii
                    not_found = False
                ii = ii + 1
        else:
            end_ind = len(self.times) - 1
        print(end_ind)
        # Save to Magnet object attributes
        self.beg_ind = beg_ind
        self.end_ind = end_ind
        print(beg_ind, end_ind)
        self.magnet = self.magnet_orig[beg_ind:end_ind]
        self.times = self.times_orig[beg_ind:end_ind]
        
        # Grab baseline during OFF state using first 1000 ms
        self.rough_baseline = np.mean(self.magnet[beg_ind:beg_ind + 1000]);

        if plot:
            plt.figure()
            plt.title('Find beg/end indices')
            plt.plot(self.times_orig, self.magnet_orig, alpha=0.5)
            plt.vlines([self.times_orig[beg_ind], self.times_orig[end_ind]], 
                       min(self.magnet_orig), max(self.magnet_orig), 
                       color='r')
            plt.hlines(magnet_off_median, self.times_orig[magnet_off_indices[0]],
                       self.times_orig[magnet_off_indices[1]], color='r',linewidth=3)
        
    def detect_press_boundaries(self, top_ind=1, bottom_ind=-2, bins=10):
        """Find typical top and bottom boundaries of presses based on
        frequency of hall sensor values. Discard least common 
        values (these will be during magnet off), then sort by hall sensor
        values and choose top and bottom boundaries.
        """
        histogram = np.histogram(self.magnet, bins)
        sorted_hist = sorted(zip(histogram[0], histogram[1]), reverse=True)
        sorted_hist = [item for item in sorted_hist if item[0] > 1000]
        sorted_hist = sorted(sorted_hist, key=lambda x:x[1], reverse=True)
        top, bottom = sorted_hist[top_ind][1], sorted_hist[bottom_ind][1]
        print(sorted_hist)
        print(bottom_ind)
        # Save data
        self.sorted_hist = sorted_hist
        self.top = top
        self.bottom = bottom
        
    
    def detect_first_last_press(self,first_last_ind=(0, -2),first_threshold=0.4):
        if self.weird_last_press:
            first_last_ind=(0, -1)
        """Detect first and last press (down) indices."""
        self.first_last_target = self.top - first_threshold * (self.top - self.bottom)
        bool_mask = self.magnet < self.first_last_target
        crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
        self.all_down_idxs = np.where(np.invert(bool_mask[:-1]) & crossings)
        self.all_up_idxs = np.where(bool_mask[:-1] & crossings )
        self.first_down_idx = self.all_down_idxs[0][first_last_ind[0]]
        self.last_down_idx = self.all_down_idxs[0][first_last_ind[1]]
            
        
    def scale_medpc(self):
        magnet_time_difference = (
            self.times[self.last_down_idx] - self.times[self.first_down_idx]
            )
        medpc_time_difference = (self.RAW_LPs[-1] - self.RAW_LPs[0]) * 1000
        self.ratio = magnet_time_difference / medpc_time_difference
        scaled_LPs = [
            press * self.ratio * 1000 for press in self.RAW_LPs]
        self.scaled_LPs = [press - scaled_LPs[0] + self.times[self.first_down_idx] 
                           for press in scaled_LPs] # center them on first press
        scaled_rewards = [
            reward * self.ratio * 1000 for reward in self.RAW_REWARDS]
        self.scaled_rewards = [(reward - scaled_LPs[0] + self.times[self.first_down_idx])
                               for reward in scaled_rewards]
        
    def create_baselines(self, length=1000, independent=5000):
        """Set up baselines for presses that are more than 5000 ms apart.
        For closer presses, use previous baseline
        """
        m, t, LPs = self.magnet, self.times, self.scaled_LPs
        baselines = np.zeros(len(self.scaled_LPs))
        if self.first_down_idx-length > 0:
            baselines[0] = np.median(m[
                self.first_down_idx-length:self.first_down_idx])
        else:
            baselines[0] = np.median(m[:self.first_down_idx-length])
        
        for ind, press, baseline in zip(
                range(1, len(baselines)),
                LPs[1:], baselines[1:]):
            end = np.where(t == t[t >= press][0])[0][0] # Baseline ends at press boundary
            start = end - length
            if press - LPs[ind - 1] > independent:
                curr_baseline = np.median(m[start:end])
            else:
                curr_baseline = baselines[ind - 1]
            baselines[ind] = curr_baseline
        self.independent_threshold = independent
        self.baseline_length = length
        self.baselines = baselines
        
    def optimize_thresholds(self, percent=0.6, plot=False):
        """Define threshold for each baseline by starting at a fixed
        distance and moving until atleast 1 crossing is found.
        """
        m, t, LPs, baselines = (self.magnet, self.times, self.scaled_LPs, 
                                self.baselines)
        thresholds = np.zeros(len(baselines))
        for ind, press, baseline in zip(range(len(LPs)), 
                                        LPs, baselines):
            # Assign beg/end search indices
            if ind < len(LPs) - 1:
                search_times = t[(t > press) & (t < LPs[ind + 1])]
                if search_times.size:
                    search_start_ind = np.where(t == search_times[0])[0][0]
                    search_end_ind = np.where(t == search_times[-1])[0][0]
                else:
                    print(f'Very short press, {ind}, using previous press')
                    thresholds[ind] = 9999
                    # in case there is no data in between medpc time stamps
                    continue
                    
            else:
                search_times = t[t > press]
                search_start_ind = np.where(t == search_times[0])[0][0]
                search_end_ind = len(m) - 1
                    
                
            # Conduct search
            search_magnet = m[search_start_ind:search_end_ind]
            curr_baseline = baselines[ind]
            curr_offset = percent * (self.top-self.bottom)
            move = 0.05 * (self.top-self.bottom)
            original_move = move
            
            # Attempt #1
            while (curr_target := curr_baseline - curr_offset) < curr_baseline-(2*move):
                if plot:
                    plt.hlines(curr_target, t[search_start_ind], t[search_end_ind], color='k',
                                label='target', linestyle ='--')
                bool_mask = search_magnet < curr_target 
                crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
                down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
                up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
                if np.where(down_xings)[0].size > 0 or np.where(up_xings)[0].size > 0:
                    break
                curr_offset -= move
                
            # Attempt #2
            if np.where(down_xings)[0].size == 0 and (np.where(up_xings)[0].size == 0):
                curr_offset = percent * (self.top-self.bottom)
                move = original_move
                while (curr_target := curr_baseline - curr_offset) > self.bottom-10:
                    if plot:
                        plt.hlines(curr_target, t[search_start_ind], t[search_end_ind], color='k',
                                    label='target', linestyle ='--')
                    bool_mask = search_magnet < curr_target 
                    crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
                    down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
                    up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
                    if np.where(down_xings)[0].size > 0 or np.where(up_xings)[0].size > 0:
                        break
                    curr_offset += move
            
            # Attempt #3
            # Repeat in case still nothing found
            # I need to make this code more efficient...
            if np.where(down_xings)[0].size == 0 and (np.where(up_xings)[0].size == 0):
                print(f'Extending baseline beyond search end: {ind}')
                search_magnet = m[search_start_ind:search_end_ind + 500]
                curr_baseline = baselines[ind]
                curr_offset = percent * (self.top-self.bottom)
                move = 0.05 * (self.top-self.bottom)
                original_move = move
                while (curr_target := curr_baseline - curr_offset) < curr_baseline:
                    # print(curr_target, curr_offset)
                    search_magnet = m[search_start_ind:search_end_ind + 500]
                    if plot:
                        plt.hlines(curr_target, t[search_start_ind], t[search_end_ind + 500],
                                    label='target', linestyle ='-',alpha=0.5,
                                    color='r')
                    bool_mask = search_magnet < curr_target 
                    crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
                    down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
                    up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
                    if np.where(down_xings)[0].size > 0 or np.where(up_xings)[0].size > 0:
                        break
                    search_end_ind +=50
            if plot:
                if np.where(down_xings)[0].size:
                    plt.gca().plot(t[np.where(down_xings)[0][0]], m[np.where(down_xings)[0][0]],
                              color='k', markersize=3)
                if np.where(up_xings)[0].size:
                    plt.gca().plot(t[np.where(up_xings)[0][0]], m[np.where(up_xings)[0][0]],
                              color='k', markersize=3)
            thresholds[ind] = curr_target
        self.move = original_move
        self.thresholds = thresholds
            
    
    def detect_press_ind(self, smooth=15, plot=True, down_offset=50, up_offset=50,
                         hill_start=50, hill_end=100, num_hill_checks=2,
                         alignment='left'):
        """"Main Search for press indices."""
        m, t, LPs, baselines = (self.magnet, self.times, self.scaled_LPs, 
                                self.baselines)
        final_press_ind = {ind:(0,0) for ind in range(len(LPs))}
        for ind, press, threshold, curr_baseline in zip(range(len(LPs)), 
                                        LPs, self.thresholds,
                                        baselines):
            
            
            if threshold == 9999: 
                final_press_ind[ind] = final_press_ind[ind-1]
                continue
            if ind < 1:
                prev_up = 0
                next_press = LPs[ind + 1]
            elif ind < len(LPs) - 1:
                prev_up = final_press_ind[ind-1][1]
                next_press = LPs[ind + 1]
            else:
                prev_up = final_press_ind[ind-1][1]
                next_press = t[-1]
            
            # Define search vector and look for crossings in that vector
            searching = True
            search_times = t[(t > press) & (t < next_press)]
            search_start_ind = np.where(t == search_times[0])[0][0]
            search_end_ind = np.where(t == search_times[-1])[0][0]
            while searching == True:
                search_magnet = m[search_start_ind:search_end_ind]
                bool_mask = search_magnet < threshold
                crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
                down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
                up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
                num_down = np.where(down_xings)[0].size
                num_up = np.where(up_xings)[0].size
                num_detected = num_down + num_up
                down_choice, up_choice = 0, 0
                if num_detected > 0:
                    searching = False
                    break
                search_end_ind += 25
                # print(f'Extending search end: {ind}')
                if plot:
                    plt.hlines(threshold, t[search_start_ind], t[search_end_ind + 500],
                                linestyle ='-',alpha=0.5)
                
            
            if self.weird_last_press and (ind == len(LPs) - 1):
                down_crossing = search_start_ind + np.where(down_xings)[0][0]
                final_down_idx = self.down_search(down_crossing, 
                                             baselines[ind], smooth)
                final_up_idx = final_down_idx + 750
                plt.plot(t[final_down_idx:final_up_idx], 
                         m[final_down_idx:final_up_idx], color='r', 
                         linewidth=2, alpha = 0.75)
                final_press_ind[ind] = (final_down_idx, final_up_idx)
                continue
            
            # Look for indices from crossing points
            # If you cut off the right half of the press, start "up idx"
            # search at the end of the search vector (plus the searcch_offset) 
            # Same idea for if you cut off left half of the press
            if num_detected == 1:
                if num_down:
                    down_crossing = search_start_ind + np.where(down_xings)[0][0]
                    down_choice, up_choice = (down_crossing,
                                              search_end_ind + up_offset)
                    if plot:
                        plt.plot(t[down_crossing], m[down_crossing],
                                 marker='o', color='k')
                        plt.plot(t[up_choice], m[up_choice],
                                 marker='*', color='r', markersize=10)
                if num_up:
                    up_crossing = search_start_ind + np.where(up_xings)[0][0]
                    down_choice, up_choice = (search_start_ind - down_offset,
                                              up_crossing)
                    if plot: 
                        plt.plot(t[up_crossing], m[up_crossing],
                                 marker='o', color='k')
            elif num_detected == 2:
                down_crossing = search_start_ind + np.where(down_xings)[0][0]
                up_crossing = search_start_ind + np.where(up_xings)[0][0]
                if down_crossing < up_crossing:
                    down_choice, up_choice = down_crossing, up_crossing
                else:
                    # This is setup to detect the leftmost press detected
                    # If the previous up index has already been found in the search vector,
                    # then look for next press
                    if ind ==len(LPs):
                        down_choice, up_choice = (search_start_ind - down_offset,
                                                  up_crossing)
                    elif alignment == 'right':
                        down_choice, up_choice = (down_crossing,
                                                  search_end_ind + up_offset)
                    
                    else:
                        down_choice, up_choice = (search_start_ind - down_offset,
                                                  up_crossing)
            elif num_detected > 2:
                down_crossing = search_start_ind + np.where(down_xings)[0]
                up_crossing = search_start_ind + np.where(up_xings)[0]
                if down_crossing[0] < prev_up:
                    if len(up_crossing) >= 2:
                        final_down_idx, up_choice = prev_up, up_crossing[-1]
                    else:
                        final_down_idx, up_choice = prev_up, search_end_ind + up_offset
                elif down_crossing[0] > up_crossing[0]:
                    # Could indicate a second press detected
                    if ind == len(LPs) -1 or alignment == 'left':
                        down_choice, up_choice = (search_start_ind - down_offset,
                                                  up_crossing[0])
                    else:
                        down_choice, up_choice = down_crossing[0], up_crossing[-1]
                else:
                    down_choice, up_choice = down_crossing[0], up_crossing[-1]
            else:
                print(f'Error in crossings: {ind}')
                continue
            
            
            ### Perform main search based on choices for where to start ###
            if down_choice:
                final_down_idx = self.down_search(down_choice, 
                                             baselines[ind], smooth)
                if plot:
                    plt.plot(t[down_choice], m[down_choice],
                             marker='^', color='b', markersize=3)
            if up_choice:
                final_up_idx = self.up_search(up_choice,
                                         baselines[ind], smooth)
                if plot:
                    plt.plot(t[up_choice], m[up_choice],
                             marker='^', color='r', markersize=3)
            
            
            ### Check for indices being stuck on "hills" during search ###
            # First need to check if far away from previous press and not too close to baseline
            
            # hill check for down indices
            for _ in range(num_hill_checks):
                hill_check_down = m[
                    (final_down_idx - hill_end):(final_down_idx - hill_start)
                    ]
                if abs(m[final_down_idx] - curr_baseline) > (15 * self.move): 
                    if np.mean(hill_check_down) > m[final_down_idx] and m[final_down_idx] < curr_baseline:
                        if plot:
                            plt.plot(t[(final_down_idx - hill_end):(final_down_idx - hill_start)],
                                     hill_check_down,
                                     color='r')
                            plt.plot(t[final_down_idx], m[final_down_idx],
                                         marker='o', color='k', 
                                         markersize=4, linestyle='None')
                        final_down_idx = self.down_search(final_down_idx, 
                                                     baselines[ind], smooth)
            
            # hill check for up indices
            for _ in range(num_hill_checks):
                hill_check_up = m[
                    (final_up_idx + hill_start):(final_up_idx + hill_end)
                    ]
                if ind < len(LPs) - 1:
                    if abs(m[final_up_idx] - curr_baseline) > (15 * self.move):
                        if np.mean(hill_check_up) > m[final_up_idx] and m[final_up_idx] < curr_baseline:
                            if plot:
                                plt.plot(t[(final_up_idx + hill_start):(final_up_idx + hill_end)],
                                         hill_check_up,
                                         color='r')
                                plt.plot(t[final_up_idx], m[final_up_idx],
                                             marker='o', color='k', 
                                             markersize=4, linestyle='None')
                            final_up_idx = self.up_search(final_up_idx,
                                                         baselines[ind], smooth)
            
            assert final_up_idx > final_down_idx
            ### PLOT ###
            if plot:
                # final down and up indices
                plt.plot(t[final_down_idx], m[final_down_idx],
                             marker='o', color='b', markersize=4)
                plt.plot(t[final_up_idx], m[final_up_idx],
                         marker='o', color='r', markersize=6)
                
                # highlight press
                # if ind in np.ravel(self.in_seq_ind):
                #     color = 'm'
                # else:
                #     color = 'g'
                
                if ind %2:
                    color = 'm'
                else:
                    color = 'g'
                plt.plot(t[final_down_idx:final_up_idx], 
                         m[final_down_idx:final_up_idx], color=color, 
                         linewidth=2, alpha = 0.75)
                
                # Press indices
                plt.text(t[final_down_idx], m[final_down_idx] + 3, f'{ind} D',
                             fontsize=8)
                plt.text(t[final_up_idx], m[final_up_idx], f'{ind} U',
                             fontsize=8)
                
                # Crossings
                if num_down:
                    plt.plot(t[down_crossing], m[down_crossing],
                             marker='o', color='k', linestyle = 'None')
                if num_up:
                    plt.plot(t[up_crossing], m[up_crossing],
                             marker='o', color='k', linestyle = 'None')
                    
            
            # Save to dictionary
            final_press_ind[ind] = (final_down_idx, final_up_idx)
        self.final_press_ind = final_press_ind
        
    
    def down_search(self, start_ind, baseline, smooth_factor=10):
        m = self.magnet
        final_down_idx = start_ind
        if (m[final_down_idx:final_down_idx+smooth_factor] <= m[final_down_idx-1]).all():
            while (
                    np.mean(
                        m[final_down_idx:final_down_idx+smooth_factor] <= m[final_down_idx-1]
                        )
                    and m[final_down_idx] < baseline
                    ):
                final_down_idx -= 1 
        else:
            while (
                    np.mean(
                        m[final_down_idx:final_down_idx+smooth_factor] > m[final_down_idx-1]
                        )
                    and m[final_down_idx] < baseline
                    ):
                final_down_idx -= 1 
            while (
                    np.mean(
                    m[final_down_idx:final_down_idx+(smooth_factor)] <= m[final_down_idx-1]
                    )
                    and m[final_down_idx] < baseline
                    ):
                final_down_idx -= 1 
        
        return final_down_idx


    def up_search(self, start_ind, baseline, smooth_factor=10):
        final_up_idx = start_ind
        m = self.magnet
        if (m[final_up_idx-smooth_factor:final_up_idx] <= m[final_up_idx+1]).all():
            while (
                    np.mean(
                        m[final_up_idx-smooth_factor:final_up_idx] <= m[final_up_idx+1]
                        )
                    and m[final_up_idx] < baseline
                    ):
                final_up_idx += 1
        else:
            while np.mean(
                    m[final_up_idx-smooth_factor:final_up_idx] > m[final_up_idx+1]
                    ):
                final_up_idx += 1
            while (
                    np.mean(
                        m[final_up_idx-smooth_factor:final_up_idx] <= m[final_up_idx+1]
                        )
                    and m[final_up_idx] < baseline
                    ):
                final_up_idx += 1
        return final_up_idx
    
    def plot_sequences(self):
        plt.figure()
        starts = [self.final_press_ind[seq[0]][0] for seq in self.in_seq_ind]
        ends = [self.final_press_ind[seq[4]][1] for seq in self.in_seq_ind]
        for start, end in zip(starts, ends):
            plt.plot(self.magnet[start:end])
    
    def plot_all_presses(self):
        plt.figure()
        downs = [press[0] for press in self.final_press_ind.values()]
        ups = [press[1] for press in self.final_press_ind.values()]
        for d, u in zip(downs, ups):
            plt.plot(self.magnet[d:u], alpha= 0.5, color ='b')

### Outside the Class ###

def load_final_LP_vectors(magnet, final_press_ind,override_zscore=(False,'n/a')):
    """Returns a dictionary where keys are press index and values
    are arrays of presses."""
    if override_zscore[0]:
        magnet = sp.stats.zscore(magnet[:override_zscore[1]])
    else:
        magnet = sp.stats.zscore(magnet)
    final_LP_vectors = {ind: [] for ind in range(len(final_press_ind))}
    for ind, press_ind in final_press_ind.items():
        down_idx, up_idx = press_ind[0], press_ind[1]
        assert down_idx > 0 and up_idx > 0
        if up_idx - down_idx > 15_000:
            final_LP_vectors[ind] = magnet[down_idx:down_idx+15_000]
            print(f'{ind} is a long press: {up_idx - down_idx}')
        else:
            final_LP_vectors[ind] = magnet[down_idx:up_idx]
        assert len(final_LP_vectors[ind])
    return final_LP_vectors

def day_by_day(mouse_dict, mouse, length=500, vmin=0, vmax=1, traces=False, plot=False,
               save=False):
    all_matrices = {}
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    final_corr_matrix = np.zeros((len(dates), len(dates)))
    for date, vectors in mouse_dict.items():
        all_matrices[date] = np.zeros((len(vectors), length))
        for press_ind, press in enumerate(vectors.values()):
            if len(press) < length:
                all_matrices[date][press_ind,:len(press)] = press
            else:
                all_matrices[date][press_ind,:length] = press[:length]
    
    for pair in product(date_indices, repeat=2):
        i, j = pair[0], pair[1]
        print(f'{i}: {dates[i]} /// {j}: {dates[j]}')
        day_1, day_2 = all_matrices[dates[i]], all_matrices[dates[j]]
        pcorr = manual_pairwise_pearsonr(day_1.T, day_2.T)
        flat_pcorr=np.ravel(pcorr)
        drop_Ones = flat_pcorr[flat_pcorr!=1]
        final_corr_matrix[i,j]=np.median(drop_Ones)
        final_corr_matrix[j,i]=np.median(drop_Ones)
    if plot:
        plt.figure()
        plt.imshow(final_corr_matrix, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f'Day by day: #{mouse}, length={length}')
        if save:
            plt.savefig(f'day_by_day_{mouse}_{length}.png', dpi=500)
    return all_matrices, final_corr_matrix

def colorful_traces(mouse_dict, mouse, date, num_presses=100, 
                    x_max=1000, save=False):
    colors = ['#ff0000', '#ffa500', '#ffff00', '#008000', '#0000ff',
              '#ee82ee', '#4b0082']
    plt.figure()
    plt.title(f'{mouse}, {date}')
    plt.xlabel('time (ms)')
    plt.xlim((0, x_max))
    # plt.title(f'{mouse}, {date}, #LPs = {len(vectors)}')
    vectors = mouse_dict[date]
    sample_ind = [i for i in range(len(mouse_dict[date]))]
    from random import sample
    for press_ind in sample(sample_ind, k=num_presses):
        press = vectors[press_ind]
        length = len(press)
        x_list = [i for i in range(0, length, int(length/len(colors)))]
        for ind, (x, color) in enumerate(zip(x_list[:-1], colors)):
            if ind == len(colors):
                plt.gca().plot([i for i in range(x, x_list[ind + 1])],
                         press[x:x_list[ind + 1]], alpha=0.25, color=color)
            else:
                plt.gca().plot([i for i in range(x, x_list[ind + 1] + 1)],
                         press[x:x_list[ind + 1] + 1], alpha=0.25, color=color)


def mean_traces(mouse_dict, mouse, date, length=2000, save=False,
                color='k', label=''):
    vectors = mouse_dict[date]
    mean_trace = np.zeros((len(vectors), length))
    for press_ind, press in enumerate(vectors.values()):
        if len(press) < length:
            mean_trace[press_ind,:len(press)] = press
        else:
            mean_trace[press_ind,:length] = press[:length]
    mean_trace = np.mean(mean_trace,axis=0)
    plt.gca().plot(mean_trace, alpha=0.5, color=color,label=label)
    plt.title(f'{mouse}')
    plt.xlabel('time (ms)')
    return mean_trace

def press_lengths(mouse, save=False, color='k', label=''):
    with open(f'magnet_presses_{mouse}.pkl', 'rb') as handle:
        mouse_dict = pickle.load(handle)
    dates = sorted(list(mouse_dict.keys()))
    lengths = []
    avg_lengths = []
    for date in dates:
        vectors = mouse_dict[date]
        curr_day = []
        for press_ind, press in enumerate(vectors.values()):
            curr_day.append(len(press))
        lengths.append(curr_day)
        avg_lengths.append(np.mean(curr_day))
    plt.plot(avg_lengths, color=color, alpha=0.5, label=f'{mouse}')
    return lengths, avg_lengths


def color_hist(mouse, cutoff, bin_width=100,cmap='hot'):
    lst = []
    lengths, avg_lengths=press_lengths(mouse)
    for i, length in enumerate(lengths):
        y = np.histogram(lengths[i],density=True, 
                         bins=[i for i in range(0,cutoff,bin_width)])
        lst.append(y[0])
    plt.title(f'#{mouse}')
    plt.imshow(np.flip(np.array(lst).T, axis=0), cmap=cmap,
               extent=[0, 1000, 0, cutoff])

# def mean_traces(all_matrices, mouse, date, color='k', length=2000, save=False):
#     vectors = mouse_dict[date]
#     max_length = max([len(vector) for vector in vectors.values()])
#     all_presses = np.zeros((len(vectors), max_length))
#     all_presses[:] = np.NaN
#     mean_trace = []
#     stds = []
#     # a[np.invert(np.isnan(a))][0]
#     for press_ind, press in enumerate(vectors.values()):
#         all_presses[press_ind,:len(press)] = press
#     for col_ind in range(max_length):
#         col = all_presses[:,col_ind]
#         col_avg = np.mean(col[np.invert(np.isnan(col))])
#         col_std = np.std(col[np.invert(np.isnan(col))])
#         stds.append(col_std)
#         mean_trace.append(col_avg)
#         if col_ind == 250: 
#             print(len(col[np.invert(np.isnan(col))]))
#     plt.plot(mean_trace, color=color)
#     # lower = np.array(mean_trace) - np.array(stds)
#     # upper = np.array(mean_trace) + np.array(stds)
#     # plt.gca().fill_between(mean_trace, lower, upper,
#     #                        alpha =0.5, facecolor='b')
#     return all_presses, mean_trace,stds
def by_press_no(all_matrices, mouse_dict, master_df, mouse, length=1000,
                vmin=0, vmax=1, save=False, plot=False):
    """"Each square is median value of correlation of all first (second...)
    presses from one session and all first (second...) presses from another 
    session.
    """
    mouse_df = master_df[master_df['Mouse'] == mouse]
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    matrices_by_press_no = {i:[[] for _ in range(len(dates))] for i in range(5)}
    all_in_seq_ind = {date:[] for date in dates}
    
    # Organize data by press no.
    for date_ind, date in enumerate(dates):
        press_matrix = all_matrices[date]
        LPs = np.array(mouse_df[mouse_df['Date'] == date]['Lever'].values[0][1:])
        rewards = np.array(mouse_df[mouse_df['Date'] == date]['Reward'].values[0])
        in_seq_ind = []
        for rwd in rewards:
            in_seq_ind.append([ind for ind in np.where(LPs <= rwd)[0][-5:]])
        all_in_seq_ind[date] = in_seq_ind
        order_indices = {press:np.where(seq==press)[0][0] for seq in in_seq_ind for press in seq}
        for press_num in range(5):
            matrices_by_press_no[press_num][date_ind] = np.zeros((len(rewards), length))
        for seq_ind, seq in enumerate(in_seq_ind):
            for press_ind in seq:
                order, press = order_indices[press_ind], press_matrix[press_ind]
                matrices_by_press_no[order][date_ind][seq_ind,:] = press
    
    # Correlations and plotting
    for n in range(5):
        plt.figure()
        plt.title(f'{mouse}, press no. #{n+1}, length={length}')
        matrices = matrices_by_press_no[n]
        curr_corr_matrix = np.zeros( (len(dates), len(dates)) )
        for pair in product(date_indices, repeat=2):
            i, j = pair[0], pair[1]
            print(dates[i],dates[j])
            day_1, day_2 = matrices[i], matrices[j]
            pcorr = manual_pairwise_pearsonr(day_1.T, day_2.T)
            flat_pcorr=np.ravel(pcorr)
            drop_Ones = flat_pcorr[flat_pcorr!=1]
            curr_corr_matrix[i,j]=np.median(drop_Ones)
            curr_corr_matrix[j,i]=np.median(drop_Ones)
        if plot:
            plt.imshow(curr_corr_matrix, vmin=vmin, vmax=vmax)
            plt.colorbar()
            # Save
            if save:
                plt.savefig(f'press_no_{n + 1}_{mouse}_{length}.png', dpi=500)
    return matrices_by_press_no, all_in_seq_ind
            
        
def in_vs_out_seq_corr(all_matrices, all_in_seq_ind, mouse_dict, master_df, 
                       mouse, length=500,vmin_in=0, vmax_in=1,
                       vmin_out=0,vmax_out=1, save=False, plot=False):
    mouse_df = master_df[master_df['Mouse'] == mouse]
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    in_seq_corr_matrix = np.zeros((len(dates), len(dates)))
    out_seq_corr_matrix = np.zeros((len(dates), len(dates)))
    in_seq_matrices = {date:[] for date in dates}
    out_seq_matrices = {date:[] for date in dates}
    
    for i, date in enumerate(dates):
        in_seq_ind = all_in_seq_ind[date]
        press_matrix = all_matrices[date]
        LPs = np.array(mouse_df[mouse_df['Date'] == date]['Lever'].values[0][1:])
        set_in_seq_ind = set([press for seq in in_seq_ind for press in seq])
        all_ind = set(range(len(LPs)))
        out_seq_ind = all_ind - set_in_seq_ind
        in_seq_matrices[date] = np.zeros((len(in_seq_ind) * 5, length))
        out_seq_matrices[date] = np.zeros((len(out_seq_ind), length))
        
        for j, ind in enumerate(list(set_in_seq_ind)):
            in_seq_matrices[date][j] = press_matrix[ind]
            
        for k, ind in enumerate(list(out_seq_ind)):
            out_seq_matrices[date][k] = press_matrix[ind]
            
    
    # In sequence presses
    plt.figure()
    for pair in product(date_indices, repeat=2):
        i, j = pair[0], pair[1]
        print(dates[i], dates[j])
        day_1, day_2 = in_seq_matrices[dates[i]], in_seq_matrices[dates[j]]
        pcorr = manual_pairwise_pearsonr(day_1.T, day_2.T)
        flat_pcorr=np.ravel(pcorr)
        drop_Ones = flat_pcorr[flat_pcorr!=1]
        in_seq_corr_matrix[i,j]=np.median(drop_Ones)
        in_seq_corr_matrix[j,i]=np.median(drop_Ones)
    if plot:
        plt.imshow(in_seq_corr_matrix, vmin=vmin_in, vmax=vmax_in)
        plt.title(f'In sequence presses - #{mouse}, length={length}')
        plt.colorbar()
    
    # Save
    if save:
        plt.savefig(f'in_seq_{mouse}_{length}.png', dpi=500)
    
    # Out of sequence resses
    plt.figure()
    for pair in product(date_indices, repeat=2):
        i, j = pair[0], pair[1]
        print(dates[i], dates[j])
        day_1, day_2 = out_seq_matrices[dates[i]], out_seq_matrices[dates[j]]
        pcorr = manual_pairwise_pearsonr(day_1.T, day_2.T)
        flat_pcorr=np.ravel(pcorr)
        drop_Ones = flat_pcorr[flat_pcorr!=1]
        out_seq_corr_matrix[i,j]=np.median(drop_Ones)
        out_seq_corr_matrix[j,i]=np.median(drop_Ones)
    if plot:
        plt.imshow(out_seq_corr_matrix, vmin=vmin_out, vmax=vmax_out)
        plt.colorbar()
        plt.title(f'Out of sequence presses - #{mouse}, length={length}')
    
    if save:
        plt.savefig(f'out_seq_{mouse}_{length}.png', dpi=500)
    return in_seq_matrices, out_seq_matrices

def within_seq_corr(all_matrices, mouse_dict, all_in_seq_ind, in_seq_matrices,
                    matrices_by_press_no, length=1000, full=False):
    dates_strings = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates_strings))]
    dates = {date_str:date_ind for date_ind, date_str in zip(date_indices, dates_strings)}
    all_press_pcorrs = []
    pcorrs_by_day = []
    all_seq = []
    all_seq_by_day = {}
    total_in_seq_presses = 0
    
    for date_str, vectors in all_matrices.items():
        in_seq_ind = all_in_seq_ind[date_str]
        curr_day = []
        all_seq_by_day[date_str] = []
        for seq_num, seq in enumerate(in_seq_ind):
            curr_seq = np.zeros((5, length))
            for press_no, press_ind in enumerate(seq):
                total_in_seq_presses += 1
                curr_seq[press_no, :length] = matrices_by_press_no[press_no][dates[date_str]][seq_num][:length]
            all_seq.append(curr_seq)
            all_seq_by_day[date_str].append(curr_seq)
            pcorr = manual_pairwise_pearsonr(curr_seq.T, curr_seq.T)
            flat_pcorr = np.ravel(pcorr)
            drop_Ones = flat_pcorr[flat_pcorr!=1]
            curr_day.append(np.median(drop_Ones))
            all_press_pcorrs.append(np.median(drop_Ones))
            print(len(curr_seq[0]))
        pcorrs_by_day.append(np.mean(curr_day))
    
    plt.figure()
    plt.plot(all_press_pcorrs)
    plt.figure()
    plt.plot(pcorrs_by_day)
    print(f'Total sequences: {total_in_seq_presses / 5}')
    print(f'Total presses: {total_in_seq_presses}')
    
    if full:
        all_seq_corr_matrix = np.zeros((int(total_in_seq_presses / 5), int(total_in_seq_presses / 5)))
        all_seq_indices = list(range(len(all_seq)))
        for pair in product(all_seq_indices, repeat=2):
            i, j = pair[0], pair[1]
            seq_1, seq_2 = all_seq[i], all_seq[j]
            pcorr = manual_pairwise_pearsonr(seq_1.T, seq_2.T)
            flat_pcorr=np.ravel(pcorr)
            drop_Ones = flat_pcorr[flat_pcorr!=1]
            all_seq_corr_matrix[i,j]=np.median(drop_Ones)
            all_seq_corr_matrix[j,i]=np.median(drop_Ones)
        plt.figure()
        plt.imshow(all_seq_corr_matrix)
    
    # within_by_day_pcorr = np.zeros((len(dates), len(dates)))
    # for pair in product(date_indices, repeat=2):
    #     i, j = pair[0], pair[1]
    #     print(dates_strings[i], dates_strings[j])
    #     day_1, day_2 = all_seq_by_day[dates_strings[i]], all_seq_by_day[dates_strings[j]]
    #     temp = [[(i,j) for j in range(len(day_2))] for i in range(len(day_1))]
    #     indices = []
    #     for row in temp:
    #         for col in row:
    #             indices.append(col)
    #     curr_square = np.zeros((len(day_1), len(day_2)))
    #     for pair in indices:
    #         k, m = pair[0], pair[1]
    #         seq_1, seq_2 = day_1[k], day_2[m]
    #         pcorr = manual_pairwise_pearsonr(seq_1.T, seq_2.T)
    #         flat_pcorr=np.ravel(pcorr)
    #         if dates_strings[i] == dates_strings[j] and k == m:
    #             drop_Ones = flat_pcorr[flat_pcorr!=1]
    #             curr_square[k,m]=np.median(drop_Ones)
    #         else:
    #             curr_square[k,m]=np.median(flat_pcorr)
    #     within_by_day_pcorr[i, j] = np.median(curr_square)
    #     if i ==0 and j == 0:
    #         plt.figure()
    #         plt.imshow(curr_square)
    #     if i == len(dates) and j == len(dates):
    #         break
    # plt.figure()
    # plt.imshow(within_by_day_pcorr)
    return pcorrs_by_day, all_seq_by_day
    
    
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

import cv2

def all_presses_corr(mouse_dict, mouse, sigma= 1, length=1000, vmin=0, vmax=1, plot=False,
                blur=101, save=False):
    num_presses = sum([len(value) for value in mouse_dict.values()])
    all_presses = np.zeros((num_presses, length))
    dates = sorted(list(mouse_dict.keys()))
    row = 0
    
    for date in dates:
        vectors = mouse_dict[date]
        for press_ind, press in enumerate(vectors.values()):
            if len(press) < length:
                all_presses[row,:len(press)] = press
            else:
                all_presses[row,:length] = press[:length]
            row += 1
    corr_mat = manual_pairwise_pearsonr(all_presses.T, all_presses.T)
    blurred = cv2.GaussianBlur(corr_mat, (blur,blur), 0, borderType=cv2.BORDER_REFLECT_101)
    if plot:
        plt.figure()
        plt.imshow(blurred,cmap='jet',vmin=0,vmax=1)
        plt.title(f'{mouse}, {date}')
    plt.hist(np.ravel(blurred), bins=50)
    return all_presses, corr_mat, blurred
    
def scan_files(mice, date):
    for mouse in mice:
        magnet_file = (
            '/Users/emma-fuze-grace/Lab/hall_sensor_data'+ 
            f'/{date}/HallSensor_{date}_{mouse}.csv'
            )
        try:
            magnet_df = pd.read_csv(magnet_file)
        except FileNotFoundError:
            print(f'FileNotFoundError: {mouse}')
            continue
        magnet_df = pd.read_csv(magnet_file)
        magnet_df.columns = ['Hall_sensor','Magnet_block','Time']
        plt.figure()
        plt.title(f'{mouse}, {date}')
        plt.plot(magnet_df.Hall_sensor.values)
    
def test(mouse, date, freq=0.000000005):
    magnet_file = (
        '/Users/emma-fuze-grace/Lab/hall_sensor_data'+ 
        f'/{date}/HallSensor_{date}_{mouse}.csv'
        )
    magnet_df = pd.read_csv(magnet_file)
    magnet_df.columns = ['Hall_sensor','Magnet_block','Time']
    plt.figure()
    from scipy.ndimage.filters import uniform_filter1d
    from scipy.fftpack import rfft, irfft, fftfreq
    W = fftfreq(len(magnet_df.Hall_sensor), 
                d=magnet_df.Time.values[-1] - magnet_df.Time.values[0])
    f_signal = rfft(magnet_df.Hall_sensor.values)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>freq)] = 0
    cut_signal = irfft(cut_f_signal)
    smooth_magnet = uniform_filter1d(cut_signal, size=50)
    plt.title(f'{mouse}, {date}')
    plt.plot(smooth_magnet)
    return smooth_magnet
    
def first_last_stats(mice, dates, length=500, color='b'):
    final = {mouse: (0,0) for mouse in mice}
    path=(
        '/Users/emma-fuze-grace/Lab/Behavior_VarSeq/magnet_parameters_and_indxs'
        )
    for mouse, date_pair in zip(mice, dates):
        
        if mouse in [4392,4407,4410,4411]:
            with open(f'{path}/magnet_presses_{mouse}.pkl', 'rb') as handle:
                mouse_dict = pickle.load(handle)
        else:
            with open(f'{path}/mouse_{mouse}.pkl', 'rb') as handle:
                mouse_dict = pickle.load(handle)
        first = {date_pair[0]: mouse_dict[date_pair[0]]}
        last = {date_pair[1]: mouse_dict[date_pair[1]]}
        print(mouse_dict.keys())
        all_presses, corr_mat_1,blurred = all_presses_corr(first, mouse, length=length, blur=5,plot=False)
        all_presses, corr_mat_2,blurred = all_presses_corr(last, mouse, length=length, blur=5,plot=False)
        flat_1 = np.ravel(corr_mat_1)
        flat_2 = np.ravel(corr_mat_2)
        drop_1 = flat_1[flat_1 != 1]
        drop_2 = flat_2[flat_2 != 1]
        final[mouse] = (np.median(drop_1), np.median(drop_2))
    firsts = [corr_vals[0] for corr_vals in final.values()]
    lasts = [corr_vals[1] for corr_vals in final.values()]
    plt.figure()
    for mouse, values in final.items():
        plt.plot(values, alpha=0.5, color=color)
    plt.plot([0, 1],[np.mean(firsts), np.mean(lasts)], 
             linewidth=5, color=color)
    plt.gca().set_xticks([0,1])
    plt.gca().set_xticklabels(['early','late'])
    plt.title(f'Median R-Value n={len(mice)}, length={length}')
    return final
        
if __name__ == '__main__':
    
    ### FR5_CATEG_GROUP_1 ###
    mice = [i for i in range(4392, 4414)]
    # mice = [i for i in range(4667, 4694)]
    # mice = [4222, 4225, 4226, 4230, 4242]
    # group_1 = range(4392, 4396)
    # group_2 = range(4401, 4406)
    # group_3 = range(4407, 4410)
    # group_4 = range(4410, 4414)
    path=(
        '/Users/emma-fuze-grace/Lab/Behavior_VarSeq/magnet_parameters_and_indxs'
        )
    ### FR5 ###
    # FR5_mice_1 = [4218, 4221, 4222, 4224, 4225]
    # FR5_mice_2 = [4226, 4229, 4230, 4232, 4233]
    # FR5_mice_3 = [4237, 4240, 4242, 4243, 4244]
    # mice = FR5_mice_3
    # mice = [4222, 4224, 4225]
    
    file_dir = '/Users/emma-fuze-grace/Lab/medpc_data/medpc_FR5CATEG_old'
    master_df = create_medpc_master(mice, file_dir)
    mouse, date = 4392, '20220213'
    process = True
    # Load magnet session
    mouse_df = master_df[master_df['Mouse'] == mouse]
    date_df = mouse_df[mouse_df['Date']==date]
    
    magnet_file = (
        '/Users/emma-fuze-grace/Lab/hall_sensor_data'+ 
        f'/{date}/HallSensor_{date}_{mouse}.csv'
        )
    
    # session = Old_Magnet(magnet_file, date_df, mouse, date, process,
    #                       on_threshold=15,
    #                       off_threshold=15,
    #                       independent=10000,
    #                       magnet_off_indices=(0,1000),
    #                       first_last_ind=(0,-2),
    #                       top_ind=1,bottom_ind=-2,
    #                       normal=True,
    #                       weird_last_press=False)
    # session.plot(['medpc', 'lines'])
    # session.plot()
    import json

    with open(f'{path}/mouse_{mouse}_parameters.json') as file:
        parameters = json.load(file)
    if date in parameters['weird_last_press']:
        weird_last_press=True
    else:
        weird_last_press=False
    if date in parameters["magnet_off_indices"].keys():
        magnet_off_indices = parameters["magnet_off_indices"][date]
    else:
        magnet_off_indices = (0, 1000)
    if date in parameters["flip"]:
        flip = True
    else:
        flip = False
    session = Old_Magnet(magnet_file, date_df, mouse, date, process,
                          on_threshold=parameters['on_thresholds'][date],
                          off_threshold=parameters['off_thresholds'][date],
                          independent=parameters['independent_thresholds'][date],
                          weird_last_press=weird_last_press,
                          magnet_off_indices=magnet_off_indices,
                          first_last_ind=parameters['first_last_ind'][date],
                          flip=flip, first_threshold=0.4,bottom_ind=-2)
    
    session.plot()
    session.optimize_thresholds(percent=parameters['percents'][date], plot=True)
    session.detect_press_ind(plot=True, 
                              down_offset=parameters['down_offset'][date], 
                              up_offset=parameters['up_offset'][date], 
                              hill_start=parameters['hill_start'][date],
                              num_hill_checks=parameters['num_hill_checks'][date],
                              hill_end=parameters['hill_end'][date],
                              alignment=parameters['alignment'][date] )
    session.plot(['medpc','presses', 'press_boundaries'])
    # session.optimize_thresholds(percent=0.85, plot=True)
    # session.detect_press_ind(plot=True, 
    #                           down_offset=15, 
    #                           up_offset=15, 
    #                           hill_start=50,
    #                           num_hill_checks=2,
    #                           hill_end=100,
    #                           alignment='left')
    mice = [4674, 4677, 4678, 4679, 4680, 4681, 4688, 4692, 4693]
    dates = [
        ('20220309','20220325'),
        ('20220331','20220413'),
        ('20220331','20220413'),
        ('20220325','20220413'),
        ('20220309','20220325'),
        ('20220311','20220330'),
        ('20220311','20220401'),
        ('20220311','20220330'),
        ('20220310','20220330')
        ]
    
    # FR5
    mice = [4222, 4225, 4226, 4230,4242]
    dates = [
        ('20211210','20211215'),
        ('20211210','20211215'),
        ('20211210','20211214'),
        ('20211210','20211215'),
        ('20211210','20211215'),
        ]
    
    # CATEG
    mice = [4392, 4407, 4410, 4411, 4393, 4394, 4401]
    dates = [
        ('20220214','20220224'),
        ('20220214','20220224'),
        ('20220214','20220224'),
        ('20220214','20220224'),
        ('20220216','20220224'),
        ('20220216','20220223'),
        ('20220216','20220223'),
        ('20220216','20220223'),
        ]
    final_LP_vectors = load_final_LP_vectors(session.magnet, 
                                              session.final_press_ind) 
    [[4387, 4396, 4397, 4667, 4668, 4682],
     [4388, 4389, 4398, 4399, 4669, 4670, 4684, 4685],
     [4392, 4393,4394,4395,4401,4402,4403,4404,4405,4407,
      4408,4409,4410,4411]]
    fr5 = [4222, 4224, 4225, 4226, 4229, 4230, 4233, 4240, 4242]
    # session.plot(['presses'])
    # plt.plot(session.times_orig[session.beg_ind],session.magnet_orig[session.beg_ind],markersize=10,marker='*')
    # plt.plot(session.times_orig[session.end_ind],session.magnet_orig[session.end_ind],markersize=10,marker='*')
    # cv2, image smoothing
    # import pickle
    # with open(f'magnet_presses_{mouse}.pkl', 'rb') as handle:
    #     mouse_dict = pickle.load(handle)
    #     # lengths = [300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500]
    #     length = 1000
    #     vmin, vmax = 0, 1
        # all_matrices, final_corr_matrix = day_by_day(mouse_dict, mouse=mouse, length=length, 
        #                                               vmin=vmin, vmax=vmax, plot=True,traces=True,
        #                                               save=False)
    #     matrices_by_press_no, all_in_seq_ind = by_press_no(all_matrices, mouse_dict, master_df, mouse, 
    #                                               length=length, vmin=vmin, vmax=vmax,
    #                                               save=False, plot=True)
    #     in_seq_matrices, out_seq_matrices = in_vs_out_seq_corr(all_matrices, all_in_seq_ind,
    #                                                             mouse_dict,
    #                                                             master_df, mouse,
    #                                                             length=length,
    #                                                             vmin_in=vmin, vmax_in=vmax,
    #                                                             vmin_out=vmin, vmax_out=vmax,
    #                                                             save=False, plot=True)
    #     pcorrs_by_day, all_seq_by_day = within_seq_corr(all_matrices, mouse_dict,
    #                                                                          all_in_seq_ind, in_seq_matrices,
    #                                                                          matrices_by_press_no, length=length, full=False)
        # for i, date in enumerate(dates):

        #     mean_traces(mouse_dict, mouse, date, length=2000, save=False, 
        #     color=(0, float(i/len(dates)), 1))
# TODOs:
    # rolling average
    # day by day
    # show example sequence
    