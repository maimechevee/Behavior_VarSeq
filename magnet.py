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


class Old_Magnet():
    
    def __init__(self, magnet_file, session_df, mouse, date, process=False, 
                 keep=True, RAW_DF=None, smooth_df=None, bins=10,
                 on_off_threshold=15, independent=5000,weird_last_press=False):
        # Save hall sensor files and medpc df (from master_df)
        self.file = magnet_file
        self.session_df = session_df
        self.mouse = mouse
        self.date = date
        self.keep = keep
        self.weird_last_press = weird_last_press
        self.RAW_LPs = np.array(date_df['Lever'].values[0][1:]) # starts with a 0
        self.RAW_REWARDS = np.array(date_df['Reward'].values[0])
        in_seq_ind = []
        for rwd in self.RAW_REWARDS:
            in_seq_ind.append([ind for ind in np.where(self.RAW_LPs <= rwd)[0][-5:]])
        self.in_seq_ind = in_seq_ind
        
        # Load hall sensor file
        self.RAW_DF = pd.read_csv(magnet_file)
        self.RAW_DF.columns = ['Hall_sensor','Magnet_block','Time']
        
        # Flip data if necessary
        RAW_MAGNET = self.RAW_DF.Hall_sensor.values
        if RAW_MAGNET[0] > np.mean(RAW_MAGNET):
            self.RAW_DF.Hall_sensor = self.RAW_DF.Hall_sensor * (-1)
        
        if np.mean(RAW_MAGNET) < 1:
            self.RAW_DF.Hall_sensor = self.RAW_DF.Hall_sensor + abs(min(self.RAW_DF.Hall_sensor))
        
        self.smooth()
        # Call other class functions to process data
        if process:
            self.detect_magnet_ON_OFF(on_off_threshold)
            self.detect_press_boundaries()
            self.detect_first_last_press(weird_last_press)
            self.scale_medpc()
            self.create_baselines(independent=independent)
    

    def __repr__(self):
        return ('Magnet Session\n' +
                f'File: {self.file}\n'
                f'Mouse: {self.mouse}\n' +
                f'Date: {self.date}\n' +
                f'Keep: {self.keep}\n'
                f'Press Boundaries: {self.top, self.bottom}' + 
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
            # plt.hlines(self.first_last_target, self.times[0], self.times[-1], 
            #            linewidth=2, color='k')
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
            plt.vlines(self.scaled_LPs, self.bottom, self.top, alpha=0.75,
                       color ='k')
            for ind, press in enumerate(self.scaled_LPs):
                plt.text(press, self.top+7, f'{ind}',
                             fontsize=8)
                
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
                    # ind in np.ravel(self.in_seq_ind):
                    if ind % 2:
                        color = 'm'
                    else:
                        color = 'g'
                    plt.plot(t[down:up],m[down:up], color=color, 
                             linewidth=2, alpha = 0.75)
                plt.text(t[down], m[down]+3, f'{ind} D',
                             fontsize=8)
                plt.text(t[up], m[up], f'{ind} U',
                             fontsize=8)
                    
                    
            
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
        

    def detect_magnet_ON_OFF(self, on_off_threshold=15, plot=False):
        """Find indices for when magnet turns on and off.
        Note: indices correspond to original, untruncated indexing scheme.
        """
        
        # Grab data from first second of file when magnet is still off
        magnet_off_median = np.median(self.magnet[
            self.times < self.times[0] + 1000])
        magnet_off_std = np.std(
            self.magnet[self.times < self.times[0] + 1000])
        threshold = on_off_threshold * magnet_off_std
    
        # Identify magnet onset
        ii = 1    #counter   
        not_found = True
        beg_ind = 1
        #loop through data until you deflect from OFF to ON
        while not_found :
            if (abs(self.magnet[ii] - magnet_off_median) > threshold) :
                beg_ind = ii
                not_found = False
            ii = ii + 1

        # Identify magnet offset    
        not_found = True
        end_ind = beg_ind 
        while not_found :#loop through data until you deflect from ON to OFF
            if (abs(self.magnet[ii] - magnet_off_median) < threshold) :
                end_ind = ii
                not_found = False
            ii = ii + 1
        
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
        print(sorted_hist)
        top, bottom = sorted_hist[top_ind][1], sorted_hist[bottom_ind][1]
        
        # Save data
        self.sorted_hist = sorted_hist
        self.top = top
        self.bottom = bottom
        
    
    def detect_first_last_press(self,weird_last_press=False):
        """Detect first and last press (down) indices."""
        self.first_last_target = self.top - .5 * (self.top - self.bottom)
        bool_mask = self.magnet < self.first_last_target
        crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
        self.all_down_idxs = np.where(np.invert(bool_mask[:-1]) & crossings)
        self.all_up_idxs = np.where(bool_mask[:-1] & crossings )
        self.first_down_idx = self.all_down_idxs[0][0]
        if weird_last_press:
            self.last_down_idx = self.all_down_idxs[0][-1]
        else:
            self.last_down_idx = self.all_down_idxs[0][-2]
        
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
        baselines[0] = np.median(m[
            self.first_down_idx-length:self.first_down_idx])
        
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
                search_start_ind = np.where(t == search_times[0])[0][0]
                search_end_ind = np.where(t == search_times[-1])[0][0]
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
            while (curr_target := curr_baseline - curr_offset) < curr_baseline:
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
            
            if np.where(down_xings)[0].size:
                plt.gca().plot(t[np.where(down_xings)[0][0]], m[np.where(down_xings)[0][0]],
                          color='k', markersize=3)
            if np.where(up_xings)[0].size:
                plt.gca().plot(t[np.where(up_xings)[0][0]], m[np.where(up_xings)[0][0]],
                          color='k', markersize=3)
            thresholds[ind] = curr_target
        self.move = original_move
        self.thresholds = thresholds
            
    
    def detect_press_ind(self, smooth=15, plot=True, search_offset=50,
                         hill_start=50, hill_end=100, num_hill_checks=5):
        """"Main Search for press indices."""
        m, t, LPs, baselines = (self.magnet, self.times, self.scaled_LPs, 
                                self.baselines)
        final_press_ind = {ind:(0,0) for ind in range(len(LPs))}
        for ind, press, threshold, curr_baseline in zip(range(len(LPs)), 
                                        LPs, self.thresholds,
                                        baselines):
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
            search_times = t[(t > press) & (t < next_press)]
            search_start_ind = np.where(t == search_times[0])[0][0]
            search_end_ind = np.where(t == search_times[-1])[0][0]
            search_magnet = m[search_start_ind:search_end_ind]
            bool_mask = search_magnet < threshold
            crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
            down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
            up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
            num_down = np.where(down_xings)[0].size
            num_up = np.where(up_xings)[0].size
            num_detected = num_down + num_up
            down_choice, up_choice = 0, 0
            
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
                                              search_end_ind + search_offset)
                    if plot:
                        plt.plot(t[down_crossing], m[down_crossing],
                                 marker='o', color='k')
                if num_up:
                    up_crossing = search_start_ind + np.where(up_xings)[0][0]
                    down_choice, up_choice = (search_start_ind - search_offset,
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
                    down_choice, up_choice = (search_start_ind - search_offset,
                                              up_crossing)
            elif num_detected > 2:
                down_crossing = search_start_ind + np.where(down_xings)[0]
                up_crossing = search_start_ind + np.where(up_xings)[0]
                if down_crossing[0] < prev_up:
                    final_down_idx, up_choice = prev_up, up_crossing[-1]
                elif abs(search_end_ind - down_crossing[-1]) < 250:
                    # Could indicate a second press detected
                    down_choice, up_choice = (search_start_ind - search_offset,
                                              up_crossing[0])
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
                             marker='*', color='m')
            if up_choice:
                final_up_idx = self.up_search(up_choice,
                                         baselines[ind], smooth)
                if plot:
                    plt.plot(t[up_choice], m[up_choice],
                             marker='*', color='m')
            
            
            ### Check for indices being stuck on "hills" during search ###
            # First need to check if far away from previous press and not too close to baseline
            
            # hill check for down indices
            for _ in range(num_hill_checks):
                hill_check_down = m[
                    (final_down_idx - hill_end):(final_down_idx - hill_start)
                    ]
                if abs(m[final_down_idx] - curr_baseline) > (1 * self.move): 
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
                    if abs(m[final_up_idx] - curr_baseline) > (1 * self.move):
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
        
        # plt.figure()
        # down = [press[0] for press in final_press_ind.values()]
        # up = [press[1] for press in final_press_ind.values()]
        # for d, u in zip(down, up):
        #     plt.plot(magnet[d:u])

### Outside the Class ###

def load_final_LP_vectors(magnet, final_press_ind):
    """Returns a dictionary where keys are press index and values
    are arrays of presses."""
    magnet = sp.stats.zscore(magnet)
    final_LP_vectors = {ind: [] for ind in range(len(final_press_ind))}
    for ind, press_ind in final_press_ind.items():
        down_idx, up_idx = press_ind[0], press_ind[1]
        if down_idx > 0 and up_idx > 0:
            final_LP_vectors[ind] = magnet[down_idx:up_idx]
        else:
            final_LP_vectors[ind] = 'error'
    return final_LP_vectors

def day_by_day(mouse_dict, length=500, vmin=0, vmax=1, traces=False, plot=False,
               save=False):
    all_matrices = {}
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    final_corr_matrix = np.zeros((len(dates), len(dates)))
    for date, vectors in mouse_dict.items():
        all_matrices[date] = np.zeros((len(vectors), length))
        if traces:
            plt.figure()
            plt.title(f'{mouse}, {date}, #LPs = {len(vectors)}')
            # plt.vlines(length, -5,1)
        for press_ind, press in enumerate(vectors.values()):
            if len(press) < length:
                all_matrices[date][press_ind,:len(press)] = press
            else:
                all_matrices[date][press_ind,:length] = press[:length]
            # if traces:
            #     plt.plot(press)
        if traces:
            data = [len(press) for press in vectors.values()]
            plt.hist(data, density=True,
                      bins=list(range(0, 5000, 100)))
            plt.xlim((0, 5000))
            plt.ylim((0, 0.003))
            if save:
                plt.savefig(f'press_hist_{mouse}_{date}.png', dpi=500)
    
    for pair in product(date_indices, repeat=2):
        i, j = pair[0], pair[1]
        print(dates[i], dates[j])
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

def by_press_no(all_matrices, mouse_dict, master_df, mouse, length=1000,
                vmin=0, vmax=1, save=False, plot=False):
    """"Each square is median value of correlation of all first (second...)
    presses from one session and all first (second...) presses from another 
    session.
    """
    mouse_df = master_df[master_df['Mouse'] == mouse]
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    final_matrices = {i:[[] for _ in range(len(dates))] for i in range(5)}
    
    # Organize data by press no.
    for i, date in enumerate(dates):
        press_matrix = all_matrices[date]
        LPs = np.array(mouse_df[mouse_df['Date'] == date]['Lever'].values[0][1:])
        rewards = np.array(mouse_df[mouse_df['Date'] == date]['Reward'].values[0])
        in_seq_ind = []
        for rwd in rewards:
            in_seq_ind.append([ind for ind in np.where(LPs <= rwd)[0][-5:]])
        order_indices = {press:np.where(seq==press)[0][0] for seq in in_seq_ind for press in seq}
        for j in range(5):
            final_matrices[j][i] = np.zeros((len(rewards), length))
        for seq_ind, seq in enumerate(in_seq_ind):
            for press_ind in seq:
                order, press = order_indices[press_ind], press_matrix[press_ind]
                final_matrices[order][i][seq_ind,:] = press
    
    # Correlations and plotting
    for n in range(5):
        plt.figure()
        plt.title(f'{mouse}, press no. #{n+1}, length={length}')
        matrices = final_matrices[n]
        curr_corr_matrix = np.zeros( (len(dates), len(dates)) )
        for pair in product(date_indices, repeat=2):
            j, k = pair[0], pair[1]
            print(dates[j],dates[k])
            day_1, day_2 = matrices[j], matrices[k]
            pcorr = manual_pairwise_pearsonr(day_1.T, day_2.T)
            flat_pcorr=np.ravel(pcorr)
            drop_Ones = flat_pcorr[flat_pcorr!=1]
            curr_corr_matrix[j,k]=np.median(drop_Ones)
            curr_corr_matrix[k,j]=np.median(drop_Ones)
        if plot:
            plt.imshow(curr_corr_matrix, vmin=vmin, vmax=vmax)
            plt.colorbar()
            # Save
            if save:
                plt.savefig(f'press_no_{n + 1}_{mouse}_{length}.png', dpi=500)
    return final_matrices
            
        
def in_vs_out_seq_corr(all_matrices, mouse_dict, master_df, mouse, length=500,vmin_in=0, vmax_in=1,
                       vmin_out=0,vmax_out=1, save=False, plot=False):
    mouse_df = master_df[master_df['Mouse'] == mouse]
    dates = sorted(list(mouse_dict.keys()))
    date_indices = [date_ind for date_ind in range(len(dates))]
    in_seq_corr_matrix = np.zeros((len(dates), len(dates)))
    out_seq_corr_matrix = np.zeros((len(dates), len(dates)))
    in_seq_matrices = {i:[] for i in range(len(dates))}
    out_seq_matrices = {i:[] for i in range(len(dates))}
    
    for i, date in enumerate(dates):
        press_matrix = all_matrices[date]
        LPs = np.array(mouse_df[mouse_df['Date'] == date]['Lever'].values[0][1:])
        rewards = np.array(mouse_df[mouse_df['Date'] == date]['Reward'].values[0])
        in_seq_ind = []
        for rwd in rewards:
            in_seq_ind.append([ind for ind in np.where(LPs <= rwd)[0][-5:]])
        set_in_seq_ind = set([press for seq in in_seq_ind for press in seq])
        all_ind = set(range(len(LPs)))
        out_seq_ind = all_ind - set_in_seq_ind
        in_seq_matrices[i] = np.zeros((len(in_seq_ind) * 5, length))
        out_seq_matrices[i] = np.zeros((len(out_seq_ind), length))
        
        for j, ind in enumerate(list(set_in_seq_ind)):
            in_seq_matrices[i][j] = press_matrix[ind]
            
        for k, ind in enumerate(list(out_seq_ind)):
            out_seq_matrices[i][k] = press_matrix[ind]
            
    
    # In sequence presses
    plt.figure()
    for pair in product(date_indices, repeat=2):
        i, j = pair[0], pair[1]
        print(dates[i], dates[j])
        day_1, day_2 = in_seq_matrices[i], in_seq_matrices[j]
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
        day_1, day_2 = out_seq_matrices[i], out_seq_matrices[j]
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


    
if __name__ == '__main__':
    
    ### FR5_CATEG_GROUP_1 ###
    mice = [i for i in range(4392, 4414)]
    # group_1 = range(4392, 4396)
    # group_2 = range(4401, 4406)
    # group_3 = range(4407, 4410)
    # group_4 = range(4410, 4414)
    
    ### FR5 ###
    # FR5_mice_1 = [4218, 4221, 4222, 4224, 4225]
    # FR5_mice_2 = [4226, 4229, 4230, 4232, 4233]
    # FR5_mice_3 = [4237, 4240, 4242, 4243, 4244]
    # mice = FR5_mice_3
    # mice = [4222, 4224, 4225]
    
    file_dir = '/Users/emma-fuze-grace/Lab/medpc_data/medpc_FR5CATEG_old'
    master_df = create_medpc_master(mice, file_dir)
    mouse, date = 4410, '20220222'
    process = True

    # Load magnet session
    mouse_df = master_df[master_df['Mouse'] == mouse]
    date_df = mouse_df[mouse_df['Date']==date]
    
    magnet_file = (
        '/Users/emma-fuze-grace/Lab/hall_sensor_data/hall_sensor_FR5CATEG_old'+ 
        f'/HallSensor_{date}_{mouse}.csv'
        )
    
    import json

    with open(f'mouse_{mouse}_parameters.json') as file:
        parameters = json.load(file)
    
    session = Old_Magnet(magnet_file, date_df, mouse, date, process,
                          on_off_threshold=parameters['on_off_thresholds'][date],
                          independent=parameters['independent_thresholds'][date],
                          weird_last_press=False)
    
    session.optimize_thresholds(percent=parameters['percents'][date], plot=False)
    session.detect_press_ind(plot=False, 
                             search_offset=parameters['search_offset'][date], 
                             hill_start=parameters['hill_start'][date],
                              num_hill_checks=parameters['num_hill_checks'][date],
                              hill_end=parameters['hill_end'][date])
    final_LP_vectors = load_final_LP_vectors(session.magnet, 
                                              session.final_press_ind) 
    session.plot(['medpc', 'presses', 'press_boundaries'])
    # import pickle
    # with open('magnet_presses_4392.pkl', 'rb') as handle:
    #     mouse_dict = pickle.load(handle)
    #     # lengths = [300, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]
    #     length = 2000
    #     all_matrices, final_corr_matrix = day_by_day(mouse_dict, length=length, 
    #                                                   vmin=0, vmax=1, plot=True,traces=False,
    #                                                   save=False)
    #     final_matrices = by_press_no(all_matrices, mouse_dict, master_df, mouse, 
    #                                               length=length, vmin=0, vmax=1,
    #                                               save=False, plot=True)
    #     in_seq_matrices, out_seq_matrices = in_vs_out_seq_corr(all_matrices, mouse_dict,
    #                                                             master_df, mouse,
    #                                                             length=length,
    #                                                             vmin_in=0, vmax_in=1,
    #                                                             vmin_out=0, vmax_out=1,
    #                                                             save=False, plot=True)
    
