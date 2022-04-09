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
        
        self.smooth()
        # Call other class functions to process data
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
                f'Keep: {self.keep}\n'
                f'Press Boundaries: {self.top, self.bottom}'
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
        if 'scale_LPs' in plot_what:
            plt.plot(self.scaled_LPs, 
                [self.top + 5 for _ in range(len(self.scaled_LPs))],
                linestyle='--', marker='*', markersize=10, color='m')
            plt.vlines(self.scaled_LPs, self.bottom, self.top, alpha=0.75,
                       color ='k')
        if 'baselines' in plot_what:
            t, LPs = self.times, self.scaled_LPs
            for ind, press, baseline in zip(
                    range(1, len(self.baselines[1:])),
                    LPs[1:], self.baselines[1:]):
                end = np.where(t == t[t >= press][0])[0][0]
                end_last = np.where(t == t[t >= LPs[ind - 1]][0])[0][0]
                start = end - self.baseline_length
                if abs(press - LPs[ind - 1]) > self.independent_threshold:
                    plt.hlines(baseline, t[start], t[end],
                               linewidth=2, color='r', label='baseline')
                else:
                    plt.hlines(baseline, t[end_last], t[end],
                                linewidth=2, linestyle='--', color='r')
            
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
        top, bottom = sorted_hist[top_ind][1], sorted_hist[bottom_ind][1]
        
        # Save data
        self.sorted_hist = sorted_hist
        self.top = top
        self.bottom = bottom
        
    
    def detect_first_last_press(self):
        """Detect first and last press (down) indices."""
        self.first_last_target = self.top - .5 * (self.top - self.bottom)
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
        
    def create_baselines(self, length=500, independent=5000):
        """Set up baselines for presses that are more than 5000 ms apart.
        For closer presses, use previous baseline
        """
        m, t, LPs = self.magnet, self.times, self.scaled_LPs
        baselines = np.zeros(len(self.scaled_LPs))
        baselines[0] = np.median(m[
            self.first_down_idx-length:self.first_down_idx])
        
        for ind, press, baseline in zip(
                range(1, len(baselines[1:])),
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
        
    def optimize_thresholds(self, percent=0.6):
        """Define threshold for each baseline by starting at a fixed
        distance and moving until atleast 1 crossing is found.
        """
        m, t, LPs, baselines = (self.magnet, self.times, self.scaled_LPs, 
                                self.baselines)
        thresholds = np.zeros(len(baselines))
        for ind, press, baseline in zip(range(1, len(LPs[1:])), 
                                        LPs[1:], baselines[1:]):
            search_times = t[(t > press) & (t < LPs[ind + 1])]
            search_start_ind = np.where(t == search_times[0])[0][0]
            search_end_ind = np.where(t == search_times[-1])[0][0]
            search_magnet = m[search_start_ind:search_end_ind]
            curr_baseline = baselines[ind]
            curr_offset = percent * (self.top-self.bottom)
            move = 0.05 * (self.top-self.bottom)
            sign, original_move = 1, move
            while (curr_target := curr_baseline - curr_offset) < curr_baseline:
                plt.hlines(curr_target, t[search_start_ind], t[search_end_ind], color='k',
                            label='target', linestyle ='--')
                bool_mask = search_magnet < curr_target 
                crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
                down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
                up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
                if np.where(down_xings)[0].size > 0 or np.where(up_xings)[0].size > 0:
                    break
                sign *= -1
                move += original_move
                curr_offset += (sign * move)
                # plt.plot(t[np.where(down_xings)[0][0]], m[np.where(down_xings)[0][0]],
                #          color='k', markersize=3)
                # plt.plot(t[np.where(up_xings)[0][0]], m[np.where(up_xings)[0][0]],
                #          color='k', markersize=3)
            thresholds[ind] = curr_target
        self.move = move
        self.thresholds = thresholds
            
    
    def detect_press_ind(self, smooth=15):
        """"Main Search for press indices."""
        m, t, LPs, baselines = (self.magnet, self.times, self.scaled_LPs, 
                                self.baselines)
        for ind, press, threshold, curr_baseline in zip(range(1, len(LPs[1:])), 
                                        LPs[1:], self.thresholds[1:],
                                        baselines[1:]):
            search_times = t[(t > press) & (t < LPs[ind + 1])]
            search_start_ind = np.where(t == search_times[0])[0][0]
            search_end_ind = np.where(t == search_times[-1])[0][0]
            search_magnet = m[search_start_ind:search_end_ind]
            
            bool_mask = search_magnet < threshold
            crossings = np.invert( bool_mask[:-1] == bool_mask[1:] )
            down_xings = np.invert(bool_mask[:-1]) & crossings # crossing and first is greater than threshold = going down
            up_xings = bool_mask[:-1] & crossings # crossing and first is less than threshold = going up
            final_press_ind = {ind:(0,0) for ind in range(len(LPs))}
            num_down = np.where(down_xings)[0].size
            num_up = np.where(up_xings)[0].size
            num_detected = num_down + num_up
            prev_up = final_press_ind[ind-1][1]
            search_offset = 25
            if num_detected == 1:
                if num_down:
                    down_crossing = search_start_ind + np.where(down_xings)[0][0]
                    final_down_idx = self.down_search(down_crossing,
                                             baselines[ind], smooth)
                    plt.plot(t[down_crossing], m[down_crossing],
                             marker='o', color='k')
                    final_up_idx = self.up_search(search_end_ind + search_offset,
                                             baselines[ind], smooth)
                if num_up:
                    up_crossing = search_start_ind + np.where(up_xings)[0][0]
                    plt.plot(t[up_crossing], m[up_crossing],
                             marker='o', color='k')
                    if search_start_ind < prev_up:
                        final_down_idx = prev_up
                        final_up_idx = self.up_search(up_crossing,
                                                 baselines[ind], smooth)
                    else: 
                        final_down_idx = self.down_search(search_start_ind - search_offset, 
                                                     baselines[ind],
                                                     smooth)
                        final_up_idx = self.up_search(up_crossing,
                                                 baselines[ind], smooth)
            elif num_detected == 2:
                down_crossing = search_start_ind + np.where(down_xings)[0][0]
                up_crossing = search_start_ind + np.where(up_xings)[0][0]
                plt.plot(t[down_crossing], m[down_crossing],
                         marker='o', color='k')
                plt.plot(t[up_crossing], m[up_crossing],
                         marker='o', color='k')
                if down_crossing < up_crossing:
                    final_down_idx = self.down_search(down_crossing, 
                                                 baselines[ind], smooth)
                    final_up_idx = self.up_search(up_crossing, 
                                             baselines[ind], smooth)
                else: 
                    if up_crossing < prev_up:
                        final_down_idx = prev_up
                        final_up_idx = self.up_search(search_end_ind + search_offset, 
                                                 baselines[ind], smooth)
                    else:
                        final_down_idx = self.down_search(search_start_ind - search_offset, 
                                                     baselines[ind], smooth)
                        final_up_idx = self.up_search(up_crossing, 
                                                 baselines[ind], smooth)
            elif num_detected > 2:
                down_crossing = search_start_ind + np.where(down_xings)[0]
                up_crossing = search_start_ind + np.where(up_xings)[0]
                plt.plot(t[down_crossing], m[down_crossing],
                         marker='o', color='k', linestyle='None')
                plt.plot(t[up_crossing], m[up_crossing],
                         marker='o', color='k', linestyle='None')
                if down_crossing[0] < prev_up:
                    final_down_idx = prev_up
                    final_up_idx = self.up_search(up_crossing[-1], 
                                             baselines[ind], smooth)
                    plt.plot(t[up_crossing[-1]], m[up_crossing[-1]],
                             marker='*', color='g')
                else:
                    if num_up % 2 == 0 and num_down % 2 == 0 and up_crossing[0] < down_crossing[0]:
                        # if start search < prev_up, down = prev_up
                        final_down_idx = self.down_search(search_start_ind - 50, 
                                                     baselines[ind], smooth)
                        final_up_idx = self.up_search(up_crossing[0],
                                                 baselines[ind], smooth)
                        plt.plot(t[search_start_ind - 50], m[search_start_ind - 50],
                                 marker='*', color='m')
                        plt.plot(t[up_crossing[1]], m[up_crossing[1]],
                                 marker='*', color='g')
                    else:
                        final_down_idx = self.down_search(down_crossing[0],
                                                     baselines[ind], smooth)
                        final_up_idx = self.up_search(up_crossing[-1],
                                                 baselines[ind], smooth)
                        plt.plot(t[down_crossing[0]], m[down_crossing[0]],
                                 marker='*', color='m')
                        plt.plot(t[up_crossing[-1]], m[up_crossing[-1]],
                                 marker='*', color='g')
            else:
                print(f'Error in crossings: {ind}, {num_detected}')
            
            ### Check for indices being stuck on "hills" during search ###
            # First need to check if far away from previous press and not too close to baseline
            hill_end = 100
            hill_start = 50
            
            # hill check for down indices
            for _ in range(2):
                hill_check_down = m[
                    (final_down_idx - hill_end):(final_down_idx - hill_start)
                    ]
                if abs(final_down_idx - prev_up) > 100 and abs(
                        m[final_down_idx] - curr_baseline) > (1 * self.move): 
                    if np.mean(hill_check_down) > m[final_down_idx] and m[final_down_idx] < curr_baseline:
                        plt.plot(t[(final_down_idx - hill_end):(final_down_idx - hill_start)],
                                 hill_check_down,
                                 color='r')
                        plt.plot(t[final_down_idx], m[final_down_idx],
                                     marker='o', color='k', markersize=4)
                        final_down_idx = self.down_search(final_down_idx, 
                                                     baselines[ind], smooth)
            # hill check for up indices
            for _ in range(2):
                hill_check_up = m[
                    (final_up_idx + hill_start):(final_up_idx + hill_end)
                    ]
                if abs(press - LPs[ind + 1]) > 200:
                    if abs(m[final_up_idx] - curr_baseline) > (1 * self.move):
                        if np.mean(hill_check_up) > m[final_up_idx] and m[final_up_idx] < curr_baseline:
                            plt.plot(t[(final_up_idx + hill_start):(final_up_idx + hill_end)],
                                     hill_check_up,
                                     color='r')
                            plt.plot(t[final_up_idx], m[final_up_idx],
                                         marker='o', color='k', markersize=4)
                            final_up_idx = self.up_search(final_up_idx,
                                                         baselines[ind], smooth)
                    
            # Plot down and up idx
            if ind % 2:
                color = 'm'
            else:
                color = 'g'
            plt.plot(t[final_down_idx], m[final_down_idx],
                         marker='o', color='b', markersize=4)
            plt.plot(t[final_up_idx], m[final_up_idx],
                     marker='o', color='r', markersize=6)
            plt.plot(t[final_down_idx:final_up_idx], 
                     m[final_down_idx:final_up_idx], color=color, 
                     linewidth=2, alpha=0.5)
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
        
if __name__ == '__main__':
    
    ### FR5_CATEG_GROUP_1 ###
    mice = [i for i in range(4392, 4414)]
    group_1 = range(4392, 4396)
    group_2 = range(4401, 4406)
    group_3 = range(4407, 4410)
    group_4 = range(4410, 4414)
    
    ### FR5 ###
    # FR5_mice_1 = [4218, 4221, 4222, 4224, 4225]
    # FR5_mice_2 = [4226, 4229, 4230, 4232, 4233]
    # FR5_mice_3 = [4237, 4240, 4242, 4243, 4244]
    # mice = FR5_mice_3
    # mice = [4222, 4224, 4225]
    
    file_dir = '/Users/emma-fuze-grace/Lab/medpc_FR5CATEG_old'
    master_df = create_medpc_master(mice, file_dir)
    date = '20220209'
    process = True

    # Load magnet session
    for mouse in [4411]:
        mouse_df = master_df[master_df['Mouse'] == mouse]
        date_df = mouse_df[mouse_df['Date']==date]
        
        magnet_file = (
            '/Users/emma-fuze-grace/Lab/hall_sensor_FR5CATEG_OLD' + 
            f'/HallSensor_{date}_{mouse}.csv'
            )
        
        ### Process data ###
        session = Magnet(magnet_file, date_df, mouse, date, process)
        if process:
            plot_what = ['press_boundaries', 'first_last_press', 
                          'scale_LPs', 'baselines']
            session.plot(plot_what)
        else:
            session.plot()
    
    session.optimize_thresholds()
    session.detect_press_ind()
    
    
