#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:24:35 2022

@author: emma-fuze-grace
"""

"""For code that is often reused in medpc graphs."""

import numpy as np
    
def summaries(master_df, mice, experiment, group, event, axis, kind='mean'):
    """Returns list of lists of session means or medians for any mouse that 
    trained on a given day.
    """
    by_mouse = [] # List of lists where each list is one mouse's median or mean 
                  #  values from a given event over each day trained
    for mouse, info in mice.items():
        experiments = info['experiments']
        if experiment in experiments.keys():
            if experiments[experiment] == group:
                mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
                all_session_lists = mouse_df[event].values
                mouse_summary = [np.mean(day) for day in all_session_lists if day != 'N/A']
                by_mouse.append(mouse_summary)
    if axis == 'day':
        most_days_trained = max([len(training) for training in by_mouse])
        by_day = [[] for day in range(most_days_trained)]
        for mouse in by_mouse:
            for ind, day in enumerate(mouse):
                by_day[ind].append(day)
        return by_day
    elif axis == 'mouse':
        return by_mouse
    
    