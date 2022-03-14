#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:35:20 2022

@author: emma-fuze-grace
"""

"""FR5 Try #2 From February."""

import medpc
import mouse_dict
from templates import summaries
import numpy as np
from matplotlib import pyplot as plt


tar_var_mice = mouse_dict.create_mice('mice.txt')
file_dir = '/Users/emma-fuze-grace/Lab/2022-02_TarVar_Categ_01/2022-02_TarVar_Categ_01_data'
master_df = medpc.create_medpc_master([i for i in range(4386, 4414)], file_dir)
event = 'Variance'
by_mouse = summaries(master_df, tar_var_mice, 'TarVar3', 'CATEG', event, 'mouse')
by_day = summaries(master_df, tar_var_mice, 'TarVar3', 'CATEG', event, 'day')
for mouse in by_mouse:
    plt.plot(mouse, color='cornflowerblue')
plt.plot([np.mean(day) for day in by_day], color='tomato')
plt.yscale('log')
