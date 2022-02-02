#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:16:35 2022

@author: emma-fuze-grace
"""

import matplotlib.pyplot as plt
import matplotlib
from create_medpc_master import create_medpc_master
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice=[4234]
dates = ['20220130']
master_df = create_medpc_master(mice,dates)

fig,ax=plt.subplots(1,1)

for mouse in mice:
    curr_mouse_df = (master_df[master_df['Mouse']==mouse]).reset_index()
    plt.plot(range(len(curr_mouse_df['Variance'][0])), curr_mouse_df['Variance'][0])
    plt.yscale('log')
    
    
plt.yscale('log')
plt.xlabel('Trial Num', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Individual Trial Variances', size=16)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()