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


mice=[4219,4222,4224,4225,4226,
      4229,4230,4231,4234,4239,4240,4241] 
filedir='G:/Behavior study Dec2021/All medpc together'
master_df = create_medpc_master(mice, filedir)

#plot progression of Variance on each day, one plot per mouse
for mouse in mice:
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    plt.figure()
    for day in [mouse_df.index[0], mouse_df.index[-1]]:
        day_df=mouse_df.loc[day,:]
        plt.plot(range(len(day_df['Variance'])), day_df['Variance'])
        plt.yscale('log')
        
#plot mean variance across days
plt.figure()
All_Variance=np.empty((len(mice), 12))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    Variance=mouse_df['Variance'].values
    Mean_variance_across_days=[np.median(x) for x in Variance]
    while len(Mean_variance_across_days)<12:
        Mean_variance_across_days.append(float('nan'))
    All_Variance[i,:]=Mean_variance_across_days
    plt.plot(Mean_variance_across_days, color='b', alpha=0.5)
plt.plot(np.nanmean(All_Variance, axis=0), linewidth=2, color='k')
plt.yscale('log')  
   
        
plt.yscale('log')
plt.xlabel('Trial Num', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Individual Trial Variances', size=16)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

###############################################################################
# plot variance starting with first FR5/Va5 and into CATEG
###############################################################################
plt.figure()
All_Variance=np.empty((len(mice), 26))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    protocol_colors=['k' if 'CATEG' not in x else 'r' for x in mouse_df['Protocol'].values]
    Mean_variance_across_days=[np.median(x) for x in Variance]
    while len(Mean_variance_across_days)<26:
        Mean_variance_across_days.append(float('nan'))
        protocol_colors.append('b')
    All_Variance[i,:]=Mean_variance_across_days
    plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c=protocol_colors, alpha=0.5)
plt.yscale('log')  
 
plt.plot(np.nanmean(All_Variance, axis=0), linewidth=2, color='k')
plt.yscale('log')  

###############################################################################
# mousewise heatmaps variance starting with first FR5/Va5 and into CATEG
###############################################################################
for i,mouse in enumerate(mice):
    plt.figure(figsize=(3,6))
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    
    max_session_len=max([len(x) for x in Variance])
    matrix=np.zeros((max_session_len,len(mouse_df)))
    for j,session in enumerate(Variance):
        if len(session)==0:
            continue
        while len(session)<max_session_len:
            session=np.insert(session,-1,0)
        matrix[:,j]=session
    plt.imshow(matrix, cmap='rainbow',norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    
###############################################################################
# unified mean heatmaps variance starting with first FR5/Va5 and into CATEG
# scaled time version
###############################################################################
master_matrix=np.zeros((100,26))
master_N=np.zeros((100,26))
plt.figure(figsize=(3,6))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    
    Number_of_days=len(Variance) #store which columns were contributed to
    master_N[:,:Number_of_days]+=1
    matrix=np.zeros((100,26))
    for j,session in enumerate(Variance):
        if len(session)==0:
            continue
        for index in range(100): #extrapolate values
            matrix[index,j]=session[int(index*(len(session)/100))]
    master_matrix+=matrix
    
plt.imshow(np.divide(master_matrix,master_N), cmap='rainbow',norm=matplotlib.colors.LogNorm(), vmin=0.1, vmax=10000)
plt.colorbar()  
  
###############################################################################
# unified mean heatmaps variance starting with first FR5/Va5 and into CATEG
# NONSCALED time
###############################################################################
master_matrix=np.zeros((110,26))
master_N=np.zeros((110,26))
plt.figure(figsize=(3,6))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    
    Number_of_days=len(Variance) #store which columns were contributed to
    
    max_session_len=max([len(x) for x in Variance])
    matrix=np.zeros((110,26))
    for j,session in enumerate(Variance):
        if len(session)==0:
            continue
        master_N[:len(session),j]+=1

        matrix[:len(session),j]=session
    master_matrix+=matrix
    
plt.imshow(np.divide(master_matrix,master_N), cmap='rainbow',norm=matplotlib.colors.LogNorm(), vmin=0.1, vmax=10000)
plt.colorbar()  

###############################################################################
# unified MEDIAN heatmaps variance starting with first FR5/Va5 and into CATEG
# NONSCALED time
###############################################################################
master_matrix=np.empty((110,26,len(mice)))
plt.figure(figsize=(3,6))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    
    Number_of_days=len(Variance) #store which columns were contributed to
    
    max_session_len=max([len(x) for x in Variance])
    matrix=np.empty((110,26)) #trials, days, mice
    for j,session in enumerate(Variance):
        if len(session)==0:
            continue
        for k,v in enumerate(session):
            matrix[k,j]=v
    master_matrix[:,:,i]=matrix
matrix_of_medians=np.median(master_matrix,axis=2)   
plt.imshow(matrix_of_medians, cmap='rainbow',norm=matplotlib.colors.LogNorm())
plt.colorbar()   

###############################################################################
# plot IPI (all) starting with first FR5/Va5 and into CATEG
###############################################################################
plt.figure()
All_Variance=np.empty((len(mice), 26))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['IPI'].values
    protocol_colors=['k' if 'CATEG' not in x else 'r' for x in mouse_df['Protocol'].values]
    Mean_variance_across_days=[np.mean(x) for x in Variance]
    while len(Mean_variance_across_days)<26:
        Mean_variance_across_days.append(float('nan'))
        protocol_colors.append('b')
    All_Variance[i,:]=Mean_variance_across_days
    plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c=protocol_colors, alpha=0.5)
plt.yscale('log')  
 
plt.plot(np.nanmean(All_Variance, axis=0), linewidth=2, color='k')
plt.yscale('log')  