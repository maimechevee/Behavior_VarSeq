#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:16:35 2022

@author: emma-fuze-grace
"""
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Maxime_Tools')
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Behavior_VarSeq')

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
# Same as above but split in two plots
###############################################################################
fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_Variance=np.empty((len(mice), 15))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    protocol_specific_variance=[v for v,x in zip(Variance,mouse_df['Protocol'].values) if 'CATEG' not in x]
    Mean_variance_across_days=[np.median(x) for x in protocol_specific_variance]
    while len(Mean_variance_across_days)<15:
        Mean_variance_across_days.append(float('nan'))
    All_Variance[i,:]=Mean_variance_across_days
    #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
    plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.3)
plt.yscale('log')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xticks([0,4,9,14],['1','5','10','15'],  size=16)
plt.xlabel('Time on FR5 schedule (days)', size=20)
plt.ylabel('Median within sequence \n inter-press interval', size=20)
plt.title(str(len(mice)) + ' mice')


mean=np.nanmean(All_Variance, axis=0)
std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
plt.plot(mean, linewidth=3, color='cornflowerblue')
plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color='cornflowerblue', linewidth=3)
plt.ylim(0,10000)
plt.yscale('log')  

fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_Variance=np.empty((len(mice), 10))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    protocol_specific_variance=[v for v,x in zip(Variance,mouse_df['Protocol'].values) if 'CATEG'  in x]
    Mean_variance_across_days=[np.median(x) for x in protocol_specific_variance]
    while len(Mean_variance_across_days)<10:
        Mean_variance_across_days.append(float('nan'))
    All_Variance[i,:]=Mean_variance_across_days
    #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
    plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='tomato',alpha=0.3)
plt.yscale('log')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xticks([0,4,9,14],['1','5','10','15'],  size=16)
plt.xlabel('Time on variance-targetted FR5 schedule (days)', size=20)
plt.ylabel('Median within sequence \n inter-press interval', size=20)
plt.title(str(len(mice)) + ' mice')


mean=np.nanmean(All_Variance, axis=0)
std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
plt.plot(mean, linewidth=3, color='tomato')
plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color='tomato', linewidth=3)
plt.ylim(0,10000)
plt.yscale('log')  

###############################################################################
# example plots for single sessions
###############################################################################
mouse=4224
mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
mouse_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
date='20220131'
#for date in np.unique(mouse_df['Date']):
plt.figure()
date_df = mouse_df[mouse_df['Date']==date].reset_index()

Variance=date_df['Variance'].values[0]
plt.plot(Variance)
plt.yscale('log')

Target=[np.median(Variance[i-5:i]) for i in np.arange(5,len(Variance))]
while len(Target)<len(Variance):
    Target.insert(0,float('nan'))
plt.plot(Target)

Rewarded_trials_index= np.where(Variance<Target)[0]
Rewarded_trials_index=[x for x in Rewarded_trials_index]
while Rewarded_trials_index[0]!=0:
    Rewarded_trials_index.insert(0,Rewarded_trials_index[0]-1)
plt.vlines(Rewarded_trials_index, np.zeros_like(Rewarded_trials_index), Variance[Rewarded_trials_index], linestyles='dotted')

#get the LP times for an example reward
trial=np.where([x==80 for x in Rewarded_trials_index])[0][0]
plt.figure()
reward_time=date_df['Reward'].values[0][trial]
LP_times=date_df['Lever'].values[0]
temp=np.where(LP_times<=reward_time)[0]
plt.vlines(LP_times[temp[-5:]], 0,1)
variance_IPI=np.var(np.diff(LP_times[temp[-5:]]))
plt.xlim(2550, 2557)
print(variance_IPI)
print(Variance[Rewarded_trials_index[trial]])

#22 is unrewarded, before 23 which is.
trial=np.where([x==33 for x in Rewarded_trials_index])[0][0]
plt.figure()
reward_time=date_df['Reward'].values[0][trial]
LP_times=date_df['Lever'].values[0]
temp=np.where(LP_times<=reward_time)[0]
plt.vlines(LP_times[temp[-10:-5]], 0,1)
plt.xlim(608,615)


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