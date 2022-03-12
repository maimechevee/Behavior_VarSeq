#############################################
# BASAL GANGLIA GRC POSTER
#############################################

import matplotlib
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import sys
import math
from sklearn import metrics
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy.matlib
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Maxime_Tools')
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Behavior_VarSeq')
from create_medpc_master import create_medpc_master

###############################################################################
# Load data
###############################################################################

mice=[4217,4218,4219,4220,
      4221,4222,4223,4224,
      4225,4226,4227,4228,
      4229,4230,4231,4232,
      4233,4234,4235,4236,
      4237,4238,4239,4240,
      4241,4242,4243,4244,
      4386,4387,4388,4389,
      4390,4391,4392,4393,
      4394,4395,4396,4397,
      4398,4399,4400,4401,
      4402,4403,4404,4405,
      4406,4407,4408,4409,
      4410,4411,4412,4413] 
file_dir='G:/Behavior study Dec2021/All medpc together'
master_df = create_medpc_master(mice, file_dir)

#drop mouse/days based on google doc notes
discard_list=[
[4240, '20220120'], #wrong mouse run
[4388, '20220208'], #wrong protocol
[4388, '20220209'], #wrong protocol
[4393, '20220209'], #wrong protocol
[4394, '20220209'], #wrong protocol
[4398, '20220208'], #wrong protocol
[4398, '20220209'], #wrong protocol
[4401, '20220209'], #wrong protocol
[4403, '20220209'], #wrong protocol
[4405, '20220208'], #wrong protocol
[4405, '20220209'], #wrong protocol
[4394, '20220211'], #wrong protocol
[4225, '20211204'], #protocol was not capturing Vaiance yet
[4234, '20211204'], #wrong protocol
]
master_df = discard_day(master_df, discard_list)

#drop mice doing the noLight, noForcedReward, Neither
discard_list=[
    4386,4387,4388,4389,
    4390,4391,4396,4397,
    4398,4399,4400, 4406]
master_df=discard_mice(master_df, discard_list)

#drop the extra test training done on a subset of animals
mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
dates=['20220118','20220120','20220121','20220124','20220125','20220126','20220127','20220128','20220130','20220131','20220201']
for mouse in mice:
    for date in dates:
        master_df = discard_day(master_df, [[mouse,date]])

#starting dataset:
len(np.unique(master_df['Mouse']))#44
mice=np.unique(master_df['Mouse'])
backup=master_df
###############################################################################
# plots that will decide which mice are included
###############################################################################
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

#Make reward vs days plot and generate All_rewards, All_protocols
All_rewards=np.zeros((len(np.unique(master_df['Mouse'])), 20)) #
All_protocols=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        try:
            if math.isnan(sum(sum(date_df['Reward'].values))):
                mouse_rewards[i] = 0
            else:
                mouse_rewards[i] = len(date_df['Reward'].values[0])
        except:
            #mouse_rewards[i] = len(date_df['Reward'].values[0])
            print('Error in reward column:' + f'{mouse}' + date)
        if mouse==4241 and date=='20220130':
            continue
        try :
            len(date_df['Reward'].values[0])>1 #will fail if Nan
            mouse_rewards[i]=len(date_df['Reward'].values[0]) 
        except:
            mouse_rewards[i]=0
            print(mouse)
            print(date)
            
        mouse_protocols.append(date_df['Protocol'].values)
    while len(mouse_rewards)<20:
        mouse_rewards=np.append(mouse_rewards,float('nan'))
    print(mouse)
    print(mouse_rewards)
    All_rewards[j,:]=mouse_rewards
    All_protocols.append(mouse_protocols)
    
fig,ax=plt.subplots(1,1)
plt.plot(All_rewards.transpose(), color='k', alpha=0.5)

# Number of days on FR1
Time_on_FR1=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR1' in x[0]]
    except:
        print('Mask problem: ' + f'{mouse}')
    Time_on_FR1.append(len(mask))
fig,ax=plt.subplots(1,1)
plt.hist(Time_on_FR1, bins=30)
# based on shape: cut off at 6 days of FR1
colors=[]
discard_list=[]
for each,mouse in zip(Time_on_FR1, np.unique(master_df['Mouse'])):
    if each>5:
        colors.append('r')
        discard_list.append(mouse)
    else:
        colors.append('k')

# Number of days on FR5
Time_on_FR5=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
    except:
        print('Mask problem: ' + f'{mouse}')
    Time_on_FR5.append(len(mask))
fig,ax=plt.subplots(1,1)
plt.hist(Time_on_FR5, bins=30)

# Number of days on FR5 vs FR1 (obvious)
fig,ax=plt.subplots(1,1)
plt.scatter(Time_on_FR1,Time_on_FR5, c=colors)

#DISCARD based on number of days available for analysis
master_df=discard_mice(master_df, discard_list)
keep_index=[i for i,x in enumerate(mice) if x not in discard_list]
#New dataset:
len(np.unique(master_df['Mouse']))#35
mice=np.unique(master_df['Mouse'])
All_rewards=All_rewards[keep_index]
All_protocols=[All_protocols[i] for i in keep_index]


# #Check overall performance
# fig,ax=plt.subplots(1,1)
# total_discarded=0
# discard_list=[]
# for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
#     try:
#         mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
#     except:
#         print('Mask problem: ' + f'{mouse}')
#     cum_data=Cumulative(mouse_data[mask])
#     #print(cum_data)
#     last_data=[x for x in cum_data if not math.isnan(x)][-1]
#     if last_data>315:
#         print(mouse)
#         #print(cum_data)
#         discard_list.append(mouse)
#         total_discarded+=1
#         color='r'
#     else:
#         color='k'
#     plt.plot(cum_data, color=color, linestyle='dotted')
# print('Discard List: ')
# print(discard_list)
# plt.xlabel('Time from first FR5 session (day)', size=16)
# plt.xticks(fontsize=14)
# plt.ylabel('Cumulstive rewards obtained (#)', size=16)
# plt.yticks(fontsize=14)
# plt.legend(['N='+str(len(np.unique(master_df['Mouse']))-total_discarded)+' mice', 'N='+str(total_discarded)+' mice'])
# leg = ax.get_legend()
# leg.legendHandles[0].set_color('k')
# leg.legendHandles[1].set_color('r')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

#DISCARD based on mice not learning
discard_list=[4217]
master_df=discard_mice(master_df, discard_list)
#New dataset:
len(np.unique(master_df['Mouse']))#34
mice=np.unique(master_df['Mouse'])
###############################################################################
# cumulative reward across days, split by group (Still part of checking the dataset)
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,10))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Reward'].values))):
        #     mouse_rewards[i]=0
        # else:
        mouse_rewards[i]=len(date_df['Reward'].values[0])
    
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        cum_mouse_rewards=Cumulative(mouse_rewards)
        plt.plot(cum_mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR5.append(cum_mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        cum_mouse_rewards=Cumulative(mouse_rewards)
        plt.plot(cum_mouse_rewards, linestyle='dotted', color='green')
        All_rewards_CATEG.append(cum_mouse_rewards)
    else:
        print(mouse)
        print(mouse_rewards)
        cum_mouse_rewards=Cumulative(mouse_rewards)
        plt.plot(cum_mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_Var5.append(cum_mouse_rewards)

step=0
meanFR5=[]
semFR5=[]
while All_rewards_FR5:
    step_values=[x[step] for x in All_rewards_FR5]
    step_length=len(step_values)
    meanFR5.append(np.mean(step_values))
    semFR5.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_FR5=[x[1:] for x in All_rewards_FR5]
    All_rewards_FR5=[x for x in All_rewards_FR5 if sum(x)>0]
    
step=0
meanVar=[]
semVar=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar.append(np.mean(step_values))
    semVar.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
    All_rewards_Var5=[x for x in All_rewards_Var5 if sum(x)>0]

step=0
meanCATEG=[]
semCATEG=[]
while All_rewards_CATEG:
    step_values=[x[step] for x in All_rewards_CATEG]
    step_length=len(step_values)
    meanCATEG.append(np.mean(step_values))
    semCATEG.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_CATEG=[x[1:] for x in All_rewards_CATEG]
    All_rewards_CATEG=[x for x in All_rewards_CATEG if sum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='green')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='green', linewidths=2) 

#plt.vlines(3.5,0,600, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Cumulstive rewards obtained (#)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# PART1: Differences in reinforcement/learning?
###############################################################################

###############################################################################
# reward rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=[]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Reward'].values))):
        #     mouse_rewards[i]=0
        # else:
        mouse_rewards.append(len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
    while len(mouse_rewards)<15:
        mouse_rewards.append(float('nan'))
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR5.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='green')
        All_rewards_CATEG.append(mouse_rewards)
    else:
        print(mouse)
        print(mouse_rewards)
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_Var5.append(mouse_rewards)

step=0
meanFR5=[]
semFR5=[]
while All_rewards_FR5:
    step_values=[x[step] for x in All_rewards_FR5]
    step_length=len(step_values)
    meanFR5.append(np.nanmean(step_values))
    semFR5.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_FR5=[x[1:] for x in All_rewards_FR5]
    All_rewards_FR5=[x for x in All_rewards_FR5 if np.nansum(x)>0]
    
step=0
meanVar5=[]
semVar5=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar5.append(np.nanmean(step_values))
    semVar5.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
    All_rewards_Var5=[x for x in All_rewards_Var5 if np.nansum(x)>0]

step=0
meanCATEG=[]
semCATEG=[]
while All_rewards_CATEG:
    step_values=[x[step] for x in All_rewards_CATEG]
    step_length=len(step_values)
    meanCATEG.append(np.nanmean(step_values))
    semCATEG.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_CATEG=[x[1:] for x in All_rewards_CATEG]
    All_rewards_CATEG=[x for x in All_rewards_CATEG if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar5, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar5)), [a-b for a,b in zip(meanVar5,semVar5)], [a+b for a,b in zip(meanVar5,semVar5)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='green')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='green', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# LP rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=[]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Reward'].values))):
        #     mouse_rewards[i]=0
        # else:
        mouse_rewards.append(len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
    while len(mouse_rewards)<15:
        mouse_rewards.append(float('nan'))
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR5.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='green')
        All_rewards_CATEG.append(mouse_rewards)
    else:
        print(mouse)
        print(mouse_rewards)
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_Var5.append(mouse_rewards)

step=0
meanFR5=[]
semFR5=[]
while All_rewards_FR5:
    step_values=[x[step] for x in All_rewards_FR5]
    step_length=len(step_values)
    meanFR5.append(np.nanmean(step_values))
    semFR5.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_FR5=[x[1:] for x in All_rewards_FR5]
    All_rewards_FR5=[x for x in All_rewards_FR5 if np.nansum(x)>0]
    
step=0
meanVar5=[]
semVar5=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar5.append(np.nanmean(step_values))
    semVar5.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
    All_rewards_Var5=[x for x in All_rewards_Var5 if np.nansum(x)>0]

step=0
meanCATEG=[]
semCATEG=[]
while All_rewards_CATEG:
    step_values=[x[step] for x in All_rewards_CATEG]
    step_length=len(step_values)
    meanCATEG.append(np.nanmean(step_values))
    semCATEG.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_CATEG=[x[1:] for x in All_rewards_CATEG]
    All_rewards_CATEG=[x for x in All_rewards_CATEG if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar5, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar5)), [a-b for a,b in zip(meanVar5,semVar5)], [a+b for a,b in zip(meanVar5,semVar5)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='green')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='green', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


###############################################################################
# Reward rate vs LP rate
###############################################################################
fig,ax=plt.subplots(1,1, figsize=(5,5))

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
All_LPs_FR5=[]
All_LPs_Var5=[]
All_LPs_CATEG=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,10))[0]
    mouse_LPs=np.zeros((1,10))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]
        if len(date_df['Lever'].values[0]) ==0:
            mouse_rewards[i]=0
            mouse_LPs[i]=0
        else:
            mouse_rewards[i]=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60) #divide by the last reward timestamps to et the rate
            mouse_LPs[i]=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)
            
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='tomato')
        plt.scatter(mouse_LPs, mouse_rewards, c='tomato', s=4)
        All_rewards_FR5.append(mouse_rewards)
        All_LPs_FR5.append(mouse_LPs)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='green')
        plt.scatter(mouse_LPs, mouse_rewards, c='green', s=4)
        All_rewards_CATEG.append(mouse_rewards)
        All_LPs_CATEG.append(mouse_LPs)
    else:
        # print(mouse)
        # print(mouse_rewards)
        #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='cornflowerblue')
        plt.scatter(mouse_LPs, mouse_rewards, c='cornflowerblue', s=4)
        All_rewards_Var5.append(mouse_rewards)
        All_LPs_Var5.append(mouse_LPs)

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('LP rate (press/min)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[1].set_color('green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# PART2: Differences in sequencinf ability?
###############################################################################

###############################################################################
# Number of extra presses across days (maybe not that useful)
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_Var=[]
All_rewards_CATEG=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,10))[0]
    
    #First days
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Lever'].values))):
        #     mouse_rewards[i]=0
        # else:
        relevant_presses=len(date_df['Reward'].values[0])*5
        total_presses=len(date_df['Lever'].values[0])
        extra_presses=total_presses-relevant_presses
        extra_press_per_seq=extra_presses/len(date_df['Reward'].values[0])
        mouse_rewards[i]=extra_press_per_seq 
        
   
           
    print(mouse)
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='green')
        All_rewards_CATEG.append(mouse_rewards)
    else: #the rest in both types of Var
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_Var.append(mouse_rewards)


step=0
meanFR5=[]
semFR5=[]
while All_rewards_FR:
    step_values=[x[step] for x in All_rewards_FR]
    step_length=len(step_values)
    meanFR5.append(np.mean(step_values))
    semFR5.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_FR=[x[1:] for x in All_rewards_FR]
    All_rewards_FR=[x for x in All_rewards_FR if sum(x)!=0]
    
step=0
meanVar=[]
semVar=[]
while All_rewards_Var:
    step_values=[x[step] for x in All_rewards_Var]
    step_length=len(step_values)
    meanVar.append(np.mean(step_values))
    semVar.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_Var=[x[1:] for x in All_rewards_Var]
    All_rewards_Var=[x for x in All_rewards_Var if sum(x)!=0]
    
step=0
meanCATEG=[]
semCATEG=[]
while All_rewards_CATEG:
    step_values=[x[step] for x in All_rewards_CATEG]
    step_length=len(step_values)
    meanCATEG.append(np.mean(step_values))
    semCATEG.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_CATEG=[x[1:] for x in All_rewards_CATEG]
    All_rewards_CATEG=[x for x in All_rewards_CATEG if sum(x)!=0]    

plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='green')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='green', linewidths=2) 
plt.yscale('log')
#plt.vlines(3.5,-5,2, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('log(extraLP/reward)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('green')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

###############################################################################
# IPI distribution
###############################################################################
All_IPIs_FR_seq=[]
All_IPIs_FR_rest=[]
All_IPIs_Var_seq=[]
All_IPIs_Var_rest=[]
All_IPIs_CATEG_seq=[]
All_IPIs_CATEG_rest=[]

test_mice=[4409,4230,4223]
for j,mouse in enumerate(test_mice):
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,15))
    mouse_heatmap_rest=np.zeros((40,15))
    seq_day_mean=[]
    rest_day_mean=[]
    # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        IPIs=np.array(date_df['IPI'].values[0])
        #find the index of 5 presses preceding each reward
        rewards=date_df['Reward'].values[0]
        LPs=np.array(date_df['Lever'].values[0])
        indices=[]
        for rwd in rewards:
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            IPI_indices=LP_indices[1:]
            indices.append(IPI_indices)
        Seq_IPI_indices=[x for l in indices for x in l]
        Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices][1:]#dont count the first item, it's a zero
        Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
        Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
        
        seq_day_mean.append(np.log10(np.median(Seq_IPIs)))
        rest_day_mean.append(np.log10(np.median(Rest_IPIs)))

        seq_data,edges=np.histogram(np.log10(Seq_IPIs), bins=40, range=(-1,3), density=True)
        rest_data,edges=np.histogram(np.log10(Rest_IPIs), bins=40,  range=(-1,3), density=True)
        
        mouse_heatmap_seq[:,i]=seq_data
        mouse_heatmap_rest[:,i]=rest_data
        
    # #convert to RGB
    # blue_mouse_heatmap_seq=np.zeros((50,10,3))
    # blue_mouse_heatmap_seq[:,:,0]=mouse_heatmap_seq/np.max(mouse_heatmap_seq)
    # blue_mouse_heatmap_seq[:,:,1]=mouse_heatmap_seq/np.max(mouse_heatmap_seq)
    # #blue_mouse_heatmap_seq=1-blue_mouse_heatmap_seq
    # red_mouse_heatmap_rest=np.zeros((50,10,3))
    # red_mouse_heatmap_rest[:,:,1]=mouse_heatmap_rest/np.max(mouse_heatmap_rest)
    # red_mouse_heatmap_rest[:,:,2]=mouse_heatmap_rest/np.max(mouse_heatmap_rest)
    # #red_mouse_heatmap_rest=1-red_mouse_heatmap_rest
    
    # combined_mouse_heatmap=np.add(blue_mouse_heatmap_seq, red_mouse_heatmap_rest)/2
    # combined_mouse_heatmap=1-combined_mouse_heatmap
    # # mask=np.sum(combined_mouse_heatmap, axis=2)==0
    # # combined_mouse_heatmap[mask]=np.array([1,1,1])
    # plt.imshow(combined_mouse_heatmap)
    
    plt.imshow(mouse_heatmap_rest, alpha=0.5, cmap='Blues')
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='Reds')
    plt.plot([x*10+10 for x in seq_day_mean], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.plot([x*10+10 for x in rest_day_mean], color='b') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values])
    plt.xlabel('Sessions (#)')
    fig,[ax1, ax2]=plt.subplots(2,1, figsize=(5,10))
    ax1.bar(edges[1:],mouse_heatmap_seq[:,0], alpha=0.5, color='r', width=0.1)
    ax1.bar(edges[1:],mouse_heatmap_rest[:,0],  alpha=0.5,  color='b', width=0.1)
    ax1.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.bar(edges[1:],mouse_heatmap_seq[:,9],  alpha=0.5,  color='r', width=0.1)
    ax2.bar(edges[1:],mouse_heatmap_rest[:,9], alpha=0.5,  color='b', width=0.1)
    ax2.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.set_xlabel('IPI (s)')
      
All_mouse_heatmap_seq=np.zeros((40,15))
All_mouse_heatmap_rest=np.zeros((40,15))
All_seq_IPIs=[]
All_rest_IPIs=[]
counter=0
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,15))
    mouse_heatmap_rest=np.zeros((40,15))
    seq_day_IPIs=[]
    rest_day_IPIs=[]
    #if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
        for i,date in enumerate(np.unique(mouse_df['Date'])):
            date_df=mouse_df[mouse_df['Date']==date]
            IPIs=np.array(date_df['IPI'].values[0])
            #find the index of 5 presses preceding each reward
            rewards=date_df['Reward'].values[0]
            LPs=np.array(date_df['Lever'].values[0])
            indices=[]
            for rwd in rewards:
                LP_indices=np.where(LPs<=rwd)[0][-5:]
                IPI_indices=LP_indices[1:]
                indices.append(IPI_indices)
            Seq_IPI_indices=[x for l in indices for x in l]
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices][1:]#dont count the first item, it's a zero
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
            
            seq_day_IPIs.append(np.log10(Seq_IPIs))
            rest_day_IPIs.append(np.log10(Rest_IPIs))
    
            seq_data,edges=np.histogram(np.log10(Seq_IPIs), bins=40, range=(-1,3), density=True)
            rest_data,edges=np.histogram(np.log10(Rest_IPIs), bins=40,  range=(-1,3), density=True)
            
            mouse_heatmap_seq[:,i]=seq_data
            mouse_heatmap_rest[:,i]=rest_data
        All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
        All_mouse_heatmap_rest=np.add(All_mouse_heatmap_rest, mouse_heatmap_rest)
        All_seq_IPIs.append(seq_day_IPIs)
        All_rest_IPIs.append(rest_day_IPIs)
        counter+=1
Mean_mouse_heatmap_seq=All_mouse_heatmap_seq/counter
Mean_mouse_heatmap_rest=All_mouse_heatmap_rest/counter 
Median_seq=[]
for day in range(15):
    values=[]
    for mouse in range(len(All_seq_IPIs)):
        if len(All_seq_IPIs[mouse])>day:
            values.append( All_seq_IPIs[mouse][day])
    values=[x for l in values for x in l]
    Median_seq.append(np.median(values))
Median_rest=[]
for day in range(15):
    values=[]
    for mouse in range(len(All_rest_IPIs)):
        if len(All_rest_IPIs[mouse])>day:
            values.append( All_rest_IPIs[mouse][day])
    values=[x for l in values for x in l]
    Median_rest.append(np.median(values))
    
fig,ax=plt.subplots(1,1, figsize=(5,10))
plt.imshow(Mean_mouse_heatmap_rest, alpha=0.5, cmap='Blues')
plt.plot([x*10+10 for x in Median_rest], color='b') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
#plt.colorbar()
plt.imshow(Mean_mouse_heatmap_seq, alpha=0.5, cmap='Reds')
plt.plot([x*10+10 for x in Median_seq], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
#plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
plt.ylabel('IPI (s)')
log_values=[float(x) for x in edges[[0,10,20,30,40]]]
plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values])
plt.xlabel('Sessions (#)')
fig,[ax1, ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.bar(edges[1:],Mean_mouse_heatmap_seq[:,0], alpha=0.5, color='r', width=0.1)
ax1.bar(edges[1:],Mean_mouse_heatmap_rest[:,0],  alpha=0.5,  color='b', width=0.1)
ax1.set_xticks(edges[[0,10,20,30,40]])
ax1.set_xticklabels([str(10**x) for x in log_values])
ax2.bar(edges[1:],Mean_mouse_heatmap_seq[:,9],  alpha=0.5,  color='r', width=0.1)
ax2.bar(edges[1:],Mean_mouse_heatmap_rest[:,9], alpha=0.5,  color='b', width=0.1)
ax2.set_xticks(edges[[0,10,20,30,40]])
ax2.set_xticklabels([str(10**x) for x in log_values])
ax2.set_xlabel('IPI (s)')
    
###############################################################################
# within sequence IPI  vs inter IPIs (plot - accounts for mouse)
###############################################################################
protocol_df=master_df[master_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
mice=np.unique(protocol_df['Mouse'])

protocol_df=master_df[master_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarget_FR5']
mice=np.unique(protocol_df['Mouse'])

protocol_df=master_df[master_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR5']
protocol_df=protocol_df[protocol_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
protocol_df=protocol_df[protocol_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1']
mice=np.unique(protocol_df['Mouse'])

fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_SeqIPIs=np.empty((len(mice), 15))
All_RestIPIs=np.empty((len(mice), 15))
for i,mouse in enumerate(mice):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,15))
    mouse_heatmap_rest=np.zeros((40,15))
    seq_day_IPIs=[]
    rest_day_IPIs=[]
    for j,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        IPIs=np.array(date_df['IPI'].values[0])
        #find the index of 5 presses preceding each reward
        rewards=date_df['Reward'].values[0]
        LPs=np.array(date_df['Lever'].values[0])
        indices=[]
        for rwd in rewards:
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            IPI_indices=LP_indices[1:]
            indices.append(IPI_indices)
        Seq_IPI_indices=[x for l in indices for x in l]
        Rest_IPI_indices=[k for k in range(len(IPIs)) if k not in Seq_IPI_indices][1:]#dont count the first item, it's a zero
        if len(Seq_IPI_indices)>0:
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
        else:
            Seq_IPIs=[float('nan')]
        if len(Rest_IPI_indices)>0:
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
        else:
            print('x')
            Rest_IPIs=[float('nan')]
        seq_day_IPIs.append(Seq_IPIs)
        rest_day_IPIs.append(Rest_IPIs)
    
    Median_SeqIPIs_across_days=[np.median(x) for x in seq_day_IPIs]
    Median_RestIPIs_across_days=[np.median(x) for x in rest_day_IPIs]
    while len(Median_SeqIPIs_across_days)<15:
        Median_SeqIPIs_across_days.append(float('nan'))
    while len(Median_RestIPIs_across_days)<15:
        Median_RestIPIs_across_days.append(float('nan'))
    All_SeqIPIs[i,:]=Median_SeqIPIs_across_days
    All_RestIPIs[i,:]=Median_RestIPIs_across_days
    #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
    plt.plot(np.arange(len(Median_SeqIPIs_across_days)), Median_SeqIPIs_across_days, c='tomato',alpha=0.3)
    plt.plot(np.arange(len(Median_RestIPIs_across_days)), Median_RestIPIs_across_days, c='cornflowerblue',alpha=0.3)
plt.yscale('log')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xticks([0,4,9,14],['1','5','10','15'],  size=16)
plt.xlabel('Time on FR5 schedule (days)', size=20)
plt.ylabel('Median inter-press interval', size=20)
plt.title(str(len(mice)) + ' mice')


mean=np.nanmean(All_SeqIPIs, axis=0)
std=np.nanstd(All_SeqIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_SeqIPIs[:,i]]) for i in range(np.shape(All_SeqIPIs)[1])] )
plt.plot(mean, linewidth=3, color='tomato')
plt.vlines(range(np.shape(All_SeqIPIs)[1]), mean-std, mean+std, color='tomato', linewidth=3)


mean=np.nanmean(All_RestIPIs, axis=0)
std=np.nanstd(All_RestIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_RestIPIs[:,i]]) for i in range(np.shape(All_RestIPIs)[1])] )
plt.plot(mean, linewidth=3, color='cornflowerblue')
plt.vlines(range(np.shape(All_RestIPIs)[1]), mean-std, mean+std, color='cornflowerblue', linewidth=3)

plt.ylim(0,100)
plt.yscale('log') 
###############################################################################
# PART3: How stereotypical are sequences?
###############################################################################

###############################################################################
# within sequence LP rate
###############################################################################
All_LPrates_FR_seq=[]
All_LPrates_Var_seq=[]
All_LPrates_CATEG_seq=[]

test_mice=[4409,4230,4223, 4394]
for j,mouse in enumerate(test_mice):
    fig,ax=plt.subplots(1,1, figsize=(5,15))
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((60,15))
    seq_day_median=[]
    # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        #find the index of 5 presses preceding each reward
        rewards=date_df['Reward'].values[0]
        LPs=np.array(date_df['Lever'].values[0])
        day_rates=[]
        for rwd in rewards:
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            seq_duration=LPs[LP_indices[-1]] - LPs[LP_indices[0]]
            day_rates.append(5/seq_duration)
        
        seq_day_median.append(np.log10(np.median(day_rates)))

        seq_data,edges=np.histogram(np.log10(day_rates), range=(-2,1),bins=60,  density=True)#range=(-1,3),
        
        mouse_heatmap_seq[:,i]=seq_data
        
    # #convert to RGB
    # blue_mouse_heatmap_seq=np.zeros((50,10,3))
    # blue_mouse_heatmap_seq[:,:,0]=mouse_heatmap_seq/np.max(mouse_heatmap_seq)
    # blue_mouse_heatmap_seq[:,:,1]=mouse_heatmap_seq/np.max(mouse_heatmap_seq)
    # #blue_mouse_heatmap_seq=1-blue_mouse_heatmap_seq
    # red_mouse_heatmap_rest=np.zeros((50,10,3))
    # red_mouse_heatmap_rest[:,:,1]=mouse_heatmap_rest/np.max(mouse_heatmap_rest)
    # red_mouse_heatmap_rest[:,:,2]=mouse_heatmap_rest/np.max(mouse_heatmap_rest)
    # #red_mouse_heatmap_rest=1-red_mouse_heatmap_rest
    
    # combined_mouse_heatmap=np.add(blue_mouse_heatmap_seq, red_mouse_heatmap_rest)/2
    # combined_mouse_heatmap=1-combined_mouse_heatmap
    # # mask=np.sum(combined_mouse_heatmap, axis=2)==0
    # # combined_mouse_heatmap[mask]=np.array([1,1,1])
    # plt.imshow(combined_mouse_heatmap)
    
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='jet')
    plt.plot([x*20+40 for x in seq_day_median], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values])
    plt.xlabel('Sessions (#)')
    fig,[ax1, ax2]=plt.subplots(2,1, figsize=(5,10))
    ax1.bar(edges[1:],mouse_heatmap_seq[:,0], alpha=0.5, color='r', width=0.1)
    ax1.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.bar(edges[1:],mouse_heatmap_seq[:,9],  alpha=0.5,  color='r', width=0.1)
    ax2.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.set_xlabel('IPI (s)')

All_mouse_heatmap_seq=np.zeros((60,10))
All_seq_LPrates=[]
counter=0
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((60,10))
    seq_day_LPrates=[]
    #if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
        print(mouse)
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
            date_df=mouse_df[mouse_df['Date']==date]
            #find the index of 5 presses preceding each reward
            rewards=date_df['Reward'].values[0]
            LPs=np.array(date_df['Lever'].values[0])
            day_rates=[]
            for rwd in rewards:
                LP_indices=np.where(LPs<=rwd)[0][-5:]
                seq_duration=LPs[LP_indices[-1]] - LPs[LP_indices[0]]
                day_rates.append(5/seq_duration)

                
            seq_day_LPrates.append(np.log10(day_rates))
            if len(seq_day_LPrates[0])<2:
                seq_data=np.zeros((1,60))
            else:
                seq_data,edges=np.histogram(np.log10(day_rates), bins=60, range=(-2,1), density=True)
            mouse_heatmap_seq[:,i]=seq_data
            
  
        All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
        All_seq_LPrates.append(seq_day_LPrates)
        counter+=1
Mean_mouse_heatmap_seq=All_mouse_heatmap_seq/counter
Median_seq=[]
for day in range(10):
    values=[]
    for mouse in range(len(All_seq_LPrates)):
        if len(All_seq_LPrates[mouse])>day:
            values.append( All_seq_LPrates[mouse][day])
    values=[x for l in values for x in l]
    Median_seq.append(np.median(values))

    
fig,ax=plt.subplots(1,1, figsize=(5,10))
#plt.colorbar()
plt.imshow(Mean_mouse_heatmap_seq, alpha=0.5, cmap='jet')
plt.plot([x*20+40 for x in Median_seq], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
#plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
plt.ylabel('within sequence LP rate (press/s)')
log_values=[float(x) for x in edges[[0,20,40, 60]]]
plt.yticks([0,20,40, 60],[str(10**x) for x in log_values])
plt.xlabel('Sessions (#)')
fig,ax=plt.subplots(1,1, figsize=(5,5))
ax.bar(edges[1:],Mean_mouse_heatmap_seq[:,0], alpha=0.5, color='orange', width=0.05)
ax.bar(edges[1:],Mean_mouse_heatmap_seq[:,9],  alpha=0.5,  color='teal', width=0.05)
ax.set_xticks(edges[[0,20,40, 60]])
ax.set_xticklabels([str(10**x) for x in log_values])
ax.set_xlabel('within sequence LP rate (press/s)')


###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
###############################################################################
High_resp_CATEG=[4392,4394,4395,4403,4404,4405,4408,4409,4410]
test_mice=[4409,4230,4223]

All_mouse_heatmap_seq=np.zeros((70,15))
All_seq_LPrates=[]
counter=0
for j,mouse in enumerate(np.unique(master_df['Mouse'])):#enumerate(High_resp_CATEG)
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((70,15))
    seq_day_LPrates=[]
    #if 1:
    if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    #if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
        print(mouse)
        for i,date in enumerate(np.unique(mouse_df['Date'])):
            date_df=mouse_df[mouse_df['Date']==date]
            IPIs=np.array(date_df['IPI'].values[0])
            #find the index of 5 presses preceding each reward
            rewards=date_df['Reward'].values[0]
            #LPs=np.array(date_df['Variance'].values[0])
            # day_rates=[]
            # for rwd in rewards:
            #     LP_indices=np.where(LPs<=rwd)[0][-5:]
            #     seq_duration=LPs[LP_indices[-1]] - LPs[LP_indices[0]]
            #     day_rates.append(5/seq_duration)
            LPs=np.array(date_df['Lever'].values[0])
            day_variances=[]
            for rwd in rewards:
                LP_indices=np.where(LPs<=rwd)[0][-5:]
                IPI_indices=LP_indices[1:]
                variance=np.var(IPIs[IPI_indices])/np.mean(IPIs[IPI_indices])
                day_variances.append(variance)
           
                
            seq_day_LPrates.append(np.log10(day_variances))
            if len(seq_day_LPrates[0])<2:
                seq_data=np.zeros((1,100))
            else:
                seq_data,edges=np.histogram(np.log10(day_variances), bins=70, range=(-3,4), density=True)
            mouse_heatmap_seq[:,i]=seq_data
            
  
        All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
        All_seq_LPrates.append(seq_day_LPrates)
        counter+=1
        
        fig,ax=plt.subplots(1,1, figsize=(5,10))
        plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='jet')
        plt.plot([x*10+30 for x in [np.median(x) for x in seq_day_LPrates]], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
        plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
        plt.ylabel('IPI (s)')
        log_values=[float(x) for x in edges[[0,10,20,30,40]]]
        plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values])
        plt.xlabel('Sessions (#)')
       
        
        
Mean_mouse_heatmap_seq=All_mouse_heatmap_seq/counter
Median_seq=[]
for day in range(10):
    values=[]
    for mouse in range(len(All_seq_LPrates)):
        if len(All_seq_LPrates[mouse])>day:
            values.append( All_seq_LPrates[mouse][day])
    values=[x for l in values for x in l]
    Median_seq.append(np.median(values))

    
fig,ax=plt.subplots(1,1, figsize=(5,10))
#plt.colorbar()
plt.imshow(Mean_mouse_heatmap_seq, alpha=0.5, cmap='jet')
plt.plot([x*10+30 for x in Median_seq], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
#plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
plt.ylabel('within sequence IPI variance')
log_values=[float(x) for x in edges[[0,20,40, 60, 80, 100]]]
plt.yticks([0,20,40, 60, 80, 100],[str(10**x) for x in log_values])
plt.xlabel('Sessions (#)')
fig,ax=plt.subplots(1,1, figsize=(5,5))
ax.bar(edges[1:],Mean_mouse_heatmap_seq[:,0], alpha=0.5, color='orange', width=0.1)
ax.bar(edges[1:],Mean_mouse_heatmap_seq[:,9],  alpha=0.5,  color='teal', width=0.1)
ax.set_xticks(edges[[0,20,40, 60, 80, 100]])
ax.set_xticklabels([str(10**x) for x in log_values])
ax.set_xlabel('within sequence IPI variance')

###############################################################################
# within sequence IPI variance (plot - accounts for mouse)
###############################################################################
fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_Variance=np.empty((len(mice), 15))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    Variance=mouse_df['Variance'].values
    protocol_specific_variance=[v for v,x in zip(Variance,mouse_df['Protocol'].values) if 'LongWinVarTarget_FR5' in x]
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

#

protocol_df=master_df[master_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
mice=np.unique(protocol_df['Mouse'])

protocol_df=master_df[master_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarget_FR5']
mice=np.unique(protocol_df['Mouse'])

protocol_df=master_df[master_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR5']
protocol_df=protocol_df[protocol_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
protocol_df=protocol_df[protocol_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1']
mice=np.unique(protocol_df['Mouse'])

fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_Variance=np.empty((len(mice), 15))
for i,mouse in enumerate(mice):
    mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_variances=[]
    for j,date in enumerate(np.unique(mouse_df['Date'])[:15]):
        date_df=mouse_df[mouse_df['Date']==date]
        IPIs=np.array(date_df['IPI'].values[0])
        #find the index of 5 presses preceding each reward
        rewards=date_df['Reward'].values[0]
        #day_variances=np.array(date_df['Variance'].values[0])
        day_rates=[]
        for rwd in rewards:
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            seq_duration=LPs[LP_indices[-1]] - LPs[LP_indices[0]]
            day_rates.append(5/seq_duration)
        LPs=np.array(date_df['Lever'].values[0])
        day_variances=[]
        for rwd in rewards:
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            IPI_indices=LP_indices[1:]
            variance=np.var(IPIs[IPI_indices])/np.mean(IPIs[IPI_indices])
            day_variances.append(variance)
        mouse_variances.append(day_variances)
    
    
    
    #Variance=mouse_df['Variance'].values
    protocol_specific_variance=[v for v,x in zip(np.array(mouse_variances),mouse_df['Protocol'].values) if 'FR5' in x] #LongWinVarTarget_FR5
    if len(protocol_specific_variance)==0:
        continue
    else:
        print(mouse_df['Protocol'].values)
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
plt.ylim(0,1000)
plt.yscale('log') 


###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
# ON MICE THAT DID BOTH FR/Var FOLLOWED BY CATEG
###############################################################################
mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
file_dir='G:/Behavior study Dec2021/All medpc together'
master_df2 = create_medpc_master(mice, file_dir)
###############################################################################
# plot variance starting with first FR5/Va5 and into CATEG
###############################################################################

fig,ax=plt.subplots(1,1,figsize=(10,5))
plt.sca(ax)
All_Variance=np.empty((len(mice), 15))
for i,mouse in enumerate(mice):
    mouse_df = master_df2[master_df2['Mouse']==mouse].reset_index()
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
    mouse_df = master_df2[master_df2['Mouse']==mouse].reset_index()
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
# Example rasters of LPs
###############################################################################

test_mice=[4219]#[4219, 4225,4230,4239]
for j,mouse in enumerate(test_mice):
    mouse_protocols=[]
    mouse_df=master_df2[master_df2['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
    counter=0
    for i,date in enumerate(['20211215','20220201']):
        
        plt.figure()
        Range=(0,5)
        plt.title(mouse)
        date_df=mouse_df[mouse_df['Date']==date]
        #find the index of 5 presses preceding each reward
        rewards=date_df['Reward'].values[0]
        LPs=np.array(date_df['Lever'].values[0])
        Presses=np.zeros((len(rewards), 5))
        Trialwise_LPs=[]
        for k,rwd in enumerate(rewards):
            LP_indices=np.where(LPs<=rwd)[0][-5:]
            trial_LPs=LPs[LP_indices]-LPs[LP_indices[0]]
            Trialwise_LPs.append(trial_LPs)
            Presses[k,:]=trial_LPs
            plt.scatter(trial_LPs, np.ones_like(trial_LPs)+counter, c=['k','b','y','r','g'])
            counter+=1
        plt.xlim(Range)
        plt.figure()
        plt.hist(Presses[:,1], bins=30, range=Range, color='b', alpha=0.5)
        plt.hist(Presses[:,2], bins=30, range=Range, color='y', alpha=0.5)
        plt.hist(Presses[:,3], bins=30, range=Range, color='r', alpha=0.5)
        plt.hist(Presses[:,4], bins=30, range=Range, color='g', alpha=0.5)
    
       