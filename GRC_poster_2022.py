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
# Males=[4217,4218,4219,4220,
#       4221,4222,4223,4224,
#       4225,4226,4227,4228,
#       4229,4396,4397,
#       4398,4399,4400,
#       4406,4407,4408,4409,
#       4410,4411,4412,4413]
# Females=[4230,4231,4232,
#         4233,4234,4235,4236,
#         4237,4238,4239,4240,
#         4241,4242,4243,4244,
#         4386,4387,4388,4389,
#         4390,4391,4392,4393,
#         4394,4395,4401,
#         4402,4403,4404,4405]

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
      4410,4411,4412,4413,
      4673,	4674,	4675,	4676,	4677,	4678,	4679,	4680,	4681,
      4688,	4689,	4690,	4691,	4692,	4693,	4694] 
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
[4411, '20220222'], #wrong protocol
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
    
# fig,ax=plt.subplots(1,1)
# plt.plot(All_rewards.transpose(), color='k', alpha=0.5)

# # Number of days on FR1
# Time_on_FR1=[]
# for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
#     try:
#         mask=[i for i,x in enumerate(mouse_protocols) if 'FR1' in x[0]]
#     except:
#         print('Mask problem: ' + f'{mouse}')
#     Time_on_FR1.append(len(mask))
# fig,ax=plt.subplots(1,1)
# plt.hist(Time_on_FR1, bins=30)
# # based on shape: cut off at 6 days of FR1
# colors=[]
# discard_list=[]
# for each,mouse in zip(Time_on_FR1, np.unique(master_df['Mouse'])):
#     if each>5:
#         colors.append('r')
#         discard_list.append(mouse)
#     else:
#         colors.append('k')

# Number of days on FR5
Time_on_FR5=[]
discard_list=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
    except:
        print('Mask problem: ' + f'{mouse}')
    print( len(mask))
    if len(mask)<7:
        discard_list.append(mouse)
    Time_on_FR5.append(len(mask))
fig,ax=plt.subplots(1,1)
plt.hist(Time_on_FR5, bins=30)

# # Number of days on FR5 vs FR1 (obvious)
# fig,ax=plt.subplots(1,1)
# plt.scatter(Time_on_FR1,Time_on_FR5, c=colors)

#DISCARD based on number of days available for analysis (at least 7 days of FR5)
master_df=discard_mice(master_df, discard_list)
keep_index=[i for i,x in enumerate(mice) if x not in discard_list]
#New dataset:
len(np.unique(master_df['Mouse']))#40
mice=np.unique(master_df['Mouse'])
All_rewards=All_rewards[keep_index]
All_protocols=[All_protocols[i] for i in keep_index]


#Check overall performance
fig,ax=plt.subplots(1,1)
total_discarded=0
discard_list=[]
Total_reward_acquired=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
        print(len(mask))
    except:
        print('Mask problem: ' + f'{mouse}')
    cum_data=Cumulative(mouse_data[mask])
    #print(cum_data)
    last_data=[x for x in cum_data if not math.isnan(x)][-1]
    Total_reward_acquired.append(last_data)
    if last_data<220:
        print(mouse) 
        discard_list.append(mouse)
        total_discarded+=1
        color='r'
    else:
        color='k'
    plt.plot(cum_data, color=color, linestyle='dotted')
print('Discard List: ')
print(discard_list)
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Cumulstive rewards obtained (#)', size=16)
plt.yticks(fontsize=14)
plt.legend(['N='+str(len(np.unique(master_df['Mouse']))-total_discarded)+' mice', 'N='+str(total_discarded)+' mice'])
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('r')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.figure()
plt.hist(Total_reward_acquired, bins=20)
#DISCARD based on mice not learning
master_df=discard_mice(master_df, discard_list)
#New dataset:
len(np.unique(master_df['Mouse']))#33: 10FR5, 14CATEG, 9Var(not using), 14 FR5>CATEG
mice=np.unique(master_df['Mouse'])

#remove var5
discard_list=[4217,4219,4220,4223,4227,4228,4231,4234,4235,4236,4238,4239,4241]
master_df=discard_mice(master_df, discard_list)
#final length: 38
 FR5CATEG_mice = [4673,	4674,	4675,	4676,	4677,	4678,	4679,	4680,	4681,
 4688,	4689,	4690,	4691,	4692,	4693,	4694] 
 FR5_mice=[4232,4233,4237,4240,4242,4243,4244,4229,4218,4221,4222,4224,4225,4226,4230]
 CATEG_mice=[4392,4393,4394,4395,4401,4402,4403,4404,4405,4407,4408,4409,4410,4411,4412,4413]
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
All_rewards_FR5CATEG=[]
for group in [FR5_mice, CATEG_mice, FR5CATEG_mice]:
    mice=group
    for j,mouse in enumerate(mice):
        # if mouse in Females:
        #     continue
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_rewards=[]
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
            date_df=mouse_df[mouse_df['Date']==date]
            # if math.isnan(sum(sum(date_df['Reward'].values))):
            #     mouse_rewards[i]=0
            # else:
            mouse_rewards.append(len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
            #mouse_rewards.append(sum([x<1800 for x in date_df['Reward'].values[0]]) /1800*60) #divide by the last reward timestamps to et the rate
    
        while len(mouse_rewards)<10:
            mouse_rewards.append(float('nan'))
        #print(date_df['Protocol'].values[0])
        if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
            if mouse in FR5CATEG_mice:
                plt.plot(mouse_rewards, linestyle='dotted', color='k')
                All_rewards_FR5CATEG.append(mouse_rewards)
            else:
                plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
                All_rewards_FR5.append(mouse_rewards)
        elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
            if mouse in FR5CATEG_mice:
                plt.plot(mouse_rewards, linestyle='dotted', color='k')
                All_rewards_FR5CATEG.append(mouse_rewards)
            else:
                plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
                All_rewards_CATEG.append(mouse_rewards)
        # else:
        #     print(mouse)
        #     print(mouse_rewards)
        #     plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        #     All_rewards_Var5.append(mouse_rewards)

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
    
# step=0
# meanVar5=[]
# semVar5=[]
# while All_rewards_Var5:
#     step_values=[x[step] for x in All_rewards_Var5]
#     step_length=len(step_values)
#     meanVar5.append(np.nanmean(step_values))
#     semVar5.append(np.nanstd(step_values)/np.sqrt(step_length))
#     All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
#     All_rewards_Var5=[x for x in All_rewards_Var5 if np.nansum(x)>0]

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

step=0
meanFR5CATEG=[]
semFR5CATEG=[]
while All_rewards_FR5CATEG:
    step_values=[x[step] for x in All_rewards_FR5CATEG]
    step_length=len(step_values)
    meanFR5CATEG.append(np.nanmean(step_values))
    semFR5CATEG.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_FR5CATEG=[x[1:] for x in All_rewards_FR5CATEG]
    All_rewards_FR5CATEG=[x for x in All_rewards_FR5CATEG if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
# plt.plot(meanVar5, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(meanVar5)), [a-b for a,b in zip(meanVar5,semVar5)], [a+b for a,b in zip(meanVar5,semVar5)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanFR5CATEG, linewidth=2, color='k')
plt.vlines(range(len(meanFR5CATEG)), [a-b for a,b in zip(meanFR5CATEG,semFR5CATEG)], [a+b for a,b in zip(meanFR5CATEG,semFR5CATEG)], colors='k', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5_mice]))+' mice',
                               'CATEG, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in CATEG_mice]))+' mice',
                                'CATEG, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5CATEG_mice]))+' mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# LP rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
All_rewards_FR5CATEG=[]
for group in [FR5_mice, CATEG_mice, FR5CATEG_mice]:
    mice=group
    for j,mouse in enumerate(mice):
        # if mouse in Females:
        #     continue
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_rewards=[]
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
            date_df=mouse_df[mouse_df['Date']==date]
            # if math.isnan(sum(sum(date_df['Reward'].values))):
            #     mouse_rewards[i]=0
            # else:
            mouse_rewards.append(len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
            #mouse_rewards.append(sum([x<1800 for x in date_df['Reward'].values[0]]) /1800*60) #divide by the last reward timestamps to et the rate
    
        while len(mouse_rewards)<10:
            mouse_rewards.append(float('nan'))
        #print(date_df['Protocol'].values[0])
        if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
            if mouse in FR5CATEG_mice:
                plt.plot(mouse_rewards, linestyle='dotted', color='k')
                All_rewards_FR5CATEG.append(mouse_rewards)
            else:
                plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
                All_rewards_FR5.append(mouse_rewards)
        elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
            if mouse in FR5CATEG_mice:
                plt.plot(mouse_rewards, linestyle='dotted', color='k')
                All_rewards_FR5CATEG.append(mouse_rewards)
            else:
                plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
                All_rewards_CATEG.append(mouse_rewards)
        # else:
        #     print(mouse)
        #     print(mouse_rewards)
        #     plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        #     All_rewards_Var5.append(mouse_rewards)

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
    
# step=0
# meanVar5=[]
# semVar5=[]
# while All_rewards_Var5:
#     step_values=[x[step] for x in All_rewards_Var5]
#     step_length=len(step_values)
#     meanVar5.append(np.nanmean(step_values))
#     semVar5.append(np.nanstd(step_values)/np.sqrt(step_length))
#     All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
#     All_rewards_Var5=[x for x in All_rewards_Var5 if np.nansum(x)>0]

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

step=0
meanFR5CATEG=[]
semFR5CATEG=[]
while All_rewards_FR5CATEG:
    step_values=[x[step] for x in All_rewards_FR5CATEG]
    step_length=len(step_values)
    meanFR5CATEG.append(np.nanmean(step_values))
    semFR5CATEG.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_FR5CATEG=[x[1:] for x in All_rewards_FR5CATEG]
    All_rewards_FR5CATEG=[x for x in All_rewards_FR5CATEG if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
# plt.plot(meanVar5, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(meanVar5)), [a-b for a,b in zip(meanVar5,semVar5)], [a+b for a,b in zip(meanVar5,semVar5)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanCATEG, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanFR5CATEG, linewidth=2, color='k')
plt.vlines(range(len(meanFR5CATEG)), [a-b for a,b in zip(meanFR5CATEG,semFR5CATEG)], [a+b for a,b in zip(meanFR5CATEG,semFR5CATEG)], colors='k', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5_mice]))+' mice',
                               'CATEG, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in CATEG_mice]))+' mice',
                                'CATEG, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5CATEG_mice]))+' mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


###############################################################################
# Reward rate vs LP rate
###############################################################################
# fig,ax=plt.subplots(1,1, figsize=(5,5))

# All_rewards_FR5=[]
# All_rewards_Var5=[]
# All_rewards_CATEG=[]
# All_LPs_FR5=[]
# All_LPs_Var5=[]
# All_LPs_CATEG=[]
# for j,mouse in enumerate(np.unique(master_df['Mouse'])):
#     mouse_protocols=[]
#     mouse_df=master_df[master_df['Mouse']==mouse]
#     mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
#     mouse_rewards=np.zeros((1,10))[0]
#     mouse_LPs=np.zeros((1,10))[0]
#     for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
#         date_df=mouse_df[mouse_df['Date']==date]
#         if len(date_df['Lever'].values[0]) ==0:
#             mouse_rewards[i]=0
#             mouse_LPs[i]=0
#         else:
#             mouse_rewards[i]=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60) #divide by the last reward timestamps to et the rate
#             mouse_LPs[i]=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)
            
#     if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
#         #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='tomato')
#         plt.scatter(mouse_LPs, mouse_rewards, c='tomato', s=4)
#         All_rewards_FR5.append(mouse_rewards)
#         All_LPs_FR5.append(mouse_LPs)
#     elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
#         #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='green')
#         plt.scatter(mouse_LPs, mouse_rewards, c='green', s=4)
#         All_rewards_CATEG.append(mouse_rewards)
#         All_LPs_CATEG.append(mouse_LPs)
#     else:
#         # print(mouse)
#         # print(mouse_rewards)
#         #plt.plot(mouse_LPs, mouse_rewards, linestyle='dotted', color='cornflowerblue')
#         plt.scatter(mouse_LPs, mouse_rewards, c='cornflowerblue', s=4)
#         All_rewards_Var5.append(mouse_rewards)
#         All_LPs_Var5.append(mouse_LPs)

# #plt.vlines(3.5,0,6, color='k', linestyle='dashed')
# plt.xlabel('LP rate (press/min)', size=16)
# plt.xticks(fontsize=14)
# plt.ylabel('Reward rate (#/min)', size=16)
# plt.yticks(fontsize=14)
# plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
# leg = ax.get_legend()
# leg.legendHandles[0].set_color('tomato')
# leg.legendHandles[1].set_color('cornflowerblue')
# leg.legendHandles[1].set_color('green')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

###############################################################################
# PART2: Differences in sequencinf ability?
###############################################################################

###############################################################################
# Number of extra presses across days (maybe not that useful)
###############################################################################
# fig,ax=plt.subplots(1,1)

# All_rewards_FR=[]
# All_rewards_Var=[]
# All_rewards_CATEG=[]
# for j,mouse in enumerate(np.unique(master_df['Mouse'])):
#     mouse_protocols=[]
#     mouse_df=master_df[master_df['Mouse']==mouse]
#     mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
#     mouse_rewards=np.zeros((1,10))[0]
    
#     #First days
#     for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
#         date_df=mouse_df[mouse_df['Date']==date]
#         # if math.isnan(sum(sum(date_df['Lever'].values))):
#         #     mouse_rewards[i]=0
#         # else:
#         relevant_presses=len(date_df['Reward'].values[0])*5
#         total_presses=len(date_df['Lever'].values[0])
#         extra_presses=total_presses-relevant_presses
#         extra_press_per_seq=extra_presses/len(date_df['Reward'].values[0])
#         mouse_rewards[i]=extra_press_per_seq 
        
   
           
#     print(mouse)
#     if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
#         plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
#         All_rewards_FR.append(mouse_rewards)
#     elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
#         plt.plot(mouse_rewards, linestyle='dotted', color='green')
#         All_rewards_CATEG.append(mouse_rewards)
#     else: #the rest in both types of Var
#         plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
#         All_rewards_Var.append(mouse_rewards)


# step=0
# meanFR5=[]
# semFR5=[]
# while All_rewards_FR:
#     step_values=[x[step] for x in All_rewards_FR]
#     step_length=len(step_values)
#     meanFR5.append(np.mean(step_values))
#     semFR5.append(np.std(step_values)/np.sqrt(step_length))
#     All_rewards_FR=[x[1:] for x in All_rewards_FR]
#     All_rewards_FR=[x for x in All_rewards_FR if sum(x)!=0]
    
# step=0
# meanVar=[]
# semVar=[]
# while All_rewards_Var:
#     step_values=[x[step] for x in All_rewards_Var]
#     step_length=len(step_values)
#     meanVar.append(np.mean(step_values))
#     semVar.append(np.std(step_values)/np.sqrt(step_length))
#     All_rewards_Var=[x[1:] for x in All_rewards_Var]
#     All_rewards_Var=[x for x in All_rewards_Var if sum(x)!=0]
    
# step=0
# meanCATEG=[]
# semCATEG=[]
# while All_rewards_CATEG:
#     step_values=[x[step] for x in All_rewards_CATEG]
#     step_length=len(step_values)
#     meanCATEG.append(np.mean(step_values))
#     semCATEG.append(np.std(step_values)/np.sqrt(step_length))
#     All_rewards_CATEG=[x[1:] for x in All_rewards_CATEG]
#     All_rewards_CATEG=[x for x in All_rewards_CATEG if sum(x)!=0]    

# plt.plot(meanFR5, linewidth=2, color='tomato')
# plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
# plt.plot(meanVar, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 
# plt.plot(meanCATEG, linewidth=2, color='green')
# plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='green', linewidths=2) 
# plt.yscale('log')
# #plt.vlines(3.5,-5,2, color='k', linestyle='dashed')
# plt.xlabel('Time from first FR5 session (day)', size=16)
# plt.xticks(fontsize=14)
# plt.ylabel('log(extraLP/reward)', size=16)
# plt.yticks(fontsize=14)
# plt.legend(['FR5, N=9 mice', 'Var, N=11 mice', 'CATEG, N=15 mice'], loc='upper left')
# leg = ax.get_legend()
# leg.legendHandles[0].set_color('tomato')
# leg.legendHandles[1].set_color('cornflowerblue')
# leg.legendHandles[2].set_color('green')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.show()

###############################################################################
# IPI distribution
###############################################################################
All_IPIs_FR_seq=[]
All_IPIs_FR_rest=[]
# All_IPIs_Var_seq=[]
All_IPIs_Var_rest=[]
All_IPIs_CATEG_seq=[]
All_IPIs_CATEG_rest=[]
mice=np.unique(master_df['Mouse'])
test_mice=[4409,4230,4223]
test_mice=mice
for j,mouse in enumerate(test_mice):
    #fig,ax=plt.subplots(1,1, figsize=(5,10))
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
        for r,rwd in enumerate(rewards):
            #LP_indices=np.where(LPs<=rwd)[0][-5:]
            LP_indices = np.where( (LPs<=rwd) & (LPs>rewards[r-1]) )[0] #get the indices of the LPs between teh two rewards. must be a multiple of 5
            IPI_indices=LP_indices[1:]
            while len(IPI_indices)>1:
                indices.append(IPI_indices[-4:]) #count the last 4 ipis (5LPS) as within sequence
                IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq
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
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='Reds')
    plt.plot([x*10+10 for x in seq_day_mean], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.plot([x*10+10 for x in rest_day_mean], color='b') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values])
    plt.xlabel('Sessions (#)')
    
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    plt.imshow(mouse_heatmap_rest, alpha=0.5, cmap='Blues')
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
    ax2.bar(edges[1:],mouse_heatmap_seq[:,-1],  alpha=0.5,  color='r', width=0.1)
    ax2.bar(edges[1:],mouse_heatmap_rest[:,-1], alpha=0.5,  color='b', width=0.1)
    ax2.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.set_xlabel('IPI (s)')
      
All_mouse_heatmap_seq=np.zeros((40,15))
All_mouse_heatmap_rest=np.zeros((40,15))
All_seq_IPIs=[]
All_rest_IPIs=[]
counter=0
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    if mouse in Males:
        continue
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,15))
    mouse_heatmap_rest=np.zeros((40,15))
    seq_day_IPIs=[]
    rest_day_IPIs=[]
    if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    #if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
        for i,date in enumerate(np.unique(mouse_df['Date'])):
            date_df=mouse_df[mouse_df['Date']==date]
            IPIs=np.array(date_df['IPI'].values[0])
            #find the index of 5 presses preceding each reward
            rewards=date_df['Reward'].values[0]
            LPs=np.array(date_df['Lever'].values[0])
            indices=[]
            for r,rwd in enumerate(rewards):
                #LP_indices=np.where(LPs<=rwd)[0][-5:]
                LP_indices = np.where( (LPs<=rwd) & (LPs>rewards[r-1]) )[0] #get the indices of the LPs between teh two rewards. must be a multiple of 5
                IPI_indices=LP_indices[1:]
                while len(IPI_indices)>1:
                    indices.append(IPI_indices[-4:]) #count the last 4 ipis (5LPS) as within sequence
                    IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq
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


for group in [FR5_mice, CATEG_mice, FR5CATEG_mice]:
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    fig,ax=plt.subplots(1,1,figsize=(10,5))
    plt.sca(ax)
    All_SeqIPIs=np.empty((len(mice), 10))
    All_RestIPIs=np.empty((len(mice), 10))
    All_InterFailedIPIs=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):

        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_heatmap_seq=np.zeros((40,10))
        mouse_heatmap_rest=np.zeros((40,10))
        seq_day_IPIs=[]
        rest_day_IPIs=[]
        inter_failed_day_IPIs=[]
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
            date_df=mouse_df[mouse_df['Date']==date]
            IPIs=np.array(date_df['IPI'].values[0])
            #find the index of 5 presses preceding each reward
            rewards=date_df['Reward'].values[0]
            LPs=np.array(date_df['Lever'].values[0])
            indices=[]
            inter_failed=[]
            for r,rwd in enumerate(rewards):
                #LP_indices=np.where(LPs<=rwd)[0][-5:]
                LP_indices = np.where( (LPs<=rwd) & (LPs>rewards[r-1]) )[0] #get the indices of the LPs between teh two rewards. must be a multiple of 5
                IPI_indices=LP_indices[1:]
                while len(IPI_indices)>1:
                    indices.append(IPI_indices[-4:]) #count the last 4 ipis (5LPS) as within sequence
                    if len( IPI_indices)!=4:
                        try: 
                            inter_failed.append(IPI_indices[-5])
                        except:
                            IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq
                            continue
                    IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq
            Seq_IPI_indices=[x for l in indices for x in l]
            Rest_IPI_indices=[k for k in range(len(IPIs)) if k not in Seq_IPI_indices][1:]#dont count the first item, it's a zero
            Inter_failed_IPI_indices=inter_failed 
            if len(Seq_IPI_indices)>0:
                Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            else:
                Seq_IPIs=[float('nan')]
            if len(Rest_IPI_indices)>0:
                Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
            else:
                print('x')
                Rest_IPIs=[float('nan')]
            if len(Inter_failed_IPI_indices)>0:
                Inter_failed_IPIs=IPIs[np.array(Inter_failed_IPI_indices)]
            else:
                print('x')
                Inter_failed_IPIs=[float('nan')]
            seq_day_IPIs.append(Seq_IPIs)
            rest_day_IPIs.append(Rest_IPIs)
            inter_failed_day_IPIs.append(Inter_failed_IPIs)
        
        Median_SeqIPIs_across_days=[np.nanmean(x) for x in seq_day_IPIs]
        Median_RestIPIs_across_days=[np.nanmean(x) for x in rest_day_IPIs]
        Median_InterfailedIPIs_across_days=[np.nanmean(x) for x in inter_failed_day_IPIs]
        while len(Median_SeqIPIs_across_days)<10:
            Median_SeqIPIs_across_days.append(float('nan'))
        while len(Median_RestIPIs_across_days)<10:
            Median_RestIPIs_across_days.append(float('nan'))
        while len(Median_InterfailedIPIs_across_days)<10:
            Median_InterfailedIPIs_across_days.append(float('nan'))
        All_SeqIPIs[j,:]=Median_SeqIPIs_across_days
        All_RestIPIs[j,:]=Median_RestIPIs_across_days
        All_InterFailedIPIs[j,:]=Median_InterfailedIPIs_across_days
        #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
        plt.plot(np.arange(len(Median_SeqIPIs_across_days)), Median_SeqIPIs_across_days, c='tomato',alpha=0.3)
        plt.plot(np.arange(len(Median_RestIPIs_across_days)), Median_RestIPIs_across_days, c='cornflowerblue',alpha=0.3)
        if mouse not in FR5_mice:
            plt.plot(np.arange(len(Median_InterfailedIPIs_across_days)), Median_InterfailedIPIs_across_days, c='b',alpha=0.3)
    plt.yscale('log')  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.xticks([0,4,9],['1','5','10'],  size=16)
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
    
    if mouse not in FR5_mice:
        mean=np.nanmean(All_InterFailedIPIs, axis=0)
        std=np.nanstd(All_InterFailedIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_InterFailedIPIs[:,i]]) for i in range(np.shape(All_InterFailedIPIs)[1])] )
        plt.plot(mean, linewidth=3, color='b')
        plt.vlines(range(np.shape(All_InterFailedIPIs)[1]), mean-std, mean+std, color='b', linewidth=3)
    
    plt.ylim(0,100)
    plt.yscale('log') 
###############################################################################
# PART3: How stereotypical are sequences?
###############################################################################


###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
###############################################################################
High_resp_CATEG=[4392,4394,4395,4403,4404,4405,4408,4409,4410]
test_mice=[4409,4230,4223]

All_mouse_heatmap_seq=np.zeros((70,10))
All_seq_LPrates=[]
counter=0
for j,mouse in enumerate(mice):#enumerate(High_resp_CATEG)   np.unique(master_df['Mouse'])
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days

    mouse_heatmap_seq=np.zeros((70,10))
    seq_day_LPrates=[]
    if 1:
    #if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    #if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
        print(mouse)
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
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
            
            #This loop takes all rewarded sequences
            for rwd in rewards:
                LP_indices=np.where(LPs<=rwd)[0][-5:]
                IPI_indices=LP_indices[1:]
                variance=np.var(IPIs[IPI_indices])/np.mean(IPIs[IPI_indices])
                day_variances.append(variance)
            
            #This loop takes all attempted sequences
            # for r,rwd in enumerate(rewards):
            #     #LP_indices=np.where(LPs<=rwd)[0][-5:]
            #     LP_indices = np.where( (LPs<=rwd) & (LPs>rewards[r-1]) )[0] #get the indices of the LPs between teh two rewards. must be a multiple of 5
            #     IPI_indices=LP_indices[1:]
            #     while len(IPI_indices)>1:
            #         indices.append(IPI_indices[-4:]) #count the last 4 ipis (5LPS) as within sequence
            #         variance=np.var(IPIs[IPI_indices])/np.mean(IPIs[IPI_indices])
            #         day_variances.append(variance)
            #         if len( IPI_indices)!=4:
            #             try: 
            #                 IPI_indices[-5]
            #             except:
            #                 IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq
            #                 continue
            #         IPI_indices = IPI_indices[:-5] #drop them + the previous one, which is INTER seq

                
           
                
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

for group in [FR5_mice, CATEG_mice, FR5CATEG_mice]:
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    
    All_Variance=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):

        mouse_df = master_df[master_df['Mouse']==mouse].reset_index()
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
        if mouse not in FR5_mice:
            mouse_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
        mouse_variances=[]
        #Variance=mouse_df['Variance'].values[:10]
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
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
            
            #This loop takes all rewarded sequences
            for rwd in rewards:
                LP_indices=np.where(LPs<=rwd)[0][-5:]
                IPI_indices=LP_indices[1:]
                variance=np.var(IPIs[IPI_indices])/np.mean(IPIs[IPI_indices])
                day_variances.append(variance)
            mouse_variances.append(day_variances)
                
        Mean_variance_across_days=[np.median(x) for x in mouse_variances]
        while len(Mean_variance_across_days)<10:
            Mean_variance_across_days.append(float('nan'))
        All_Variance[j,:]=Mean_variance_across_days
        #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
        if mouse in FR5_mice:
            color='tomato'
        elif mouse in CATEG_mice:
            color='k'
        elif mouse in FR5CATEG_mice:
            color='cornflowerblue'
        plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c=color,alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.xticks([0,4,9],['1','5','10'],  size=16)
    plt.xlabel('Time on FR5 schedule (days)', size=20)
    plt.ylabel('Median within sequence \n inter-press interval', size=20)
    plt.title(str(len(mice)) + ' mice')
    mean=np.nanmean(All_Variance, axis=0)
    std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
    plt.plot(mean, linewidth=3, color=color)
    plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color=color, linewidth=3)
    plt.yscale('log') 



###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
# ON MICE THAT DID BOTH FR/Var FOLLOWED BY CATEG
###############################################################################
mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
file_dir='G:/Behavior study Dec2021/All medpc together'
master_df2 = create_medpc_master(mice, file_dir)

###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
###############################################################################
All_mouse_heatmap_seq=np.zeros((70,25))
All_seq_LPrates=[]
counter=0
for j,mouse in enumerate(mice):#enumerate(High_resp_CATEG)   np.unique(master_df['Mouse'])
    mouse_protocols=[]
    mouse_df=master_df2[master_df2['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((70,25))
    seq_day_LPrates=[]
    if 1:
    #if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
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

test_mice=[4229]#[4219, 4225,4230,4239]
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
    
###############################################################################
# PART 3.1: cohort of FR5>CATEG
############################################################################### 
 
###############################################################################
# PART 4: MAGNET
############################################################################### 

###############################################################################
# PART 5: NoLight, NoForcedReward, Neither
############################################################################### 

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
      4410,4411,4412,4413, 
      4667,4668,4669,4670,
      4671,4672,4673,4674,
      4675,4676,4677,4678,4679,
      4680,4681,4682,4683,
      4684,4685,4686,4687,
      4688,4689,4690,4691,
      4692,4693, 4694] 
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
[4668, '20220317'], #wrong protocol
]
master_df = discard_day(master_df, discard_list)

#drop mice doing Variance targetted
discard_list=[
    4217,4219,4220,4223,
    4227,4228,4231,4234,
    4235,4236,4238,4239,
    4241,4392,4393,
    4394,4395,4401,
    4402,4403,4404,4405,4407,4408,4409,
    4410,4411,4412,4413, 4673,4674,
    4675,4676,4677,4678,4679,
    4680,4681, 4688,4689,4690,4691,
    4692,4693, 4694]
master_df=discard_mice(master_df, discard_list)

#drop mice based on google spreadsheet
discard_list=[4683 ,4691,4694]
master_df=discard_mice(master_df, discard_list)

#drop the extra test training done on a subset of animals
mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
dates=['20220118','20220120','20220121','20220124','20220125','20220126','20220127','20220128','20220130','20220131','20220201']
for mouse in mice:
    for date in dates:
        master_df = discard_day(master_df, [[mouse,date]])

#starting dataset:
len(np.unique(master_df['Mouse']))#38
mice=np.unique(master_df['Mouse'])
backup=master_df.copy()
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
Time_on_FR1_FR5=[]
Time_on_FR1_noLight=[]
Time_on_FR1_noForcedR=[]
Time_on_FR1_neither=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR1' in x[0]]
    except:
        print('Mask problem: ' + f'{mouse}')
    if mouse_protocols[-1][0] == 'MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLight':
        Time_on_FR1_noLight.append(len(mask))
    elif mouse_protocols[-1][0] == 'MC_magbase_ForcedReward_LongWinVarTarget_FR5_noForcedR':
        Time_on_FR1_noForcedR.append(len(mask))
    elif mouse_protocols[-1][0] == 'MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLightnoForcedR':
        Time_on_FR1_neither.append(len(mask))
    else:
        Time_on_FR1_FR5.append(len(mask))
fig,ax=plt.subplots(1,1)
plt.hist([Time_on_FR1_FR5, Time_on_FR1_noLight, Time_on_FR1_noForcedR, Time_on_FR1_neither],stacked=True, bins=10)

# colors=[]
# discard_list=[]
# for each,mouse in zip(Time_on_FR1, np.unique(master_df['Mouse'])):
#     if each>5:
#         colors.append('r')
#         discard_list.append(mouse)
#     else:
#         colors.append('k')

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


colors=[]
discard_list=[]
for each,mouse in zip(Time_on_FR5, np.unique(master_df['Mouse'])):
    if each<7:
        colors.append('r')
        discard_list.append(mouse)
    else:
        colors.append('k')
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


###############################################################################
# reward rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_noLight=[]
All_rewards_noForcedR=[]
All_rewards_neither=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
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
        plt.plot(mouse_rewards, linestyle='dotted', color='k')
        All_rewards_FR5.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLightnoForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_neither.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLight':
        plt.plot(mouse_rewards, linestyle='dotted', color='green')
        All_rewards_noLight.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_noForcedR.append(mouse_rewards)


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
mean_noLight=[]
sem_noLight=[]
while All_rewards_noLight:
    step_values=[x[step] for x in All_rewards_noLight]
    step_length=len(step_values)
    mean_noLight.append(np.nanmean(step_values))
    sem_noLight.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noLight=[x[1:] for x in All_rewards_noLight]
    All_rewards_noLight=[x for x in All_rewards_noLight if np.nansum(x)>0]

step=0
mean_noForcedR=[]
sem_noForcedR=[]
while All_rewards_noForcedR:
    step_values=[x[step] for x in All_rewards_noForcedR]
    step_length=len(step_values)
    mean_noForcedR.append(np.nanmean(step_values))
    sem_noForcedR.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noForcedR=[x[1:] for x in All_rewards_noForcedR]
    All_rewards_noForcedR=[x for x in All_rewards_noForcedR if np.nansum(x)>0]
    
step=0
mean_neither=[]
sem_neither=[]
while All_rewards_neither:
    step_values=[x[step] for x in All_rewards_neither]
    step_length=len(step_values)
    mean_neither.append(np.nanmean(step_values))
    sem_neither.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_neither=[x[1:] for x in All_rewards_neither]
    All_rewards_neither=[x for x in All_rewards_neither if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='k')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='k', linewidths=2) 
plt.plot(mean_noLight, linewidth=2, color='green')
plt.vlines(range(len(mean_noLight)), [a-b for a,b in zip(mean_noLight,sem_noLight)], [a+b for a,b in zip(mean_noLight,sem_noLight)], colors='green', linewidths=2) 
plt.plot(mean_noForcedR, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(mean_noForcedR)), [a-b for a,b in zip(mean_noForcedR,sem_noForcedR)], [a+b for a,b in zip(mean_noForcedR,sem_noForcedR)], colors='cornflowerblue', linewidths=2) 
plt.plot(mean_neither, linewidth=2, color='tomato')
plt.vlines(range(len(mean_neither)), [a-b for a,b in zip(mean_neither,sem_neither)], [a+b for a,b in zip(mean_neither,sem_neither)], colors='tomato', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=12 mice', 'noLight, N=4 mice', 'noForcedR, N=4 mice', 'neither, N=4 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('cornflowerblue')
leg.legendHandles[3].set_color('tomato')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# LP rate
###############################################################################
fig,ax=plt.subplots(1,1, figsize=(8,5))

All_rewards_FR5=[]
All_rewards_noLight=[]
All_rewards_noForcedR=[]
All_rewards_neither=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
    mouse_rewards=[]
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]

        # mouse_rewards.append(len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
        mouse_rewards.append(sum([x<1800 for x in date_df['Lever'].values[0]]) /1800*60) #divide by the last reward timestamps to et the rate
    while len(mouse_rewards)<15:
        mouse_rewards.append(float('nan'))
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        # plt.plot(mouse_rewards, linestyle='dotted', color='k', alpha=0.2)
        All_rewards_FR5.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLightnoForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato', alpha=0.2)
        All_rewards_neither.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLight':
        plt.plot(mouse_rewards, linestyle='dotted', color='green', alpha=0.2)
        All_rewards_noLight.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue', alpha=0.2)
        All_rewards_noForcedR.append(mouse_rewards)


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
mean_noLight=[]
sem_noLight=[]
while All_rewards_noLight:
    step_values=[x[step] for x in All_rewards_noLight]
    step_length=len(step_values)
    mean_noLight.append(np.nanmean(step_values))
    sem_noLight.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noLight=[x[1:] for x in All_rewards_noLight]
    All_rewards_noLight=[x for x in All_rewards_noLight if np.nansum(x)>0]

step=0
mean_noForcedR=[]
sem_noForcedR=[]
while All_rewards_noForcedR:
    step_values=[x[step] for x in All_rewards_noForcedR]
    step_length=len(step_values)
    mean_noForcedR.append(np.nanmean(step_values))
    sem_noForcedR.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noForcedR=[x[1:] for x in All_rewards_noForcedR]
    All_rewards_noForcedR=[x for x in All_rewards_noForcedR if np.nansum(x)>0]
    
step=0
mean_neither=[]
sem_neither=[]
while All_rewards_neither:
    step_values=[x[step] for x in All_rewards_neither]
    step_length=len(step_values)
    mean_neither.append(np.nanmean(step_values))
    sem_neither.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_neither=[x[1:] for x in All_rewards_neither]
    All_rewards_neither=[x for x in All_rewards_neither if np.nansum(x)>0]
    
# plt.plot(meanFR5, linewidth=2, color='k')
# plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='k', linewidths=2) 
plt.plot(mean_noLight, linewidth=2, color='green')
plt.vlines(range(len(mean_noLight)), [a-b for a,b in zip(mean_noLight,sem_noLight)], [a+b for a,b in zip(mean_noLight,sem_noLight)], colors='green', linewidths=2) 
plt.plot(mean_noForcedR, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(mean_noForcedR)), [a-b for a,b in zip(mean_noForcedR,sem_noForcedR)], [a+b for a,b in zip(mean_noForcedR,sem_noForcedR)], colors='cornflowerblue', linewidths=2) 
plt.plot(mean_neither, linewidth=2, color='tomato')
plt.vlines(range(len(mean_neither)), [a-b for a,b in zip(mean_neither,sem_neither)], [a+b for a,b in zip(mean_neither,sem_neither)], colors='tomato', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=12 mice', 'noLight, N=7 mice', 'noForcedR, N=8 mice', 'neither, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('cornflowerblue')
leg.legendHandles[3].set_color('tomato')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# LP rate vs Reward rate
###############################################################################
fig,ax=plt.subplots(1,1)

for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
    mouse_rewards=[]
    mouse_LPs=[]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Reward'].values))):
        #     mouse_rewards[i]=0
        # else:
        mouse_LPs.append(len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
        mouse_rewards.append(len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
  

    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.scatter(mouse_LPs,mouse_rewards, linestyle='dotted', color='k')
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLightnoForcedR':
        plt.scatter(mouse_LPs,mouse_rewards, linestyle='dotted', color='tomato')
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLight':
        plt.scatter(mouse_LPs,mouse_rewards, linestyle='dotted', color='green')
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noForcedR':
        plt.scatter(mouse_LPs,mouse_rewards, linestyle='dotted', color='cornflowerblue')


#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=12 mice', 'noLight, N=4 mice', 'noForcedR, N=4 mice', 'neither, N=4 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('cornflowerblue')
leg.legendHandles[3].set_color('tomato')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# #Questions: 
#     - Do they learn faster? (Days on FR1)
#     - Do they produce sequences of 5? (IPIs, extra presses (for light only), COllect in_between sequence (for noForced and neither))

###############################################################################
# Number of extra presses across days (maybe not that useful)
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_noLight=[]
All_rewards_noForcedR=[]
All_rewards_neither=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
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
        
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='k')
        All_rewards_FR5.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLightnoForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_neither.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noLight':
        plt.plot(mouse_rewards, linestyle='dotted', color='green')
        All_rewards_noLight.append(mouse_rewards)
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5_noForcedR':
        plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_noForcedR.append(mouse_rewards)       
   

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
mean_noLight=[]
sem_noLight=[]
while All_rewards_noLight:
    step_values=[x[step] for x in All_rewards_noLight]
    step_length=len(step_values)
    mean_noLight.append(np.nanmean(step_values))
    sem_noLight.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noLight=[x[1:] for x in All_rewards_noLight]
    All_rewards_noLight=[x for x in All_rewards_noLight if np.nansum(x)>0]

step=0
mean_noForcedR=[]
sem_noForcedR=[]
while All_rewards_noForcedR:
    step_values=[x[step] for x in All_rewards_noForcedR]
    step_length=len(step_values)
    mean_noForcedR.append(np.nanmean(step_values))
    sem_noForcedR.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_noForcedR=[x[1:] for x in All_rewards_noForcedR]
    All_rewards_noForcedR=[x for x in All_rewards_noForcedR if np.nansum(x)>0]
    
step=0
mean_neither=[]
sem_neither=[]
while All_rewards_neither:
    step_values=[x[step] for x in All_rewards_neither]
    step_length=len(step_values)
    mean_neither.append(np.nanmean(step_values))
    sem_neither.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_neither=[x[1:] for x in All_rewards_neither]
    All_rewards_neither=[x for x in All_rewards_neither if np.nansum(x)>0]
    
plt.plot(meanFR5, linewidth=2, color='k')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='k', linewidths=2) 
plt.plot(mean_noLight, linewidth=2, color='green')
plt.vlines(range(len(mean_noLight)), [a-b for a,b in zip(mean_noLight,sem_noLight)], [a+b for a,b in zip(mean_noLight,sem_noLight)], colors='green', linewidths=2) 
plt.plot(mean_noForcedR, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(mean_noForcedR)), [a-b for a,b in zip(mean_noForcedR,sem_noForcedR)], [a+b for a,b in zip(mean_noForcedR,sem_noForcedR)], colors='cornflowerblue', linewidths=2) 
plt.plot(mean_neither, linewidth=2, color='tomato')
plt.vlines(range(len(mean_neither)), [a-b for a,b in zip(mean_neither,sem_neither)], [a+b for a,b in zip(mean_neither,sem_neither)], colors='tomato', linewidths=2) 
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


test_mice=[4386,4387,4396,4397, 4667,4668,4682] #noLight
test_mice=[4388,4389,4398,4399, 4669,4670,4684,4685] #noForcedR
test_mice=[4390,4391,4400,4406, 4671,4672,4686,4687] #noLightnoForcedR

for j,mouse in enumerate(test_mice):
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
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
 
###############################################################################
# within sequence IPI  vs inter IPIs (summary plot: Delat inter-intra)
###############################################################################
fig,ax=plt.subplots(1,1,figsize=(8,5))
Grouped_mice=[[4386,4387,4396,4397, 4667,4668,4682], #noLight
              [4388,4389,4398,4399, 4669,4670,4684,4685], #noForcedR
              [4390,4391,4400,4406, 4671,4672,4686,4687]] #noLightnoForcedR
colors=['green','cornflowerblue','tomato']
for k,test_mice in enumerate(Grouped_mice):
    Deltas=np.empty((len(test_mice), 10))
    for j,mouse in enumerate(test_mice):
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
    
        seq_day_mean=[]
        rest_day_mean=[]
        
        # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
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
            
        Deltas[j,:i+1]=[b-a for a,b in zip(seq_day_mean, rest_day_mean)]
    mean_Delta=np.nanmean(Deltas, axis=0)
    sem_Delta=np.nanstd(Deltas, axis=0)/np.sqrt(len(test_mice))
    plt.plot(mean_Delta, color=colors[k], linewidth=2)
    plt.vlines(range(len(mean_Delta)), [a-b for a,b in zip(mean_Delta,sem_Delta)], [a+b for a,b in zip(mean_Delta,sem_Delta)], color=colors[k], linewidths=2) 
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('interIPI - intraIPI (s)', size=16)
plt.yticks(fontsize=14)
plt.legend(['noLight, N=7 mice', 'noForcedR, N=8 mice', 'neither, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('tomato')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
###############################################################################
# Number of presses while reward is sitting unconsumed
# need to plot across days to see if at first they suck and naturally learn
###############################################################################

Grouped_mice=[[4386,4387,4396,4397, 4667,4668,4682], #noLight
              [4388,4389,4398,4399, 4669,4670,4684,4685], #noForcedR
              [4390,4391,4400,4406, 4671,4672,4686,4687]] #noLightnoForcedR
All_collection_RT=[]
All_extra_LP=[]
for test_mice in Grouped_mice:
    Group_collection_RT=[]
    Group_extra_LP=[]
    for j,mouse in enumerate(test_mice):
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days

        mouse_rewards=np.zeros((1,10))[0]
        Mouse_collection_RT=[]
        Mouse_extra_LP=[]
        #First days
        for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
            Collection_RT=[]
            Extra_LPs=[]
            date_df=mouse_df[mouse_df['Date']==date]
            # if math.isnan(sum(sum(date_df['Lever'].values))):
            #     mouse_rewards[i]=0
            # else:
            
            #get reward
            reward_times=np.asarray(date_df['Reward'].values[0])
            lick_times=np.asarray(date_df['Lick'].values[0])
            LP_times=np.asarray(date_df['Lever'].values[0])
            for reward in reward_times:
                #get the first lick following reward
                try: 
                    idx=np.where(lick_times>reward)[0][0]
                except: 
                    continue
                first_lick=lick_times[idx]
                Collection_RT.append(first_lick-reward)
                #get LPs between reward delivery and fist lick
                mask1=LP_times>reward
                mask2=LP_times<first_lick
                extra_LPs=np.sum(mask1&mask2)
                Extra_LPs.append(extra_LPs)
            relevant_presses=len(date_df['Reward'].values[0])*5
            total_presses=len(date_df['Lever'].values[0])
            extra_presses=total_presses-relevant_presses
            extra_press_per_seq=extra_presses/len(date_df['Reward'].values[0])
            mouse_rewards[i]=extra_press_per_seq 
            Mouse_collection_RT.append(Collection_RT)
            Mouse_extra_LP.append(Extra_LPs)
        Group_collection_RT.append(Mouse_collection_RT)  
        Group_extra_LP.append(Mouse_extra_LP)
    All_collection_RT.append(Group_collection_RT)  
    All_extra_LP.append(Group_extra_LP)

fig,ax=plt.subplots(1,1,figsize=(8,5))
colors=['green','cornflowerblue','tomato']
for k ,(group_RTs, group_LPs) in enumerate(zip(All_collection_RT, All_extra_LP)):
    step=0
    mean_group=[]
    sem_group=[]
    while group_RTs:
        step_values=[np.median(x[step]) for x in group_RTs]
        step_length=len(step_values)
        mean_group.append(np.nanmean(step_values))
        print(np.nanmean(step_values))
        
        sem_group.append(np.nanstd(step_values)/np.sqrt(step_length))
        group_RTs=[x[1:] for x in group_RTs]
        group_RTs=[x for x in group_RTs if len(x)>0]
       
    plt.plot(mean_group,color=colors[k], linewidth=2)
    plt.vlines(range(len(mean_group)), [a-b for a,b in zip(mean_group,sem_group)], [a+b for a,b in zip(mean_group,sem_group)], colors=colors[k], linewidths=2) 
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('reward collection delay (s)', size=16)
plt.yticks(fontsize=14)
plt.legend(['noLight, N=7 mice', 'noForcedR, N=8 mice', 'neither, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('cornflowerblue')
leg.legendHandles[2].set_color('tomato')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
        
        

        
for group_RTs, group_LPs in zip(All_collection_RT, All_extra_LP):
    for RTs in group_RTs:
        fig, ax=plt.subplots(1,1,figsize=(5,5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        data = np.arange(1000, 0, -1).reshape(100, 10)
        im = cax.imshow(data, cmap='magma')
        plt.sca(cax)
        plt.xticks([])
        plt.yticks([99,1], ['First', 'Last'])
        plt.xticks(visible=False)
        plt.ylabel('Day')
        
        Cmap_index=np.linspace(0,1,len(RTs))
        for i,R in enumerate(RTs):
            Cum=stats.cumfreq(R, numbins=40, defaultreallimits=(0,20))
            x= np.linspace(0, Cum.binsize*Cum.cumcount.size, Cum.cumcount.size)
            x= np.insert([np.log(a+1) for a in x][1:], 0,0)
            ax.plot(x,Cum.cumcount/np.size(R), color=cmap(Cmap_index[i]), linestyle='dotted') #color=cmap(Cmap_index[j])
        
     
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
cmap = cm.get_cmap('magma', 12)
