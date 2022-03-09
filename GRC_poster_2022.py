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
[4405, '20220209'] #wrong protocol
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
#     print(cum_data)
#     last_data=[x for x in cum_data if not math.isnan(x)][-1]
#     if last_data<330:
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

# #DISCARD based on mice not learning
# master_df=discard_mice(master_df, discard_list)
# #New dataset:
# len(np.unique(master_df['Mouse']))#38
# mice=np.unique(master_df['Mouse'])
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
    mouse_rewards=np.zeros((1,10))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Reward'].values))):
        #     mouse_rewards[i]=0
        # else:
        mouse_rewards[i]=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60) #divide by the last reward timestamps to et the rate

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
    meanFR5.append(np.mean(step_values))
    semFR5.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_FR5=[x[1:] for x in All_rewards_FR5]
    All_rewards_FR5=[x for x in All_rewards_FR5 if sum(x)>0]
    
step=0
meanVar5=[]
semVar5=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar5.append(np.mean(step_values))
    semVar5.append(np.std(step_values)/np.sqrt(step_length))
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
leg.legendHandles[1].set_color('green')
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
    mouse_rewards=np.zeros((1,10))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
        date_df=mouse_df[mouse_df['Date']==date]
        if len(date_df['Lever'].values[0]) ==0:
            mouse_rewards[i]=0
        else:
            mouse_rewards[i]=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60) #divide by the last reward timestamps to et the rate

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
    meanFR5.append(np.mean(step_values))
    semFR5.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_FR5=[x[1:] for x in All_rewards_FR5]
    All_rewards_FR5=[x for x in All_rewards_FR5 if sum(x)>0]
    
step=0
meanVar5=[]
semVar5=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar5.append(np.mean(step_values))
    semVar5.append(np.std(step_values)/np.sqrt(step_length))
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
    mouse_heatmap_seq=np.zeros((40,10))
    mouse_heatmap_rest=np.zeros((40,10))
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
        Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
        Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
        Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]

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
    
    plt.imshow(mouse_heatmap_rest, alpha=1, cmap='Blues')
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='Reds')
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
      
All_mouse_heatmap_seq=np.zeros((40,10))
All_mouse_heatmap_rest=np.zeros((40,10))
counter=0
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,10))
    mouse_heatmap_rest=np.zeros((40,10))
    #if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5': #'MC_magbase_ForcedReward_LongWinVarTarget_FR5'
    if mouse_df['Protocol'].values[0]not in ['MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5','MC_magbase_ForcedReward_LongWinVarTarget_FR5'   ]: 
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            seq_data,edges=np.histogram(np.log10(Seq_IPIs), bins=40, range=(-1,3), density=True)
            rest_data,edges=np.histogram(np.log10(Rest_IPIs), bins=40,  range=(-1,3), density=True)
            
            mouse_heatmap_seq[:,i]=seq_data
            mouse_heatmap_rest[:,i]=rest_data
        All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
        All_mouse_heatmap_rest=np.add(All_mouse_heatmap_rest, mouse_heatmap_rest)
        counter+=1
Mean_mouse_heatmap_seq=All_mouse_heatmap_seq/counter
Mean_mouse_heatmap_rest=All_mouse_heatmap_rest/counter 
fig,ax=plt.subplots(1,1, figsize=(5,10))
plt.imshow(Mean_mouse_heatmap_rest, alpha=1, cmap='Blues')
plt.colorbar()
plt.imshow(Mean_mouse_heatmap_seq, alpha=0.5, cmap='Reds')
plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
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
    
    
    
    
    
All_IPIs_FR_first_seq=[x for l in All_IPIs_FR_first_seq for x in l]
All_IPIs_Var_first_seq=[x for l in All_IPIs_Var_first_seq for x in l]
All_IPIs_CATEG_first_seq=[x for l in All_IPIs_CATEG_first_seq for x in l]
All_IPIs_FR_last_seq=[x for l in All_IPIs_FR_last_seq for x in l]
All_IPIs_Var_last_seq=[x for l in All_IPIs_Var_last_seq for x in l]
All_IPIs_CATEG_last_seq=[x for l in All_IPIs_CATEG_last_seq for x in l]   

All_IPIs_FR_first_rest=[x for l in All_IPIs_FR_first_rest for x in l]
All_IPIs_Var_first_rest=[x for l in All_IPIs_Var_first_rest for x in l]
All_IPIs_CATEG_first_rest=[x for l in All_IPIs_CATEG_first_rest for x in l]
All_IPIs_FR_last_rest=[x for l in All_IPIs_FR_last_rest for x in l]
All_IPIs_Var_last_rest=[x for l in All_IPIs_Var_last_rest for x in l]
All_IPIs_CATEG_last_rest=[x for l in All_IPIs_CATEG_last_rest for x in l]


fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_FR_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_FR_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_FR_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_FR_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)

fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_Var_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_Var_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_Var_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_Var_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)

fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_CATEG_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_CATEG_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_CATEG_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_CATEG_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)









All_IPIs_FR_first_seq=[]
All_IPIs_FR_first_rest=[]
All_IPIs_FR_last_seq=[]
All_IPIs_FR_last_rest=[]

All_IPIs_Var_first_seq=[]
All_IPIs_Var_first_rest=[]
All_IPIs_Var_last_seq=[]
All_IPIs_Var_last_rest=[]

All_IPIs_CATEG_first_seq=[]
All_IPIs_CATEG_first_rest=[]
All_IPIs_CATEG_last_seq=[]
All_IPIs_CATEG_last_rest=[]

for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    #mouse_rewards=np.zeros((1,10))[0]
    if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        #First days
        for date in [np.unique(mouse_df['Date'])[0]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax1.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax1.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_FR_first_seq.append(Seq_IPIs)
            All_IPIs_FR_first_rest.append(Rest_IPIs)
        #Last days
        for date in [np.unique(mouse_df['Date'])[-1]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax2.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax2.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_FR_last_seq.append(Seq_IPIs)
            All_IPIs_FR_last_rest.append(Rest_IPIs)
        ax1.set_title(str(mouse)+ ' FR5')
        
    elif date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5':
        #First days
        for date in [np.unique(mouse_df['Date'])[0]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax1.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax1.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_CATEG_first_seq.append(Seq_IPIs)
            All_IPIs_CATEG_first_rest.append(Rest_IPIs)
        #Last days
        for date in [np.unique(mouse_df['Date'])[-1]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax2.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax2.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_CATEG_last_seq.append(Seq_IPIs)
            All_IPIs_CATEG_last_rest.append(Rest_IPIs)
        ax1.set_title(str(mouse)+ ' CATEG')
    else:
        #First days
        for date in [np.unique(mouse_df['Date'])[0]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax1.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax1.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_Var_first_seq.append(Seq_IPIs)
            All_IPIs_Var_first_rest.append(Rest_IPIs)
        #Last days
        for date in [np.unique(mouse_df['Date'])[-1]]:
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices]
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
    
            ax2.hist(np.log(Seq_IPIs), bins=20, alpha=0.5, range=(-2,8))
            ax2.hist(np.log(Rest_IPIs), bins=20, alpha=0.5, range=(-2,8))
            
            All_IPIs_Var_last_seq.append(Seq_IPIs)
            All_IPIs_Var_last_rest.append(Rest_IPIs)
        ax1.set_title(str(mouse)+ ' Var')

All_IPIs_FR_first_seq=[x for l in All_IPIs_FR_first_seq for x in l]
All_IPIs_Var_first_seq=[x for l in All_IPIs_Var_first_seq for x in l]
All_IPIs_CATEG_first_seq=[x for l in All_IPIs_CATEG_first_seq for x in l]
All_IPIs_FR_last_seq=[x for l in All_IPIs_FR_last_seq for x in l]
All_IPIs_Var_last_seq=[x for l in All_IPIs_Var_last_seq for x in l]
All_IPIs_CATEG_last_seq=[x for l in All_IPIs_CATEG_last_seq for x in l]   

All_IPIs_FR_first_rest=[x for l in All_IPIs_FR_first_rest for x in l]
All_IPIs_Var_first_rest=[x for l in All_IPIs_Var_first_rest for x in l]
All_IPIs_CATEG_first_rest=[x for l in All_IPIs_CATEG_first_rest for x in l]
All_IPIs_FR_last_rest=[x for l in All_IPIs_FR_last_rest for x in l]
All_IPIs_Var_last_rest=[x for l in All_IPIs_Var_last_rest for x in l]
All_IPIs_CATEG_last_rest=[x for l in All_IPIs_CATEG_last_rest for x in l]


fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_FR_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_FR_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_FR_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_FR_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)

fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_Var_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_Var_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_Var_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_Var_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)

fig,[ax1,ax2]=plt.subplots(2,1, figsize=(5,10))
ax1.hist(np.log(All_IPIs_CATEG_first_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax1.hist(np.log(All_IPIs_CATEG_first_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_CATEG_last_seq), bins=20, alpha=0.5, range=(-2,8))#, density=True)
ax2.hist(np.log(All_IPIs_CATEG_last_rest), bins=20, alpha=0.5, range=(-2,8))#, density=True)