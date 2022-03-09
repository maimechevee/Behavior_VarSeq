import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math
import matplotlib
from create_medpc_master import *
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice=[4219,4222,4224,4225,4226,
      4229,4230,4231,4234,4239,4240,4241] 
dates=['20220118', '20220120','20220121','20220124','20220125', 
       '20220126', '20220127', '20220128', '20220130', '20220131', '20220201']
filename='G:/Behavior study Dec2021/TarVar CATEG Medpc'
master_df=create_medpc_master(mice, dates, filename)

master_df = create_medpc_master(mice,dates)

#Make reward vs days plot
All_rewards=np.zeros((len(np.unique(master_df['Mouse'])), len(np.unique(master_df['Date']))))
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
    print(mouse)
    print(mouse_rewards)
    All_rewards[j,:]=mouse_rewards
    All_protocols.append(mouse_protocols)
    
fig,ax=plt.subplots(1,1)
plt.plot(All_rewards.transpose(), color='k', alpha=0.5)

def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]
#Full training
for mouse,mouse_data in zip(mice,All_rewards):
    cum_data=Cumulative(mouse_data)
    if cum_data[-1]<400:
        print(mouse)
    plt.plot(cum_data, color='k', linestyle='dotted')
 
#Only FR5
fig,ax=plt.subplots(1,1)
total_discarded=0
discard_list=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    try:
        mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
    except:
        print('Mask problem: ' + f'{mouse}')
    cum_data=Cumulative(mouse_data[mask])
    if cum_data[-1]<330:
        print(str(mouse)+str(cum_data[-1]))
        print(cum_data)
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
plt.legend(['N='+str(28-total_discarded)+' mice', 'N='+str(total_discarded)+' mice'])
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('r')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#Only FR5 CATEG
fig,ax=plt.subplots(1,1)
total_discarded=0
discard_list=[]
for mouse,mouse_data, mouse_protocols in zip(mice,All_rewards, All_protocols):
    mask=[i for i,x in enumerate(mouse_protocols) if 'CATEG' in x[0]]
    cum_data=Cumulative(mouse_data[mask])
    if cum_data[-1]<330:
        print(str(mouse)+str(cum_data[-1]))
        print(cum_data)
        discard_list.append(mouse)
        total_discarded+=1
        color='r'
    else:
        color='k'
    plt.plot(cum_data, color=color, linestyle='dotted')   
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Cumulstive rewards obtained (#)', size=16)
plt.yticks(fontsize=14)
plt.legend(['N='+str(28-total_discarded)+' mice', 'N='+str(total_discarded)+' mice'])
leg = ax.get_legend()
leg.legendHandles[0].set_color('k')
leg.legendHandles[1].set_color('r')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# Discard the mice below the threshold from the master dataframe
###############################################################################
indices=[]
for mouse in discard_list:
    indices.append(master_df[master_df['Mouse']==mouse].index)
indices=[x for l in indices for x in l]
master_df=master_df.drop(indices, axis=0)

###############################################################################
# cumulative reward across days, split by group
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_Var=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        if math.isnan(sum(sum(date_df['Reward'].values))):
            mouse_rewards[i]=0
        else:
            mouse_rewards[i]=len(date_df['Reward'].values[0])
    print(mouse)
    print(mouse_rewards)
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        cum_mouse_rewards=Cumulative(mouse_rewards)
        plt.plot(cum_mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR.append(cum_mouse_rewards)
    else: #the rest in both types of Var
        cum_mouse_rewards=Cumulative(mouse_rewards)
        plt.plot(cum_mouse_rewards, linestyle='dotted', color='cornflowerblue')
        All_rewards_Var.append(cum_mouse_rewards)

step=0
meanFR5=[]
semFR5=[]
while All_rewards_FR:
    step_values=[x[step] for x in All_rewards_FR]
    step_length=len(step_values)
    meanFR5.append(np.mean(step_values))
    semFR5.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_FR=[x[1:] for x in All_rewards_FR]
    All_rewards_FR=[x for x in All_rewards_FR if sum(x)>0]
    
step=0
meanVar=[]
semVar=[]
while All_rewards_Var:
    step_values=[x[step] for x in All_rewards_Var]
    step_length=len(step_values)
    meanVar.append(np.mean(step_values))
    semVar.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_Var=[x[1:] for x in All_rewards_Var]
    All_rewards_Var=[x for x in All_rewards_Var if sum(x)>0]

plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 

plt.vlines(3.5,0,600, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Cumulstive rewards obtained (#)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='lower right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# reward rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_Var=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        if math.isnan(sum(sum(date_df['Reward'].values))):
            mouse_rewards[i]=0
        else:
            mouse_rewards[i]=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60) #divide by the last reward timestamps to et the rate
    print(mouse)
    print(mouse_rewards)
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR.append(mouse_rewards)
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
    All_rewards_FR=[x for x in All_rewards_FR if sum(x)>0]
    
step=0
meanVar=[]
semVar=[]
while All_rewards_Var:
    step_values=[x[step] for x in All_rewards_Var]
    step_length=len(step_values)
    meanVar.append(np.mean(step_values))
    semVar.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_Var=[x[1:] for x in All_rewards_Var]
    All_rewards_Var=[x for x in All_rewards_Var if sum(x)>0]

plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 

plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# LP rate, which can only be different than reward rate if mice press extra before collectibg reward
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_Var=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
    for i,date in enumerate(np.unique(mouse_df['Date'])):
        date_df=mouse_df[mouse_df['Date']==date]
        if math.isnan(sum(sum(date_df['Lever'].values))):
            mouse_rewards[i]=0
        else:
            mouse_rewards[i]=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60) #divide by the last reward timestamps to et the rate
    print(mouse)
    print(mouse_rewards)
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
        All_rewards_FR.append(mouse_rewards)
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
    All_rewards_FR=[x for x in All_rewards_FR if sum(x)>0]
    
step=0
meanVar=[]
semVar=[]
while All_rewards_Var:
    step_values=[x[step] for x in All_rewards_Var]
    step_length=len(step_values)
    meanVar.append(np.mean(step_values))
    semVar.append(np.std(step_values)/np.sqrt(step_length))
    All_rewards_Var=[x[1:] for x in All_rewards_Var]
    All_rewards_Var=[x for x in All_rewards_Var if sum(x)>0]

plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 

plt.vlines(3.5,0,30, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('LP rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################################################################
# Some plots to assess "sequencing"
###############################################################################

# Number of extra presses on first day vs last day

All_rewards_FR=[]
All_rewards_Var=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,2))[0]
    
    #First days
    for date in [np.unique(mouse_df['Date'])[0]]:
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Lever'].values))):
        #     mouse_rewards[0]=0
        # else:
        relevant_presses=len(date_df['Reward'].values[0])*5
        total_presses=len(date_df['Lever'].values[0])
        extra_presses=total_presses-relevant_presses
        extra_press_per_seq=extra_presses/len(date_df['Reward'].values[0])
        mouse_rewards[0]=extra_press_per_seq 
        
    #Last days
    for date in [np.unique(mouse_df['Date'])[-1]]:
        date_df=mouse_df[mouse_df['Date']==date]
        # if math.isnan(sum(sum(date_df['Lever'].values))):
        #     mouse_rewards[1]=0
        # else:
        relevant_presses=len(date_df['Reward'].values[0])*5
        total_presses=len(date_df['Lever'].values[0])
        extra_presses=total_presses-relevant_presses
        extra_press_per_seq=extra_presses/len(date_df['Reward'].values[0])
        mouse_rewards[1]=extra_press_per_seq 
           
    print(mouse)
    if date_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
        All_rewards_FR.append(mouse_rewards)
    else: #the rest in both types of Var
        All_rewards_Var.append(mouse_rewards)

fig,ax=plt.subplots(1,1)
[plt.plot(x/x[0], color='tomato') for x in All_rewards_FR]
[plt.plot(x/x[0], color='cornflowerblue') for x in All_rewards_Var]
plt.xlim(-0.5,1.5)
plt.ylim(0,4)

# Number of extra presses across days
fig,ax=plt.subplots(1,1)

All_rewards_FR=[]
All_rewards_Var=[]
for j,mouse in enumerate(np.unique(master_df['Mouse'])):
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
    
    #First days
    for i,date in enumerate(np.unique(mouse_df['Date'])):
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
        plt.plot(np.log(mouse_rewards), linestyle='dotted', color='tomato')
        All_rewards_FR.append(np.log(mouse_rewards))
    else: #the rest in both types of Var
        plt.plot(np.log(mouse_rewards), linestyle='dotted', color='cornflowerblue')
        All_rewards_Var.append(np.log(mouse_rewards))


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

plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanVar)), [a-b for a,b in zip(meanVar,semVar)], [a+b for a,b in zip(meanVar,semVar)], colors='cornflowerblue', linewidths=2) 

plt.vlines(3.5,-5,2, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('log(extraLP/reward)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()
