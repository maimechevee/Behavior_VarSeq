#############################################
# Sequential behavior paper
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
matplotlib.rcParams['font.sans-serif'] = "Arial"
import numpy.matlib
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Maxime_Tools')
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Behavior_VarSeq')
from create_medpc_master import create_medpc_master

###############################################################################
# Number of mice, days to acquisition, etc. just math with data from spreqdsheets
###############################################################################

#FR5 w/LightCue
F=[3,3,2,5]
np.mean(F) #3.25
np.std(F)/np.sqrt(len(F)-1) # 0.6291528696058959
M=[3,5,2,4] 
np.mean(M)#3.5
np.std(M)/np.sqrt(len(M)-1) #0.6454972243679029
A=F+M
np.mean(A) #3.375
np.std(A)/np.sqrt(len(A)-1) #0.4199277148680302

#FR5 w/MustCollect
F=[2, 9, 6]
np.mean(F) #5.666666666666667
np.std(F)/np.sqrt(len(F)-1) # 2.0275875100994063
M=[6, 4, 5] 
np.mean(M)#5.0
np.std(M)/np.sqrt(len(M)-1) #0.5773502691896257
A=F+M
np.mean(A) #5.857142857142857
np.std(A)/np.sqrt(len(A)-1) #0.9618576131773409

#FR5 w/MustCollect and LightCue
F=[4,3,4]
np.mean(F) #3.6666666666666665
np.std(F)/np.sqrt(len(F)-1) # 0.3333333333333333
M=[3,3,4,2,3,3] 
np.mean(M)#3.0
np.std(M)/np.sqrt(len(M)-1) #0.2581988897471611
A=F+M
np.mean(A) #3.2222222222222223
np.std(A)/np.sqrt(len(A)-1) #0.22222222222222218

#LowVariance
F=[4,5,5,4,3,6,3,5,2]
np.mean(F) #4.111111111111111
np.std(F)/np.sqrt(len(F)-1) # 0.4230985058813282
M=[3,3,5,5,3] 
np.mean(M)#3.8
np.std(M)/np.sqrt(len(M)-1) #0.4898979485566356
A=F+M
np.mean(A) #4.0
np.std(A)/np.sqrt(len(A)-1) #0.3144854510165755

###############################################################################
# Load data
###############################################################################
mice=[4218,4221,4222,4224,
      4225,4226,4230,4232,
      4233,4237,4240,4242,
      4243,4244,4229,
      4386,4387,4388,4389,
      4392,4393,4394,4395, 
      4396,4397,4398,4399,
      4401,4402,4403,4404,
      4405,4407,4408,4409,
      4410,4411,4412,4413,
      4667,4668, 4669,4670,
      4682,4683,4684,4685,
      ] 
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

#drop the extra test training done on a subset of animals
mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
dates=['20220118','20220120','20220121','20220124','20220125','20220126','20220127','20220128','20220130','20220131','20220201']
for mouse in mice:
    for date in dates:
        master_df = discard_day(master_df, [[mouse,date]])
#drop based on google doc
mice=[4410]
dates=['20220213'] #wrong protocol
for mouse in mice:
    for date in dates:
        master_df = discard_day(master_df, [[mouse,date]])
#starting dataset:
len(np.unique(master_df['Mouse']))#47
mice=np.unique(master_df['Mouse'])


FR5_mice=[4232,4233,4237,4240,4242,4243,4244,4229,4218,4221,4222,4224,4225,4226,4230] #15
CATEG_mice=[4392,4393,4394,4395,4401,4402,4403,4404,4405,4407,4408,4409,4410,4411,4412,4413] #16
noLight_mice=[4386, 4387,4396,4397, 4667,4668,4682, 4683] #8
noForcedR_mice= [4388,4389,4398,4399, 4669,4670,4684,4685] #8

###############################################################################
###############################################################################
############################## FIGURE 1 #######################################
###############################################################################
###############################################################################

###############################################################################
# Days to acquisition and renoval of mice that didn't learn
###############################################################################
def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

# generate All_rewards, All_protocols
colors=['tab:orange','tab:blue','tab:red',  'tab:green']
Anova_df=pd.DataFrame(columns=['feature','group', 'subject'])
fig,ax=plt.subplots(1,1,figsize=(3,3))
All_rewards=[]
All_protocols=[]
counter=0
for c,group in enumerate([FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]):
    mice=group
    Group_rewards=np.zeros((len(group), 30)) #
    Group_protocols=[]
    #Get the time to 2x51 in FR1 for each group
    Group_days_to_criteria=[]
    for m,mouse in enumerate(mice):
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
        for i,date in enumerate(np.unique(mouse_df['Date'])[:30]):
            date_df=mouse_df[mouse_df['Date']==date]
            try :
                len(date_df['Reward'].values[0])>1 #will fail if Nan
                mouse_rewards[i]=len(date_df['Reward'].values[0]) 
            except:
                mouse_rewards[i]=0
                # print(mouse)
                # print(date)
                
            mouse_protocols.append(date_df['Protocol'].values)
        while len(mouse_rewards)<30:
            mouse_rewards=np.append(mouse_rewards,float('nan'))
            
        day=[i for i,(a,b) in enumerate(zip(mouse_rewards[:-1], mouse_rewards[1:])) if (a>=46) & (b>=46) ]
        try: 
            Group_days_to_criteria.append(day[0])
            Anova_df.at[counter,'feature']=day[0]
            Anova_df.at[counter,'group']=colors[c]
            Anova_df.at[counter,'subject']=mouse
        except: 
            Group_days_to_criteria.append(np.float('nan'))
            Anova_df.at[counter,'feature']=100
            Anova_df.at[counter,'group']=colors[c]
            Anova_df.at[counter,'subject']=mouse
        counter+=1
        

        # print(mouse)
        # print(mouse_rewards)
        
        Group_rewards[m,:]=mouse_rewards
        Group_protocols.append(mouse_protocols)
    All_protocols.append(Group_protocols)
    All_rewards.append(Group_rewards)
        

    Cum=stats.cumfreq(Group_days_to_criteria, numbins=15, defaultreallimits=(0,15))
    print(Group_days_to_criteria)
    print(colors[c])
    print('The mean:')
    print(np.nanmean(Group_days_to_criteria))
    print("The sem:")
    print(np.nanstd(Group_days_to_criteria)/np.sqrt(len(Group_days_to_criteria)-1))
    print("N:")
    print( len(Group_days_to_criteria))
    x= np.linspace(0, Cum.binsize*Cum.cumcount.size, Cum.cumcount.size)
    plt.plot(x,Cum.cumcount/np.size(Group_days_to_criteria),color=colors[c],  linewidth=2) #color=cmap(Cmap_index[j])
plt.xlabel('Number of days to criteria', size=16)
plt.xticks([1,3,5,7,9,11,13,15],['2','4','6','8','10','12','14', '16'],fontsize=14)
plt.ylabel('Cumulative fraction', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=15 mice', 'noLight, N=8 mice', 'noForcedR, N=8 mice',  'CATEG, N=16 mice'], loc='lower right')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
   

# Days to criteria all together (for methods)
days= [2, 0, 2, 1, 1, 2, 0, 1, 1, 4, 2, 3, 7, 0, 3, 0, 3, 0, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 4, 1, 3, 0, 1, 1, 3, 3, 1]
np.mean(days)
# 1.7297297297297298
np.std(days)/np.sqrt(len(days)-1)
# 0.2377593373044441
import pingouin as pg
# Compute the two-way mixed-design ANOVA
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])
aov = pg.anova(dv='feature', between='group', data=Anova_df,
               detailed=True)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# When performed on all mice, including the ones that were not included in final dataset
# =============
# ANOVA SUMMARY
# =============

# Source          SS    DF       MS        F    p-unc      np2
# --------  --------  ----  -------  -------  -------  -------
# group     1242.436     3  414.145    2.150    0.108    0.130
# Within    8283.308    43  192.635  nan      nan      nan

#when performed on only the data we are using
# =============
# ANOVA SUMMARY
# =============

# Source        SS    DF     MS        F    p-unc      np2
# --------  ------  ----  -----  -------  -------  -------
# group     16.843     3  5.614    3.170    0.037    0.224
# Within    58.454    33  1.771  nan      nan      nan

posthocs = pg.pairwise_tukey(dv='feature',  between='group',
                              data=Anova_df)
pg.print_table(posthocs)

#only on included data
# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green       3.167      1.714   1.452  0.649   2.236      0.134     1.045
# tab:blue    tab:orange      3.167      1.111   2.056  0.701   2.930      0.030     1.454
# tab:blue    tab:red         3.167      1.375   1.792  0.719   2.493      0.080     1.260
# tab:green   tab:orange      1.714      1.111   0.603  0.569   1.061      0.695     0.437
# tab:green   tab:red         1.714      1.375   0.339  0.590   0.575      0.900     0.245
# tab:orange  tab:red         1.111      1.375  -0.264  0.647  -0.408      0.900    -0.188
# tab:orange
# The mean:
# 1.1111111111111112
# The sem:
# 0.26057865332352387
# N:
# 9
# tab:blue
# The mean:
# 3.1666666666666665
# The sem:
# 0.945750730607407
# N:
# 6
# tab:red
# The mean:
# 1.375
# The sem:
# 0.4199277148680302
# N:
# 8
# tab:green
# The mean:
# 1.7142857142857142
# The sem:
# 0.3043380752555216
# N:
#14


# #Do a chi-square to test acquisition
# from scipy.stats import chi2_contingency 

# g, p, dof, expctd=chi2_contingency([[15,7,8,16],[0,1,0,0]])
# # (4.980978260869565,
# #  0.17319532020189932,
# #  3,
# #  array([[14.68085106,  7.82978723,  7.82978723, 15.65957447],
# #         [ 0.31914894,  0.17021277,  0.17021277,  0.34042553]]))
# fig,ax=plt.subplots(1,1, figsize=(3,3))
# ax.bar([0,1,2,3], [1,0.75,1,1])
# ax.bar([0,1,2,3],[0,0.25,0,0], bottom=[1,0.75,1,1])

#Run this to exclude mice who did not perform the task for at least 10 days
# Number of days on FR5
All_groups=[FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]
Time_on_FR5=[]
discard_list=[]
for mice,group_data, group_protocols in zip(All_groups,All_rewards, All_protocols):
    for mouse, mouse_data,mouse_protocols in zip ( mice,group_data, group_protocols):
        try:
            mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
        except:
            print('Mask problem: ' + f'{mouse}')
        print( len(mask))
        if len(mask)<10:
            discard_list.append(mouse)
        Time_on_FR5.append(len(mask))
fig,ax=plt.subplots(1,1)
plt.hist(Time_on_FR5, bins=30)


#DISCARD based on number of days available for analysis (at least 10 days of FR5)
master_df=discard_mice(master_df, discard_list)

###############################################################################
#Rerun Figure 1 script to generate the figures in the paper, which are based only on mice included.
###############################################################################

################################################################################
FR5_mice=[x for x in FR5_mice if x in np.unique(master_df['Mouse'])] #9
CATEG_mice=[x for x in CATEG_mice if x in np.unique(master_df['Mouse'])] #14
noLight_mice=[x for x in noLight_mice if x in np.unique(master_df['Mouse'])] #6
noForcedR_mice= [x for x in noForcedR_mice if x in np.unique(master_df['Mouse'])] #8
################################################################################

###############################################################################
###############################################################################
############################## FIGURE 2 #######################################
###############################################################################
###############################################################################

# #############
# # EXTRA #
# #############

# #Compare LP rate on last day of FR1 to show they are slow at learning but still learn the same
# #manually
# colors=['tab:orange','tab:blue','tab:red',  'tab:green']
# Anova_df=pd.DataFrame(columns=['feature','group', 'subject'])

# counter=0
# for c,group in enumerate([FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]):
#     mice=group
#     Group_rewards=np.zeros((len(group), 30)) #
#     Group_protocols=[]
#     #Get the time to 2x51 in FR1 for each group
#     Group_days_to_criteria=[]
#     for m,mouse in enumerate(mice):
#         mouse_protocols=[]
#         mouse_df=master_df[master_df['Mouse']==mouse]
#         mouse_rewards=np.zeros((1,len(np.unique(mouse_df['Date']))))[0]
#         for i,date in enumerate(np.unique(mouse_df['Date'])[:30]):
#             date_df=mouse_df[mouse_df['Date']==date]
#             try :
#                 len(date_df['Reward'].values[0])>1 #will fail if Nan
#                 mouse_rewards[i]=len(date_df['Reward'].values[0]) 
#             except:
#                 mouse_rewards[i]=0
#                 # print(mouse)
#                 # print(date)
                
#             mouse_protocols.append(date_df['Protocol'].values)
#         while len(mouse_rewards)<30:
#             mouse_rewards=np.append(mouse_rewards,float('nan'))
            
#         day=[i for i,(a,b) in enumerate(zip(mouse_rewards[:-1], mouse_rewards[1:])) if (a>=46) & (b>=46) ]
 
#         Group_days_to_criteria.append(day[0])
#         Anova_df.at[counter,'feature']=len(mouse_df['Lever'].values[day[0]]) / (mouse_df['Lever'].values[day[0]][-1]/60)
#         Anova_df.at[counter,'group']=colors[c]
#         Anova_df.at[counter,'subject']=mouse
#         counter+=1
        
        
# fig,ax=plt.subplots(1,1, figsize=(3,3))
# i=0
# for j,group in enumerate(['tab:orange',  'tab:blue','tab:red','tab:green' ]):
#     group_df=Anova_df[Anova_df['group']==group]
#     for subject in np.unique(group_df['subject']):
#         subject_df=group_df[group_df['subject']==subject]
#         plt.scatter(i,subject_df['feature'].values, color=colors[j], alpha=0.3)

#     mean0=np.mean(group_df['feature'])
#     sem0=np.std(group_df['feature'])/np.sqrt(len(group_df)-1)
#     print("Means:")
#     print(mean0)
#     print("SEMs:")
#     print(sem0)
#     plt.scatter(i, mean0,  c=colors[j])
#     plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j])
#     i+=1
# plt.xlabel('Groups', size=16)
# plt.xticks([0,1,2,3], ['FR5','noLight', 'noRwdCol',  'Variance'],fontsize=14)
# plt.ylabel(' Lever press rate (#/sec)', size=16)
# plt.yticks(fontsize=14)
# #plt.ylim(0,20)
# plt.xlim(-1,4)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# # Means:
# # 1.7876447008999226
# # SEMs:
# # 0.18877139388616454
# # Means:
# # 1.4206953847605892
# # SEMs:
# # 0.3140997136499106
# # Means:
# # 1.3252791648780562
# # SEMs:
# # 0.0848490157439139
# # Means:
# # 2.1231497784686577
# # SEMs:
# # 0.2455985149686317

# # One way ANOVA 
# Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])

# aov = pg.anova(dv='feature', between='group', data=Anova_df,
#                 detailed=True)# Pretty printing of ANOVA summary
# pg.print_table(aov)
# # =============
# # ANOVA SUMMARY
# # =============

# # Source        SS    DF     MS        F    p-unc      np2
# # --------  ------  ----  -----  -------  -------  -------
# # group      4.055     3  1.352    2.638    0.066    0.193
# # Within    16.907    33  0.512  nan      nan      nan


                    #############
                    # FIGURE 2D #
                    #############
#Check overall performance
fig,ax=plt.subplots(1,1)
total_discarded=0
Total_reward_acquired=[]
Anova_df=pd.DataFrame(columns=['feature','group', 'subject'])
counter=0
colors=['tab:orange','tab:blue','tab:red',  'tab:green']
for c,(mice,group_data, group_protocols) in enumerate(zip(All_groups,All_rewards, All_protocols)):
    Group_reward_acquired=[]
    for mouse,mouse_data, mouse_protocols in zip(mice,group_data, group_protocols):
        if mouse in np.unique(master_df['Mouse']):
            try:
                mask=[i for i,x in enumerate(mouse_protocols) if 'FR5' in x[0]]
                print(len(mask))
            except:
                print('Mask problem: ' + f'{mouse}')
            cum_data=Cumulative(mouse_data[mask])
            #print(cum_data)
            last_data=[x for x in cum_data if not math.isnan(x)][9]
            Group_reward_acquired.append(last_data)
            plt.plot(cum_data, color=colors[c], linestyle='dotted')
            
            Anova_df.at[counter,'feature']=last_data
            Anova_df.at[counter,'group']=colors[c]
            Anova_df.at[counter,'subject']=mouse
            counter+=1
    Total_reward_acquired.append(Group_reward_acquired)

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

# Compute the two-way mixed-design ANOVA
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])
aov = pg.anova(dv='feature', between='group', data=Anova_df,
               detailed=True)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# =============
# ANOVA SUMMARY
# =============

# Source            SS    DF         MS        F    p-unc             np2
# --------  ----------  ----  ---------  -------  -------            -------
# group     295148.699     3  98382.900   17.078    7.170001e-07    0.608
# Within    190101.192    33   5760.642  nan      nan                nan

posthocs = pg.pairwise_tukey(dv='feature',  between='group',
                              data=Anova_df)
pg.print_table(posthocs)

# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)      diff      se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  --------  ------  ------  ---------  --------
# tab:blue    tab:green     295.833    285.071    10.762  37.035   0.291      0.900     0.136
# tab:blue    tab:orange    295.833    492.778  -196.944  40.002  -4.923      0.001    -2.442
# tab:blue    tab:red       295.833    426.625  -130.792  40.990  -3.191      0.016    -1.613
# tab:green   tab:orange    285.071    492.778  -207.706  32.428  -6.405      0.001    -2.638
# tab:green   tab:red       285.071    426.625  -141.554  33.639  -4.208      0.001    -1.794
# tab:orange  tab:red       492.778    426.625    66.153  36.880   1.794      0.295     0.827

i=0
fig,ax=plt.subplots(1,1, figsize=(3,3))
for c,group in enumerate(['tab:orange','tab:blue','tab:red',  'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature'].values, color=colors[c], alpha=0.3)

    mean0=np.mean(group_df['feature'])
    sem0=np.std(group_df['feature'])/np.sqrt(len(group_df)-1)
    print("Mean:")
    print(mean0)
    print("SEM:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[c])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[c])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2,3], [ 'FR5', 'noLight','noMustCollect', 'LowVariance'],fontsize=14)
plt.ylabel(' Total rewards acquired (#)', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,550)
plt.xlim(-1,4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Mean:
# 492.77777777777777
# SEM:
# 5.569704205635305
# Mean:
# 295.8333333333333
# SEM:
# 32.631698971671355
# Mean:
# 426.625
# SEM:
# 32.083004791144035
# Mean:
# 285.07142857142856
# SEM:
# 23.238009855424718

# #############
# # Extra #
# #############
# ###############################################################################
# # reward rate
# ###############################################################################
# fig,ax=plt.subplots(1,1)
# mice=np.unique(master_df['Mouse'])
# Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
# All_rewards=[]
# counter=0
# for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
#     test_mice=[x for x in mice if x in group]
#     group_rewards=np.zeros((len(test_mice), 10))
#     for j,mouse in enumerate(test_mice):
#         # if mouse in Females:
#         #     continue
#         mouse_protocols=[]
#         mouse_df=master_df[master_df['Mouse']==mouse]
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
#         mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
#         mouse_rewards=[]
       
#         for i,date in enumerate(np.unique(mouse_df['Date'])[:10]):
#             date_df=mouse_df[mouse_df['Date']==date]
#             # if math.isnan(sum(sum(date_df['Reward'].values))):
#             #     mouse_rewards[i]=0
#             # else:
#             mouse_rewards.append(len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
#             #mouse_rewards.append(sum([x<1800 for x in date_df['Reward'].values[0]]) /1800*60) #divide by the last reward timestamps to et the rate
            
#             if (i==0) | (i==8):
#                 Anova_df.at[counter,'feature']=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)
#                 Anova_df.at[counter,'group']=colors[c]
#                 Anova_df.at[counter,'subject']=mouse
#                 Anova_df.at[counter,'time']=i
#                 counter+=1
#             elif  (i==9):
#                 Anova_df.at[counter-1,'feature']= (Anova_df.at[counter-1,'feature'] + (len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60))) /2

#         if c==1:
#             plt.plot(mouse_rewards, linestyle='dotted',alpha=0.5, color=colors[c])
#             print(mouse)
#         group_rewards[j,:]=mouse_rewards
#     All_rewards.append(group_rewards)

# for c,group in enumerate(All_rewards):
#     meanFR5=np.mean(group, axis=0)
#     semFR5=np.std(group,axis=0)/np.sqrt(np.shape(group)[0]-1)
#     plt.plot(meanFR5, linewidth=2, color=colors[c])
#     plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors=colors[c], linewidths=2) 
# #plt.vlines(3.5,0,6, color='k', linestyle='dashed')
# plt.xlabel('Time from first FR5 session (day)', size=16)
# plt.xticks(fontsize=14)
# plt.ylabel('Reward rate (#/min)', size=16)
# plt.yticks(fontsize=14)
# plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
# leg = ax.get_legend()
# leg.legendHandles[0].set_color(colors[0])
# leg.legendHandles[1].set_color(colors[1])
# leg.legendHandles[2].set_color(colors[2])
# leg.legendHandles[3].set_color(colors[3])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()


# #manually
# fig,ax=plt.subplots(1,1)
# i=0
# for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
#     group_df=Anova_df[Anova_df['group']==group]
#     for subject in np.unique(group_df['subject']):
#         subject_df=group_df[group_df['subject']==subject]
#         plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
       
#     #test
#     s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==8]['feature'])
#     print(p)
#     mean0=np.mean(group_df[group_df['time']==0]['feature'])
#     mean9=np.mean(group_df[group_df['time']==8]['feature'])
#     sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
#     sem9=np.std(group_df[group_df['time']==8]['feature'])/np.sqrt(len(group_df)-1)
#     plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
#     plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
#     plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
#     i+=2
# plt.xlabel('Groups', size=16)
# plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
# plt.ylabel('Reward rate (#/min)', size=16)
# plt.yticks(fontsize=14)
# plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
# leg = ax.get_legend()
# leg.legendHandles[0].set_color(colors[0])
# leg.legendHandles[1].set_color(colors[1])
# leg.legendHandles[2].set_color(colors[2])
# leg.legendHandles[3].set_color(colors[3])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()
# # t-tests for each group
# # 0.26721081416801556
# # 0.03658372594275172
# # 0.029391381452190387
# # 0.00011432198479542105

# # One way ANOVA on difference
# new_Anova_df=Anova_df[Anova_df['time']==8].copy()
# new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==8]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
# new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
# #drop the group that was not significant
# new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']

# fig,ax=plt.subplots(1,1)
# i=0
# for j,group in enumerate(['tab:red','tab:orange', 'tab:green' ]):
#     group_df=new_Anova_df[new_Anova_df['group']==group]
#     for subject in np.unique(group_df['subject']):
#         subject_df=group_df[group_df['subject']==subject]
#         plt.scatter(i,subject_df['feature_diff'].values, color=colors[j+1], alpha=0.3)

#     mean0=np.mean(group_df['feature_diff'])
#     sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
#     plt.scatter(i, mean0,  c=colors[j+1])
#     plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j+1])
#     i+=1
# plt.xlabel('Groups', size=16)
# plt.xticks([0,1,2], [ 'noRwdCol', 'FR5', 'Variance'],fontsize=14)
# plt.ylabel(' DELTA Reward rate (#/min)', size=16)
# plt.yticks(fontsize=14)
# plt.ylim(-1,5)
# plt.xlim(-0.5,2.5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout()

# aov = pg.anova(dv='feature_diff', between='group', data=new_Anova_df,
#                detailed=True)# Pretty printing of ANOVA summary
# pg.print_table(aov)
# # =============
# # ANOVA SUMMARY
# # =============

# # Source        SS    DF     MS        F    p-unc      np2
# # --------  ------  ----  -----  -------  -------  -------
# # group      5.485     2  2.742    1.025    0.372    0.068
# # Within    74.894    28  2.675  nan      nan      nan

                    #############
                    # FIGURE 2A #
                    #############
###############################################################################
# LP rate
###############################################################################
fig,ax=plt.subplots(1,1)
mice=np.unique(master_df['Mouse'])
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
All_rewards=[]
counter=0
for c,group in enumerate([FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]):
    test_mice=[x for x in mice if x in group]
    group_rewards=np.zeros((len(test_mice), 10))
    for j,mouse in enumerate(test_mice):
        # if mouse in Females:
        #     continue
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
            # if math.isnan(sum(sum(date_df['Reward'].values))):
            #     mouse_rewards[i]=0
            # else:
            mouse_rewards.append(len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)) #divide by the last reward timestamps to et the rate
            #mouse_rewards.append(sum([x<1800 for x in date_df['Reward'].values[0]]) /1800*60) #divide by the last reward timestamps to et the rate
            if (i==0) | (i==8):
                Anova_df.at[counter,'feature']=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1
            elif  (i==9):
                Anova_df.at[counter-1,'feature']= (Anova_df.at[counter-1,'feature'] + (len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60))) /2


        #plt.plot(mouse_rewards, linestyle='dotted',alpha=0.5, color=colors[c])
        group_rewards[j,:]=mouse_rewards
    All_rewards.append(group_rewards)

for c,group in enumerate(All_rewards):
    meanFR5=np.mean(group, axis=0)
    semFR5=np.std(group,axis=0)/np.sqrt(np.shape(group)[0]-1)
    plt.plot(meanFR5, linewidth=2, color=colors[c])
    plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors=colors[c], linewidths=2) 


#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend([ 'FR5, N=9 mice', 'noLight, N=6 mice', 'noForcedR, N=8 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()


#manually
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange', 'tab:blue','tab:red', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    #test
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==8]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==8]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==8]['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0, mean9)
    print("SEMs:")
    print(sem0, sem9)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# 0.03202962648775018
# Means:
# 5.7173171209255464 11.774507606395769
# SEMs:
# 0.4181831095600726 1.2635838884774486
# 0.246493734406131
# Means:
# 2.2170241617562136 3.3504223224004206
# SEMs:
# 0.1798171186945579 0.5815021767644364
# 0.035334875370374386
# Means:
# 4.307988825208579 7.567826672500988
# SEMs:
# 0.526048479173935 1.25360812071322
# 4.091272124995579e-05
# Means:
# 2.318384967794326 5.6864167733689115
# SEMs:
# 0.254340125060456 0.40863121750186043

#ttest results
# 0.03202962648775018  FR5
# 0.246493734406131  noLight
# 0.035334875370374386  noMustColect
# 4.091272124995579e-05 LowVariance

                    #############
                    # FIGURE 2C #
                    #############

#Compare LP rate on last day across groups
#manually
new_Anova_df=Anova_df[Anova_df['time']==8].copy()
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange',  'tab:blue','tab:red','tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature'].values, color=colors[j], alpha=0.3)

    mean0=np.mean(group_df['feature'])
    sem0=np.std(group_df['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[j])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2,3], ['FR5','noLight', 'noRwdCol',  'Variance'],fontsize=14)
plt.ylabel(' Lever press rate (#/sec)', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,20)
plt.xlim(-1,4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# Means:
# 11.774507606395767
# SEMs:
# 1.8419742172033928
# Means:
# 3.3504223224004206
# SEMs:
# 0.8625071127126496
# Means:
# 7.567826672500988
# SEMs:
# 1.8350943846793573
# Means:
# 5.6864167733689115
# SEMs:
# 0.5889002620112813
# One way ANOVA on fold change
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])

aov = pg.anova(dv='feature', between='group', data=Anova_df,
                detailed=True)# Pretty printing of ANOVA summary
pg.print_table(aov)
#    Source           SS  DF          MS          F     p-unc      np2
# 0   group  342.933798   3  114.311266  8.933313  0.000043  0.276859
# 1  Within  895.724681  70   12.796067       NaN       NaN       NaN

posthocs = pg.pairwise_tukey(dv='feature', between='group',
                              data=Anova_df)
pg.print_table(posthocs)
# # ==============
# # POST HOC TESTS
# # ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green       2.784      4.002  -1.219  1.234  -0.987      0.732    -0.334
# tab:blue    tab:orange      2.784      8.746  -5.962  1.333  -4.472      0.001    -1.622
# tab:blue    tab:red         2.784      5.938  -3.154  1.366  -2.309      0.106    -0.856
# tab:green   tab:orange      4.002      8.746  -4.744  1.081  -4.389      0.001    -1.303
# tab:green   tab:red         4.002      5.938  -1.936  1.121  -1.727      0.318    -0.531
# tab:orange  tab:red         8.746      5.938   2.808  1.229   2.285      0.112     0.766

                    #############
                    # FIGURE 2B #
                    #############
# One way ANOVA on fold change
new_Anova_df=Anova_df[Anova_df['time']==8].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==8]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
#drop the group that was not significant
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange', 'tab:red', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j+1], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[j+1])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j+1])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2], [ 'FR5', 'noRwdCol',  'Variance'],fontsize=14)
plt.ylabel(' Fold change LPrate', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,10)
plt.xlim(-0.5, 2.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

aov = pg.anova(dv='feature_diff', between='group', data=new_Anova_df,
               detailed=True)# Pretty printing of ANOVA summary
pg.print_table(aov)
# =============
# ANOVA SUMMARY
# =============

# Source        SS    DF     MS        F    p-unc      np2
# --------  ------  ----  -----  -------  -------  -------
# group      8.527     2  4.263    1.457    0.250    0.094
# Within    81.956    28  2.927  nan      nan      nan

###############################################################################
###############################################################################
############################## FIGURE 3 #######################################
###############################################################################
###############################################################################

                    #############
                    # FIGURE 3 B-E #
                    #############
                
###############################################################################
# IPI distribution
###############################################################################
mice=np.unique(master_df['Mouse'])
test_mice=[4387,4684, 4230,4394] 
for j,mouse in enumerate(test_mice):
    #fig,ax=plt.subplots(1,1, figsize=(5,10))
    mouse_protocols=[]
    mouse_df=master_df[master_df['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((40,10))
    mouse_heatmap_rest=np.zeros((40,10))
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
        
        mouse_heatmap_seq[:,i]=seq_data[::-1]
        mouse_heatmap_rest[:,i]=rest_data[::-1]
        
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
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='Oranges', vmin=0, vmax=2)
    plt.plot([40-(x*10+10) for x in seq_day_mean], color='orange') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.plot([40-(x*10+10) for x in rest_day_mean], color='mediumpurple') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values][::-1])
    plt.xticks([1,3,5,7,9], ['2','4','6','8','10'])
    plt.xlabel('Sessions (#)')
    plt.colorbar()
    
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    plt.imshow(mouse_heatmap_rest, alpha=0.5, cmap='Purples', vmin=0, vmax=2)
    plt.plot([40-(x*10+10) for x in seq_day_mean], color='orange') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.plot([40-(x*10+10) for x in rest_day_mean], color='mediumpurple') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values][::-1])
    plt.xticks([1,3,5,7,9], ['2','4','6','8','10'])
    plt.xlabel('Sessions (#)')
    plt.colorbar()
    
    
    fig,[ax1, ax2]=plt.subplots(2,1, figsize=(5,10))
    ax1.bar(edges[1:],mouse_heatmap_seq[::-1,0], alpha=0.5, color='orange', width=0.1)
    ax1.vlines(seq_day_mean[0], 0,3, color='orange')
    ax1.bar(edges[1:],mouse_heatmap_rest[::-1,0],  alpha=0.5,  color='mediumpurple', width=0.1)
    ax1.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax1.vlines(rest_day_mean[0], 0,3, color='mediumpurple')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.bar(edges[1:],mouse_heatmap_seq[::-1,-1],  alpha=0.5,  color='orange', width=0.1)
    ax2.vlines(seq_day_mean[-1], 0,3, color='orange')
    ax2.bar(edges[1:],mouse_heatmap_rest[::-1,-1], alpha=0.5,  color='mediumpurple', width=0.1)
    ax2.vlines(rest_day_mean[-1], 0,3, color='mediumpurple')
    ax2.set_xticks(edges[[0,10,20,30,40]],[str(10**x) for x in log_values])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel('IPI (s)')
      

for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    fig,ax=plt.subplots(1,1,figsize=(6,12))
    plt.sca(ax)
    All_SeqIPIs=np.empty((len(mice), 10))
    All_RestIPIs=np.empty((len(mice), 10))
    All_InterFailedIPIs=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):
        # if mouse in Males:
        #     continue

        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedReward_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_NoForcedRew_NoLight_DynRespWin_1R'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLight'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noForcedR'] #do not count the FR1 early days
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1_noLightnoForcedR'] #do not count the FR1 early days
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
            # else:
            #     print('x')
            #     Inter_failed_IPIs=[float('nan')]
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
        plt.plot(np.arange(len(Median_SeqIPIs_across_days)), Median_SeqIPIs_across_days, c='orange',alpha=0.3)
        plt.plot(np.arange(len(Median_RestIPIs_across_days)), Median_RestIPIs_across_days, c='mediumpurple',alpha=0.3)
    plt.yscale('log')  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.xticks([0,4,9],['1','5','10'],  size=16)
    plt.xlabel('Time on FR5 schedule (days)', size=20)
    plt.ylabel('Median inter-press interval', size=20)
    plt.title(str(len(mice)) + ' mice')
    
    
    mean=np.nanmean(All_SeqIPIs, axis=0)
    std=np.nanstd(All_SeqIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_SeqIPIs[:,i]]) for i in range(np.shape(All_SeqIPIs)[1])] )
    plt.plot(mean, linewidth=3, color='orange')
    plt.vlines(range(np.shape(All_SeqIPIs)[1]), mean-std, mean+std, color='orange', linewidth=3)
    
    
    mean=np.nanmean(All_RestIPIs, axis=0)
    std=np.nanstd(All_RestIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_RestIPIs[:,i]]) for i in range(np.shape(All_RestIPIs)[1])] )
    plt.plot(mean, linewidth=3, color='mediumpurple')
    plt.vlines(range(np.shape(All_RestIPIs)[1]), mean-std, mean+std, color='mediumpurple', linewidth=3)
    

    plt.yscale('log') 
    plt.ylim(0.5,200)


                    #############
                    # FIGURE 3F #
                    #############

#Plot within IPI
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
Interfail_IPIs=0
fig,ax=plt.subplots(1,1,figsize=(8,5))
for c,group in enumerate([ FR5_mice,noLight_mice, noForcedR_mice, CATEG_mice]):
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    Deltas=np.empty((len(mice), 10))
    Deltas_failed=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):
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
        interfail_day_mean=[]
        
        # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices+inter_failed][1:]#dont count the first item, it's a zero
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
            if mouse in CATEG_mice:
                if np.sum(inter_failed)>0:
                    Interfail_IPIs=IPIs[np.array(inter_failed)]
            
            seq_day_mean.append(np.median(Seq_IPIs))
            rest_day_mean.append(np.median(Rest_IPIs))
            interfail_day_mean.append(np.median(Interfail_IPIs))
            
            if (i==0) | (i==9):
                Anova_df.at[counter,'feature']= np.median(Seq_IPIs)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1
            
        Deltas[j,:i+1]=[b/a for a,b in zip(seq_day_mean, rest_day_mean)]
        Deltas_failed[j,:i+1]=[b/a for a,b in zip(seq_day_mean, interfail_day_mean)]
        # print(mouse)
        # print(date)
        # print(Deltas[j,:])
    mean_Delta=np.nanmean(Deltas, axis=0)
    sem_Delta=np.nanstd(Deltas, axis=0)/np.sqrt(len(test_mice))
    plt.plot(mean_Delta, color=colors[c], linewidth=2)
    plt.vlines(range(len(mean_Delta)), [a-b for a,b in zip(mean_Delta,sem_Delta)], [a+b for a,b in zip(mean_Delta,sem_Delta)], color=colors[c], linewidths=2) 
    if mouse in CATEG_mice:
        mean_Delta=np.nanmean(Deltas_failed, axis=0)
        sem_Delta=np.nanstd(Deltas_failed, axis=0)/np.sqrt(len(test_mice))
        plt.plot(mean_Delta, color=colors[c], linewidth=2, alpha=0.5)
        plt.vlines(range(len(mean_Delta)), [a-b for a,b in zip(mean_Delta,sem_Delta)], [a+b for a,b in zip(mean_Delta,sem_Delta)], color=colors[c], linewidths=2, alpha=0.5) 

plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.xticks([1,3,5,7,9], ['2','4','6','8','10'])
plt.ylabel('interIPI / intraIPI', size=16)
plt.yticks(fontsize=14)
plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()


#manually
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange', 'tab:blue','tab:red','tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0, mean9)
    print("SEMs:")
    print(sem0, sem9)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('IntraIPI (s)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice', 'noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
#ttest
# 0.0006284894913309088 FR5
# 0.46346253629259143 no Light
# 0.03463462175099151 noMustCollect
# 0.0007543134413713983 LowVariance


# Compute the two-way mixed-design ANOVA
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
#drop the group that was not significant
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']
aov = pg.anova(dv='feature_diff',  between='group',  data=new_Anova_df)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# =============
# ANOVA SUMMARY
# =============

# Source      ddof1    ddof2      F    p-unc    np2
# --------  -------  -------  -----  -------  -----
# group           2       28  3.076    0.062  0.180

fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange','tab:red', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j+1], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[j+1])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j+1])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2], [ 'FR5','noRwdCol',  'Variance'],fontsize=14)
plt.ylabel(' Fold change withinIPI', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,1)
plt.xlim(-0.5, 2.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# Means:
# 0.31558025503753523
# SEMs:
# 0.05481722440381783
# Means:
# 0.3258753446823451
# SEMs:
# 0.08941673801101366
# Means:
# 0.16789743328936715
# SEMs:
# 0.028785916329506313


                    #############
                    # FIGURE 3G #
                    #############
#########################
#Same, but plotting FOLD CHANGE
#########################
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
fig,ax=plt.subplots(1,1,figsize=(8,5))
for c,group in enumerate([FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]):
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    Deltas=np.empty((len(mice), 10))
    Deltas_failed=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):
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
        interfail_day_mean=[]
        
        # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices+inter_failed][1:]#dont count the first item, it's a zero
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
            if mouse in CATEG_mice:
                if np.sum(inter_failed)>0:
                    Interfail_IPIs=IPIs[np.array(inter_failed)]
            
            seq_day_mean.append(np.median(Seq_IPIs))
            rest_day_mean.append(np.median(Rest_IPIs))
            interfail_day_mean.append(np.median(Interfail_IPIs))
            
            if (i==0) | (i==9):
                Anova_df.at[counter,'feature']= np.median(Rest_IPIs) / np.median(Seq_IPIs)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1
            
        Deltas[j,:i+1]=[b/a for a,b in zip(seq_day_mean, rest_day_mean)]
        Deltas_failed[j,:i+1]=[b/a for a,b in zip(seq_day_mean, interfail_day_mean)]
        # print(mouse)
        # print(date)
        # print(Deltas[j,:])
    mean_Delta=np.nanmean(Deltas, axis=0)
    sem_Delta=np.nanstd(Deltas, axis=0)/np.sqrt(len(test_mice))
    plt.plot(mean_Delta, color=colors[c], linewidth=2)
    plt.vlines(range(len(mean_Delta)), [a-b for a,b in zip(mean_Delta,sem_Delta)], [a+b for a,b in zip(mean_Delta,sem_Delta)], color=colors[c], linewidths=2) 
    if mouse in CATEG_mice:
        mean_Delta=np.nanmean(Deltas_failed, axis=0)
        sem_Delta=np.nanstd(Deltas_failed, axis=0)/np.sqrt(len(test_mice))
        plt.plot(mean_Delta, color=colors[c], linewidth=2, alpha=0.5)
        plt.vlines(range(len(mean_Delta)), [a-b for a,b in zip(mean_Delta,sem_Delta)], [a+b for a,b in zip(mean_Delta,sem_Delta)], color=colors[c], linewidths=2, alpha=0.5) 

plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.xticks([1,3,5,7,9], ['2','4','6','8','10'])
plt.ylabel('interIPI / intraIPI', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice','noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()


#manually
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange','tab:blue','tab:red', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0, mean9)
    print("SEMs:")
    print(sem0, sem9)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('InterIPI / IntraIPI (s)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice','noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

# 0.0021597527236944295
# Means:
# 3.327445980118493 12.627839702379998
# SEMs:
# 0.27188093798783686 1.3610556918183614
# 0.7134422191208979
# Means:
# 2.724570395511201 3.4213456851893795
# SEMs:
# 0.4755611239604243 1.0423714565720807
# 0.003554854251798505
# Means:
# 2.94614533765874 16.17504597549524
# SEMs:
# 0.2859220029601658 2.048621974392677
# 0.00167249499727292
# Means:
# 2.072026892582841 19.163789892388166
# SEMs:
# 0.18610895472464137 3.123240736889657

#ttest
# 0.0021597527236944295 FR5
# 0.7134422191208979 noLight
# 0.003554854251798505 noMustCollect
# 0.00167249499727292 LowVariance

new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
#drop the group that was not significant
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']

aov = pg.anova(dv='feature_diff',  between='group',  data=new_Anova_df)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# =============
# ANOVA SUMMARY (on three groups)
# =============

# Source      ddof1    ddof2      F    p-unc    np2
# --------  -------  -------  -----  -------  -----
# group           2       28  2.226    0.127  0.137


fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange','tab:red', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j+1], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[j+1])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j+1])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2], [ 'FR5', 'noRwdCol',  'Variance'],fontsize=14)
plt.ylabel(' Fold change across/within', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,35)
plt.xlim(-0.5, 2.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

###############################################################################
###############################################################################
############################## FIGURE 6 #######################################
###############################################################################
###############################################################################

                    #############
                    # FIGURE 6B #
                    #############
#plot the three types of IPIs for CATEG
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
for group in [ CATEG_mice]:
    c=3
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    Deltas=np.empty((len(mice), 10))
    Deltas_failed=np.empty((len(mice), 10))
    for j,mouse in enumerate(mice):
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
        interfail_day_mean=[]
        
        # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
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
            Rest_IPI_indices=[i for i in range(len(IPIs)) if i not in Seq_IPI_indices+inter_failed][1:]#dont count the first item, it's a zero
            Seq_IPIs=IPIs[np.array(Seq_IPI_indices)]
            Rest_IPIs=IPIs[np.array(Rest_IPI_indices)]
            
            if mouse in CATEG_mice:
                if np.sum(inter_failed)>0:
                    Interfail_IPIs=IPIs[np.array(inter_failed)]
            
            print(Interfail_IPIs)
            
            seq_day_mean.append(np.median(Seq_IPIs))
            rest_day_mean.append(np.median(Rest_IPIs))
            interfail_day_mean.append(np.median(Interfail_IPIs))
            
            if (i==0) | (i==9):
                for IPItype in ['r', 'b', 'k']:
                    if IPItype =='r':
                        Anova_df.at[counter,'feature']= np.median(Seq_IPIs)
                        Anova_df.at[counter,'group']='r'
                        Anova_df.at[counter,'subject']=mouse
                        Anova_df.at[counter,'time']=i
                        counter+=1
                    elif IPItype =='b':
                        Anova_df.at[counter,'feature']= np.median(Rest_IPIs)
                        Anova_df.at[counter,'group']='b'
                        Anova_df.at[counter,'subject']=mouse
                        Anova_df.at[counter,'time']=i
                        counter+=1
                    elif IPItype =='k':
                        Anova_df.at[counter,'feature']= np.median(Interfail_IPIs)
                        Anova_df.at[counter,'group']='k'
                        Anova_df.at[counter,'subject']=mouse
                        Anova_df.at[counter,'time']=i
                        counter+=1
            

#manually
#drop outlier
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['r','b','k' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        if subject == 4393:
            continue
        plt.plot([1+i,2+i],subject_df['feature'].values, color=group, alpha=0.3)
        # print(subject)
        # print(subject_df['feature'].values)
    group_df=group_df[group_df['subject']!=4393]
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0, mean9)
    print("SEMs:")
    print(sem0, sem9)
    plt.plot([1+i,2+i], [mean0, mean9],  color=group)
    plt.scatter([1+i,2+i], [mean0, mean9],  c=group)
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=group)
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,], ['First','Last', 'First','Last', 'First','Last', ],fontsize=14)
plt.xlim(0,7)
plt.ylabel('IPI (s)', size=16)
plt.yticks(fontsize=14)
plt.legend(['intra', 'inter', 'inter failed'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('r')
leg.legendHandles[1].set_color('b')
leg.legendHandles[2].set_color('k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# ttest
# 3.6821492031004345e-05 intra
# 0.4581617930200178 inter 
# 0.00038409493637690275 inter failed

                    #############
                    # FIGURE 6C #
                    #############
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
new_Anova_df=new_Anova_df[new_Anova_df['group']!='b']

fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['r','k' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=group, alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=group)
    plt.vlines(i, mean0-sem0,mean0+sem0, color=group)
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1], [ 'FR5',   'Variance'],fontsize=14)
plt.ylabel(' Fold change across/within', size=16)
plt.yticks(fontsize=14)
#plt.ylim(0,35)
plt.xlim(-0.5, 2.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

# Means:
# 0.16789743328936715
# SEMs:
# 0.028785916329506313
# Means:
# 0.11382100191395124
# SEMs:
# 0.024454798311776865

posthocs = pg.pairwise_ttests(dv='feature_diff',  between='group', data=new_Anova_df)
pg.print_table(posthocs)

# ==============
# POST HOC TESTS
# ==============

# Contrast    A    B    Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
# ----------  ---  ---  --------  ------------  ------  ------  ---------  -------  ------  --------
# group       k    r    False     True          -1.432  26.000  two-sided    0.164   0.752    -0.525



###############################################################################
###############################################################################
############################## FIGURE 4 #######################################
###############################################################################
###############################################################################

                    #############
                    # FIGURE 4A,B #
                    #############

#The current raster examples are actually based on the extra training done after, should change in final version
sequential_mice=[4219,4224,4225,4226,4222,4230,4231,4239,4234,4240,4241,4229]
mice=[x for x in FR5_mice if x in sequential_mice]
file_dir='G:/Behavior study Dec2021/All medpc together'
master_df2 = create_medpc_master(mice, file_dir)

#drop mouse/days based on google doc notes
discard_list=[
[4240, '20220124'], #wrong protocol
[4224, '20220121'], #wrong protocol
[4224, '20220124'], #wrong protocol
[4225, '20220124'], #wrong protocol
[4226, '20220124'], #wrong protocol
[4221, '20220121'], #wrong protocol
[4221, '20220124'], #wrong protocol
[4230, '20220121'], #wrong protocol
[4230, '20220124'], #wrong protocol
[4229, '20220121'], #wrong protocol
]
master_df2 = discard_day(master_df2, discard_list)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

test_mice=[4229]#[4219, 4225,4230,4239]
for j,mouse in enumerate(test_mice):
    mouse_protocols=[]
    mouse_df=master_df2[master_df2['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    # if mouse_df['Protocol'].values[0]=='MC_magbase_ForcedReward_LongWinVarTarget_FR5':
    counter=0
    Variances=[]
    for i,date in enumerate(['20211215','20220201']):
        
        fig,ax1=plt.subplots(1,1)
        fig,ax2=plt.subplots(1,1, figsize=(6,3))
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
            ax1.scatter(trial_LPs, np.ones_like(trial_LPs)+counter, c=['k','b','y','r','g'])
            trial_IPI_var=np.var(np.diff(trial_LPs))
            Variances.append(trial_IPI_var)
            counter+=1
        ax1.set_xlim(Range)
        ax2.scatter(np.arange(len(Variances)), Variances)
        ax2.plot(np.arange(len(Variances))[3:-3], moving_average(Variances, n=7), color='r')
        ax2.set_yscale('log')
        ax2.set_ylim(0.0001, 1000)
        fig, ax3=plt.subplots(1,1)
        sns.distplot( np.log10(Variances))
        plt.ylim(0,0.4)
        plt.vlines(np.median(np.log10(Variances)), 0,0.4)
        plt.figure()
        plt.hist(Presses[:,1], bins=30, range=Range, color='b', alpha=0.5)
        plt.hist(Presses[:,2], bins=30, range=Range, color='y', alpha=0.5)
        plt.hist(Presses[:,3], bins=30, range=Range, color='r', alpha=0.5)
        plt.hist(Presses[:,4], bins=30, range=Range, color='g', alpha=0.5)  
        

                    #############
                    # FIGURE 4C #
                    #############
###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
###############################################################################
test_mice=[4397,4398, 4233, 4394]
[noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]
All_mouse_heatmap_seq=np.zeros((70,10))
All_seq_LPrates=[]
counter=0
for j,mouse in enumerate(test_mice):#enumerate(High_resp_CATEG)   np.unique(master_df['Mouse'])
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
        
        # #This loop takes all attempted sequences
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
        mouse_heatmap_seq[:,i]=seq_data[::-1]
        
        
        
  
    All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
    All_seq_LPrates.append(seq_day_LPrates)
    counter+=1
    
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='rainbow', vmin=0, vmax=2)
    plt.plot([70-(x*10+30) for x in [np.median(x) for x in seq_day_LPrates]], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI Variance')
    plt.xticks([0,4,9],['1','5','10'],  size=16)
    log_values=[float(x) for x in edges[[0,10,20,30,40, 50, 60, 70]]]
    plt.yticks([0,10,20,30,40, 50, 60, 70],[str(10**x) for x in log_values][::-1])
    plt.xlabel('Sessions (#)')
    plt.colorbar()


                    #############
                    # FIGURE 4D #
                    #############
###############################################################################
# within sequence IPI variance (plot - accounts for mouse)
###############################################################################
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
fig,ax=plt.subplots(1,1,figsize=(6,12))
plt.sca(ax)

for c,group in enumerate([FR5_mice, noLight_mice, noForcedR_mice,  CATEG_mice]):
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
            
            
            # #This loop takes all attempted sequences
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
            
            mouse_variances.append(day_variances)
            
            if (i==0) | (i==8):
                Anova_df.at[counter,'feature']= np.mean(day_variances)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1
            if (i==9):
                Anova_df.at[counter-1,'feature']= ( Anova_df.at[counter-1,'feature'] + np.mean(day_variances)) /2
                
        Mean_variance_across_days=[np.mean(x) for x in mouse_variances]
       
        All_Variance[j,:]=Mean_variance_across_days

        #plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c=colors[c],alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.xticks([0,4,9],['1','5','10'],  size=16)
    plt.xlabel('Time on FR5 schedule (days)', size=20)
    plt.ylabel('Variance', size=20)
    mean=np.nanmean(All_Variance, axis=0)
    std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
    plt.plot(mean, linewidth=3, color=colors[c])
    plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color=colors[c], linewidth=3)
plt.yscale('log') 
plt.legend(['FR5, N=9 mice','noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.ylim(1,150)
    



#manually
fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange','tab:blue','tab:red', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==8]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==8]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==8]['feature'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0, mean9)
    print("SEMs:")
    print(sem0, sem9)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('Variance', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=9 mice','noLight, N=6 mice', 'noForcedR, N=8 mice',  'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
#ttest rewarded sequences
# 0.048965658519248015 FR5
# 0.17532219549988848 noLight
# 0.8238111413778587 noMustCollect
# 0.015427787128954446 LowVariance
# 0.048965658519248015
# Means:
# 11.872763651433704 5.046695523863196
# SEMs:
# 1.4325585210324436 0.9089368201816495
# 0.17532219549988848
# Means:
# 33.057368418961296 48.677023066080075
# SEMs:
# 4.760966619700077 10.61061556898379
# 0.8238111413778587
# Means:
# 18.502746092397135 16.51950639371423
# SEMs:
# 3.836668874127844 4.617010187150874
# 0.015427787128954446
# Means:
# 26.332745267667754 4.428954975417443
# SEMs:
# 5.400273347429245 0.6767154447511144

# #ttest attempted sequences
# 0.024030133780860646 FR5
# 0.19261278960264208 noLight
# 0.8752826260292234 noMustCollect
# 0.06868906739865306 LowVariance

                    #############
                    # FIGURE 4E #
                    #############
new_Anova_df=Anova_df[Anova_df['time']==8].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==8]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:red']

# aov = pg.anova(dv='feature_diff',  between='group',  data=new_Anova_df)
# # Pretty printing of ANOVA summary
# pg.print_table(aov)

posthocs = pg.pairwise_ttests(dv='feature_diff',  between='group', data=new_Anova_df)
pg.print_table(posthocs)

# Contrast    A          B           Paired    Parametric         T    dof  Tail         p-unc    BF10    hedges
# ----------  ---------  ----------  --------  ------------  ------  -----  ---------  -------  ------  --------
# group       tab:green  tab:orange  False     True          -1.125  9.733  two-sided    0.288   0.603    -0.546


fig,ax=plt.subplots(1,1, figsize=(3,3))
i=0
for j,group in enumerate(['tab:orange', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j+1], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    print("Means:")
    print(mean0)
    print("SEMs:")
    print(sem0)
    plt.scatter(i, mean0,  c=colors[j+1])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j+1])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1], [ 'FR5',   'LowVariance'],fontsize=14)
plt.ylabel(' Fold change in IPI variance', size=16)
plt.yticks(fontsize=14)
# plt.ylim(0,35)
plt.xlim(-0.5, 1.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

# Means:
# 0.6566335306579596
# SEMs:
# 0.2619116719412309
# Means:
# 0.3467330087233512
# SEMs:
# 0.08563433419212924


###############################################################################
###############################################################################
############################## FIGURE 5 #######################################
###############################################################################
###############################################################################

                    #############
                    # FIGURE 5A,B #
                    #############
#GET EMMA'S CODE

                    #############
                    # FIGURE 5C #
                    #############
#r values and length values from EMMA
filename= 'G:/Behavior study Dec2021/Data figures/figure_4.xlsx'
CATEG_r_early=[0.275237546,0.560863314,0.352037791,0.482394646,0.369892867,0.385851979,0.484050033,0.38972061,0.355992621]
CATEG_r_late=[0.601730283,0.654595408,0.347238602,0.567046371,0.372602478,0.497437372,0.536403723,0.507716717,0.371354927]
FR5_r_early=[0.361770762,0.769085397,0.588147948,0.65719664,0.376133277,0.551317678]
FR5_r_late=[0.608256238,0.73773793,0.676257644,0.674332659,0.498371187,0.494933979]

CATEG_length_early=[665.5,405,773,437.5,607.5,642,538,494,838]
CATEG_length_late=[363.5,345,1044.5,340,665.5,418,462,439,735]
FR5_length_early=[608,345.5,495,387,628,402]
FR5_length_late=[356,333,383,318,435,582]

Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0

Earlys=[CATEG_r_early,FR5_r_early]
Lates=[CATEG_r_late,FR5_r_late]
subjects=[ [1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15] ]
fig,ax=plt.subplots(1,1,figsize=(3,3))
x1=0
x2=1
for i,(early, late) in enumerate(zip(Earlys, Lates)):
    for j,(a,b) in enumerate(zip(early, late)):
        plt.plot([x1, x2], [a,b], alpha=0.5, color='r')
        Anova_df.at[counter,'feature']= a
        Anova_df.at[counter,'group']=i
        Anova_df.at[counter,'subject']=subjects[i][j]
        Anova_df.at[counter,'time']='early'
        counter+=1
        Anova_df.at[counter,'feature']= b
        Anova_df.at[counter,'group']=i
        Anova_df.at[counter,'subject']=subjects[i][j]
        Anova_df.at[counter,'time']='late'
        counter+=1
        
    mean_early=np.mean(early)
    print("mean early:")
    print(mean_early)
    sem_early=np.std(early)/np.sqrt(len(early)-1)
    print("sem early:")
    print(sem_early)
    
    mean_late=np.mean(late)
    print("mean late:")
    print(mean_late)
    sem_late=np.std(late)/np.sqrt(len(late)-1)
    print("sem late:")
    print(sem_late)
    
    plt.plot([x1, x2], [mean_early, mean_late], color='r')
    plt.scatter([x1, x2], [mean_early, mean_late], color='r')
    plt.vlines([x1,x2],[mean_early-sem_early, mean_late-sem_late], [mean_early+sem_early, mean_late+sem_late])
    
    s,p=sp.stats.wilcoxon(early, late)
    print(p)
    x1+=2
    x2+=2
    
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])    
aov = pg.mixed_anova(dv='feature', between='group',
                  within='time', subject='subject', data=Anova_df)

# mean early:
# 0.406226823
# sem early:
# 0.028962690618695757
# mean late:
# 0.4951250978888888
# sem late:
# 0.0365219109372289
# 0.01171875
# mean early:
# 0.550608617
# sem early:
# 0.06496260307150127
# mean late:
# 0.6149816061666667
# sem late:
# 0.04098823756054954
# 0.3125

#         Source        SS  DF1  DF2        MS         F     p-unc       np2  eps
# 0        group  0.125679    1   13  0.125679  6.247682  0.026612  0.324594  NaN
# 1         time  0.046912    1   13  0.046912  8.478629  0.012130  0.394747  1.0
# 2  Interaction  0.001083    1   13  0.001083  0.195678  0.665503  0.014829  NaN

aov = pg.pairwise_ttests(dv='feature', between='group',
                  within='time', subject='subject', data=Anova_df)

#        Contrast   time      A     B  ...       Tail     p-unc   BF10    hedges
# 0          time      -  early  late  ...  two-sided  0.009566  5.777 -0.600230
# 1         group      -      0     1  ...  two-sided  0.047447  2.145 -1.239879
# 2  time * group  early      0     1  ...  two-sided  0.081851  1.527 -1.133010
# 3  time * group   late      0     1  ...  two-sided  0.050461  1.816 -1.062862
