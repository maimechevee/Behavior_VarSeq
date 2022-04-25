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
import numpy.matlib
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Maxime_Tools')
sys.path.append('C:/Users/cheveemf/Documents/GitHub\Behavior_VarSeq')
from create_medpc_master import create_medpc_master

###############################################################################
# Data to be included: FR5, noLight, noForcedR, CATEG 
###############################################################################

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
dates=['20220213']
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
# Figure 1
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
colors=['tab:blue','tab:red', 'tab:orange', 'tab:green']
Anova_df=pd.DataFrame(columns=['feature','group', 'subject'])
fig,ax=plt.subplots(1,1,figsize=(8,5))
All_rewards=[]
All_protocols=[]
counter=0
for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
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
                print(mouse)
                print(date)
                
            mouse_protocols.append(date_df['Protocol'].values)
        while len(mouse_rewards)<30:
            mouse_rewards=np.append(mouse_rewards,float('nan'))
            
        day=[i for i,(a,b) in enumerate(zip(mouse_rewards[:-1], mouse_rewards[1:])) if (a>=48) & (b>=48) ]
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
        

        print(mouse)
        print(mouse_rewards)
        
        Group_rewards[m,:]=mouse_rewards
        Group_protocols.append(mouse_protocols)
    All_protocols.append(Group_protocols)
    All_rewards.append(Group_rewards)
        

    Cum=stats.cumfreq(Group_days_to_criteria, numbins=15, defaultreallimits=(0,15))
    x= np.linspace(0, Cum.binsize*Cum.cumcount.size, Cum.cumcount.size)
    plt.plot(x,Cum.cumcount/np.size(Group_days_to_criteria),color=colors[c],  linewidth=2) #color=cmap(Cmap_index[j])
plt.xlabel('Number of days to criteria', size=16)
plt.xticks([1,3,5,7,9,11,13,15],['2','4','6','8','10','12','14', '16'],fontsize=14)
plt.ylabel('Cumulative fraction', size=16)
plt.yticks(fontsize=14)
plt.legend(['noLight, N=8 mice', 'noForcedR, N=8 mice', 'FR5, N=15 mice', 'CATEG, N=16 mice'], loc='lower right')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
    

# #degrees of freedom
# N = len(Anova_df.feature)
# df_a = len(Anova_df.virus.unique()) - 1
# df_b = len(Anova_df.injection.unique()) - 1
# df_axb = df_a*df_b 
# df_w = N - (len(Anova_df.virus.unique())*len(Anova_df.injection.unique()))


# #Sum of squares
# grand_mean = Anova_df['feature'].mean()
# ssq_a = sum([(Anova_df[Anova_df.virus ==l].feature.mean()-grand_mean)**2 for l in Anova_df.virus])
# ssq_b = sum([(Anova_df[Anova_df.injection ==l].feature.mean()-grand_mean)**2 for l in Anova_df.injection])
# ssq_t = sum((Anova_df.feature - grand_mean)**2)

# #Sum of squares within
# dreadd = Anova_df[Anova_df.virus == 1]
# control = Anova_df[Anova_df.virus == 0]
# dreadd_injection_means = [dreadd[dreadd.injection == d].feature.mean() for d in dreadd.injection]
# control_injection_means = [control[control.injection == d].feature.mean() for d in control.injection]
# ssq_w = sum((control.feature - control_injection_means)**2) +sum((dreadd.feature - dreadd_injection_means)**2)

# #Sum of squares interaction
# ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w

# #Mean Square A
# ms_a = ssq_a/df_a
# #Mean Square B
# ms_b = ssq_b/df_b
# #Mean Square AxB
# ms_axb = ssq_axb/df_axb
# #Mean Square Within/Error/Residual
# ms_w = ssq_w/df_w

# #F-ratio
# f_a = ms_a/ms_w
# f_b = ms_b/ms_w
# f_axb = ms_axb/ms_w

# #p-values
# p_a = stats.f.sf(f_a, df_a, df_w)
# p_b = stats.f.sf(f_b, df_b, df_w)
# p_axb = stats.f.sf(f_axb, df_axb, df_w)


# #RESULTS
# results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
#            'df':[df_a, df_b, df_axb, df_w],
#            'F':[f_a, f_b, f_axb, 'NaN'],
#             'PR(&gt;F)':[p_a, p_b, p_axb, 'NaN']}
# columns=['sum_sq', 'df', 'F', 'PR(&gt;F)']
# aov_table1 = pd.DataFrame(results, columns=columns,
#                           index=['virus', 'injection', 
#                           'virus:injection', 'Residual'])

# def eta_squared(aov):
#     aov['eta_sq'] = 'NaN'
#     aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
#     return aov
# def omega_squared(aov):
#     mse = aov['sum_sq'][-1]/aov['df'][-1]
#     aov['omega_sq'] = 'NaN'
#     aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
#     return aov
# eta_squared(aov_table1)
# omega_squared(aov_table1)
# print(aov_table1)


import pingouin as pg
# Compute the two-way mixed-design ANOVA
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])
aov = pg.anova(dv='feature', between='group', data=Anova_df,
               detailed=True)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# =============
# ANOVA SUMMARY
# =============

# Source          SS    DF       MS        F    p-unc      np2
# --------  --------  ----  -------  -------  -------  -------
# group     1242.436     3  414.145    2.150    0.108    0.130
# Within    8283.308    43  192.635  nan      nan      nan


# Number of days on FR5
All_groups=[noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]
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

################################################################################
FR5_mice=[x for x in FR5_mice if x in np.unique(master_df['Mouse'])] #9
CATEG_mice=[x for x in CATEG_mice if x in np.unique(master_df['Mouse'])] #14
noLight_mice=[x for x in noLight_mice if x in np.unique(master_df['Mouse'])] #6
noForcedR_mice= [x for x in noForcedR_mice if x in np.unique(master_df['Mouse'])] #8
################################################################################

#Check overall performance
fig,ax=plt.subplots(1,1)
total_discarded=0
Total_reward_acquired=[]
Anova_df=pd.DataFrame(columns=['feature','group', 'subject'])
counter=0
colors=['tab:blue','tab:red', 'tab:orange', 'tab:green']
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

fig,ax=plt.subplots(1,1, figsize=(5,5))
for c,group in enumerate(Total_reward_acquired):
    sns.distplot(group, rug=True,hist=False, color=colors[c])
plt.xlabel('Total rewards acquired over 10 days', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Density', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,0.02)
plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

###############################################################################
# reward rate
###############################################################################
fig,ax=plt.subplots(1,1)
mice=np.unique(master_df['Mouse'])
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
All_rewards=[]
counter=0
for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
    test_mice=[x for x in mice if x in group]
    group_rewards=np.zeros((len(test_mice), 10))
    for j,mouse in enumerate(test_mice):
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
            
            if (i==0) | (i==9):
                Anova_df.at[counter,'feature']=len(date_df['Reward'].values[0]) / (date_df['Reward'].values[0][-1]/60)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1

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
plt.ylabel('Reward rate (#/min)', size=16)
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




#make ANOVA data plot
ax = sns.violinplot(x="group", y="feature", hue='time',inner='point', dodge=True, data=Anova_df)
#manually
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    
    #test
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
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
# t-tests for each group
# 0.22808885357730663
# 0.11082666368724045
# 0.011344422162947657
# 0.0024144925853634893

# One way ANOVA on difference
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.subtract(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])

fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    plt.scatter(i, mean0,  c=colors[j])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2,3], ['noLight', 'noRwdCol', 'FR5', 'Variance'],fontsize=14)
plt.ylabel(' DELTA Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.ylim(-1,4)
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
# group      9.017     3  3.006    4.282    0.012    0.280
# Within    23.165    33  0.702  nan      nan      nan
posthocs = pg.pairwise_tukey(dv='feature_diff', between='group',
                              data=new_Anova_df)
pg.print_table(posthocs)

# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green       0.202      0.348  -0.146  0.409  -0.358      0.900    -0.167
# tab:blue    tab:orange      0.202      1.500  -1.298  0.442  -2.940      0.029    -1.459
# tab:blue    tab:red         0.202      0.587  -0.386  0.452  -0.852      0.810    -0.431
# tab:green   tab:orange      0.348      1.500  -1.152  0.358  -3.219      0.015    -1.326
# tab:green   tab:red         0.348      0.587  -0.240  0.371  -0.645      0.900    -0.275
# tab:orange  tab:red         1.500      0.587   0.913  0.407   2.242      0.133     1.034

###############################################################################
# LP rate
###############################################################################
fig,ax=plt.subplots(1,1)
mice=np.unique(master_df['Mouse'])
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
All_rewards=[]
counter=0
for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
    test_mice=[x for x in mice if x in group]
    group_rewards=np.zeros((len(test_mice), 10))
    for j,mouse in enumerate(test_mice):
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
            if (i==0) | (i==9):
                Anova_df.at[counter,'feature']=len(date_df['Lever'].values[0]) / (date_df['Lever'].values[0][-1]/60)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1

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
plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

#make ANOVA data plot
ax = sns.violinplot(x="group", y="feature", hue='time',inner='point', dodge=True, data=Anova_df)
#manually
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    #test
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('Lever press rate (#/min)', size=16)
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
#ttest results
# 0.00813314131315286
# 0.024057059686878924
# 0.012822938760560737
# 0.0006387784411372044

#Compare LP rate on last day across groups
#manually
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature'].values, color=colors[j], alpha=0.3)

    mean0=np.mean(group_df['feature'])
    sem0=np.std(group_df['feature'])/np.sqrt(len(group_df)-1)
    plt.scatter(i, mean0,  c=colors[j])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2,3], ['noLight', 'noRwdCol', 'FR5', 'Variance'],fontsize=14)
plt.ylabel(' Lever press rate (#/sec)', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
# One way ANOVA on fold change
Anova_df['feature'] = pd.to_numeric(Anova_df['feature'])

aov = pg.anova(dv='feature', between='group', data=new_Anova_df,
               detailed=True)# Pretty printing of ANOVA summary
pg.print_table(aov)
#    Source           SS  DF          MS          F     p-unc      np2
# 0   group   547.306072   3  182.435357  11.378034  0.000004  0.32779
# 1  Within  1122.379783  70   16.033997        NaN       NaN      NaN

posthocs = pg.pairwise_tukey(dv='feature', between='group',
                              data=new_Anova_df)
pg.print_table(posthocs)
# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green       1.761      4.079  -2.318  1.382  -1.678      0.343    -0.567
# tab:blue    tab:orange      1.761      9.444  -7.683  1.492  -5.148      0.001    -1.867
# tab:blue    tab:red         1.761      3.344  -1.583  1.529  -1.035      0.706    -0.384
# tab:green   tab:orange      4.079      9.444  -5.365  1.210  -4.435      0.001    -1.317
# tab:green   tab:red         4.079      3.344   0.735  1.255   0.586      0.900     0.180
# tab:orange  tab:red         9.444      3.344   6.100  1.376   4.434      0.001     1.487

# One way ANOVA on fold change
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.subtract(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])

fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=new_Anova_df[new_Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.scatter(i,subject_df['feature_diff'].values, color=colors[j], alpha=0.3)

    mean0=np.mean(group_df['feature_diff'])
    sem0=np.std(group_df['feature_diff'])/np.sqrt(len(group_df)-1)
    plt.scatter(i, mean0,  c=colors[j])
    plt.vlines(i, mean0-sem0,mean0+sem0, color=colors[j])
    i+=1
plt.xlabel('Groups', size=16)
plt.xticks([0,1,2,3], ['noLight', 'noRwdCol', 'FR5', 'Variance'],fontsize=14)
plt.ylabel(' Fold change LPrate', size=16)
plt.yticks(fontsize=14)
plt.ylim(0,20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()

aov = pg.anova(dv='feature_diff', between='group', data=new_Anova_df,
               detailed=True)# Pretty printing of ANOVA summary
pg.print_table(aov)
# =============
# ANOVA SUMMARY
# =============
#
#    Source          SS  DF          MS         F     p-unc      np2
# 0   group  406.485040   3  135.495013  9.571501  0.000108  0.46528
# 1  Within  467.150932  33   14.156089       NaN       NaN      NaN
posthocs = pg.pairwise_tukey(dv='feature_diff', between='group',
                              data=new_Anova_df)
pg.print_table(posthocs)
# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green      10.099      3.050   7.049  1.836   3.839      0.003     1.794
# tab:blue    tab:orange     10.099      2.785   7.314  1.983   3.688      0.004     1.830
# tab:blue    tab:red        10.099      9.520   0.579  2.032   0.285      0.900     0.144
# tab:green   tab:orange      3.050      2.785   0.266  1.607   0.165      0.900     0.068
# tab:green   tab:red         3.050      9.520  -6.469  1.668  -3.879      0.003    -1.654
# tab:orange  tab:red         2.785      9.520  -6.735  1.828  -3.684      0.004    -1.699


#results for difference
# =============
# ANOVA SUMMARY
# =============

# Source         SS    DF      MS        F    p-unc      np2
# --------  -------  ----  ------  -------  -------  -------
# group     119.557     3  39.852    1.866    0.155    0.145
# Within    704.848    33  21.359  nan      nan      nan
###############################################################################
# Figure 2
###############################################################################

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


#Plot within IPI
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
fig,ax=plt.subplots(1,1,figsize=(8,5))
for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
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

# Compute the two-way mixed-design ANOVA
new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])

aov = pg.anova(dv='feature_diff',  between='group',  data=new_Anova_df)
# Pretty printing of ANOVA summary
pg.print_table(aov)

# =============
# ANOVA SUMMARY
# =============
#   Source  ddof1  ddof2          F     p-unc       np2
# 0  group      3     33  15.805969  0.000002  0.589644

posthocs = pg.pairwise_tukey(dv='feature_diff', between='group', data=new_Anova_df)
pg.print_table(posthocs)

# ==============
# POST HOC TESTS
# ==============

# A           B             mean(A)    mean(B)    diff     se       T    p-tukey    hedges
# ----------  ----------  ---------  ---------  ------  -----  ------  ---------  --------
# tab:blue    tab:green       1.137      0.168   0.969  0.144   6.737      0.001     3.149
# tab:blue    tab:orange      1.137      0.316   0.822  0.155   5.287      0.001     2.623
# tab:blue    tab:red         1.137      0.326   0.811  0.159   5.095      0.001     2.576
# tab:green   tab:orange      0.168      0.316  -0.148  0.126  -1.172      0.633    -0.483
# tab:green   tab:red         0.168      0.326  -0.158  0.131  -1.209      0.613    -0.515
# tab:orange  tab:red         0.316      0.326  -0.010  0.143  -0.072      0.900    -0.033


#make ANOVA data plot
ax = sns.violinplot(x="group", y="feature", hue='time',inner='point', dodge=True, data=Anova_df)
#manually
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
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
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('IntraIPI (s)', size=16)
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
#ttest
# 0.46346253629259143
# 0.03463462175099151
# 0.0006284894913309088
# 0.0007543134413713983

#########################
#Same, but plotting DELTA
#########################
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
fig,ax=plt.subplots(1,1,figsize=(8,5))
for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
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
plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()


#make ANOVA data plot
ax = sns.violinplot(x="group", y="feature", hue='time',inner='point', dodge=True, data=Anova_df)
#manually
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
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
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('InterIPI / IntraIPI (s)', size=16)
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
#ttest
# 0.7134422191208979
# 0.003554854251798505
# 0.0021597527236944295
# 0.00167249499727292

new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.divide(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
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


################################################################################
# Delay to reward collection? TBD
###############################################################################

################################################################################
# Figure 3
###############################################################################

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
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values][::-1])
    plt.xlabel('Sessions (#)')
    plt.colorbar()


###############################################################################
# within sequence IPI variance (plot - accounts for mouse)
###############################################################################
Anova_df=pd.DataFrame(columns=['feature','group', 'subject', 'time'])
counter=0
fig,ax=plt.subplots(1,1,figsize=(6,12))
plt.sca(ax)

for c,group in enumerate([noLight_mice, noForcedR_mice, FR5_mice, CATEG_mice]):
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
            mouse_variances.append(day_variances)
            
            if (i==0) | (i==9):
                Anova_df.at[counter,'feature']= np.mean(day_variances)
                Anova_df.at[counter,'group']=colors[c]
                Anova_df.at[counter,'subject']=mouse
                Anova_df.at[counter,'time']=i
                counter+=1
                
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
plt.legend(['noLight, N=6 mice', 'noForcedR, N=8 mice', 'FR5, N=9 mice', 'CATEG, N=14 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color(colors[0])
leg.legendHandles[1].set_color(colors[1])
leg.legendHandles[2].set_color(colors[2])
leg.legendHandles[3].set_color(colors[3])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.ylim(1,150)
    


#make ANOVA data plot
ax = sns.violinplot(x="group", y="feature", hue='time',inner='point', dodge=True, data=Anova_df)
#manually
fig,ax=plt.subplots(1,1)
i=0
for j,group in enumerate(['tab:blue','tab:red','tab:orange', 'tab:green' ]):
    group_df=Anova_df[Anova_df['group']==group]
    for subject in np.unique(group_df['subject']):
        subject_df=group_df[group_df['subject']==subject]
        plt.plot([1+i,2+i],subject_df['feature'].values, color=colors[j], alpha=0.3)
    s,p=sp.stats.ttest_rel(group_df[group_df['time']==0]['feature'], group_df[group_df['time']==9]['feature'])
    print(p)
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean0=np.mean(group_df[group_df['time']==0]['feature'])
    mean9=np.mean(group_df[group_df['time']==9]['feature'])
    sem0=np.std(group_df[group_df['time']==0]['feature'])/np.sqrt(len(group_df)-1)
    sem9=np.std(group_df[group_df['time']==9]['feature'])/np.sqrt(len(group_df)-1)
    plt.plot([1+i,2+i], [mean0, mean9],  color=colors[j])
    plt.scatter([1+i,2+i], [mean0, mean9],  c=colors[j])
    plt.vlines([1+i,2+i], [mean0-sem0, mean9-sem9],[mean0+sem0, mean9+sem9], color=colors[j])
    i+=2
plt.xlabel('Groups', size=16)
plt.xticks([1,2,3,4,5,6,7,8], ['First','Last', 'First','Last', 'First','Last', 'First','Last'],fontsize=14)
plt.ylabel('Variance', size=16)
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
#ttest
# 0.1967182944234653
# 0.7228736090906663
# 0.03121847148172749
# 0.015432206129699371


new_Anova_df=Anova_df[Anova_df['time']==9].copy()
new_Anova_df['feature_diff']= np.subtract(Anova_df[Anova_df['time']==9]['feature'].values, Anova_df[Anova_df['time']==0]['feature'].values)
new_Anova_df['feature_diff'] = pd.to_numeric(new_Anova_df['feature_diff'])
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:blue']
new_Anova_df=new_Anova_df[new_Anova_df['group']!='tab:red']

# aov = pg.anova(dv='feature_diff',  between='group',  data=new_Anova_df)
# # Pretty printing of ANOVA summary
# pg.print_table(aov)

posthocs = pg.pairwise_ttests(dv='feature_diff',  between='group', data=new_Anova_df)
pg.print_table(posthocs)

# Contrast    A          B           Paired    Parametric         T     dof  Tail         p-unc    BF10    hedges
# ----------  ---------  ----------  --------  ------------  ------  ------  ---------  -------  ------  --------
# group       tab:green  tab:orange  False     True          -1.669  16.514  two-sided    0.114   1.012    -0.569


###############################################################################
# Figure 4: magnet?
###############################################################################

###############################################################################
# Figure 5: sequential
###############################################################################

###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
# ON MICE THAT DID BOTH FR/Var FOLLOWED BY CATEG
###############################################################################
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

###############################################################################
# within sequence IPI variance (heatmaps - all mice treated equal)
###############################################################################
All_mouse_heatmap_seq=np.zeros((70,18))
All_seq_LPrates_FR5=[]
All_seq_LPrates_CATEG=[]
counter=0
for j,mouse in enumerate(np.unique(master_df2['Mouse'])):#enumerate(High_resp_CATEG)   np.unique(master_df['Mouse'])
    mouse_protocols=[]
    mouse_df=master_df2[master_df2['Mouse']==mouse]
    mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
    mouse_heatmap_seq=np.zeros((70,18))
    seq_day_LPrates_FR5=[]
    seq_day_LPrates_CATEG=[]

    #first grab the last 10 days of FR5 for each mouse
    protocol_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarget_FR5']

    for i,date in enumerate(np.unique(protocol_df['Date'])[:10]):
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
       
            
        seq_day_LPrates_FR5.append(np.log10(day_variances))
        if len(seq_day_LPrates_FR5[0])<2:
            seq_data=np.zeros((1,100))
        else:
            seq_data,edges=np.histogram(np.log10(day_variances), bins=70, range=(-3,4), density=True)
        mouse_heatmap_seq[:,i]=seq_data[::-1]
        
     #second grab the CATEG data
    protocol_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']

    for k,date in enumerate(np.unique(protocol_df['Date'])[:7]):
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
        
             
         seq_day_LPrates_CATEG.append(np.log10(day_variances))
         if len(seq_day_LPrates_CATEG[0])<2:
             seq_data=np.zeros((1,100))
         else:
             seq_data,edges=np.histogram(np.log10(day_variances), bins=70, range=(-3,4), density=True)
         mouse_heatmap_seq[:,i+k+2]=seq_data[::-1]
        
  
    All_mouse_heatmap_seq=np.add(All_mouse_heatmap_seq,mouse_heatmap_seq)
    All_seq_LPrates_FR5.append(seq_day_LPrates_FR5)
    All_seq_LPrates_CATEG.append(seq_day_LPrates_CATEG)
    counter+=1
    
    fig,ax=plt.subplots(1,1, figsize=(5,10))
    plt.imshow(mouse_heatmap_seq, alpha=0.5, cmap='jet')
    plt.plot([70-(x*10+30) for x in [np.median(x) for x in seq_day_LPrates_FR5+seq_day_LPrates_CATEG]], color='r') #10=40(bins)/(3-(-1)) (range) +10 (origin=-1) (histogram adjustements)
    plt.title(str(mouse)+np.unique(mouse_df['Protocol'])[-1])
    plt.ylabel('IPI (s)')
    log_values=[float(x) for x in edges[[0,10,20,30,40]]]
    plt.yticks([0,10,20,30,40],[str(10**x) for x in log_values][::-1])
    plt.xlabel('Sessions (#)')
       


###############################################################################
# plot variance starting with first FR5/Va5 and into CATEG
###############################################################################

fig,ax=plt.subplots(1,1,figsize=(5,5))
plt.sca(ax)
All_Variance=np.empty((len(np.unique(master_df2['Mouse'])), 10))
for i,mouse in enumerate(np.unique(master_df2['Mouse'])):
    mouse_df = master_df2[master_df2['Mouse']==mouse].reset_index()
    protocol_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarget_FR5']
    Mean_variance_across_days=[]
    for j,date in enumerate(np.unique(protocol_df['Date'])[:10]):
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
        Mean_variance_across_days.append(np.median(day_variances))
  
    All_Variance[i,:]=Mean_variance_across_days
    #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
    plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.3)
plt.yscale('log')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xticks([0,4,9],['1','5','10'],  size=16)
plt.xlabel('Time on FR5 schedule (days)', size=20)
plt.ylabel('Median within sequence \n inter-press interval', size=20)
plt.title(str(len(mice)) + ' mice')


mean=np.nanmean(All_Variance, axis=0)
std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
plt.plot(mean, linewidth=3, color='cornflowerblue')
plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color='cornflowerblue', linewidth=3)
plt.ylim(0.01,100)

fig,ax=plt.subplots(1,1,figsize=(5,5))
plt.sca(ax)
All_Variance=np.empty((len(np.unique(master_df2['Mouse'])), 7))
for i,mouse in enumerate(np.unique(master_df2['Mouse'])):
    mouse_df = master_df2[master_df2['Mouse']==mouse].reset_index()
    protocol_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
    Mean_variance_across_days=[]
    for j,date in enumerate(np.unique(protocol_df['Date'])[:7]):
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
        Mean_variance_across_days.append(np.median(day_variances))
  
    All_Variance[i,:]=Mean_variance_across_days
    #plt.scatter(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.5)
    plt.plot(np.arange(len(Mean_variance_across_days)), Mean_variance_across_days, c='cornflowerblue',alpha=0.3)
plt.yscale('log')  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
plt.xticks([0,4,9],['1','5','10'],  size=16)
plt.xlabel('Time on FR5 schedule (days)', size=20)
plt.ylabel('Median within sequence \n inter-press interval', size=20)
plt.title(str(len(mice)) + ' mice')


mean=np.nanmean(All_Variance, axis=0)
std=np.nanstd(All_Variance, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_Variance[:,i]]) for i in range(np.shape(All_Variance)[1])] )
plt.plot(mean, linewidth=3, color='cornflowerblue')
plt.vlines(range(np.shape(All_Variance)[1]), mean-std, mean+std, color='cornflowerblue', linewidth=3)
plt.ylim(0.01,100)
###############################################################################
# Example rasters of LPs
###############################################################################
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
        plt.figure()
        plt.hist(Presses[:,1], bins=30, range=Range, color='b', alpha=0.5)
        plt.hist(Presses[:,2], bins=30, range=Range, color='y', alpha=0.5)
        plt.hist(Presses[:,3], bins=30, range=Range, color='r', alpha=0.5)
        plt.hist(Presses[:,4], bins=30, range=Range, color='g', alpha=0.5)    