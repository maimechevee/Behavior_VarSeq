import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from create_medpc_master import create_medpc_master
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice = [i for i in range(4386, 4414)]
file_dir = ('/Users/emma-fuze-grace/Lab/Behavior_VarSeq'
            '/2022-02_TarVar_Categ_01/2022-02_TarVar_Categ_01_data')
master_df = create_medpc_master(mice, file_dir)
num_days_training = len(np.unique(master_df['Date']))
mice = np.unique(master_df['Mouse'])  

# Contains median session variance for each mouse/day combo   
Variance_FR5 = [[] for i in range(num_days_training - 2)]
Variance_FR5var = [[] for i in range(num_days_training - 2)]

fig,ax=plt.subplots(1,1)

for i in range(len(mice)):
    curr_mouse = mice[i]
    curr_mouse_df = (master_df[master_df['Mouse']==curr_mouse]).reset_index()
    exclude_FR1 = [curr_mouse_df['Variance'][ind] is not None for ind in range(len(curr_mouse_df))]
    curr_mouse_df = (curr_mouse_df[exclude_FR1]).reset_index()
    curr_mouse_variance = curr_mouse_df['Variance']
    mouse_group = curr_mouse_df.iloc[-1]['Protocol'][41:] #Grab last day protocol to put mouse in group
    if 'va2' not in mouse_group:
        plt.plot([np.median(day) for day in curr_mouse_variance], linewidth = 2, linestyle = '--', color = 'tomato')
    else:
        plt.plot([np.median(day) for day in curr_mouse_variance], linewidth = 2, linestyle = '--', color = 'cornflowerblue')
    for j in range(curr_mouse_variance.size):
        curr_variance = curr_mouse_variance[j]
        # Can't find file for 4234 on the 13th -> so check if nan
        if 'va2' not in mouse_group and not math.isnan(np.median(curr_variance)):
            Variance_FR5[j].append(np.median(curr_variance))
        elif not math.isnan(np.median(curr_variance)):
            Variance_FR5var[j].append(np.median(curr_variance))
        

plt.vlines(3.5,0,300, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Individual session variance medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Take mean of median and median of median
Variance_means_FR5 = [np.mean(day) for day in Variance_FR5 if day]
Variance_means_FR5var = [np.mean(day) for day in Variance_FR5var if day]
Variance_med_FR5 = [np.median(day) for day in Variance_FR5 if day]
Variance_med_FR5var = [np.median(day) for day in Variance_FR5var if day]
Variance_sem_FR5 = [(np.std(day))/np.sqrt(len(day)) for day in Variance_FR5 if day]
Variance_sem_FR5var = [(np.std(day))/np.sqrt(len(day)) for day in Variance_FR5var if day]

# Set up sem lists
FR5_lower = [a-b for a,b in zip(Variance_means_FR5,Variance_sem_FR5)]
FR5_upper = [a+b for a,b in zip(Variance_means_FR5,Variance_sem_FR5)]
FR5var_lower = [a-b for a,b in zip(Variance_means_FR5var,Variance_sem_FR5var)]
FR5var_upper = [a+b for a,b in zip(Variance_means_FR5var,Variance_sem_FR5var)]

# Plot Mean of Medians

fig,ax=plt.subplots(1,1)
plt.plot(Variance_means_FR5, linewidth=2, color='tomato')
# plt.vlines(range(len(Variance_means_FR5var)), FR5_lower, FR5_upper, linewidths=2, colors='tomato') 
plt.plot(Variance_means_FR5var, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(Variance_means_FR5var)), FR5var_lower, FR5var_upper, linewidths=2, colors='cornflowerblue') 

plt.vlines(3.5,0,50, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Means of session variance means', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot Median of Medians
fig,ax=plt.subplots(1,1)
plt.plot(Variance_med_FR5, linewidth=2, color='tomato')
# plt.vlines(range(len(Variance_means_FR5)), FR5_lower, FR5_upper, linewidths=2, colors='tomato') 
plt.plot(Variance_med_FR5var, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(Variance_means_FR5var)), FR5var_lower, FR5var_upper, linewidths=2, colors='cornflowerblue') 

plt.vlines(3.5,0,30, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Median of session variance medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()



        
        
        





