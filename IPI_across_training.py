import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from create_medpc_master import create_medpc_master
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice=[4217,4218,4219,4220,4221,4222,4223,4224,4225,4226,4227,4228,
      4229,4230,4231,4232,4233,4234,4235,4236,4237,4238,4239,4240,4241,4242,4243, 4244] #(ints)
dates=['20211202', '20211203', '20211204', '20211205', '20211206', '20211207', '20211208',
       '20211209', '20211210', '20211211', '20211212', '20211213', '20211214', '20211215'] #(strs)
master_df = create_medpc_master(mice,dates)

# Had a problem with extracting data from '2021-12-12_16h47m_Subject 4241.txt'

# Taken from Reward_across_training
discard_list = [4217, 4218, 4221, 4227, 4232, 4235, 4236, 4237, 4238, 4244]     
indices=[]
for mouse in discard_list:
    indices.append(master_df[master_df['Mouse']==mouse].index)
indices=[x for l in indices for x in l]
master_df=master_df.drop(indices, axis=0)
num_days_training = len(np.unique(master_df['Date']))
mice = np.unique(master_df['Mouse'])  

# Contains median session IPI for each mouse/day combo   
IPI_FR5 = [[] for i in range(num_days_training - 2)]
IPI_FR5var = [[] for i in range(num_days_training - 2)]

fig,ax=plt.subplots(1,1)

for i in range(len(mice)):
    curr_mouse = mice[i]
    curr_mouse_df = (master_df[master_df['Mouse']==curr_mouse]).reset_index()
    exclude_FR1 = [curr_mouse_df['IPI'][ind] is not None for ind in range(len(curr_mouse_df))]
    curr_mouse_df = (curr_mouse_df[exclude_FR1]).reset_index()
    curr_mouse_IPI = curr_mouse_df['IPI']
    mouse_group = curr_mouse_df.iloc[-1]['Protocol'][41:] #Grab last day protocol to put mouse in group
    if 'va2' not in mouse_group:
            plt.plot([np.mean(day) for day in curr_mouse_IPI], linewidth = 1, linestyle = '--', color = 'tomato')
    else:
        plt.plot([np.mean(day) for day in curr_mouse_IPI], linewidth = 1, linestyle = '--', color = 'cornflowerblue')
    for j in range(curr_mouse_IPI.size):
        curr_IPI = curr_mouse_IPI[j]
        # Can't find file for 4234 on the 13th -> so check if nan
        if 'va2' not in mouse_group and not math.isnan(np.median(curr_IPI)):
            IPI_FR5[j].append(np.median(curr_IPI))
        elif not math.isnan(np.median(curr_IPI)):
            IPI_FR5var[j].append(np.median(curr_IPI))
        

plt.vlines(3.5,0,10, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Individual session IPI medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Take mean of median and median of median
IPI_means_FR5 = [np.mean(day) for day in IPI_FR5 if day]
IPI_means_FR5var = [np.mean(day) for day in IPI_FR5var]
IPI_med_FR5 = [np.median(day) for day in IPI_FR5 if day]
IPI_med_FR5var = [np.median(day) for day in IPI_FR5var]
IPI_sem_FR5 = [(np.std(day))/np.sqrt(len(day)) for day in IPI_FR5 if day]
IPI_sem_FR5var = [(np.std(day))/np.sqrt(len(day)) for day in IPI_FR5var if day]

# Set up sem lists
FR5_lower = [a-b for a,b in zip(IPI_means_FR5,IPI_sem_FR5)]
FR5_upper = [a+b for a,b in zip(IPI_means_FR5,IPI_sem_FR5)]
FR5var_lower = [a-b for a,b in zip(IPI_means_FR5var,IPI_sem_FR5var)]
FR5var_upper = [a+b for a,b in zip(IPI_means_FR5var,IPI_sem_FR5var)]

# Plot Mean of Medians

fig,ax=plt.subplots(1,1)
plt.plot(IPI_means_FR5, linewidth=2, color='tomato')
plt.vlines(range(len(IPI_means_FR5var)), FR5_lower, FR5_upper, linewidths=2, colors='tomato') 
plt.plot(IPI_means_FR5var, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(IPI_means_FR5var)), FR5var_lower, FR5var_upper, linewidths=2, colors='cornflowerblue') 

plt.vlines(3.5,0,5, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Means of session IPI medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot Median of Medians
fig,ax=plt.subplots(1,1)
plt.plot(range(len(IPI_means_FR5)), IPI_med_FR5, linewidth=2, color='tomato')
# plt.vlines(range(len(IPI_means_FR5)), FR5_lower, FR5_upper, linewidths=2, colors='tomato') 
plt.plot(IPI_med_FR5var, linewidth=2, color='cornflowerblue')
# plt.vlines(range(len(IPI_means_FR5var)), FR5var_lower, FR5var_upper, linewidths=2, colors='cornflowerblue') 

plt.vlines(3.5,0,5, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Median of session IPI medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()



        
        
        





