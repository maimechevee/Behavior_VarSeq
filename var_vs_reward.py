import matplotlib.pyplot as plt
import numpy as np
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

# -----------------------------------------------------------------------------
def rolling_avg(arr, delay):
    roll_avg = [np.mean(arr[i-delay:i+delay+1]) for i in range(delay,len(arr)-delay)]
    return roll_avg

FR5_by_mouse = []
FR5var_by_mouse = []

# Individual lines
ind_delay = 50
fig,ax = plt.subplots(1,1)
for i in range(len(mice)):
    curr_mouse = mice[i]
    curr_mouse_df = (master_df[master_df['Mouse']==curr_mouse]).reset_index()
    exclude_FR1 = [curr_mouse_df['Variance'][ind] is not None for ind in range(len(curr_mouse_df))]
    curr_mouse_df = (curr_mouse_df[exclude_FR1]).reset_index()
    curr_mouse_variance = (curr_mouse_df['Variance']).reset_index()
    temp = [curr_mouse_variance.iloc[j]['Variance'] for j in range(len(curr_mouse_variance))]
    variance_aggregate = []
    for sublist in temp:
        if np.size(sublist) > 1:
            for item in sublist:
                variance_aggregate.append(item)
    mouse_group = curr_mouse_df.iloc[-1]['Protocol'][41:]
    variance_aggregate = variance_aggregate[1:] # Just realized all of these vectors start with 0 
    roll_avg = rolling_avg(variance_aggregate, ind_delay)
    if 'va2' not in mouse_group:
        plt.plot([i + ind_delay for i in range(len(roll_avg))],
                 roll_avg, linewidth = 1, color = 'tomato')
        FR5_by_mouse.append(variance_aggregate)
    else:
        plt.plot([i + ind_delay for i in range(len(roll_avg))],
                 roll_avg, linewidth = 1, color = 'cornflowerblue')
        FR5var_by_mouse.append(variance_aggregate)

plt.title('Rolling Average of Variance (Individual)')
plt.xlabel('Total Rewards since first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Variance', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Calculate Means
max_rewards = np.max([len(mouse) for mouse in FR5_by_mouse])
FR5_by_reward = [[] for i in range(max_rewards)]
FR5var_by_reward = [[] for i in range(max_rewards)]

for i in range(max_rewards):
    for j in range(len(mice)):
        try:
            FR5_by_reward[i].append(FR5_by_mouse[j][i])
            FR5var_by_reward[i].append(FR5var_by_mouse[j][i])
        except IndexError:
            pass

FR5_means = [np.mean(day) for day in FR5_by_reward if day]
FR5var_means = [np.mean(day) for day in FR5var_by_reward if day]

# Plot Mean of Medians
group_delay = 50
fig,ax=plt.subplots(1,1)
plt.plot([i + group_delay for i in range(len(FR5_means) - 2 * group_delay)], 
         rolling_avg(FR5_means,group_delay), linewidth=2, color='tomato')
plt.plot([i + group_delay for i in range(len(FR5var_means) - 2 * group_delay)], 
         rolling_avg(FR5var_means,group_delay), linewidth=2, color='cornflowerblue')

plt.xlabel('Total Rewards since first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Variance', size=16)
plt.title('Rolling Average of Variance (Means)')
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('cornflowerblue')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()
