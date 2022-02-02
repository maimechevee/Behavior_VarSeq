import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from create_medpc_master import create_medpc_master
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice=[4219, 4224, 4225, 4226, 4222, 4230, 4231, 4239, 4234, 4240, 4241, 4229]
dates = ['20220125', '20220126', '20220127', '20220128', '20220130']
master_df = create_medpc_master(mice,dates)

num_days_training = len(np.unique(master_df['Date']))
mice = np.unique(master_df['Mouse']) 

# -----------------------------------------------------------------------------
def rolling_avg(arr, delay):
    roll_avg = [np.mean(arr[i-delay:i+delay+1]) for i in range(delay,len(arr)-delay)]
    return roll_avg

FR5_by_mouse = [] # Each List contains all variance data for each mouse
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
    variance_aggregate = [] # Contains all variance data as a single list
    for sublist in temp:
        if np.size(sublist) > 1:
            for item in sublist:
                variance_aggregate.append(item)
    roll_avg = rolling_avg(variance_aggregate, ind_delay)
    plt.plot([i + ind_delay for i in range(len(roll_avg))],
             roll_avg, linewidth = 1, color = 'tomato')
    FR5_by_mouse.append(variance_aggregate)

plt.title('Rolling Average of Variance (Individual)')
plt.xlabel('Total Rewards since first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Variance', size=16)
plt.yticks(fontsize=14)
leg = ax.get_legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Calculate Means
max_rewards = np.max([len(mouse) for mouse in FR5_by_mouse])
FR5_by_reward = [[] for i in range(max_rewards)]

for i in range(max_rewards):
    for j in range(len(mice)):
        try:
            FR5_by_reward[i].append(FR5_by_mouse[j][i])
        except IndexError:
            pass

FR5_means = [np.mean(day) for day in FR5_by_reward if day]

# Plot Mean of Means
group_delay = 50
fig,ax=plt.subplots(1,1)
plt.plot([i + group_delay for i in range(len(FR5_means) - 2 * group_delay)], 
         rolling_avg(FR5_means,group_delay), linewidth=2, color='tomato')

plt.xlabel('Total Rewards since first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Variance', size=16)
plt.title('Rolling Average of Variance (Means)')
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()
