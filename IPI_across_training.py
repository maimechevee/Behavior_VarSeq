import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from create_medpc_master import create_medpc_master
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mice=[4219, 4224, 4225, 4226, 4222, 4230, 4231, 4239, 4234, 4240, 4241, 4229]
dates = ['20220125', '20220126', '20220127', '20220128', '20220130']
master_df = create_medpc_master(mice,dates)

# Contains median session IPI for each mouse/day combo   
IPI = [[] for i in range(len(dates))]

fig,ax=plt.subplots(1,1)

for curr_mouse in mice:
    curr_mouse_df = (master_df[master_df['Mouse']==curr_mouse]).reset_index()
    exclude_FR1 = [curr_mouse_df['IPI'][ind] is not None for ind in range(len(curr_mouse_df))]
    curr_mouse_df = (curr_mouse_df[exclude_FR1]).reset_index()
    curr_mouse_IPI = curr_mouse_df['IPI']
    plt.plot([np.median(day) for day in curr_mouse_IPI], linewidth = 1, linestyle = '--', color = 'tomato')
    for j in range(curr_mouse_IPI.size):
        curr_IPI = curr_mouse_IPI[j]
        if not math.isnan(np.median(curr_IPI)):
            IPI[j].append(np.median(curr_IPI))
    
plt.yscale('log')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Individual session IPI medians', size=16)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Take mean of median and median of median
IPI_means = [np.mean(day) for day in IPI if day]
IPI_med = [np.median(day) for day in IPI if day]
IPI_sem = [(np.std(day))/np.sqrt(len(day)) for day in IPI if day]

# Set up sem lists
lower = [a-b for a,b in zip(IPI_means,IPI_sem)]
upper = [a+b for a,b in zip(IPI_means,IPI_sem)]

# Plot Mean of Medians

fig,ax=plt.subplots(1,1)
plt.plot(IPI_means, linewidth=2, color='tomato')
plt.vlines(range(len(IPI_means)), lower, upper, linewidths=2, colors='tomato') 

plt.xticks(fontsize=14)
plt.ylabel('Means of session IPI medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yscale('log')


# Plot Median of Medians
fig,ax=plt.subplots(1,1)
plt.plot(range(len(IPI_means)), IPI_med, linewidth=2, color='tomato')
# plt.vlines(range(len(IPI_means)), lower, upper, linewidths=2, colors='tomato') 

plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Median of session IPI medians', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N=10 mice', 'Var, N=8 mice'], loc='upper right')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylim(0,1.3)
plt.show()



        
        
        





