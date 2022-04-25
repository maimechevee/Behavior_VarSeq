

#################################################################################
# can we combine groups?
#################################################################################

###############################################################################
# reward/LP rate
###############################################################################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
All_rewards_FR5CATEG=[]
for group in [FR5_mice,VAR_mice]:
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
                plt.plot(mouse_rewards, linestyle='dotted', color='tomato')
                All_rewards_FR5.append(mouse_rewards)
        else:
            plt.plot(mouse_rewards, linestyle='dotted', color='grey')
            All_rewards_Var5.append(mouse_rewards)

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
meanVar5=[]
semVar5=[]
while All_rewards_Var5:
    step_values=[x[step] for x in All_rewards_Var5]
    step_length=len(step_values)
    meanVar5.append(np.nanmean(step_values))
    semVar5.append(np.nanstd(step_values)/np.sqrt(step_length))
    All_rewards_Var5=[x[1:] for x in All_rewards_Var5]
    All_rewards_Var5=[x for x in All_rewards_Var5 if np.nansum(x)>0]


    
plt.plot(meanFR5, linewidth=2, color='tomato')
plt.vlines(range(len(meanFR5)), [a-b for a,b in zip(meanFR5,semFR5)], [a+b for a,b in zip(meanFR5,semFR5)], colors='tomato', linewidths=2) 
plt.plot(meanVar5, linewidth=2, color='grey')
plt.vlines(range(len(meanVar5)), [a-b for a,b in zip(meanVar5,semVar5)], [a+b for a,b in zip(meanVar5,semVar5)], colors='grey', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('LP rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5_mice]))+' mice', 
            'Var5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in VAR_mice]))+' mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('tomato')
leg.legendHandles[1].set_color('grey')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

###############################
fig,ax=plt.subplots(1,1)

All_rewards_FR5=[]
All_rewards_Var5=[]
All_rewards_CATEG=[]
All_rewards_FR5CATEG=[]
for group in [CATEG_mice, FR5CATEG_mice]:
    mice=group
    for j,mouse in enumerate(mice):
        # if mouse in Females:
        #     continue
        mouse_protocols=[]
        mouse_df=master_df[master_df['Mouse']==mouse]
        mouse_df=mouse_df[mouse_df['Protocol']!='MC_magbase_ForcedReward_LongWinVarTarget_FR1'] #do not count the FR1 early days
        mouse_rewards=[]
        
        #important line: only include CATEG days
        mouse_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
        
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
        if mouse in CATEG_mice:
                plt.plot(mouse_rewards, linestyle='dotted', color='cornflowerblue')
                All_rewards_CATEG.append(mouse_rewards)
        elif mouse in FR5CATEG_mice:
            plt.plot(mouse_rewards, linestyle='dotted', color='grey')
            All_rewards_FR5CATEG.append(mouse_rewards)

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

plt.plot(meanCATEG, linewidth=2, color='cornflowerblue')
plt.vlines(range(len(meanCATEG)), [a-b for a,b in zip(meanCATEG,semCATEG)], [a+b for a,b in zip(meanCATEG,semCATEG)], colors='cornflowerblue', linewidths=2) 
plt.plot(meanFR5CATEG, linewidth=2, color='k')
plt.vlines(range(len(meanFR5CATEG)), [a-b for a,b in zip(meanFR5CATEG,semFR5CATEG)], [a+b for a,b in zip(meanFR5CATEG,semFR5CATEG)], colors='k', linewidths=2) 

#plt.vlines(3.5,0,6, color='k', linestyle='dashed')
plt.xlabel('Time from first FR5 session (day)', size=16)
plt.xticks(fontsize=14)
plt.ylabel('Reward rate (#/min)', size=16)
plt.yticks(fontsize=14)
plt.legend(['FR5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in CATEG_mice]))+' mice', 
            'Var5, N='+str(len([x for x in np.unique(master_df['Mouse']) if x in FR5CATEG_mice]))+' mice'], loc='upper left')
leg = ax.get_legend()
leg.legendHandles[0].set_color('cornflowerblue')
leg.legendHandles[1].set_color('grey')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

################################################################################
# IPI
################################################################################
fig,ax=plt.subplots(1,1,figsize=(6,12))
for m,group in enumerate([FR5_mice, VAR_mice]):
    if m==0:
        colors=['orange', 'mediumpurple']
    else:
        colors=['grey', 'k']
    mice=[x for x in np.unique(master_df['Mouse']) if x in group]
    
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
        # if mouse not in FR5_mice+ VAR_mice:
        #     mouse_df=mouse_df[mouse_df['Protocol']=='MC_magbase_ForcedReward_LongWinVarTarCATEG_FR5']
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
        plt.plot(np.arange(len(Median_SeqIPIs_across_days)), Median_SeqIPIs_across_days, c=colors[0],alpha=0.3)
        plt.plot(np.arange(len(Median_RestIPIs_across_days)), Median_RestIPIs_across_days, c=colors[1],alpha=0.3)
        # if mouse not in FR5_mice+VAR_mice:
        #     plt.plot(np.arange(len(Median_InterfailedIPIs_across_days)), Median_InterfailedIPIs_across_days, c='b',alpha=0.3)
    plt.yscale('log')  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.xticks([0,4,9],['1','5','10'],  size=16)
    plt.xlabel('Time on FR5 schedule (days)', size=20)
    plt.ylabel('Median inter-press interval', size=20)
    plt.title(str(len(mice)) + ' mice')
    
    
    mean=np.nanmean(All_SeqIPIs, axis=0)
    std=np.nanstd(All_SeqIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_SeqIPIs[:,i]]) for i in range(np.shape(All_SeqIPIs)[1])] )
    plt.plot(mean, linewidth=3, color=colors[0])
    plt.vlines(range(np.shape(All_SeqIPIs)[1]), mean-std, mean+std, color=colors[0], linewidth=3)
    
    
    mean=np.nanmean(All_RestIPIs, axis=0)
    std=np.nanstd(All_RestIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_RestIPIs[:,i]]) for i in range(np.shape(All_RestIPIs)[1])] )
    plt.plot(mean, linewidth=3, color=colors[1])
    plt.vlines(range(np.shape(All_RestIPIs)[1]), mean-std, mean+std, color=colors[1], linewidth=3)
    
    # if mouse not in FR5_mice:
    #     mean=np.nanmean(All_InterFailedIPIs, axis=0)
    #     std=np.nanstd(All_InterFailedIPIs, axis=0)/np.sqrt([np.sum([not math.isnan(x) for x in All_InterFailedIPIs[:,i]]) for i in range(np.shape(All_InterFailedIPIs)[1])] )
    #     plt.plot(mean, linewidth=3, color='b')
    #     plt.vlines(range(np.shape(All_InterFailedIPIs)[1]), mean-std, mean+std, color='b', linewidth=3)
    
    plt.yscale('log') 