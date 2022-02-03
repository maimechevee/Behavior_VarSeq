import pandas as pd
import numpy as np
import os
import math
import datetime

# Returns dictionary with Protocol, Run Time, Reward, Lever, Lick, IPI, Variance
# Added run time
def medpc_extract(file):
    # file = home_dir + '/' + file
    with open(file) as f:
        file_info = f.readlines()
        medpc_data = dict.fromkeys(['Protocol', 'Run Time', 'Reward', 'Lever', 'Lick', 'IPI', 'Variance'])
        medpc_data['Protocol'] = file_info[12][5:-1]
        letters = ['F:','G:','U:','W:','X:','Y:','Z:']
        
        # Find run time
        start_time = datetime.datetime.strptime(file_info[10][-9:-1],'%H:%M:%S')
        end_time = datetime.datetime.strptime(file_info[11][-9:-1],'%H:%M:%S')
        medpc_data['Run Time'] = f'{end_time - start_time}'
        row_nums = []
        for letter in letters: #This IF statement catches the case when there is no data for a requested letter. Places a 0.
            row=[ind for ind, row in enumerate(file_info) if row == f'{letter}\n']
            if row:
                row_nums += row
            else:
                row_nums += [0]
        boundaries = dict(zip(letters, row_nums))
        try:
            medpc_data['Reward'] = np.asarray([float(item) for row in file_info[(boundaries['Z:']+1):]
                                       for item in row[7:-1].split() ])
            medpc_data['Lever'] = np.asarray([float(item) for row in file_info[(boundaries['X:']+1):boundaries['Y:']] for item in row[7:-1].split() ])
            medpc_data['Lick'] = np.asarray([float(item) for row in file_info[(boundaries['Y:']+1):boundaries['Z:']] for item in row[7:-1].split() ])
        except: # Added these exceptions because 4241's file on 12/12 was giving me trouble
            print(f'Error in file: {file}')
        if 'FR5' in medpc_data['Protocol']:
            if not boundaries['F:']==0: #this catches the one day recorded before I started saving the IPIs in F
                try: # Accidentally had variance/ipi switched - Variance is in F and IPI is in U
                    medpc_data['Variance'] = np.asarray( 
                        [float(item) for row in file_info[(boundaries['F:'] + 1):boundaries['G:']] for item in row[7:-1].split()])
                    medpc_data['IPI'] = np.asarray(
                        [float(item) for row in file_info[(boundaries['U:'] + 1):boundaries['W:']] for item in
                         row[7:-1].split()])
                except:
                    print(f'Error in file: {file}')
                    medpc_data['Reward'] == []
                    medpc_data['Lever'] == []
                    medpc_data['Lick'] == []
    return medpc_data



def create_medpc_master(mice,  file_dir):
    columns = ['Mouse', 'Date', 'Protocol', 'Run Time', 'Reward', 'Lever', 'Lick', 'IPI', 'Variance']
    medpc_master = pd.DataFrame(columns=columns)
    for mouse in mice: #loop through each mouse
        fnall = [dayfile for dayfile in os.listdir(file_dir) if str(mouse) in dayfile] # find all the files that have the mouse's name in them
        for day in fnall: #loop trhough each day from the mouse
            fullfile=os.path.join(file_dir, day) #get full path to file
            day_df=pd.DataFrame(columns=columns)
            day_df.at[0,'Mouse'] = mouse
            day_df.at[0,'Date'] = day[:4]+day[5:7]+day[8:10]
            medpc_data = medpc_extract(fullfile) #call base function to extract data from txt file  
            day_df.at[0, 'Protocol'] = medpc_data['Protocol']
            day_df.at[0, 'Reward'] = medpc_data['Reward']
            day_df.at[0, 'Lever'] = medpc_data['Lever']
            day_df.at[0, 'Lick'] = medpc_data['Lick']
            day_df.at[0, 'IPI'] = medpc_data['IPI']
            day_df.at[0, 'Variance'] = medpc_data['Variance']
            day_df.at[0, 'Run Time'] = (medpc_data['Run Time'])
            medpc_master=pd.concat([medpc_master,day_df],ignore_index=True)        
        

   
    if 4225 in mice: 
        # Add IPI/Var data for 4225 (First FR5 mouse). This mouse was recorded before we started directly storing IPI 
        #as a variable. The block below fills in the missing info for this mouse.
        mouse_df = medpc_master[medpc_master['Mouse']==4225]
        
        if '20211204' not in np.unique(mouse_df['Date']): #if this day is not included in data set of interest
            return medpc_master
        date_df = mouse_df[mouse_df['Date']=='20211204']
        new_ind = date_df.index.values[0]
        lever_4225 = medpc_master.at[new_ind, 'Lever']
        reward_4225 = medpc_master.at[new_ind, 'Reward']
        IPI_4225 = [lever_4225[i] - lever_4225[i-1] for i in range(1,len(lever_4225))]
        Var_4225 = []
        for i in range(len(reward_4225)):
            curr_reward = reward_4225[i]
            sequence = lever_4225[lever_4225 < curr_reward][-5:]
            Var_4225.append(np.var(sequence))
        medpc_master.at[new_ind, 'IPI'] = IPI_4225
        medpc_master.at[new_ind, 'Variance'] = Var_4225

    return medpc_master
            
        
    