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



def create_medpc_master(mice, dates):
    columns = ['Mouse', 'Date', 'Protocol', 'Run Time', 'Reward', 'Lever', 'Lick', 'IPI', 'Variance']
    total_mice = len(mice)
    days_training=len(dates)
    mice_x_day = [y for x in mice for y in [x]*days_training] #this only works when all mice contribute to all days
    medpc_master = pd.DataFrame(columns = columns)
    medpc_master['Mouse'] = mice_x_day
    medpc_master['Date'] = [y for x in range(total_mice) for y in dates]
    file_dir = "/Users/emma-fuze-grace/Lab/Behavior_VarSeq/Medpc Data/CATEG"
    os.chdir((file_dir))
    print(f'Now in following directory: {file_dir}')
    for f in os.listdir():
        # 2021-12-02_12h13m_Subject 4217.txt (Original name format)
        file_name, file_ext = os.path.splitext(f)
        if file_name[0] != '.':
            file_date = file_name[:4]+file_name[5:7]+file_name[8:10]
            mouse_num = int(file_name[-4:])
            medpc_data = medpc_extract(f)
            curr_mouse = medpc_master[medpc_master['Mouse']==mouse_num].index
            curr_date = medpc_master[medpc_master['Date']==file_date].index
            try:
                ind = int(curr_mouse.intersection(curr_date).values)
                if not math.isnan(ind):
                    medpc_master.at[ind, 'Protocol'] = medpc_data['Protocol']
                    medpc_master.at[ind, 'Reward'] = medpc_data['Reward']
                    medpc_master.at[ind, 'Lever'] = medpc_data['Lever']
                    medpc_master.at[ind, 'Lick'] = medpc_data['Lick']
                    medpc_master.at[ind, 'IPI'] = medpc_data['IPI']
                    medpc_master.at[ind, 'Variance'] = medpc_data['Variance']
                    medpc_master.at[ind, 'Run Time'] = (medpc_data['Run Time'])
            except:
                print(curr_mouse)
                print(curr_date)
            
    return medpc_master
            
        
    
    # # Add IPI/Var data for 4225 (First FR5 mouse)
    # mouse_4225_ind = medpc_master[medpc_master['Mouse']==4225].index
    # date_12_04_ind = medpc_master[medpc_master['Date']=='20211204'].index
    # new_ind = int(mouse_4225_ind.intersection(date_12_04_ind).values)
    # lever_4225 = medpc_master.at[new_ind, 'Lever']
    # reward_4225 = medpc_master.at[new_ind, 'Reward']
    # IPI_4225 = [lever_4225[i] - lever_4225[i-1] for i in range(1,len(lever_4225))]
    # Var_4225 = []
    # for i in range(len(reward_4225)):
    #     curr_reward = reward_4225[i]
    #     sequence = lever_4225[lever_4225 < curr_reward][-5:]
    #     Var_4225.append(np.var(sequence))
    # medpc_master.at[new_ind, 'IPI'] = IPI_4225
    # medpc_master.at[new_ind, 'Variance'] = Var_4225
    # return medpc_master



