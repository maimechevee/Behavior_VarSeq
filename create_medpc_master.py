#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a master dataframe including one session (mouse / date combination)
    per row.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def extract_event(file_info, boundary_1, boundary_2):
    """Returns np array of floats for a single event given row nums in file
    where data begins and ends.

    Parameters:
        file_info (list of strs): read from file using f.readlines()
        boundary_1 (ints): beinning of search for event timestamps (inclusive)
        boundary_2 (int): end of search for event timestamps (exclusive)
            if end of file, enter 0
    """
    event = []
    if boundary_2:
        for row in file_info[(boundary_1 + 1):boundary_2]:
            for item in row[7:-1].split():
                event.append(float(item))
    else:
        for row in file_info[(boundary_1 + 1):]:
            for item in row[7:-1].split():
                event.append(float(item))
    return event


def create_session_dictionary(file):
    """ Returns dictionary of MEDPC time stamps for a single session.

        Keys: Protocol, Run Time, Reward, Lever, Lick, IPI, Variance.
        Values: Numpy arrays with time stamps as floats.
    """

    with open(file) as f:
        file_info = f.readlines()
        medpc_data = dict.fromkeys(['Protocol', 'Run Time', 'Reward', 'Lever',
                                    'Lick', 'IPI', 'Variance'])
        medpc_data['Protocol'] = file_info[12][5:-1]

        # Find run time
        start_time = datetime.strptime(file_info[10][-9:-1], '%H:%M:%S')
        end_time = datetime.strptime(file_info[11][-9:-1], '%H:%M:%S')
        medpc_data['Run Time'] = f'{end_time - start_time}'

        """Find row numbers corresponding to beginning and ending rows
        of event indices."""
        row_nums = []
        letters = ['F:', 'G:', 'U:', 'W:', 'X:', 'Y:', 'Z:']
        for letter in letters:
            row = [ind for ind, row in enumerate(file_info)
                   if row == f'{letter}\n']
            if row:  # This IF statement catches the case when there is no data
                row_nums += row           # for a requested letter. Places a 0.
            else:
                row_nums += [0]
        boundaries = dict(zip(letters, row_nums))

        """ Populate dictionary using extract_event function and row nums."""
        medpc_data['Reward'] = extract_event(file_info, boundaries['Z:'], 0)
        medpc_data['Lever'] = extract_event(file_info, 
                                            boundaries['Y:'], 
                                            boundaries['Z:'])
        medpc_data['Lick'] = extract_event(file_info, 
                                           boundaries['X:'], 
                                           boundaries['Y:'])
        medpc_data['IPI'] = extract_event(file_info, 
                                            boundaries['U:'], 
                                            boundaries['W:'])
        
        if 'FR5' in medpc_data['Protocol']:
            """The below if statement catches the day before Variance and IPI 
            were recorded in directly for the first FR5 mouse (4225).
            """
            if boundaries['F:']: 
                medpc_data['Variance'] = extract_event(file_info, 
                                                    boundaries['F:'], 
                                                    boundaries['G:'])
        else:
            """Using "N/A" for when we expect to not have timestamps for a n event
            will make it more obvious that there is a mistake if for some reason
            None is placed here.
            """
            medpc_data['Variance'] = 'N/A'
            
        for key in medpc_data:
            assert type(medpc_data[key]) is not type(None)
            # Trying to adapt to make more flexible for different protocols, 
            # not finished yet  but am working on it.
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
            medpc_data = create_session_dictionary(fullfile) #call base function to extract data from txt file  
            day_df.at[0, 'Protocol'] = medpc_data['Protocol']
            day_df.at[0, 'Reward'] = medpc_data['Reward']
            day_df.at[0, 'Lever'] = medpc_data['Lever']
            day_df.at[0, 'Lick'] = medpc_data['Lick']
            day_df.at[0, 'IPI'] = medpc_data['IPI']
            day_df.at[0, 'Variance'] = medpc_data['Variance']
            day_df.at[0, 'Run Time'] = (medpc_data['Run Time'])
            medpc_master = pd.concat([medpc_master, day_df], ignore_index=True)    
        

   
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

    assert medpc_master.empty is False, 'Empty dataframe.'
    return medpc_master

def discard_mice(master_df, discard_list):
    """Drop poorly performing mice from dataframe.
    
    For original TarVar in December:
        discard_list = [4217, 4218, 4221, 4227, 4232, 
                        4235, 4236, 4237, 4238, 4244]
    """     
    indices = []
    for mouse in discard_list:
        indices.append(master_df[master_df['Mouse']==mouse].index)
    indices = [x for l in indices for x in l]
    return master_df.drop(indices, axis=0)

if __name__ == '__main__':
    mice = [i for i in range(4386, 4414)]
    file_dir = ('/Users/emma-fuze-grace/Lab/Behavior_VarSeq'
                '/2022-02_TarVar_Categ_01/2022-02_TarVar_Categ_01_data')
    master_df = create_medpc_master(mice, file_dir)
