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
                                            boundaries['X:'],
                                            boundaries['Y:'])
        medpc_data['Lick'] = extract_event(file_info,
                                           boundaries['Y:'],
                                           boundaries['Z:'])
        medpc_data['IPI'] = extract_event(file_info,
                                          boundaries['U:'],
                                          boundaries['W:'])

        protocol = medpc_data['Protocol'].lower()
        if 'fr1' in protocol or 'noforcedr' in protocol or 'nolight' in protocol:
            """Using "N/A" for when we expect to not have timestamps for an
            event will make it more obvious that if there is a mistake, since
            None is automatically placed as the value when dict is initialized
            using only keys.
            """
            medpc_data['Variance'] = 'N/A'
        elif 'fr5' in protocol and boundaries['F:']:
            medpc_data['Variance'] = extract_event(file_info,
                                                   boundaries['F:'],
                                                   boundaries['G:'])

        if file == '2021-12-04_12h12m_Subject 4225.txt':
            return fill_missing_info(medpc_data, 4225, '20211204')

        for key, value in medpc_data.items():
            if isinstance(value, type(None)):
                print(f'File "{file}" has None for {key}.')
    return medpc_data


def create_medpc_master(mice,  file_dir):
    columns = ['Mouse', 'Date', 'Protocol', 'Run Time', 'Reward', 'Lever',
               'Lick', 'IPI', 'Variance']
    medpc_master = pd.DataFrame(columns=columns)
    for mouse in mice:  # loop through each mouse
        print(mouse)
        # find all the files that have the mouse's name in them
        fnall = [dayfile for dayfile in os.listdir(file_dir)
                 if str(mouse) in dayfile]
        for day in fnall:  # loop through each day from the mouse
            fullfile = os.path.join(file_dir, day)  # get full path to file
            day_df = pd.DataFrame(columns=columns)
            day_df.at[0, 'Mouse'] = mouse
            day_df.at[0, 'Date'] = day[:4]+day[5:7]+day[8:10]
            # call base function to extract data from txt file
            medpc_data = create_session_dictionary(fullfile)
            day_df.at[0, 'Protocol'] = medpc_data['Protocol']
            day_df.at[0, 'Reward'] = medpc_data['Reward']
            day_df.at[0, 'Lever'] = medpc_data['Lever']
            day_df.at[0, 'Lick'] = medpc_data['Lick']
            day_df.at[0, 'IPI'] = medpc_data['IPI']
            day_df.at[0, 'Variance'] = medpc_data['Variance']
            day_df.at[0, 'Run Time'] = (medpc_data['Run Time'])
            medpc_master = pd.concat([medpc_master, day_df], ignore_index=True)
    assert medpc_master.empty is False, 'Empty dataframe.'
    medpc_master = medpc_master.sort_values(by=['Mouse', 'Date'])
    return medpc_master


def fill_missing_info(medpc_data, mouse, date):
    """ Add IPI/Var data for 4225 (First FR5 mouse). This mouse was recorded
    before we started directly storing IPI as a variable. This function
    fills in the missing info for this mouse.
    """
    lever = medpc_data['lever']
    rewards = medpc_data['reward']
    medpc_data['IPI'] = [lever[i] - lever[i-1] for i in range(1, len(lever))]
    medpc_data['Variance'] = [np.var(lever[lever < reward][-5:])
                              for reward in rewards]
    return medpc_data


def discard_mice(master_df, discard_list):
    """Drop poorly performing mice from dataframe.

    For original TarVar in December:
        discard_list = [4217, 4218, 4221, 4227, 4232,
                        4235, 4236, 4237, 4238, 4244]
    """
    indices = []
    for mouse in discard_list:
        indices.append(master_df[master_df['Mouse'] == mouse].index)
    indices = [x for l in indices for x in l]
    return master_df.drop(indices, axis=0)

def discard_day(master_df, discard_list):
    """discard_list is a list of lists
    with [mouse, day]
    """
    indices = []
    for each in discard_list:
        mouse_df=master_df[master_df['Mouse'] == each[0]]
        day_df=mouse_df[mouse_df['Date'] == each[1]]
        indices.append(day_df.index)
    indices = [x for l in indices for x in l]
    return master_df.drop(indices, axis=0)

if __name__ == '__main__':
    mice = [i for i in range(4386, 4414)]
    file_dir = input('File directory: ')
    master_df = create_medpc_master(mice, file_dir)
