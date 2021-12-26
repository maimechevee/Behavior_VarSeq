import pandas as pd
import numpy as np
import os
import datetime
import math
from tkinter.filedialog import askdirectory

# Returns dictionary with Protocol, Reward, Lever, Lick, IPI, Variance
def medpc_extract(file):
    # file = home_dir + '/' + file
    with open(file) as f:
        print(f)
        file_info = f.readlines()
        medpc_data = dict.fromkeys(['Reward', 'Lever', 'Lick', 'IPI', 'Variance'])
        medpc_data['Protocol'] = file_info[12][5:-1]
        letters = ['F:','G:','U:','W:','X:','Y:','Z:']
        
        row_nums = []
        for letter in letters: #This IF statement catches the case when there is no data for a requested letter. Places a 0.
            row=[ind for ind, row in enumerate(file_info) if row == f'{letter}\n']
            if row:
                row_nums += row
            else:
                row_nums += [0]
        boundaries = dict(zip(letters, row_nums))
        medpc_data['Reward'] = np.asarray([float(item) for row in file_info[(boundaries['Z:']+1):]
                                           for item in row[7:-1].split() ])
        medpc_data['Lever'] = np.asarray([float(item) for row in file_info[(boundaries['X:']+1):boundaries['Y:']] for item in row[7:-1].split() ])
        medpc_data['Lick'] = np.asarray([float(item) for row in file_info[(boundaries['Y:']+1):boundaries['Z:']] for item in row[7:-1].split() ])
        if 'FR5' in medpc_data['Protocol']:
            if not boundaries['F:']==0: #this catches the one day recorded before I started saving the IPIs in F
                medpc_data['IPI'] = np.asarray(
                    [float(item) for row in file_info[(boundaries['F:'] + 1):boundaries['G:']] for item in row[7:-1].split()])
                medpc_data['Variance'] = np.asarray(
                    [float(item) for row in file_info[(boundaries['U:'] + 1):boundaries['W:']] for item in
                     row[7:-1].split()])
    return medpc_data



def create_medpc_master(mice, dates):
    columns = ['Mouse', 'Date', 'Protocol', 'Protocol Day', 'Run Time', 'Reward', 'Lever', 'Lick', 'IPI', 'Variance']
    total_mice = len(mice)
    days_training=len(dates)
    mice_x_day = [y  for x in mice for y in [x]*days_training] #this only works when all mice contribute to all days
    medpc_master = pd.DataFrame(columns = columns)
    medpc_master['Mouse'] = mice_x_day
    medpc_master['Date'] = [y for x in range(total_mice) for y in dates]
    file_dir = askdirectory(title='Select Folder: ',
                            initialdir='C:/Users/...')
    os.chdir((file_dir))
    print(f'Now in following directory: {file_dir}')
    for f in os.listdir():
        # 2021-12-02_12h13m_Subject 4217.txt (Original name format)
        file_name, file_ext = os.path.splitext(f)
        file_date = file_name[:4]+file_name[5:7]+file_name[8:10]
        mouse_num = int(file_name[-4:])
       # with open(f, 'r') as ff:
        medpc_data = medpc_extract(f)
        curr_mouse = medpc_master[medpc_master['Mouse']==mouse_num].index
        curr_date = medpc_master[medpc_master['Date']==file_date].index
        ind = int(curr_mouse.intersection(curr_date).values)
        if not math.isnan(ind):
            medpc_master.at[ind, 'Protocol'] = medpc_data['Protocol']
            medpc_master.at[ind, 'Reward'] = medpc_data['Reward']
            medpc_master.at[ind, 'Lever'] = medpc_data['Lever']
            medpc_master.at[ind, 'Lick'] = medpc_data['Lick']
            medpc_master.at[ind, 'IPI'] = medpc_data['IPI']
            medpc_master.at[ind, 'Variance'] = medpc_data['Variance']

    return medpc_master

mice=[4217,4218,4219,4220,4221,4222,4223,4224,4225,4226,4227,4228,
      4229,4230,4231,4232,4233,4234,4235,4236,4237,4238,4239,4240,4241,4242,4243, 4244] #(ints)
dates=['20211202', '20211203', '20211204', '20211205', '20211206', '20211207', '20211208',
       '20211209', '20211210', '20211211', '20211212', '20211213', '20211214', '20211215'] #(strs)
master_df = create_medpc_master(mice,dates)
