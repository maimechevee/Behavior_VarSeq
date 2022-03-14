#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:58:58 2022

@author: emma-fuze-grace
"""

def create_mice(file, exp_dict={}):
    """Make master dictionary of mice where keys store information on experiments the  mice
    were run in as well as whether the mouse should be discarded.
    
    To help organizing mice into groups during graphing.
    """
    mice = {}
    with open(file) as f:
        for line in f:
            lst = line[:-1].split(', ')
            mouse_num = int(lst[0])
            mice.setdefault(mouse_num, {})
            
            experiments = {}
            for item in lst[1:-1]:
                experiment = item.split('-')[0]
                group = item.split('-')[1]
                experiments.setdefault(experiment, str)
                experiments[experiment] = group
            mice[mouse_num].setdefault('experiments', dict)
            mice[mouse_num]['experiments'] = experiments
            mice[mouse_num].setdefault('discard', False)
            mice[mouse_num]['discard'] = bool(int(lst[-1]))
            
    return mice
    
    
if __name__ == '__main__':
    tar_var_mice = create_mice('tar_var_mice.txt')
    