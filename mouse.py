#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:58:58 2022

@author: emma-fuze-grace
"""

class Mouse():
    """A class to help us organize the experiments/protocols associated with
    all the different mice we've used.
    """
    
    def __init__(self, num, discard='', experiments = {}):
        """Constructor for mouse class.
        
        Num: subject number (int)
        
        Experiments: dictionary where keys are experiment, values are group
            experiments = {
                TarVar1: FR5var
                TarVar2: CATEG
                }
            
        discard: 'discard' if we are discarding, empty string otherwise
        """
        self.num = num
        self.experiments = experiments
        self.discard = discard
        
    
    def __repr__(self):
        """Override print function for debugging."""
        return f'Subject number: {self.num}\nExperiments: {self.experiments}'


def create_master_dictionary(mice, file):
    """"Read list of mice and experiments into master dictionary
    The keys are subject numbers and the values are the corresponding mouse
    objects.
    """
    pass