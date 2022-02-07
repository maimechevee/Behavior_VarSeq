#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:33:24 2022

@author: emma-fuze-grace
"""

from scipy.io import loadmat
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import seaborn as sns
import numpy as np

corr_mat = loadmat('r_vals_array.mat')
def cluster(correlation_matrix):
    Z = hierarchy.linkage(correlation_matrix,method='average') #do it once to get Z which you will use below to get clusters
    row_linkage = hierarchy.linkage( #these you're only doing for the plot
                                    distance.pdist(correlation_matrix), method='average')
    col_linkage = hierarchy.linkage( #these you're only doing for the plot
                                    distance.pdist(correlation_matrix.T), method='average')
    sns.set(font_scale=1)
    df=pd.DataFrame(data=correlation_matrix)#create df to input clustermap
    map_test=sns.clustermap(df, row_linkage=row_linkage, col_linkage=col_linkage,
                            cmap='bwr',  method="average", xticklabels=1, 
                            figsize=(15,15), annot_kws={"size": 6},
                            vmin=0.5, vmax=1)
    map_test.ax_col_dendrogram.set_xlim([0,0]) #removes top dendogram on clustemap
    
    ## Draw dendogram
    plt.figure()
    Cluster_assignment=hierarchy.fcluster(Z, 5.5, criterion='distance')
    D=hierarchy.dendrogram(Z, color_threshold=5.5) #10 groups
    return D


result = cluster(corr_mat['r_vals_array'])
