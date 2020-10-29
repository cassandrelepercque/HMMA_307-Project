# -*- coding: utf-8 -*-
"""
@author: Cassandre Lepercque

In short: Usage of Linear mixed effects models on trophic
        position of fish in different lakes.
"""

################################
# Packages needed
#################################

from download import download
import os
import pandas as pd

################################
# Download datasets
#################################

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

fish = "http://qcbs.ca/wiki/_media/qcbs_w6_data.csv"
path_fish = os.path.join(results_dir, "fish.txt")

download(fish, path_fish, replace=False)

df_fish = pd.read_csv(path_fish, sep=",", header=0)
print(df_fish)