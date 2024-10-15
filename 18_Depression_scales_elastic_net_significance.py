#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:02:51 2024

@author: hansav
"""
import os
import pandas as pd
from scipy import stats
import numpy as np

# Step 1: Specify the directory containing CSV files
folder_path = "/project_cephfs/3022017.06/ENIGMA_ANX/Z_stat/vox/Validation/Depression_bdi_permutations"

# Step 2: Initialize an empty list to hold each dataframe
dfs = []

# Step 3: Loop through the files in the folder, find CSV files, and load them
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Step 4: Combine all dataframes into one
combined_df = pd.concat(dfs, ignore_index=True)

# Step 5: Choose a column to test against the true value (e.g., R2)
permutation_r2 = combined_df['R2']

# Step 6: Define the true value (e.g., expected true R2 value)
true_df = pd.read_csv('/project_cephfs/3022017.06/ENIGMA_ANX/Z_stat/vox/Validation/Depression_BDI_Elastic_Net.csv')
true_r2_mean = true_df['R2'].mean()

n_permutations = len(permutation_r2)
extreme_count = np.sum(permutation_r2 >= true_r2_mean)

# Calculate the p-value as the proportion of permutation R2 values greater than the true R2 mean
p_value = extreme_count / n_permutations

# Step 9: Output the result
print(f"True R2 Mean: {true_r2_mean}")
print(f"Number of Permutations: {n_permutations}")
print(f"Number of Extreme Values (R2 >= True R2 Mean): {extreme_count}")
print(f"P-value: {p_value}")

# Step 10: Check if the result is significant (commonly if p-value < 0.05)
if p_value < 0.05:
    print("The true R2 mean is statistically significant.")
else:
    print("The true R2 mean is not statistically significant.")