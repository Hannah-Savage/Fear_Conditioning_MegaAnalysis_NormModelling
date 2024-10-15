#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:21:47 2024

@author: hansav
"""
import pandas as pd
import numpy as np
import os
from pcntoolkit.dataio.fileio import load as ptkload
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib as mpl
from pathlib  import Path
from pcntoolkit.dataio.fileio import save_nifti, load_nifti
from pcntoolkit.dataio.fileio import save as ptksave

import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("husl", 9)


# globals
root_dir = '/project_cephfs/3022017.06/ENIGMA_ANX/'
proc_dir = os.path.join(root_dir,'Z_stat/')
data_dir = os.path.join(proc_dir,'data/')
w_dir = os.path.join(proc_dir,'vox/')
perm_dir = os.path.join(w_dir,'Validation/Patient_status_permutations/')
mask_nii = ('/opt/fmriprep/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
ex_nii = os.path.join(data_dir, 'ENIGMA_FC_tr_1.nii.gz')


true_mean = np.load(os.path.join(w_dir +'Validation/Patient_status_mean_auc.npy'))
true_mean = true_mean.mean()

perm_means =  []

for i in range(1,101):
    filename = os.path.join(perm_dir +'Patient_status_mean_auc_'+str(i) +'.npy')
    print(filename)
    loaded = np.load(filename)
    perm_means.append(loaded)
    
    
perm_means = np.concatenate(perm_means)

p_value = (np.sum(perm_means >= true_mean) + 1) / (len(perm_means) + 1)

# Compare against the significance level
alpha = 0.05
if p_value < alpha:
    result = "The true value is statistically significant (p < 0.05)."
else:
    result = "The true value is not statistically significant (p ? 0.05)."

print(f"P-value: {p_value}")
print(result)


