#!/home/preclineu/hansav/.conda/envs/py38/bin/python

"""
Created on Thu Aug 29 11:16:53 2024

@author: hansav
"""


"""
####################
#Elastic Net Regression classification STAI score --> PERMUTATION
####################
"""

import pandas as pd
import numpy as np
import os
import sys
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import matplotlib as mpl
from pathlib  import Path
import matplotlib.ticker as ticker


# globals
root_dir = '/project_cephfs/3022017.06/ENIGMA_ANX/'
proc_dir = os.path.join(root_dir,'Z_stat/')
data_dir = os.path.join(proc_dir,'data/')
w_dir = os.path.join(proc_dir,'vox/')
save_dir = '/project_cephfs/3022017.06/ENIGMA_ANX/Z_stat/vox/Validation/Anxiety_stai_permutations/'
mask_nii = ('/opt/fmriprep/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
ex_nii = os.path.join(data_dir, 'ENIGMA_FC_tr_1.nii.gz')



#Data IO and generation
#arguments
num = str(sys.argv[1])
print(num)

#%%LOAD IN DATA AND MASK BY AVAILABLE PARTICIPANTS

# Load in the Z_est files
Z_est_control_test = ptkload(os.path.join(w_dir,'Z_estimate.pkl'), mask=mask_nii)
Z_est_clinical = ptkload(os.path.join(w_dir,'Z_predcl.pkl'), mask=mask_nii)
Full_sample_deviations = np.append(Z_est_control_test,Z_est_clinical, axis = 0)

#Load in the contingency awareness data
Measures = pd.read_csv('/project_cephfs/3022017.06/ENIGMA_ANX/Z_stat/data/all_test_validation.csv', usecols = ["Group_Dataset", 
                                                                                                                "Anxiety_instrument",
                                                                                                                "Anxiety_score",
                                                                                                                "Depression_instrument",
                                                                                                                "Depression_score", 
                                                                                                                'Principal_diagnosis_current']) 
Measures.replace(to_replace='NA/does not apply', value='NA', regex=True, inplace=True)

Measures['Anxiety_score'] = pd.to_numeric(Measures['Anxiety_score'], errors='coerce').astype('Int64')
Measures['Depression_score'] = pd.to_numeric(Measures['Depression_score'], errors='coerce').astype('Int64')


# FOR STAI- SPANISH VERSION: add 20 to make scale the same as others:
spanish_STAI_sites = ["Barcelona_Cardoner", "Barcelona_Soriano_dataset_1", "Barcelona_Soriano_dataset_2"]
# Check if instrument_value is 1 and the Group_Dataset is in the specified list
for index, row in Measures.iterrows():
    if row['Group_Dataset'] in spanish_STAI_sites and Measures.at[index, 'Anxiety_instrument'] == 'stai-trait':
        # Modify Anxiety_score by adding 20
        Measures.at[index, 'Anxiety_score'] += 20


#Mask by participants for whom CA data is available
mask_Anx = Measures["Anxiety_score"].notna() #remove NAs
mask_select_measure = Measures['Anxiety_instrument'].isin(['stai-trait']) #CHANGE THIS TO QUESTIONNIARE OF INTEREST
#beck anxiety inventory
#hamilton anxiety
#other (scared total score)
#other (staic)
#stai-state
#stai-trait

mask_exclude_diagnosis = ~Measures['Principal_diagnosis_current'].isin(['others', 'schizophrenia']) #and remove others and schizophrenia
combined_mask_Anx = mask_Anx & mask_exclude_diagnosis 
combined_mask_Anx = combined_mask_Anx & mask_select_measure

Anx_sample = Measures['Anxiety_score'][combined_mask_Anx].to_numpy()
Anx_sample_deviations = Full_sample_deviations[combined_mask_Anx]
#print('Anxiety data available for: '+str(len(Anx_sample)) +' people')
#print(Anx_sample)


#Define parameters
X1 = Anx_sample_deviations #Deviations
y = Anx_sample.ravel()
n_samples, n_features = X1.shape
random_state = np.random.RandomState(0)

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

from sklearn.model_selection import train_test_split

ypred_all = []
score_all = []
mse_all = []
ev_all = []
coefs_all = []
model_intercept_all = []


for p in range(0,10): #100 x 10 = 1000 permuations 
    print(p)
    y_shuf = shuffle(y) #SHUFFLE THE SCORES  
    
    
    model_alpha = []
    model_intercept = []
    
    ypred_cf = []
    score_cf = []
    ev_cf = []
    mse_cf = []
    coefs_cf = []
    
    for i in range(0,10):
        print('Iteration',i)
    
        alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
        xtrain, xtest, ytrain, ytest = train_test_split(X1, y_shuf, test_size=0.1)
        
        #ElasticNetCV is a cross-validation class that can search multiple alpha values 
        #and applies the best one. We'll define the model with alphas value and fit 
        #it with xtrain and ytrain data.
        
        elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
        model = elastic_cv.fit(xtrain, ytrain)
        
        #print(model.alpha_)
        model_alpha.append(model.alpha_)
        #print(model.intercept_)
        model_intercept.append(model.intercept_)
        
        ypred = model.predict(xtest)
        score = model.score(xtest, ytest)
        ev = explained_variance_score(ytest, ypred)
        mse = mean_squared_error(ytest, ypred)
        #print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
        #      .format(score, mse, np.sqrt(mse)))
        
        
        ypred_cf.append(ypred)
        score_cf.append(score)
        ev_cf.append(ev)
        mse_cf.append(mse)
        coefs_cf.append(model.coef_)
        
        ypred_all.append(np.mean(ypred_cf))
        score_all.append(np.mean(score_cf))
        ev_all.append(np.mean(ev_cf))
        mse_all.append(np.mean(mse_cf))
        coefs_all.append(np.mean(coefs_cf, axis = 0))
        model_intercept_all.append(np.mean(model_intercept))
    
#print("OVERALL: R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
#      .format(np.mean(score_cf), np.mean(mse_cf), np.mean(np.sqrt(mse))))    

print('saving files') 
combined_elastic_net = pd.DataFrame(list(zip(score_all, ev_all, mse_all, model_intercept_all)),
                                    columns=['R2','EV', 'MSE', 'Model_Intercept']) 
combined_elastic_net.to_csv(os.path.join(save_dir,'Anxiety_stai_Elastic_Net_permutations_' +num +'.csv'))

mean_coefs = np.mean(coefs_all, axis = 0)
median_coefs = np.median(coefs_all, axis = 0)
coefs_all = np.array(coefs_all)
freq_coefs = (coefs_all > 0.0001).sum(axis=0) >= 5

ptksave(coefs_all, os.path.join(save_dir,'Anxiety_stai_coefs_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(mean_coefs, os.path.join(save_dir,'Anxiety_stai_mean_coefs_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(median_coefs, os.path.join(save_dir,'Anxiety_stai_median_coefs_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)

binary_mask = (coefs_all > 0.005).sum(axis=0) >= 5
ptksave(binary_mask, os.path.join(save_dir,'Anxiety_stai_gt5_mask_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)
count_array = (coefs_all > 0.0001).sum(axis=0)
ptksave(count_array, os.path.join(save_dir,'Anxiety_stai_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)
ptksave(freq_coefs, os.path.join(save_dir,'Anxiety_stai_frequency_coefs_permutations_' +num +'.nii.gz'), example=ex_nii, mask=mask_nii)
