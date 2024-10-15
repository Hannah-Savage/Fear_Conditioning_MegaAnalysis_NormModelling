import os
import pickle
import pandas as pd
import numpy as np
import pcntoolkit as ptk 
from pcntoolkit.dataio.fileio import load as ptkload
from pcntoolkit.dataio.fileio import save as ptksave
from pcntoolkit.normative import predict, evaluate
from pcntoolkit.dataio.fileio import save_nifti, load_nifti
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, calibration_descriptives
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns
import pingouin as pg

sns.set(style='whitegrid')
warp = None


# In[ ]:


# globals
root_dir = '/project_cephfs/3022017.06/ENIGMA_ANX/'
proc_dir = os.path.join(root_dir,'Z_stat/')
data_dir = os.path.join(proc_dir,'data/')
w_dir = os.path.join(proc_dir,'vox/')
save_dir = os.path.join(w_dir,'Regression_Coef/')
mask_nii = ('/opt/fmriprep/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
ex_nii = os.path.join(data_dir, 'ENIGMA_FC_tr_1.nii.gz')

struc_coef_dir = os.path.join(w_dir, 'Structure_Coef/')


# In[ ]:
# load covariatea
print('loading covariate data ...')
df_dem = pd.read_csv(os.path.join(data_dir,'metadata_te.csv'))

#load the pkl file
yhat_est = ptkload(os.path.join(w_dir,'yhat_estimate.pkl'), mask=mask_nii)
yhat_est_transf = np.transpose(yhat_est)

cols_cov = ["Age", 
            "Sex",
            "MRI", 
            "Instructions",
            "Precond_number_trials",
            "Multiple_CSplus", 
            "Multiple_CSminus",
            "CS_type_neutral_faces",
            "CS_type_neutral_pictures",
            "CS_type_neutral_male_avatar",
            "CS_type_snakes_spiders",
            "CS_type_gabor_patch",
            "CS_type_animal_tool",
            "CS_type_affective_faces_pictures",
            "CS_type_humanoic_characters",
            "Number_CSplus_cond",
            "Number_CSminus_cond",
            "Reinforcing_rate",
            "US_type_electric_shock", 
            "US_type_auditory", 
            "US_type_visceral",
            "US_type_thermal", 
            "Average_ITI", 
            "Average_ISI",
            "Potential_US_confound"]

# In[ ]:
    
for voxel in tqdm(range(0, yhat_est.shape[1])):
    with open(os.path.join(w_dir, 'Models',f'NM_0_{voxel}_estimate.pkl'),'rb') as f:
        nm = pickle.load(f)
    s2 = np.diag(np.linalg.inv(nm.blr.A))
    
    if voxel == 0:
        M = np.zeros((len(s2),yhat_est.shape[1]))
        S = np.zeros((len(s2),yhat_est.shape[1]))
        Mt = np.zeros((len(s2),yhat_est.shape[1]))
    M[:,voxel] = nm.blr.m
    S[:, voxel] = np.sqrt(s2)
    

save_nifti(M.T, os.path.join(struc_coef_dir,'w.nii.gz'), mask = mask_nii, examplenii = mask_nii, dtype='float32')

Mt = M
Mt[(np.abs(Mt) - 0.995 * S) < 0.0001] = 0
Mt[(np.abs(Mt) - 0.995 * S) < 0.0001] = 0
save_nifti(Mt.T, os.path.join(struc_coef_dir,'wt.nii.gz'), mask = mask_nii, examplenii = mask_nii, dtype='float32')

Mtn = Mt
for i in range(Mt.shape[0]):
    Mtn[i,:] = Mtn[i,:] / np.max(np.abs(Mtn[i,:]))

save_nifti(Mtn.T, os.path.join(struc_coef_dir,'wtn.nii.gz'), mask = mask_nii, examplenii = mask_nii, dtype='float32')


# #For each covariate of interest [n x 1]
# for column in cols_cov:
#     curr_cov = df_dem[column]
#     curr_cov = curr_cov.astype(int)
#     print(curr_cov)
    
#     covariate_Rho = []
#     covariate_pRho = []
#     coviariate_Rho_Sqrd = []
    
#     covariate_Rho_array = np.empty(0)
#     covariate_pRho_array = np.empty(0)
#     coviariate_Rho_Sqrd_array = np.empty(0)    
    
#     #For each voxel of the brain [n x 1]
#     for voxel in range(0, yhat_est.shape[1]):
#         #Correlate the covariate with the predicted activity in that voxel [1 x 1 - with n data points]
#         correlation = compute_pearsonr(curr_cov, yhat_est[:,voxel])
#         #Accumulate the correlations to form one whole brain vector of 
#         # vector of correlation coefficients
#         covariate_Rho.append(correlation[0]) 
#         coviariate_Rho_Sqrd.append(correlation[0]*correlation[0]) 
#         #vector of p-values
#         covariate_pRho.append(correlation[1]) 
        
#     #Convert the lists to arrays
#     covariate_Rho_array = np.array(covariate_Rho)
#     coviariate_Rho_Sqrd_array = np.array(coviariate_Rho_Sqrd)
#     covariate_pRho_array = np.array(covariate_pRho)
      
#     #Build the filename for each output file    
#     filename_Rho = (struc_coef_dir +'Rho_' +column +'.nii.gz')
#     filename_Rho_Sqrd = (struc_coef_dir +'Rho_Sqrd_' +column +'.nii.gz')  
#     filename_pRho = (struc_coef_dir +'pRho_' +column +'.nii.gz')
       
#     #Save each output file as a nii
#     save_nifti(covariate_Rho_array, filename_Rho, mask = mask_nii, examplenii = mask_nii, dtype='float32')
#     save_nifti(coviariate_Rho_Sqrd_array, filename_Rho_Sqrd, mask = mask_nii, examplenii = mask_nii, dtype='float32')
#     save_nifti(covariate_pRho_array, filename_pRho, mask = mask_nii, examplenii = mask_nii, dtype='float32')

print('DONE')