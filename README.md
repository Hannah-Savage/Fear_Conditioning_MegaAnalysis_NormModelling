# Fear_Conditioning_MegaAnalysis_NormModelling

This repository contains code for the (forthcoming) manuscript titled: 
**_Neural Correlates of Human Fear Conditioning and Sources of Variability: A Mega-Analysis and Normative Modeling Study of fMRI Data from 2,199 Individuals_**

## CODE:
### Mega-Analysis Code:



### Normative Modelling Code:
Normative Models: Building models + test and evaluate on control test sample
* 01_prepare_wholebrain_model_wmvol_control_test.ipynb
* 02_run_wholebrain_model_wmvol_control_test.ipynb
*  03_evaluate_wholebrain_model_wmvol_control_test.ipynb
*  04_NPM_test_controls.ipynb

Normative Models: Reference cohort - structure coefficients:
* 05_structure_coefficients_control_test.py
* 06_regression_coefficient_weights_control_test.py

Normative Models: Clinical test sample (controls + patients) - test and evaluate:
* 07_prepare_wholebrain_model_wmvol_clinical_test.ipynb
* 08_predict_wholebrain_model_wmvol_clinical_test.ipynb
* 09_evaluate_wholebrain_model_wmvol_clincial_test.ipynb
* 10_NPM_test_controls.ipynb
* 10_split_Z_est_diagnosis_NPM_threshold_combine.ipynb

Compare groups:
* 11_frequency_per_group_compare.ipynb
* 12_frequency_per_diagnosis_compare.ipynb
* 13_barplots_allgroups.ipynb
* 14_Patient_status.ipynb
* 14_Patient_status_permutations.py
* 14_Patient_status_significance.py
* 15_Primary_Diagnosis.ipynb

Validation:
* 16_Contingency_Awareness.ipynb
* 16_Contingency_Awareness_permutations.py
* 16_Contingency_Awareness_significance.py
* 17_Anxiety_scales_elastic_net.ipynb
* 17_Anxiety_scales_elastic_net_significance.py
* 18_Depression_scales_elastic_net.ipynb
* 18_Depression_scales_elastic_net_permutations.py
* 18_Depression_scales_elastic_net_significance.py

## NUMERICAL SOURCE DATA:
### Mega-Analysis Fies:

### Normative Modelling Files:
The following outputs were generated for the control_test and clinical_test:
* EV_clinical_test.nii.gz
* EV_test.nii.gz
* kurtosis_clinical_test.nii.gz
* kurtosis_test.nii.gz
* skew_clinical_test.nii.gz
* skew_test.nii.gz
* SMSE_clinical_test.nii.gz
* SMSE_test.nii.gz

Normative Probability Maps (i.e number of participants with large deviations): 
* Freq_neg_dev_controls.nii.gz
* Freq_neg_dev_gad.nii.gz
* Freq_neg_dev_ocd.nii.gz
* Freq_neg_dev_patients.nii.gz
* Freq_neg_dev_ptsd.nii.gz
* Freq_neg_dev_sadnii.gz
* Freq_pos_dev_controls.nii.gz
* Freq_pos_dev_gad.nii.gz
* Freq_pos_dev_ocd.nii.gz
* Freq_pos_dev_patients.nii.gz
* Freq_pos_dev_ptsd.nii.gz
* Freq_pos_dev_sad.nii.gz

Influence of task variables:
* The structure coefficient maps (Correlation coefficients (rho) thresholded by the respective coefficients of determination (rho2>0.3)) for each input variable are included.
* The regression coefficient weights (w/wt/wtn) for each input variable are included.

The data for the Normative Probability Map Count comparisons (Figure 6) are in:
* counts_hist_affective_only.csv
* counts_per_diagnosis_affective_only.csv




