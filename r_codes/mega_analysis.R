################################################################################
# Read the individual NIfTI images                                             #
################################################################################
dat = matrix(NA, nrow = length(readNIfTI(X$mri_path[1])[]), ncol = nrow(X))
for (i in 1:nrow(X)) {
  cat("Reading image", i, "of", nrow(X), "\n")
  nifti = readNIfTI(X$mri_path[i])
  dat_i = nifti[]
  dat_i[which(dat_i == 0)] = NA
  dat[,i] = dat_i
}

################################################################################
# Find prescaling factors based on healthy controls                            #
################################################################################
i_controls = which(X$patient == 0)
i_patients = which(X$patient == 1)
prescale_controls = prescale_fit(
  dat = dat[,i_controls],
  site = factor(X$site[i_controls]),
  cov = cbind(X$age, X$sex)[i_controls,],
  n.min = 9, impute_missing_cov = TRUE
)

################################################################################
# Find prescaling factors based on patients. These factors will be discarded   #
# (because we prefer those from controls) except for a site with only patients #
################################################################################
prescale_patients = prescale_fit(
  dat = dat[,i_patients],
  site = factor(X$site[i_patients]),
  cov = cbind(X$age, X$sex)[i_patients,],
  n.min = 9, impute_missing_cov = TRUE
)
# Add the prescaling factor of the site with only patients to the prescaling
# factors from controls
prescale_controls$prescaling_factors["Austin_Cisler"] =
  prescale_patients$prescaling_factors["Austin_Cisler"]
prescale_controls$levels_batch =
  sort(c(prescale_controls$levels_batch, "Austin_Cisler"))

################################################################################
# Apply the prescaling factors                                                 #
################################################################################
dat_prescaled = prescale_apply(
  prescale_controls,
  dat = dat,
  site = factor(X$site),
  cov = cbind(X$age, X$sex, X$patient)
)$dat

################################################################################
# Fit linear mixed-effects models: mean, age, and sex                          #
################################################################################
results = lmm_fit(
  dat = dat_prescaled[,i_controls],
  site = factor(X$site[i_controls]),
  cov = scale(cbind(X$age, X$sex)[i_controls,], scale = FALSE),
  n.min = 9, impute_missing_cov = TRUE, prescaling = FALSE
)
nifti[] = results$b[[1]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/controls_b")
nifti[] = results$z[[1]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/controls_z")
nifti[] = results$b[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/age_b")
nifti[] = results$z[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/age_z")
nifti[] = results$b[[3]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/sex_b")
nifti[] = results$z[[3]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/sex_z")

################################################################################
# Fit linear mixed-effects models: patients vs controls                        #
################################################################################
results = lmm_fit(
  dat = dat_prescaled,
  site = factor(X$site),
  cov = cbind(X$patient, X$age, X$sex),
  n.min = 9, impute_missing_cov = TRUE, prescaling = FALSE
)
nifti[] = results$b[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/patients_vs_controls_b")
nifti[] = results$z[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/patients_vs_controls_z")

################################################################################
# Fit linear mixed-effects models: simple variables -e.g., instructions-       #
################################################################################
results = lmm_fit(
  dat = dat_prescaled[,i_controls],
  site = factor(X$site[i_controls]),
  cov = cbind(X$instructions, X$age, X$sex)[i_controls,],
  n.min = 9, impute_missing_cov = TRUE, prescaling = FALSE
)
nifti[] = results$b[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/instructions_b")
nifti[] = results$z[[2]]
writeNIfTI(nifti, "gitHubCode_mega_analysis/instructions_z")

################################################################################
# Fit linear mixed-effects models: multinomial variables -e.g., reinf. rate-   #
################################################################################
results = lmm_fit(
  dat = dat_prescaled[,i_controls],
  site = factor(X$site[i_controls]),
  cov = cbind(X$reinf_rate50, X$reinf_rate80, X$age, X$sex)[i_controls,],
  lin.hyp = matrix(c(0,1,0,0,0, 0,0,1,0,0), nrow = 2, byrow = TRUE),
  n.min = 9, impute_missing_cov = TRUE, prescaling = FALSE
)
nifti[] = results$lin.hyp_chisq_map
writeNIfTI(nifti, "gitHubCode_mega_analysis/reinf_rate_chisq")
nifti[] = results$lin.hyp_z_map
writeNIfTI(nifti, "gitHubCode_mega_analysis/reinf_rate_z")
