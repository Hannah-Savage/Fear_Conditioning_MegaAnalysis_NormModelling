################################################################################
# Check differences in means and variances before ComBat (in controls)         #
################################################################################
i_controls = which(X$patient == 0)
m = lm(stai ~ country + age + sex, data = X[i_controls,])
anova(m) # country p < 0.001
leveneTest(residuals(m) ~ country, data = X[i_controls,]) # p < 0.001

################################################################################
# Fit and apply ComBat, with age and sex as covariates                         #
# Note that ComBat is fitted using controls' scores but applied to all         #
################################################################################
age_sex = cbind(X$age, X$sex)
combat = combat_fit(X$stai[i_controls],
                    site = X$country[i_controls], cov = age_sex[i_controls,],
                    n.min = 8, impute_missing_cov = TRUE)
X$stai  = combat_apply(combat, X$stai, site = X$country, cov = age_sex)$dat

################################################################################
# Check differences in means and variances after ComBat (in controls)          #
################################################################################
m = lm(stai ~ country + age + sex, data = X[i_controls,])
anova(m) # country p = 0.612
leveneTest(residuals(m) ~ country, data = X[i_controls,]) # p = 0.536
