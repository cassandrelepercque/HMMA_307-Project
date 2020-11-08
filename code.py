# -*- coding: utf-8 -*-
"""
@author: Cassandre Lepercque

In short: Usage of Linear mixed effects models on trophic
        position of fish in different lakes.
"""

################################
# Packages needed
################################

from download import download
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

################################
# Download datasets
################################

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

fish = "http://qcbs.ca/wiki/_media/qcbs_w6_data.csv"
path_fish = os.path.join(results_dir, "fish.txt")

download(fish, path_fish, replace=False)

df_fish = pd.read_csv(path_fish, sep=",", header=0)
print(df_fish)

# Let's define the abline code for figures
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

###############################
#       Dataset
###############################

df_fish.describe() # quick description
array_fish = df_fish.values # table of values
lake = df_fish["Lake"]
lake.value_counts() # number of values ​​in each lake
fish_species = df_fish["Fish_Species"]
fish_species.value_counts() # number of values ​​in each fish species

#################
#   Histograms
# ~~~~~~~~~~~~~~~

# Histogram of fish length
plt.figure(figsize=(5, 5))
plt.hist(df_fish['Fish_Length'], density=True, bins=50)
plt.xlabel('Fish length')
plt.ylabel('Frequency')
plt.title("Histogram of fish length")
ax = sns.kdeplot(df_fish['Fish_Length'], shade=True, cut=0, bw=10)
ax.legend().set_visible(False)
plt.tight_layout()
plt.savefig('hist_fish_length.pdf')

# Histogram of trophic position
plt.figure(figsize=(5, 5))
plt.hist(df_fish['Trophic_Pos'], density=True, bins=50)
plt.xlabel('Trophic position')
plt.ylabel('Frequency')
plt.title("")
ax = sns.kdeplot(df_fish['Trophic_Pos'], shade=True, cut=0, bw=10)
ax.legend().set_visible(False)
plt.tight_layout()
plt.savefig('hist_trophic_pos.pdf')

##############################################
#   Evaluate collinearity between variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Pairplot
sns.pairplot(df_fish, height=1.5, hue="Fish_Species", 
             markers=["o", "s", "D"], diag_kind="hist")
plt.show()
plt.savefig('data_visual.pdf')

# correlation between the fishes length and the trophic position
np.corrcoef(df_fish["Fish_Length"], df_fish["Trophic_Pos"])

###############################
#           Data scale
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Correction of Z : (Z = (x - mean(x))/sd(x))
# Corrected length
df_fish["Z_Length"] = (df_fish["Fish_Length"] - 
       np.mean(df_fish["Fish_Length"]))/np.std(df_fish["Fish_Length"])

# Corrected trophic position
df_fish["Z_TP"] = (df_fish["Trophic_Pos"] - 
       np.mean(df_fish["Trophic_Pos"]))/np.std(df_fish["Trophic_Pos"])

###############################
#     Studies of models
###############################

#################
# Linear model
# ~~~~~~~~~~~~~~~

lm_fish = ols('Z_TP~Z_Length', data=df_fish).fit()
print(lm_fish.summary())

# We add the column "residues" to our base table
resid = lm_fish.resid_pearson
df_fish["Residual"] = lm_fish.resid_pearson

# Boxplot of fish species effect
plt.figure(figsize=(8,5))
sns.boxplot(x=df_fish['Fish_Species'], y=df_fish['Residual'],
            hue='Fish_Species', data=df_fish)
plt.title("Species effect")
abline(0,0)
plt.xlabel('Fish Species')
plt.ylabel('Standardized Residuals')
plt.tight_layout()
plt.savefig('boxplot_species_effect.pdf')

# Boxplot of lake effect
plt.figure(figsize=(10,5))
sns.boxplot(x=df_fish['Lake'], y=df_fish['Residual'],
            hue='Lake', data=df_fish)
plt.title("Lake effect")
abline(0,0)
plt.xlabel('Type of Lake')
plt.ylabel('Standardized Residuals')
plt.tight_layout()
plt.savefig('boxplot_lake_effect.pdf')

#########################
# Mixed linear model
# ~~~~~~~~~~~~~~~~~~~~~~~
# 8 Models for comparison : MLM's Z_TP by Z_Length

# Linear model without random effect
M0 = ols('Z_TP~Z_Length', data=df_fish).fit()

# Complete model with different intercepts
vc = {'Lake' : '0+C(Lake)'}
M1 = smf.mixedlm("Z_TP ~ Z_Length ", data=df_fish, vc_formula=vc, 
                 re_formula='1', 
                 groups=df_fish['Fish_Species']).fit(reml=False)

# Complete model with differents intercepts and slopes
vc = {'Lake' : '0+C(Lake)'}
M2 = smf.mixedlm('Z_TP ~ Z_Length', vc_formula=vc, 
                             re_formula='1+Z_Length', 
                             groups=df_fish['Fish_Species'], 
                             data = df_fish).fit(reml=False)

# No Lake effect, random intercept only
M3 = smf.mixedlm("Z_TP~Z_Length", data=df_fish, 
                 groups=df_fish["Fish_Species"]).fit(reml=False)

# No Species effect, random intercept only
M4 = smf.mixedlm("Z_TP~Z_Length", data=df_fish, 
                 groups=df_fish["Lake"]).fit(reml=False)

# No Lake effect, random intercept and slope
M5 = smf.mixedlm("Z_TP~Z_Length", data=df_fish, 
                 groups=df_fish["Fish_Species"], 
                 re_formula="~Z_Length").fit(reml=False)

#No Species effect, random intercept and slope
M6 = smf.mixedlm("Z_TP~Z_Length", data=df_fish, 
                 groups=df_fish["Lake"], 
                 re_formula="~Z_Length").fit(reml=False)

# Complete model with intercepts and slopes varying by lake
vc = {'Fish_Species' : '0+C(Fish_Species)'}
M7 = smf.mixedlm('Z_TP~Z_Length + Lake', vc_formula=vc, 
                 re_formula='1+Z_Length', data=df_fish, 
                 groups=df_fish['Lake']).fit(reml=False)

# Complete model with intercepts and slopes varying by species
vc = {'Lake' : '0+C(Lake)'}
M8 = smf.mixedlm('Z_TP~Z_Length + Fish_Species', vc_formula=vc, 
                 re_formula='1+Z_Length', data=df_fish, 
                 groups=df_fish['Fish_Species']).fit(reml=False)

#########################
#         AIC
# ~~~~~~~~~~~~~~~~~~~~~~~

# Deviance calculation
# Formula : deviance=−2∗log likelihood
dev_M0 = (-2)*M0.llf
dev_M1 = (-2)*M1.llf
dev_M2 = (-2)*M2.llf
dev_M3 = (-2)*M3.llf
dev_M4 = (-2)*M4.llf
dev_M5 = (-2)*M5.llf
dev_M6 = (-2)*M6.llf
dev_M7 = (-2)*M7.llf
dev_M8 = (-2)*M8.llf

# AIC calculation
# Formula : AIC=deviance+2∗(p+1), where : p = number of parameters
AIC_M0= dev_M0 + 2*(2+1)
AIC_M1 = dev_M1 + 2*(5+1)
AIC_M2 = dev_M2 + 2*(6+1)
AIC_M3 = dev_M3 + 2*(4+1)
AIC_M4 = dev_M4 + 2*(4+1)
AIC_M5 = dev_M5 + 2*(5+1)
AIC_M6 = dev_M6 + 2*(5+1)
AIC_M7 = dev_M7 + 2*(6+1)
AIC_M8 = dev_M8 + 2*(6+1)

#########################
#     Table of AIC
# ~~~~~~~~~~~~~~~~~~~~~~~

df_AIC = pd.DataFrame(["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
                       columns=['Models'])
df_AIC['AIC']=[AIC_M0, AIC_M1, AIC_M2, AIC_M3, AIC_M4, AIC_M5, \
      AIC_M6, AIC_M7, AIC_M8]
print(df_AIC)

#################################
#   Choice of the best model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# M8 and M2 have the lowest AIC
# Let's compare with REML=TRUE
vc = {'Lake' : '0+C(Lake)'}
M8 = smf.mixedlm('Z_TP~Z_Length + Fish_Species', vc_formula=vc, 
                 re_formula='1+Z_Length', data=df_fish, 
                 groups=df_fish['Fish_Species']).fit(reml=True)

vc = {'Lake' : '0+C(Lake)'}
M2 = smf.mixedlm('Z_TP ~ Z_Length', vc_formula=vc, 
                             re_formula='1+Z_Length', 
                             groups=df_fish['Fish_Species'],
                             data = df_fish).fit(reml=True)
# Model 8 is the best one.

###############################
#     Model validation
###############################

#################################
#          Homogeneity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

E1 = M8.resid
F1 = M8.fittedvalues

# Figure
plt.figure()
plt.plot(F1, E1, 'o')
abline(0,0)
plt.title('Model homogeneity')
plt.xlabel('Fitted values')
plt.ylabel('Normalized riduals')
plt.show()
plt.savefig('Model_validation_homo.pdf')

#################################
#       Independence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Fish body length
plt.figure()
plt.plot(df_fish["Z_Length"], E1, 'o')
abline(0,0)
plt.xlabel('Z_Length')
plt.ylabel('Normalized riduals')
plt.legend(loc='best')
plt.show()
plt.savefig('Model_validation_ind.pdf')

# Species
plt.figure()
sns.boxplot(x=df_fish['Fish_Species'], y=E1,
            hue='Fish_Species', data=df_fish)
plt.title("Species effect")
abline(0,0)
plt.xlabel('Fish Species')
plt.ylabel('Normalized riduals')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('boxplot_mod_vali_species.pdf')

# Lake
plt.figure()
sns.boxplot(x=df_fish['Lake'], y=E1,
            hue='Lake', data=df_fish)
plt.title("Lake effect")
abline(0,0)
plt.xlabel('Type of lake')
plt.ylabel('Normalized riduals')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('boxplot_mod_vali_lake.pdf')

#################################
#       Normality
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Histogram
plt.figure()
plt.hist(E1, density=True, bins=50)
plt.title('Histogram of the normalized residuals')
plt.xlabel('Normalized residuals')
plt.ylabel('Frequency')
plt.show()
plt.savefig('hist_check_normality.pdf')

# All data visualization
plt.figure()
plt.plot(df_fish['Z_Length'], df_fish['Z_TP'], 'o')
plt.title('All data')
plt.xlabel('Length (mm)')
plt.ylabel('Trophic position')
abline(slope = 0.423, intercept = -1.079)
plt.show()
plt.savefig('Model_visual_all_data.pdf')