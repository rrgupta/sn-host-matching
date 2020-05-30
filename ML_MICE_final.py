#!/usr/bin/env python
""" 
Plot efficiency, purity from RandomForest classifier
http://scikit-learn.org
"""
import os, sys
import numpy as np
from astropy.table import Table, join, vstack
from astropy.io.ascii import write, read
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import math as m
from matplotlib import rc
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.interpolate import interp1d
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import argparse
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

############ Parse arguments ############
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    description='This script uses a machine learning (ML) algorithm to train a binary classifier on supernova-host galaxy matched data into classes {correct match, wrong match}. Select Random Forest features to use.')
# Choose features
parser.add_argument('--ft', 
    choices=['sep','DLR','Nobj','D12','D1D2','D13','D1D3','S12','S12_DLR','S1S2','S13','S1S3','HC','A','B','BoA','photoz','MAG_I'],
    nargs='+', default=['sep','DLR','D12','D1D2','D13','D1D3','S12','S12_DLR','S1S2','S13','S1S3','HC','A','BoA','MAG_I'],
    help='Choose SN-host match features to use for classification. List them (space-separated) and select from: {%(choices)s}',
    metavar='features')
# Select number of cores
parser.add_argument('--nc', type=int, 
    choices=range(1,8), default=4, help='Number of cores to use for parallelization', metavar='n_cores')
# Choose filename
parser.add_argument('--filestr', 
    default='test', help='Choose string to append to filename')
args = parser.parse_args()


################ FUNCTIONS ##############

# get only the selected features
def get_features(features, data):
    list = [] 
    for f in features:
        list.append(data[f])
    X = np.vstack(list).T
    return X

# find index of array where value is nearest given value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# score function to compute purity at fixed efficiency (=TPR)
def score_func(probs, y):
    correct = (y==1)
    wrong = (y==0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=1)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear') # reverse-order so x is monotonically increasing
    metric = purity_func(0.98) # purity at fixed efficiency=98%
    return float(metric)

#############################################

data_train  = read('/nova-data/host_galaxies/DES_host_matching/ML/MICE100K_cuts_features_balanced_2-1-1.txt')
data_test =  read('/nova-data/host_galaxies/DES_host_matching/ML/MICE50K_cuts_10K_hostless_remainder_features.txt')
X_train = get_features(args.ft, data_train)
y_train = data_train['class'] # class 1=correct match, 0=wrong match
X_test = get_features(args.ft, data_test)
y_test = data_test['class']

print 'Training set size = ', len(y_train)
print 'Test set size = ', len(y_test)

# build a classifier
n_estimators = 100
max_features = 10
min_samples_split = 70
clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, \
      min_samples_split=min_samples_split, criterion='entropy', n_jobs=args.nc)
### fit the training data
print '\nTraining classifier across %d cores  . . . ' %args.nc
clf.fit(X_train, y_train)


### determine feature importances for tree-based estimators (RF, AdaBoost)
print '\nComputing feature importances . . .'
importances = clf.feature_importances_
F = Table([args.ft, importances], names=('Feature','Importance'), dtype=['a','f4'])
F.sort('Importance')
F.reverse()
print
print F

probs = clf.predict_proba(X_test)[:, 1] # good matches are class 1
correct = (y_test==1)
wrong = (y_test==0)

data_test['Prob'] = probs
print data_test

pur, eff, thresh = precision_recall_curve(y_test, probs, pos_label=1)
#y = thresh[::-1][5000:-1]
#x = eff[::-1][5001:-1]
y = thresh[::-1][1000:-1]
x = eff[::-1][1001:-1]


efficiency_func = interp1d(x, y, kind='linear') # reverse-order so x is monotonically increasing
P_eff98 = efficiency_func(0.98) # threshold probability at efficiency=98%
print '\nProb (eff=98%) =', P_eff98
print 'Purity (P_thresh=0) = ', pur[0]
score = score_func(probs, y_test)
print 'SCORE (pur @ eff=98%) = ', score

correct_match = data_test['Prob']>P_eff98
print 'number of correct matches with P_thresh {} = {}'.format(P_eff98, np.sum(correct_match))
print 'number of wrong matches with P_thresh {} = {}'.format(P_eff98, np.sum(~correct_match))

########### PLOTS ###################

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
p_binwidth = 0.05
p_bins = np.arange(-0.1, 1.05, p_binwidth)
plt.hist(probs[correct], bins=p_bins, histtype='stepfilled', \
                 color='green', alpha=0.5, label='true correct match')
plt.hist(probs[wrong], bins=p_bins, histtype='step', \
                 color='red', lw=2, label='true wrong match', hatch='////////')
ax1.set_xlim(0,1.0)
ax1.set_xlabel('Probability of correct match class, $P_{corr}$')
ax1.set_ylabel('Number')
ax1.legend(loc='upper left', fontsize='medium')

ax2 = fig.add_subplot(122)
ax2.plot(thresh, eff[:-1], 'b--', label='efficiency', lw=2)
ax2.plot(thresh, pur[:-1], c='r', label='purity', lw=2)
#ax2.plot(eff[:-1], pur[:-1], c='g', label='pur v. eff', lw=2)
ax2.set_xlabel('Threshold Probability for Classification, $P_{thresh}$')
ax2.set_ylabel('Efficiency, Purity [$P_{corr} \ge P_{thresh}$]')
ax2.legend(loc='lower right', fontsize='medium')
ax2.set_ylim(0.85, 1.0)
ax2.set_xlim(0, 0.5)

fig.tight_layout()
fig.savefig('eff_pur_MICE_{}.pdf'.format(args.filestr))
