# -*- coding: UTF-8 -*-

"""
.. _tut_stats_cluster_source_1samp:

=================================================================
Permutation t-test on source data with spatio-temporal clustering
=================================================================

This script is refferred from scripts from Alex. 
<alexandre.gramfort@telecom-paristech.fr> et al. You can find their
scripts: http://martinos.org/mne/stable/auto_tutorials/plot_stats_cluster_spatio_temporal.html
Tests if the evoked response is significantly different between
conditions across subjects.
The multiple comparisons problem is addressed with a cluster-level
permutation test across space and time.

"""
# Authors: Dong Qunxi <dongqunxi@gmail.com>
#         Jürgen Dammers <j.dammers@fz-juelich.de>
# License: BSD (3-clause)

import os, glob
#from dirs_manage import set_directory
from stat_cluster import stat_clus, Ara_contr, apply_inverse_ave, apply_STC_ave, mv_ave
print(__doc__)

###############################################################################
# Set parameters
# ------------------------------------------------
subjects_dir = os.environ['SUBJECTS_DIR'] 
# Set the path for storing STCs of conflicts processing
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
#set_directory(stcs_path) 
n_subjects = 14
template = 'fsaverage'
st_list = ['LLst', 'RRst', 'RLst',  'LRst']
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt']
st_max = 0.4
st_min = 0
res_min = -0.3
res_max = 0.1 
# The parameter for t-test
n_permutation = 512
# Set the option for conflicts perception or conflicts response
conf_per = False
conf_res = True

# Preparing for ROIs clusterring, if all are set false, 
# processed array is loaded directly
do_apply_invers_ave = False
do_apply_STC_ave = False
do_calc_matrix = False
   
#conflicts perception
if conf_per == True:
    evt_list = st_list
    tmin, tmax = st_min, st_max
    conf_type = 'conf_per' 
    baseline = True
    
#conflicts response
elif conf_res == True:
    evt_list = res_list
    tmin, tmax = res_min, res_max
    conf_type = 'conf_res'
    baseline = False



###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_invers_ave:
    print '>>> Calculate inverse solution ....'
    fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_LLst_bc-ave.fif')
    apply_inverse_ave(fn_evt_list)
    #for evt in st_list:
    #    fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-ave.fif' %evt)
    #    apply_inverse_ave(fn_evt_list)
    print '>>> FINISHED with inverse solution.'
    print ''
 

    
###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_STC_ave:
    print '>>> Calculate STC ....'
    for evt in evt_list:
        fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-ave.fif' %evt)
        apply_STC_ave(fn_evt_list, event=evt, baseline=baseline)
    print '>>> FINISHED with STC generation.'
    print ''
        


###############################################################################
# conflicts contrasts
# -----------------
if do_calc_matrix:
    tstep, fsave_vertices, X = Ara_contr(evt_list, tmin, tmax, conf_type, 
                                             stcs_path, n_subjects=n_subjects)
else:
    import numpy as np
    fnmat = stcs_path + conf_type + '.npz'
    npz = np.load(fnmat)
    tstep = npz['tstep'].flatten()[0]
    fsave_vertices = npz['fsave_vertices']
    X = npz['X']
    
# Moving average across timepoints, to reduce the sample size at the time dimension.
mv_window = 20 # miliseconds
overlap = 10 # miliseconds
X = mv_ave(X, mv_window, overlap, freqs=678.17)

# Left conflict contrasts
Y = X[:, :, :n_subjects, 1] - X[:, :, :n_subjects, 0]  # make paired contrast
fn_stc_out = stcs_path + 'mv_left_%s' %conf_type
stat_clus(Y, tstep, fsave_vertices, n_permutation, p_threshold=0.01, p=0.05,  n_subjects=n_subjects, fn_stc_out=fn_stc_out)
print Y.shape
del Y
# Right conflict contrasts
Z = X[:, :, n_subjects:, 1] - X[:, :, n_subjects:, 0]  # make paired contrast
fn_stc_out = stcs_path + 'mv_right_%s' %conf_type
stat_clus(Z, tstep, fsave_vertices, n_per=n_permutation, p_threshold=0.01, p=0.05, n_subjects=n_subjects, fn_stc_out=fn_stc_out)
print X.shape, Z.shape
del X, Z

###############################################################################
# plot significant clusters
# -----------------
#import mne
#fn_stc = stcs_path + 'left_conf_res-lh.stc' 
#fn_fig = fn_stc[:fn_stc.rfind('-lh.stc')] + '.tif'
#stc = mne.read_source_estimate(fn_stc)
#brain = stc.plot(subject='fsaverage', hemi='split', subjects_dir=subjects_dir,
#                                        time_label='Duration significant (ms)')
#brain.set_data_time_index(0)
#brain.show_view('lateral')
#brain.save_image(fn_fig)
# blue blobs are for condition A < condition B, red for A > B

