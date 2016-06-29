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
from stat_cluster import stat_clus, Ara_contr, apply_inverse_ave, \
                         apply_STC_ave, morph_STC, mv_ave, per2test
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
# The spatial resolution parameter for morphing, grade=4: 5124 vertices; 
# grade=5: 20484 vertices
grade = 5
# Moving average across timepoints, to reduce the sample size at the time dimension.
mv_window = 20 # miliseconds
overlap = 5 # miliseconds
nfreqs = 678.17
# The parameters for t-test
p_th = 0.001
p_v = 0.05
permutation = 1024 #8192
step_p = 0 # 0.05
# Set the option for conflicts perception or conflicts response
conf_per = False
conf_res = True

do_apply_invers_ave = False # Making inverse operator
do_apply_STC_ave = False # Inversing conduction
do_morph_STC_ave = False # STC morphing conduction
do_calc_matrix = False # Form the group matrix or load directly
do_mv_ave = True #The moving average conduction
do_ttest = True # 1sample t test conduction
do_ftest = False # 2sample f test conduction

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
    print '>>> FINISHED with inverse solution.'
    print ''
 

    
###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_STC_ave:
    print '>>> Calculate STC ....'
    for evt in evt_list:
        fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-ave.fif' %evt)
        apply_STC_ave(fn_evt_list)
    print '>>> FINISHED with STC generation.'
    print ''
        
###############################################################################
# Morph STC data for each condition
# ------------------------------------------------
if do_morph_STC_ave:
    print '>>> Calculate morphed STC ....'
    for evt in evt_list:
        fn_stc_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-lh.stc' %evt)
        morph_STC(fn_stc_list, grade=grade, event=evt, baseline=baseline)
    print '>>> FINISHED with morphed STC generation.'
    print ''

###############################################################################
# conflicts contrasts
# -----------------
if do_calc_matrix:
    print '>>> Calculate Matrix for contrasts ....'
    tstep, fsave_vertices, X = Ara_contr(evt_list, tmin, tmax, conf_type, 
                                             stcs_path, n_subjects=n_subjects)
    print '>>> FINISHED with a group matrix generation.'
    print ''

else:
    print '>>> load Matrix for contrasts ....'
    import numpy as np
    fnmat = stcs_path + conf_type + '.npz'
    npz = np.load(fnmat)
    tstep = npz['tstep'].flatten()[0]
    X = npz['X']
    print '>>> FINISHED with the group matrix loaded.'
    print ''
###############################################################################
# Moving averages across time dimension
# -----------------
if do_mv_ave:  
    print '>>> Moving averages with window length %dms ....' %(mv_window)
    conf_type = 'mv%d_' %overlap + conf_type  
    X = mv_ave(X, mv_window, overlap, freqs=nfreqs)
    print '>>> FINISHED with the smothed group matrix generation.'
    print ''
###############################################################################
# Clustering using 1sample t-test
# -----------------
if do_ttest:
    print '>>> ttest for clustering ....'
    conf_type = 'ttest_' + conf_type
    # Left conflict contrasts
    Y = X[:, :, :n_subjects, 1] - X[:, :, :n_subjects, 0]  # make paired contrast
    fn_stc_out = stcs_path + 'left_%s' %conf_type
    stat_clus(Y, tstep, n_per=permutation, p_threshold=p_th, p=p_v, step_p=step_p,
              fn_stc_out=fn_stc_out)
    print Y.shape
    del Y
    # Right conflict contrasts
    Z = X[:, :, n_subjects:, 1] - X[:, :, n_subjects:, 0]  # make paired contrast
    fn_stc_out = stcs_path + 'right_%s' %conf_type
    stat_clus(Z, tstep, n_per=permutation, p_threshold=p_th, p=p_v, step_p=step_p,
                fn_stc_out=fn_stc_out)
    print X.shape, Z.shape
    del Z
    print '>>> FINISHED with the clusters generation.'
    print ''
###############################################################################
# Clustering using 2sample f-test
# -----------------
if do_ftest:
    print '>>> 2smpletest for clustering ....'
    conf_type = 'ftest_' + conf_type
    # Left conflict contrasts
    X1 = X[:, :, :n_subjects, 1]
    X2 = X[:, :, :n_subjects, 0]
    fn_stc_out = stcs_path + 'left_%s' %conf_type
    per2test(X1, X2, p_thr=p_th, p_v=p_v, tstep=tstep, n_per=permutation, step_p=step_p, 
             fn_stc_out=fn_stc_out)
    print X1.shape
    del X1, X2
    # Right conflict contrasts
    X3 = X[:, :, n_subjects:, 1]
    X4 = X[:, :, n_subjects:, 0]
    fn_stc_out = stcs_path + 'right_%s' %conf_type
    per2test(X3, X4, p_thr=p_th, p_v=p_v, tstep=tstep, n_per=permutation, step_p=step_p,
             fn_stc_out=fn_stc_out)
    print X3.shape
    del X, X3, X4
    print '>>> FINISHED with the clusters generation.'
    print ''
###############################################################################
# plot significant clusters
# -----------------
import mne
fn_stc = stcs_path + 'left_conf_res-lh.stc' 
fn_fig = fn_stc[:fn_stc.rfind('-lh.stc')] + '.tif'
stc = mne.read_source_estimate(fn_stc)
brain = stc.plot(subject='fsaverage', hemi='split', subjects_dir=subjects_dir,
                                        time_label='Duration significant (ms)')
brain.set_data_time_index(0)
brain.show_view('lateral')
#brain.save_image(fn_fig)
# blue blobs are for condition A < condition B, red for A > B

