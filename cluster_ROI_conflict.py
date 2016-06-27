
"""
.. _tut_stats_cluster_source_1samp:

=================================================================
Permutation t-test on source data with spatio-temporal clustering
=================================================================

Tests if the evoked response is significantly different between
conditions across subjects (simulated here using one subject's data).
The multiple comparisons problem is addressed with a cluster-level
permutation test across space and time.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD (3-clause)

import os, glob
import numpy as np
from dirs_manage import set_directory
from stat_cluster import stat_clus, Ara_contr, apply_inverse_ave, apply_STC_ave
print(__doc__)

###############################################################################
# Set parameters
# ------------------------------------------------
subjects_dir = os.environ['SUBJECTS_DIR'] 
#n_subjects = 7
n_subjects = 14
ds_factor = 2
template = 'fsaverage'
st_list = ['LLst', 'RRst', 'RLst',  'LRst']
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt']
st_max = 0.4
st_min = 0
res_min = -0.3
res_max = 0.1 

# Set the option for conflicts perception or conflicts response
conf_per = False
conf_res = True
# Set the path for storing STCs of conflicts processing
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
set_directory(stcs_path) 
   
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

do_apply_invers_ave = False
do_apply_STC_ave = False
do_calc_matrix = False


###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_invers_ave:
    print '>>> Calculate inverse solution ....'
    for evt in evt_list:
        fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-ave.fif' %evt)
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
    fnmat = stcs_path + conf_type + '.npz'
    npz = np.load(fnmat)
    tstep = npz['tstep'].flatten()[0]
    fsave_vertices = npz['fsave_vertices']
    X = npz['X']
    
    

# Downsampling timepoints.
from scipy.signal import decimate
X = decimate(X, ds_factor, axis=1)
print X.shape


# Left conflict contrasts
Y = X[:, :, :n_subjects, 1] - X[:, :, :n_subjects, 0]  # make paired contrast
fn_stc_out = stcs_path + 'left_%s' %conf_type
stat_clus(Y, tstep, fsave_vertices, fn_stc_out=fn_stc_out, n_subjects=n_subjects)
print Y.shape
del Y


# Right conflict contrasts
Z = X[:, :, n_subjects:, 1] - X[:, :, n_subjects:, 0]  # make paired contrast
fn_stc_out = stcs_path + 'right_%s' %conf_type
stat_clus(Z, tstep, fsave_vertices, fn_stc_out=fn_stc_out)
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
