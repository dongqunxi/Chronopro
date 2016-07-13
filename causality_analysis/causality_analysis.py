# -*- coding: UTF-8 -*-

"""

=================================================================
Causality analysis
=================================================================



"""
# Authors: Dong Qunxi <dongqunxi@gmail.com>
#         Jürgen Dammers <j.dammers@fz-juelich.de>
# License: BSD (3-clause)

import os, glob
#from dirs_manage import set_directory
from apply_causality import apply_inverse_oper, apply_STC_epo
from apply_causality import normalize_data, model_estimation, causal_analysis
print(__doc__)

###############################################################################
# Set parameters
# ------------------------------------------------
subjects_dir = os.environ['SUBJECTS_DIR'] 
# Set the path for storing STCs of conflicts processing
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
#set_directory(stcs_path) 
template = 'fsaverage'
#st_list = ['LLst', 'RRst', 'RLst',  'LRst']
st_list = ['RRst', 'RLst',  'LRst']
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt']
#st_max = 0.4
#st_min = 0
#res_min = -0.3
#res_max = 0.1 

# Cluster operation
do_apply_invers_oper = False # Making inverse operator
do_apply_STC_epo = False # Making STCs
    
###############################################################################
# Make inverse operator for each subject
# ------------------------------------------------
if do_apply_invers_oper:
    print '>>> Calculate Inverse Operator ....'
    fn_epo_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*ocarta,evtW_LLst_bc-epo.fif')
    apply_inverse_oper(fn_epo_list)
    print '>>> FINISHED with STC generation.'
    print ''
        
###############################################################################
# Makeing STCs
# ------------------------------------------------
if do_apply_STC_epo:
    print '>>> Calculate morphed STCs ....'
    for evt in st_list:
        fn_epo = glob.glob(subjects_dir+'/*[0-9]/MEG/*ocarta,evtW_%s_bc-epo.fif' %evt)
        apply_STC_epo(fn_epo, event=evt)
    print '>>> FINISHED with morphed STC generation.'
    print ''

#fn_ts = '/home/qdong/data/Chrono/18subjects/stcs/LLst_labels_ts.npy'
#normalize_data(fn_ts) 
#MVAR model construction and evaluation, individual causality analysis for
#each condition 
fn_norm = '/home/qdong/data/Chrono/18subjects/stcs/LLst_labels_ts,1-norm.npy'
model_estimation(fn_norm)
causal_analysis(fn_norm, method='GPDC')
#Estimate the significance of each causal matrix.
#fn_cau = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s_labels_ts,1-norm,cau.npy' %evt_st)
#sig_thresh(cau_list=fn_cau, condition=evt_st, sfreq=sfreq)