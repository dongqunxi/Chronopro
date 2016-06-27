from mne import (spatial_tris_connectivity,
                 grade_to_tris)
#from mne.epochs import equalize_epoch_counts
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from scipy import stats as stats
import os, glob, mne
import numpy as np
from dirs_manage import set_directory
from jumeg.jumeg_preprocessing import get_files_from_list


subjects_dir = os.environ['SUBJECTS_DIR'] 

def apply_inverse_ave(fnevo, min_subject='fsaverage'):
    
    from mne import make_forward_solution
    from mne.minimum_norm import write_inverse_operator
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        fn_inv = fname[:fname.rfind('-ave.fif')] + ',ave-inv.fif' 
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        #min_dir = subjects_dir + '/%s' %min_subject
        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,nr,fibp1-45-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-oct-6-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        [evoked] = mne.read_evokeds(fname)
        noise_cov = mne.read_cov(fn_cov)
        noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                        mag=0.05, grad=0.05, proj=True)
        fwd = make_forward_solution(evoked.info, fn_trans, fn_src, fn_bem)
        inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2)
        write_inverse_operator(fn_inv, inv)
        
def apply_STC_ave(fnevo, method='dSPM', snr=3.0, event='LLst', 
                      baseline=True, btmin=-0.3, btmax=-0.1, 
                      min_subject='fsaverage'):
    ''' Inverse evoked data into the source space. 
        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        method:string
            Inverse method, 'MNE' or 'dSPM'
        snr: float
            Signal to noise ratio for inverse solution.
        event: string
            The event name related with evoked data.
        baseline: bool
            If true, prestimulus segment from 'btmin' to 'btmax' will be saved, 
            If false, no baseline segment is saved.
        btmin: float
            The start time point (second) of baseline.
        btmax: float
            The end time point(second) of baseline.
        min_subject: string
            The subject name as the common brain.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse, read_inverse_operator
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        name = os.path.basename(fname)
        stc_name = name[:name.rfind('-ave.fif')] 
        fn_inv = fname[:fname.rfind('-ave.fif')] + ',ave-inv.fif' 
        subject = name.split('_')[0]
        min_dir = subjects_dir + '/%s' %min_subject
        # this path used for ROI definition
        stc_path = min_dir + '/%s_ROIs/%s' %(method,subject)
        #fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
        set_directory(stc_path) 
        snr = snr
        lambda2 = 1.0 / snr ** 2 
        #noise_cov = mne.read_cov(fn_cov)
        [evoked] = mne.read_evokeds(fname)
        inv = read_inverse_operator(fn_inv)
        stc = apply_inverse(evoked, inv, lambda2, method,
                            pick_ori=None)
        # Morph STC
        stc_morph = stc.morph('fsaverage')
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
        if baseline == True:
            stc_base = stc_morph.crop(btmin, btmax)
            stc_base.save(stc_path + '/%s_%s_baseline' % (subject, event), ftype='stc')
            
def Ara_contr(evt_list, tmin, tmax, conf_type, stcs_path, n_subjects=14, template='fsaverage'):
    con_stcs = []
    for evt in evt_list[:2]:
        print 'Condition: '+evt
        fn_stc_list1 = glob.glob(subjects_dir+'/fsaverage/dSPM_ROIs/*[0-9]/*fibp1-45,evtW_%s_bc-lh.stc' %evt)
        for fn_stc1 in fn_stc_list1[:n_subjects]:
            print fn_stc1
            stc1 = mne.read_source_estimate(fn_stc1, subject=template)
            stc1.crop(tmin, tmax)
            con_stcs.append(stc1.data)
        print
    cons = np.array(con_stcs).transpose(1,2,0) 
    
    #tmin = stc1.tmin 
    tstep = stc1.tstep 
    fsave_vertices = stc1.vertices
    del stc1
    print

    incon_stcs = []
    for evt in evt_list[2:]:
        print 'Condition: '+evt
        fn_stc_list2 = glob.glob(subjects_dir+'/fsaverage/dSPM_ROIs/*[0-9]/*fibp1-45,evtW_%s_bc-lh.stc' %evt)
        for fn_stc2 in fn_stc_list2[:n_subjects]:
            print fn_stc2
            stc2 = mne.read_source_estimate(fn_stc2, subject=template)
            stc2.crop(tmin, tmax)
            incon_stcs.append(stc2.data)
        print
    incons = np.array(incon_stcs).transpose(1,2,0)  
    del stc2
    X = [cons[:, :, :], incons[:, :, :]]
    #import pdb
    #pdb.set_trace()

    # save data matrix
    X = np.array(X).transpose(1,2,3,0)
    X = np.abs(X)  # only magnitude
    np.savez(stcs_path + '%s.npz' %conf_type, X=X, tstep=tstep, fsave_vertices=fsave_vertices)

    return tstep, fsave_vertices, X
    
def stat_clus(X, tstep, fsave_vertices, p_threshold=0.01, p=0.05, n_subjects=14, 
                  fn_stc_out=None):
    print('Computing connectivity.')
    connectivity = spatial_tris_connectivity(grade_to_tris(5))
    
    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])
    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    t_threshold = stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=2,
                                        threshold=t_threshold)
    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < p)[0]
    print 'the amount of significant clusters are: %d' %good_cluster_inds.shape
    ###############################################################################
    # Visualize the clusters
    # ----------------------
    print('Visualizing clusters.')
    
    #    Now let's build a convenient representation of each cluster, where each
    #    cluster becomes a "time point" in the SourceEstimate
    stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                                vertices=fsave_vertices,
                                                subject='fsaverage')
    stc_all_cluster_vis.save(fn_stc_out)
    #return stc_all_cluster_vis
    #    Let's actually plot the first "time point" in the SourceEstimate, which
    #    shows all the clusters, weighted by duration
    # blue blobs are for condition A < condition B, red for A > B



    
