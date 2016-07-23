import os, glob, mne
import numpy as np
from jumeg.jumeg_preprocessing import get_files_from_list, reset_directory, set_directory
import math
import matplotlib.pylab as plt
subjects_dir = os.environ['SUBJECTS_DIR']
########################################
#  ROIs definition
#######################################
def grow_ROIs(fn_ver, radius, min_subject='fsaverage'):
    ''' 
        Concentrate the anatomical labels. 
        Parameter
        ---------
        fn_stc: string
            Path of common STC.
        radius: float 
            The radius size of each functional labels.
        tmin: float
            The start timepoint(second) of common STC.
        tmax: float
            The end timepoint(second) of common STC.
        min_subject: string
            The common subject.
        fn_ana_list: string
            The path of the file including pathes of anatomical labels.
    '''     
    min_path = subjects_dir+'/fsaverage'
    # Make sure the target path is exist
    funcs_path = min_path + '/dSPM_ROIs/anno_ROIs/funcs/'
    reset_directory(funcs_path)
    # Read the stc for concentrating the anatomical labels
    stc = mne.read_source_estimate(fn_stc)
    with open(fn_ana_list, 'r') as fl:
                file_list = [line.rstrip('\n') for line in fl]
    fl.close()
    for f in file_list:
        label = mne.read_label(f)
        #stc_mean_label = stc_mean.in_label(label)
        stc_label = stc.in_label(label)
        src_pow = np.sum(stc_label.data ** 2, axis=1)
        if label.hemi == 'lh':
            h = 0
        elif label.hemi == 'rh':
            h = 1
        # Get the max MNE value within each ROI
        seed_vertno = stc_label.vertices[h][np.argmax(src_pow)]
        func_label = mne.grow_labels(min_subject, seed_vertno,
                                    extents=radius, hemis=h,
                                    subjects_dir=subjects_dir,
                                    n_jobs=1)
        func_label = func_label[0]
        func_label.save(funcs_path + '%s' %label.name)

def _merge_rois(mer_path, label_list):
    """
    subfunctions of apply_merge
    Parameter
    ----------
    mer_path: str
        The directory for storing merged ROIs.
    label_list: list
        Labels to be merged
    """
    class_list = []
    class_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong is False):
            class_label = mne.read_label(class_list[i])
            label_name = class_label.name
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            overlapped = len(np.intersect1d(test_label.vertices,
                                            class_label.vertices))
            if overlapped > 0:
                com_label = test_label + class_label
                pre_test = test_label.name.split('_')[0]
                pre_class = class_label.name.split('_')[0]
                #label_name = pre_class + '_%s-%s' %(pre_test,class_label.name.split('-')[-1])
                if pre_test != pre_class:
                    pre_class += ',%s' % pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' % pre
                    new_pre += pre_class[-1]
                    label_name = '%s_' % (new_pre) + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                fn_newlabel = mer_path + '%s.label' %label_name
                if os.path.isfile(fn_newlabel):
                    fn_newlabel = fn_newlabel[:fn_newlabel.rfind('_')] + '_new,%s' %fn_newlabel.split('_')[-1]
                mne.write_label(fn_newlabel, com_label)
                class_list[i] = fn_newlabel
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)
''' 
   Once we get ROIs, we need to make a pathes file named 'func_label_list.txt' manually,
   which includs the path of each ROI.This file as the indices of ROIs for causality 
   analysis.
'''

def apply_merge(labels_path):
    ''' 
        Merge the concentrated ROIs. 
        Parameter
        ---------
        labels_path: string.
            The path of concentrated labels.
    ''' 
    import shutil
    mer_path = labels_path + '/merge/'
    reset_directory(mer_path)
    source = []
    source_path = labels_path + '/funcs/'
    source = glob.glob(os.path.join(source_path, '*.*'))
    for filename in source:
        shutil.copy(filename, mer_path)
    reducer = True
    while reducer:
        list_dirs = os.walk(mer_path)
        label_list = ['']
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label_list.append(label_fname)
        label_list = label_list[1:]
        len_class = _merge_rois(mer_path, label_list)
        if len_class == len(label_list):
            reducer = False
            
########################################
#  Causality analysis
#######################################
def apply_inverse_oper(fnepo, tmin=-0.2, tmax=0.8):
    '''  
        Parameter
        ---------
        fnepo: string or list
            The epochs file with ECG, EOG and environmental noise free.
        tmax:float
            The end timepoint(second) of each epoch.
    '''
    #Get the default subjects_dir
    from mne import make_forward_solution
    from mne.minimum_norm import write_inverse_operator

    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        subject_path = subjects_dir + '/%s' %subject
        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-oct-6-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        fn_inv = fn_path + '/%s_epo-inv.fif' %subject
        
        epochs = mne.read_epochs(fname)
        epochs.crop(tmin, tmax)
        epochs.pick_types(meg=True, ref_meg=False)
        noise_cov = mne.read_cov(fn_cov)
        fwd = make_forward_solution(epochs.info, fn_trans, fn_src, fn_bem)
        fwd['surf_ori'] = True
        inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2,
                                                     depth=0.8, limit_depth_chs=False)
        write_inverse_operator(fn_inv, inv)
        
def apply_STC_epo(fnepo, event, method='dSPM', snr=3.0, min_subject='fsaverage'): 
    
    from mne.minimum_norm import apply_inverse_epochs  
    from mne import morph_data
    from mne.minimum_norm import read_inverse_operator     
    fnlist = get_files_from_list(fnepo)
    # loop across all filenames
    for fname in fnlist:        
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        min_dir = subjects_dir + '/%s' %min_subject
        snr = snr
        lambda2 = 1.0 / snr ** 2
        stcs_path = min_dir + '/stcs/%s/%s/' % (subject,event)
        reset_directory(stcs_path)
        #fn_inv = fname[:fname.rfind('-ave.fif')] + ',ave-inv.fif' 
        fn_inv = fn_path + '/%s_epo-inv.fif' %subject

        #noise_cov = mne.read_cov(fn_cov)
        epo = mne.read_epochs(fname)
        epo.pick_types(meg=True, ref_meg=False)
        inv = read_inverse_operator(fn_inv)
        stcs = apply_inverse_epochs(epo, inv, lambda2, method,
                            pick_ori='normal')
        s = 0
        while s < len(stcs):
            stc_morph = morph_data(subject, min_subject, stcs[s])
            stc_morph.save(stcs_path + '/trial%s_fsaverage'
                            % (str(s)), ftype='stc')
            s = s + 1
        
def cal_labelts(stcs_path, fn_func_list, condition='LLst', min_subject='fsaverage'):
    '''Extract stcs from special ROIs, and store them for funther causality
       analysis.
       Parameter
       ---------
       stcs_path: string
            The path of stc's epochs.
       fn_ana_list: string
            The path of the file including pathes of functional labels.
       condition: string
            The condition for experiments.
       min_subject: the subject for common brain
    '''
    path_list = get_files_from_list(stcs_path)
    minpath = subjects_dir + '/%s' % (min_subject)
    srcpath = minpath + '/bem/fsaverage-ico-5-src.fif'
    src_inv = mne.read_source_spaces(srcpath)
    # loop across all filenames
    for stcs_path in path_list:
        caupath = stcs_path[:stcs_path.rfind('/%s' %condition)] 
        fn_stcs_labels = caupath + '/%s_labels_ts.npy' % (condition)
        _, _, files = os.walk(stcs_path).next()
        trials = len(files) / 2
        # Get unfiltered and morphed stcs
        stcs = []
        i = 0
        while i < trials:
            fn_stc = stcs_path + 'trial%s_fsaverage' % (str(i))
            stc = mne.read_source_estimate(fn_stc + '-lh.stc',
                                           subject=min_subject)
            stcs.append(stc)
            i = i + 1
        # Get common labels
        list_file = fn_func_list
        with open(list_file, 'r') as fl:
                  file_list = [line.rstrip('\n') for line in fl]
        fl.close()
        rois = []
        labels = []
        for f in file_list:
            label = mne.read_label(f)
            labels.append(label)
            rois.append(label.name)
        # Extract stcs in common labels
        label_ts = mne.extract_label_time_course(stcs, labels, src_inv,
                                                 mode='pca_flip')
        #make label_ts's shape as (sources, samples, trials)
        label_ts = np.asarray(label_ts).transpose(1, 2, 0)
        np.save(fn_stcs_labels, label_ts)
        
def normalize_ICA(fn_ICA, fs=678, pre_t=0.2, evt_list=['LLst', 'LRst', 'RRst',  'RLst']):
    path_list = get_files_from_list(fn_ICA)
    from scipy.io import savemat
    # loop across all filenames
    for fnICA in path_list:
        npz = np.load(fnICA)
        fn_classes = fnICA[:fnICA.rfind('.npz')] + ',classes'
        fnnorm = fnICA[:fnICA.rfind('.npz')] + ',norm'
        data = npz['temporal_envelope']
        data = data.squeeze()
        #ts = []
        n = data.shape[1]
        classes = []
        i = 0
        for evt in evt_list:
            iclass = [evt for x in range(n)]
            classes.append(iclass)
            i = i + 1
        classes = np.array(classes).flatten()
        #ts = np.array(ts)
        ts = np.concatenate(data, axis=0)
        d_pre = ts[:, :, :int(pre_t*fs)]
        d_pos = ts[:, :, int(pre_t*fs):]
        d_mu = d_pre.mean(axis=-1, keepdims=True)
        d_std = d_pre.std(axis=-1, ddof=1, keepdims=True)
        z_data = (d_pos - d_mu) / d_std  
        np.save(fnnorm, z_data)    
        np.save(fn_classes, classes)
        fnmat = fnnorm[:fnnorm.rfind('.npz')] + '.mat'
        mdata = {'data': z_data.transpose(1,2,0)}
        savemat(fnmat, mdata)

        
def normalize_data(ts_path, fs=678, pre_t=0.2, evt_list=['LLst', 'LRst', 'RRst',  'RLst']):
    '''
       Before causal model construction, labelts need to be normalized further:
        1) Downsampling for reducing the time consuming.
        2) Apply Z-scoring to each STC.
       Parameter
       ---------
       fnts: string
           The file name of representative STCs for each ROI.
       factor: int
          The factor for downsampling.
    '''
    path_list = get_files_from_list(ts_path)
    from scipy.io import savemat
    # loop across all filenames
    for fnpath in path_list:
        subject = fnpath.split('/')[-1]
        fnnorm = fnpath + '/%s_conditions4_norm.npy' %subject
        fn_classes = fnpath + '/%s_classes.npy' %subject
        classes = []
        ts = []
        for evt in evt_list:
            fnts = fnpath + '/%s_labels_ts.npy' %evt
            label_ts = np.load(fnts)
            iclass = [evt for x in range(label_ts.shape[-1])]
            classes.append(iclass)
            ts.append(label_ts)
        classes = np.array(classes).flatten()
        ts = np.array(ts)
        ts = np.concatenate(ts, axis=-1)  
        d_pre = ts[:, :int(pre_t*fs), :]
        d_pos = ts[:, int(pre_t*fs):, :]
        d_mu = d_pre.mean(axis=1, keepdims=True)
        d_std = d_pre.std(axis=1, ddof=1, keepdims=True)
        z_data = (d_pos - d_mu) / d_std      
        np.save(fn_classes, classes)
        np.save(fnnorm, np.transpose(z_data,(2,0,1)))
        fnmat = fnnorm[:fnnorm.rfind('.npy')] + '.mat'
        mdata = {'data': z_data}
        savemat(fnmat, mdata)

def causal_analysis(fn_norm, morder=50, method='GPDC'):
    '''
        Calculate causality matrices of real data and surrogates.
        Parameters
        ----------
        fnnorm: string
            The file name of model order estimation.
        morder: int
            The optimized model order.
        method: string
            causality measures.
    '''
    import scot.connectivity_statistics as scs
    from scot.connectivity import connectivity
    import scot
    path_list = get_files_from_list(fn_norm)
    # loop across all filenames
    for fnnorm in path_list:
        fncau = fnnorm[:fnnorm.rfind('.npy')] + ',cau.npy'
        fnsurr = fnnorm[:fnnorm.rfind('.npy')] + ',surrcau.npy'
        X = np.load(fnnorm)
        mvar = scot.var.VAR(morder)
        c_surrogates = scs.surrogate_connectivity(method, X, mvar,
                                        repeats=1000)
        mvar.fit(X)
        con = connectivity(method, mvar.coef, mvar.rescov)
        np.save(fncau, con)
        np.save(fnsurr, c_surrogates)
        
def app_causal(fn_list, morder=50, method='GPDC'):
    import scot
    for fn_norm in fn_list:
        X = np.load(fn_norm)
        fn_path = os.path.split(fn_norm)[0]
        subject = fn_path.split('/')[-1]
        cau_path = fn_path + '/cau'
        reset_directory(cau_path)
        fn_class = fn_path + '/%s_classes.npy' %subject
        classes = np.load(fn_class)
        ws = scot.Workspace({'model_order': morder}, reducedim='no_pca', fs=678.17)
        ws.set_data(X, classes)
        ws.do_mvarica(varfit='class')
        Stable_bool = ws.var_.is_stable()
        for cla in np.unique(classes): 
            ws.set_used_labels([cla])
            cau = ws.get_connectivity(method) 
            surr = ws.get_surrogate_connectivity(method, repeats=1000)
            fncau = cau_path + '/%s_%s,stabel%s,cau.npy' %(subject, cla, str(Stable_bool)) 
            fnsurr = cau_path + '/%s_%s,surrcau.npy' %(subject, cla)
            np.save(fncau, cau)
            np.save(fnsurr, surr)  


def _plot_hist(con_b, surr_b, fig_out):
    '''
     Plot the distribution of real and surrogates' causality results.
     Parameter
     ---------
     con_b: array
            Causality matrix.
     surr_b: array
            Surrogates causality matix.
     fig_out: string
            The path to store this distribution.
    '''
    import matplotlib.pyplot as pl
    fig = pl.figure('Histogram - surrogate vs real')
    c = con_b  # take a representative freq point
    fig.add_subplot(211, title='Histogram - real connectivity')
    pl.hist(c, bins=100)  # plot histogram with 100 bins (representative)
    s = surr_b
    fig.add_subplot(212, title='Histogram - surrogate connectivity')
    pl.hist(s, bins=100)  # plot histogram
    pl.show()
    pl.savefig(fig_out)
    pl.close()
    
def _plot_thr(con_b, fdr_thr, max_thr, fig_out):
    '''plot the significant threshold of causality analysis.
       Parameter
       ---------
       con_b: array
            Causality matrix.
       fdr_thr: float
           Threshold combining with FDR.
       max_thr: float
           Threshold from the maximum causality value of surrogates.
       fig_out: string
           The path to store the threshold plots.
    '''
    import matplotlib.pyplot as pl
    pl.close('all')
    c = np.unique(con_b)
    pl.plot(c, 'k', label='real con')
    xmin, xmax = pl.xlim()
    pl.hlines(fdr_thr, xmin, xmax, linestyle='--', colors='k',
              label='p=0.05(FDR):%.2f' % fdr_thr, linewidth=2)
    pl.hlines(max_thr, xmin, xmax, linestyle='--', colors='g',
              label='Max surr', linewidth=2)
    pl.legend()
    pl.xlabel('points')
    pl.ylabel('causality values')
    pl.show()
    pl.savefig(fig_out)
    pl.close()

def sig_thresh(cau_list, freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 40)],
               sfreq=678, alpha=0.05, factor=1, min_subject='fsaverage'):
    '''
       Evaluate the significance for each pair's causal interactions.
       Parameter
       ---------
       fn_cau: string
            The file path of causality matrices.
       freqs: list
            The list of interest frequency band.
       sfreq: float
            The sampling rate.
       alpha: significant factor
       factor: int
            The downsampled factor, it is used to compute the frequency
            resolution.
    '''
    from mne.stats import fdr_correction
    from scipy import stats
    path_list = get_files_from_list(cau_list)
    # loop across all filenames
    for fncau in path_list:
        pre_ = fncau[:fncau.rfind(',stable')]
        fnsurr = pre_ + ',surrcau.npy'
        cau_path = os.path.split(fncau)[0]
        sig_path = cau_path + '/sig_cau/'
        set_directory(sig_path)
        cau = np.load(fncau)
        surr = np.load(fnsurr)
        nfft = cau.shape[-1]
        delta_F = sfreq / float(2 * nfft * factor)
        #freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 70), (60, 90)]
        sig_freqs = []
        nfreq = len(freqs)
        for ifreq in range(nfreq):
            fmin, fmax = int(freqs[ifreq][0] / delta_F), int(freqs[ifreq][1] /
                                                             delta_F)
            con_band = np.mean(cau[:, :, fmin:fmax + 1], axis=-1)
            np.fill_diagonal(con_band, 0)
            surr_band = np.mean(surr[:, :, :, fmin:fmax + 1], axis=-1)
            r, s, _ = surr_band.shape
            for i in xrange(r):
                ts = surr_band[i]
                np.fill_diagonal(ts, 0)
            con_b = con_band.flatten()
            con_b = con_b[con_b > 0]
            surr_b = surr_band.reshape(r, s * s)
            surr_b = surr_b[surr_b > 0]
            zscore = (con_b - np.mean(surr_b, axis=0)) / np.std(surr_b, axis=0)
            p_values = stats.norm.pdf(zscore)
            accept, _ = fdr_correction(p_values, alpha)
            assert accept.any() == True, ('Normalized amplitude values are not\
            statistically significant. Please try with lower alpha value.')
            z_thre = np.abs(con_b[accept]).min()
            histout = sig_path + '%s,%d-%d,distribution.png'\
                                % (pre_, freqs[ifreq][0], freqs[ifreq][1])
            throut = sig_path + '%s,%d-%d,threshold.png'\
                        % (pre_, freqs[ifreq][0], freqs[ifreq][1])
            _plot_hist(con_b, surr_b, histout)
            _plot_thr(con_b, z_thre, surr_band.max(), throut)
            con_band[con_band < z_thre] = 0
            con_band[con_band >= z_thre] = 1
            sig_freqs.append(con_band)
        sig_freqs = np.array(sig_freqs)
        np.save(sig_path + '%s_sig_con_band.npy' %pre_, sig_freqs)      


def group_causality(sig_list, condition, freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 40)], 
                    min_subject='fsaverage', submount=10):

    """
        make group causality analysis, by evaluating significant matrices across
        subjects.
        ----------
        sig_list: list
            The path list of individual significant causal matrix.
        condition: string
            One condition of the experiments.
        freqs: list
            The list of interest frequency band.
        min_subject: string
            The subject for the common brain space.
        submount: int
            Significant interactions come out at least in 'submount' subjects.
    """
    
    cau_path = subjects_dir + '/%s/causality' %min_subject
    set_directory(cau_path)
    sig_caus = []
    for f in sig_list:
        sig_cau = np.load(f)
        sig_caus.append(sig_cau)
    sig_caus = np.array(sig_caus)
    sig_group = sig_caus.sum(axis=0)
    #freqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30, 40)]
    plt.close()
    for i in xrange(len(sig_group)):
        fmin, fmax = freqs[i][0], freqs[i][1]
        cau_band = sig_group[i]
        cau_band[cau_band < submount] = 0
        plt.imshow(cau_band, interpolation='nearest')
        np.save(cau_path + '/%s_%s_%sHz.npy' %
                    (condition, str(fmin), str(fmax)), cau_band)
        v = np.arange(cau_band.max())
        plt.colorbar(ticks=v)
        # plt.colorbar()
        plt.show()
        plt.savefig(cau_path + '/%s_%s_%sHz.png' %
                    (condition, str(fmin), str(fmax)), dpi=100)
        plt.close()