import os, glob, mne
import numpy as np
from jumeg.jumeg_preprocessing import get_files_from_list, reset_directory
import math
import matplotlib.pylab as plt
subjects_dir = os.environ['SUBJECTS_DIR']

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
        
def _mvdetrend(data):
    '''Multivariate polynomial detrend of time series data.
       refer:http://users.sussex.ac.uk/~lionelb/MVGC/html/mvdetrend.html.
       Parameter
       ---------
       data: narray
            multi-trial time series data, shape as (variables, obsevations,
             trials).
       Return
       ------
       Y: detrended time series.
    '''
    n, m, N = data.shape
    pdeg = 1
    pdeg = pdeg * np.ones(n)
    x = np.arange(m) + 1
    mu = np.mean(x)
    sig = np.std(x, ddof=1)
    x = (x - mu) / sig
    P = np.zeros((n, m))
    p = []
    for i in xrange(n):
        d = pdeg[i]
        d1 = int(d + 1)
        V = np.zeros((d1, m))
        V[d1 - 1, :] = np.ones(m)
        j = int(d)
        while j > 0:
            V[j - 1, :] = x * V[j, :]
            j = j - 1
        temp = np.mean(data[i, :, :], axis=-1)
        p.append(np.linalg.lstsq(V.T, temp)[0].T)
        P[i, :] = p[i][0] * np.ones((1, m))
        for j in range(1, d1):
            P[i, :] = x * P[i, :] + p[i][j]
    Y = np.zeros(data.shape)
    for r in xrange(N):
        Y[:, :, r] = data[:, :, r] - P
    return Y

def normalize_data(fn_ts, fs=678, pre_t=0.2, factor=1):
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
    from scipy import signal
    path_list = get_files_from_list(fn_ts)
    # loop across all filenames
    for fnts in path_list:
        fnnorm = fnts[:fnts.rfind('.npy')] + ',%d-norm.npy' % factor
        label_ts = np.load(fnts)
        #assert label_ts.shape[1] / factor > label_ts.shape[-1], \
        #    ('Trial length can not be smaller than the amount of trials.')
        #if factor > 1:
        #    #downsampled
        #    dw_data = signal.decimate(label_ts, q=factor, axis=1)
        #else:
        #    dw_data = label_ts
        #zscore
        #dt_data = _mvdetrend(dw_data)
        d_pre = label_ts[:, :int(pre_t*fs), :]
        d_pos = label_ts[:, int(pre_t*fs):, :]
        d_mu = d_pre.mean(axis=1, keepdims=True)
        d_std = d_pre.std(axis=1, ddof=1, keepdims=True)
        z_data = (d_pos - d_mu) / d_std
        #d_mu = dw_data.mean(axis=1, keepdims=True)
        #d_std = dw_data.std(axis=1, ddof=1, keepdims=True)
        #z_data = (dw_data - d_mu) / d_std
        np.save(fnnorm, z_data)

def _plot_morder(bic, morder, figmorder):
    '''
       Parameter
       ---------
       bic: array
           BIC values for each model order lower than 'p_max'.
       morder: int
          The optimized model order.
       figmorder: string
          The path for storing the plot.
    '''

    plt.figure()
    h0, = plt.plot(np.arange(len(bic)) + 1, bic, 'r', linewidth=3)
    plt.legend([h0], ['BIC: %d' %morder])
    plt.xlabel('order')
    plt.ylabel('BIC')
    plt.title('Model Order')
    plt.show()
    plt.savefig(figmorder, dpi=100)
    plt.close()

def _model_order(fnnorm, p_max=100):

    """ Calculate the optimized model order for VAR 
        models from time series data.

        Parameters
        ----------
        fnnorm: string
            The file name of model order estimation.
        p_max: int 
            The upper limit for model order estimation.
        Returns
        ----------
        morder: int
            The optimized BIC model order.

    """
    figmorder = fnnorm[:fnnorm.rfind('.npy')] + '.png'
    X = np.load(fnnorm)
    n, m, N = X.shape
    if p_max == 0:
        p_max = m - 1
    q = p_max
    q1 = q + 1
    XX = np.zeros((n, q1, m + q, N))
    for k in xrange(q1):
        XX[:, k, k:k + m, :] = X
    q1n = q1 * n
    bic = np.empty((q, 1))
    bic.fill(np.nan)
    I = np.identity(n)
    # initialise recursion
    AF = np.zeros((n, q1n))#forward AR coefficients
    AB = np.zeros((n, q1n))#backward AR coefficients
    k = 1
    kn = k * n
    M = N * (m - k)
    kf = range(0, kn)
    kb = range(q1n - kn, q1n)
    XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
    XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
    #import pdb
    #pdb.set_trace()
    CXF = np.linalg.cholesky(XF.dot(XF.T)).T
    CXB = np.linalg.cholesky(XB.dot(XB.T)).T
    AF[:, kf] = np.linalg.solve(CXF.T, I)
    AB[:, kb] = np.linalg.solve(CXB.T, I)
    while k <= q - 1:
        #print('model order = %d' % k)
        #import pdb
        #pdb.set_trace()
        tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
        af = AF[:, kf]
        EF = af.dot(tempF)
        tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
        ab = AB[:, kb]
        EB = ab.dot(tempB)
        CEF = np.linalg.cholesky(EF.dot(EF.T)).T
        CEB = np.linalg.cholesky(EB.dot(EB.T)).T
        R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))
        CRF = np.linalg.cholesky(I - R.dot(R.T)).T
        CRB = np.linalg.cholesky(I - (R.T).dot(R)).T
        k = k + 1
        kn = k * n
        M = N * (m - k)
        kf = np.arange(kn)
        kb = range(q1n - kn, q1n)
        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        AF[:, kf] = np.linalg.solve(CRF.T, AFPREV - R.dot(ABPREV))
        AB[:, kb] = np.linalg.solve(CRB.T, ABPREV - R.T.dot(AFPREV))
        E = np.linalg.solve(AF[:, :n], AF[:, kf]).dot(np.reshape(XX[:, :k, k:m,
                                                      :], (kn, M), order='F'))
        DSIG = np.linalg.det((E.dot(E.T)) / (M - 1))
        i = k - 1
        K = i * n * n
        L = -(M / 2) * math.log(DSIG)
        bic[i - 1] = -2 * L + K * math.log(M)
    #morder = np.nanmin(bic), np.nanargmin(bic) + 1
    morder = np.nanargmin(bic) + 1
    _plot_morder(bic, morder, figmorder)
    return morder

def _tsdata_to_var(X,p):
    """ Calculate coefficients and recovariance and noise covariance of 
        the optimized model order.
        ref: http://users.sussex.ac.uk/~lionelb/MVGC/html/tsdata_to_var.html
        Parameters
        ----------
        X: narray, shape (n_sources, n_times, n_epochs)
              The data to estimate the model order for.
        p: int, the optimized model order.
        Returns
        ----------
        A: array, coefficients of the specified model
        SIG:array, recovariance of this model
        E:  array, noise covariance of this model
    """
    n, m, N = X.shape
    p1 = p + 1
    A = np.nan
    SIG = np.nan
    E = np.nan
    q1n = p1 * n
    I = np.eye(n)
    XX = np.zeros((n, p1, m + p, N))
    for k in xrange(p1):
        XX[:, k, k:k + m, :] = X
    AF = np.zeros((n, q1n))
    AB = np.zeros((n, q1n))
    k = 1
    kn = k * n
    M = N * (m - k)
    kf = range(0, kn)
    kb = range(q1n - kn, q1n)
    XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
    XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
    CXF = np.linalg.cholesky(XF.dot(XF.T)).T
    CXB = np.linalg.cholesky(XB.dot(XB.T)).T
    AF[:, kf] = np.linalg.solve(CXF.T, I)
    AB[:, kb] = np.linalg.solve(CXB.T, I)
    while k <= p:
        tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
        af = AF[:, kf]
        EF = af.dot(tempF)
        tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
        ab = AB[:, kb]
        EB = ab.dot(tempB)
        CEF = np.linalg.cholesky(EF.dot(EF.T)).T
        CEB = np.linalg.cholesky(EB.dot(EB.T)).T
        R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))
        RF = np.linalg.cholesky(I - R.dot(R.T)).T
        RB = np.linalg.cholesky(I - (R.T).dot(R)).T
        k = k + 1
        kn = k * n
        M = N * (m - k)
        kf = np.arange(kn)
        kb = range(q1n - kn, q1n)
        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        AF[:, kf] = np.linalg.solve(RF.T, AFPREV - R.dot(ABPREV))
        AB[:, kb] = np.linalg.solve(RB.T, ABPREV - R.T.dot(AFPREV))
    E = np.linalg.solve(AFPREV[:, :n], EF)
    SIG = (E.dot(E.T)) / (M - 1)
    E = np.reshape(E, (n, m - p, N), order='F')
    temp = np.linalg.solve(-AF[:, :n], AF[:, n:])
    A = np.reshape(temp, (n, n, p), order='F')
    return A, SIG, E

# Whiteness test                      
def _erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 +
                     t * (.09678418 + t * (-.18628806 + t * (.27886807 +
                    t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 +
                                                      t * .17087277)))))))))
    if (x >= 0.):
    	return r
    else:
    	return 2. - r
    	
def _normcdf(x, mu, sigma):
    t = x - mu
    y = 0.5 * _erfcc(-t / (sigma * math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y

def _durbinwatson(X, E):
    n, m = X.shape
    dw = np.sum(np.diff(E, axis=0) ** 2) / np.sum(E ** 2, axis=0)
    A = np.dot(X, X.T)
    from scipy.signal import lfilter
    B = lfilter(np.array([-1, 2, -1]), 1, X.T, axis=0)
    temp = X[:, [0, m - 1]] - X[:, [1, m - 2]]
    B[[0, m - 1], :] = temp.T
    D = np.dot(B, np.linalg.pinv(A))
    C = X.dot(D)
    nu1 = 2 * (m - 1) - np.trace(C)
    nu2 = 2 * (3 * m - 4) - 2 * np.trace(B.T.dot(D)) + np.trace(C.dot(C))
    mu = nu1 / (m - n)
    sigma = math.sqrt(2./((m - n) * (m - n + 2)) * (nu2 - nu1 * mu))
    pval = _normcdf(dw, mu, sigma)
    pval = 2 * min(pval, 1-pval)
    return dw, pval
    
def _significance(pval, alpha=0.05):

    sig = np.zeros(pval.shape)
    sig[:] = np.nan
    nn = ~np.isnan(pval)
    p = pval[nn]
    p.sort()
    m = p.size
    thresh = (np.arange(m)+1) * alpha / m
    rej = p <= thresh
    max_id = rej.nonzero()[0]
    if max_id.size == 0:
        crit_p = 0
        h = np.logical_and(p, 0)
    else:
        max_id = max_id[-1]
        crit_p = p[max_id]
        h = p <= crit_p
    sig[nn] = h
    nowhite=sig.nonzero()[0]
    if nowhite.size == 0:
        print 'all residuals are white by Durbin-Wastson test at significance %.2f' %alpha
        return True
    else:
        print 'WARNING: autocorrelated residuals at significance %.2f for variable(s):%s' %(alpha, nowhite)
        return False
        
def _demean(X):
    n, m, N = X.shape
    U = np.ones((1, N*m))
    Y = X.reshape((n, N*m), order = 'F')
    #Y_m = Y.mean(axis=1, keepdims=True)
    Y  = Y-Y.mean(axis=1, keepdims=True)*U
    Y = Y.reshape((n, m, N), order = 'F')
    return Y
    
def _whiteness(X,E):

    """Durbin-Watson test for whiteness (no serial correlation) of
       VAR residuals.
       Prarameters
       -----------
       X: array
          Multi-trial time series data.
       E: array
          Residuals time series.

       Returns
       -------
       dw: array
           Vector of Durbin-Watson statistics.
       pval: array
             Vector of p-values.
    """
    n, m, N = X.shape
    X = _demean(X)
    dw = np.zeros(n)
    pval = np.zeros(n)
    for i in xrange(n):
        Ei = np.squeeze(E[i, :, :])
        e_a, e_b = Ei.shape
        tempX = np.reshape(X, (n, m * N), order='F')
        tempE = np.reshape(Ei, (e_a * e_b), order='F')
        dw[i], pval[i] = _durbinwatson(tempX, tempE)
    white = _significance(pval)
    return white

# Consistency test                    
def _consistency(X, E):
    '''
       Prarameters
       -----------
       X: array
          Multi-trial time series data.
       E: array
          Residuals time series.
       Returns
       -------
       cons: float
        consistency test measurement.

    '''
    n, m, N = X.shape
    p = m - E.shape[1]
    X = _demean(X)
    X = X[:, p:m, :]
    n1, m1, N1 = X.shape
    X = np.reshape(X, (n1, m1 * N1), order='F')
    E = np.reshape(E, (n1, m1 * N1), order='F')
    s = N * (m - p)
    Y = X - E
    Rr = X.dot(X.T) / (s - 1)
    Rs = Y.dot(Y.T) / (s - 1)
    cons = 1 - np.linalg.norm(Rs - Rr, 2) / np.linalg.norm(Rr, 2)
    return cons

def model_estimation(fn_norm, thr_cons=0.8, whit_min=1., whit_max=3., pmax=100):
    '''
       Check the statistical evalutions of the MVAR model corresponding the
       optimized morder.
       Reference
       ---------
       Granger Causal Connectivity Analysis: A MATLAB Toolbox, Anil K. Seth
       (2009)
       Parameters
       ----------
        fn_norm: string
            The file name of model order estimation.
        thr_cons:float
            The threshold of consistency evaluation.
        whit_min:float
            The lower limit for whiteness evaluation.
        whit_max:float
            The upper limit for whiteness evaluation.
        pmax: int
            The maximum value for model order estimation.
    '''
    import scot
    path_list = get_files_from_list(fn_norm)
    # loop across all filenames
    for fnnorm in path_list:
        fneval = fnnorm[:fnnorm.rfind('.npy')] + '_evaluation.txt'
        #morder = _model_order(fnnorm, pmax)
        morder = 21
        X = np.load(fnnorm)
        A, SIG, E = _tsdata_to_var(X, morder)
        whi = _whiteness(X, E)
        cons = _consistency(X, E)
        X = X.transpose(1, 0, 2)
        mvar = scot.var.VAR(morder)
        mvar.fit(X)
        is_st = mvar.is_stable()
        if cons < thr_cons or is_st == False or whi == False:
            print fnnorm
        assert cons > thr_cons and is_st and whi, ('Consistency, whiteness, stability:\
                                            %f, %s, %s' %(cons, str(whi), str(is_st)))
        with open(fneval, "a") as f:
            f.write('model_order, whiteness, consistency, stability: %d, %s, %f, %s\n' 
                    %(morder, str(whi), cons, str(is_st)))

def causal_analysis(fn_norm, pmax=100, method='PDC'):
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
        X = X.transpose(1, 0, 2)
        morder = _model_order(fnnorm, pmax)
        mvar = scot.var.VAR(morder)
        c_surrogates = scs.surrogate_connectivity(method, X, mvar,
                                                  repeats=1000)
        mvar.fit(X)
        con = connectivity(method, mvar.coef, mvar.rescov)
        np.save(fncau, con)
        np.save(fnsurr, c_surrogates)