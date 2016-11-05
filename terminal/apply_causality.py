
import os
import numpy as np

# to work without DISPLAY set
import matplotlib
matplotlib.use('Agg')

#from jumeg.jumeg_preprocessing import get_files_from_list

import matplotlib.pyplot as pl

def get_files_from_list(fin):
    ''' Return string of file or files as iterables lists '''
    if isinstance(fin, list):
        fout = fin
    else:
        if isinstance(fin, str):
            fout = list([fin])
        else:
            fout = list(fin)
    return fout

def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreat the directory
    ----------
    path : the target directory.
    """
    import shutil
    isexists = os.path.exists(path)
    if isexists:
        shutil.rmtree(path)
    os.makedirs(path)
    return


def set_directory(path=None):
    """
    check whether the directory exits, if no, creat the directory
    ----------
    path : the target directory.

    """
    isexists = os.path.exists(path)
    if not isexists:
        os.makedirs(path)
    return


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
    pl.ioff()
    fig = pl.figure('Histogram - surrogate vs real')
    c = con_b  # take a representative freq point
    fig.add_subplot(211, title='Histogram - real connectivity')
    pl.hist(c, bins=100)  # plot histogram with 100 bins (representative)
    s = surr_b
    fig.add_subplot(212, title='Histogram - surrogate connectivity')
    pl.hist(s, bins=100)  # plot histogram
    # pl.show()
    pl.savefig(fig_out)
    pl.close()
    return

def _plot_thr(con_b, fdr_thr, max_thr, alpha, fig_out):
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
    pl.ioff()
    pl.close('all')
    c = np.unique(con_b)
    pl.plot(c, 'k', label='real con')
    xmin, xmax = pl.xlim()
    pl.hlines(fdr_thr, xmin, xmax, linestyle='--', colors='k',
               label='per=%.2f:%.2f' % (alpha,fdr_thr), linewidth=2)
              #label='p=%.2f(FDR):%.2f' % (alpha,fdr_thr), linewidth=2)
    pl.hlines(max_thr, xmin, xmax, linestyle='--', colors='g',
              label='Max surr', linewidth=2)
    pl.legend()
    pl.xlabel('points')
    pl.ylabel('causality values')
    # pl.show()
    pl.savefig(fig_out)
    pl.close()
    return


def causal_analysis(fn_norm, method='GPDC', morder=None, repeats=1000,
                    msave=False, per=99.99,
                    sfreq=678,
                    freqs=None):
    '''
    Calculate causality matrices of real data and surrogates. And calculate
    the significant causal matrix for each frequency band.
    Parameters
    ----------
    fnnorm: string
        The file name of model order estimation.
    morder: int
        The optimized model order.
    method: string
        causality measures.
    repeats: int
        Shuffling times for surrogates.
    msave: bool
        Save the causal matrix of the whole frequency domain or not.
    per: float or int
        Percentile of the surrogates.
    sfreq: float
        The sampling rate.
    freqs: list
        The list of interest frequency bands.
    '''
    import scot.connectivity_statistics as scs
    from scot.connectivity import connectivity
    import scot
    path_list = get_files_from_list(fn_norm)
    # loop across all filenames
    for fnnorm in path_list:
        cau_path = os.path.split(fnnorm)[0]
        name = os.path.basename(fnnorm)
        condition = name.split('_')[0]
        sig_path = cau_path + '/sig_cau_%d/' % morder
        set_directory(sig_path)
        fncau = fnnorm[:fnnorm.rfind('.npy')] + ',morder%d,cau.npy' % morder
        fnsurr = fnnorm[:fnnorm.rfind('.npy')] + ',morder%d,surrcau.npy' % morder
        X = np.load(fnnorm)
        X = X.transpose(2, 0, 1)
        mvar = scot.var.VAR(morder)
        surr = scs.surrogate_connectivity(method, X, mvar,
                                          repeats=repeats)
        mvar.fit(X)
        cau = connectivity(method, mvar.coef, mvar.rescov)
        if msave:
            np.save(fncau, cau)
            np.save(fnsurr, surr)
        nfft = cau.shape[-1]
        delta_F = sfreq / float(2 * nfft)
        sig_freqs = []
        nfreq = len(freqs)
        surr_bands = []
        cau_bands = []
        for ifreq in range(nfreq):
            print 'Frequency index used..', ifreq
            fmin, fmax = int(freqs[ifreq][0] / delta_F), int(freqs[ifreq][1] /
                                                             delta_F)
            con_band = np.mean(cau[:, :, fmin:fmax + 1], axis=-1)
            np.fill_diagonal(con_band, 0)
            surr_band = np.mean(surr[:, :, :, fmin:fmax + 1], axis=-1)
            r, s, _ = surr_band.shape
            for i in xrange(r):
                ts = surr_band[i]
                np.fill_diagonal(ts, 0)
            surr_bands.append(surr_band)
            cau_bands.append(con_band)
            con_b = con_band.flatten()
            con_b = con_b[con_b > 0]
            surr_b = surr_band.reshape(r, s * s)
            surr_b = surr_b[surr_b > 0]
            thr = np.percentile(surr_band, per)
            print 'max surrogates %.4f' % thr
            con_band[con_band < thr] = 0
            con_band[con_band >= thr] = 1
            histout = sig_path + '%s,%d-%d,distribution.png'\
                % (condition, freqs[ifreq][0], freqs[ifreq][1])
            throut = sig_path + '%s,%d-%d,threshold.png'\
                % (condition, freqs[ifreq][0], freqs[ifreq][1])
            _plot_hist(con_b, surr_b, histout)
            # _plot_thr(con_b, thr, surr_band.max(), alpha, throut)
            _plot_thr(con_b, thr, surr_band.max(), per, throut)
            # con_band[con_band < z_thre] = 0
            # con_band[con_band >= z_thre] = 1
            sig_freqs.append(con_band)

        sig_freqs = np.array(sig_freqs)
        print 'Saving computed arrays..'
        np.save(sig_path + '%s_sig_con_band.npy' % condition, sig_freqs)
        cau_bands = np.array(cau_bands)
        np.save(fncau, cau_bands)
        surr_bands = np.array(surr_bands)
        np.save(fnsurr, surr_bands)

    return


def sig_thresh(cau_list):
    path_list = get_files_from_list(cau_list)
    # loop across all filenames
    for fncau in path_list:
        fnsurr = fncau[:fncau.rfind(',cau.npy')] + ',surrcau.npy'
        sig_path = os.path.split(fncau)[0]
        name = os.path.basename(fncau)
        condition = name.split(',')[0]
        cau = np.load(fncau)
        surr = np.load(fnsurr)
        i = 0
        sig_freqs = []
        for i in range(len(cau)):
            icau = cau[i]
            isurr = surr[i]
            thr = isurr.max()
            print 'max surrogates %.4f' %thr
            icau[icau < thr] = 0
            icau[icau >= thr] = 1
            cau[i] = icau
            sig_freqs.append(icau)
        sig_freqs = np.array(sig_freqs)
        np.save(sig_path + '/%s_sig_con_band.npy' %condition, sig_freqs) 
            
def group_causality(sig_list, condition, freqs, ROI_labels=None,
                    out_path=None, submount=10):

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
    print 'Running group causality...'
    set_directory(out_path)
    sig_caus = []

    for f in sig_list:
        sig_cau = np.load(f)
        print sig_cau.shape[-1]
        sig_caus.append(sig_cau)

    sig_caus = np.array(sig_caus)
    sig_group = sig_caus.sum(axis=0)
    pl.close()
    
#    cdict3 = {'red':  ((0.0, 0.0, 0.0),
#                   (0.25, 0.0, 0.0),
#                   (0.5, 0.8, 1.0),
#                   (0.75, 1.0, 1.0),
#                   (1.0, 0.4, 1.0)),
#
#         'green': ((0.0, 0.0, 0.0),
#                   (0.25, 0.0, 0.0),
#                   (0.5, 0.9, 0.9),
#                   (0.75, 0.0, 0.0),
#                   (1.0, 0.0, 0.0)),
#
#         'blue':  ((0.0, 0.0, 0.4),
#                   (0.25, 1.0, 1.0),
#                   (0.5, 1.0, 0.8),
#                   (0.75, 0.0, 0.0),
#                   (1.0, 0.0, 0.0))}
#
#    cdict3['alpha'] = ((0.0, 1.0, 1.0),
#                    #   (0.25,1.0, 1.0),
#                    (0.5, 0.3, 0.3),
#                    #   (0.75,1.0, 1.0),
#                    (1.0, 1.0, 1.0)) 
#                    
#    pl.register_cmap(name='BlueRedAlpha', data=cdict3)                                    
    
    
    for i in xrange(len(sig_group)):
        fmin, fmax = freqs[i][0], freqs[i][1]
        cau_band = sig_group[i]
        #cau_band[cau_band < submount] = 0
        cau_band[cau_band < submount] = 0
        #fig, ax = pl.subplots()
        #cmap = pl.get_cmap('hot', cau_band.max()+1-submount)
        cmap = pl.get_cmap('hot')
        #cmap.set_under('gray')
        pl.matshow(cau_band, interpolation='nearest', vmin=submount-1, cmap=cmap)
        if ROI_labels == None:
            ROI_labels = np.arange(cau_band.shape[0]) + 1
        pl.xticks(np.arange(cau_band.shape[0]), ROI_labels, fontsize=9, rotation='vertical')
        pl.yticks(np.arange(cau_band.shape[0]), ROI_labels, fontsize=9)
        #pl.imshow(cau_band, interpolation='nearest')
        #pl.set_cmap('BlueRedAlpha')
        np.save(out_path + '/%s_%s_%sHz.npy' %
                    (condition, str(fmin), str(fmax)), cau_band)
        v = np.arange(submount, cau_band.max()+1, 1)
    
        #cax = ax.scatter(x, y, c=z, s=100, cmap=cmap, vmin=10, vmax=z.max())
        #fig.colorbar(extend='min')        
        
        pl.colorbar(ticks=v, extend='min')
        #pl.show()
        pl.savefig(out_path + '/%s_%s_%sHz.png' %
                    (condition, str(fmin), str(fmax)), dpi=300)
        pl.close()
    return

        
def plt_conditions(cau_path, st_list, nfreqs = [(4, 8), (8, 12), (12, 18), (18, 30), (30,40)]):
    #lbls = ['1', 'R2', 'R3','R4', 'R5', 'R6','R7', 'R8', 'R9','R10', 'R11', 'R12',
     #      'R13', 'R14', 'R15','R16', 'R17', 'R18','R19', 'R20', 'R21','R22', 'R23', 'R28']
    lbls = np.arange(16) + 1
    for ifreq in nfreqs:
        fmin, fmax = ifreq[0], ifreq[1]
        fig_fobj = cau_path + '/conditions4_%d_%dHz.tiff' %(fmin,fmax)
        fig, axar = pl.subplots(2,2)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) 
        for i,ax in enumerate(axar.flat):
            X = np.load(cau_path + '/%s_%d_%dHz.npy' %(st_list[i],fmin,fmax))
            ax.imshow(X, interpolation='nearest'
                    , origin='lower') 
            title = st_list[i]    
            ax.grid(False)
            ax.set_title(title)
            ax.set_yticks(np.arange(16))
            ax.set_xticks(np.arange(16))
            ax.set_xticklabels(lbls)
            ax.set_yticklabels(lbls)
        #fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout()
        #fig.savefig(fig_fobj)
        #plt.close()
        
def _save_csv(data, fn_csv):
    import csv
    csvfile = file(fn_csv, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(data.T)
    csvfile.close()  
          
def diff_mat(lfreqs=None, hfreqs=None, freqs=None, mat_dir=None, 
             hig_low=False, comp_con=True, incon_event=['LRst', 'RLst'], 
             con_event=['LLst', 'RRst']):
    """
        make comparisons between two conditions' group causal matrices
        ----------
        con_event: list
            The list of congruent conditions.
        incon_event: string
            The name of incongruent condition.  
        fmin, fmax:int
            The interest bandwidth.    
        min_subject: string
            The subject for the common brain space.
        
    """
    if comp_con:
        cong_mat = 0
        incong_mat = 0
        fn_csv = mat_dir + '/incon_con.csv'
        for ifreq in freqs:
            fmin, fmax = ifreq[0], ifreq[1]
            fn_con1 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[0], fmin, fmax)
            cong_mat = cong_mat + np.load(fn_con1)
            fn_con2 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[1], fmin, fmax)
            cong_mat = cong_mat + np.load(fn_con2)
            fn_con3 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[0], fmin, fmax)
            incong_mat = incong_mat + np.load(fn_con3)
            fn_con4 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[1], fmin, fmax)
            incong_mat = incong_mat + np.load(fn_con4)
        cong_mat[cong_mat > 0] = 1
        incong_mat[incong_mat > 0] = 2
        dif_mat = incong_mat + cong_mat
        #incong_mat[incong_mat > 0] = 1
        #dif_mat = incong_mat - cong_mat
        #dif_mat[dif_mat < 0] = 0
        _save_csv(dif_mat, fn_csv)
        
    if hig_low:
        fn_con_csv = mat_dir + '/con_high_low.csv'
        fn_incon_csv = mat_dir + '/incon_high_low.csv'
        con_high_mat, incon_high_mat = 0, 0
        con_low_mat, incon_low_mat = 0, 0
        for ifreq in hfreqs:
            fmin, fmax = ifreq[0], ifreq[1]
            fn_con1 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[0], fmin, fmax)
            con_high_mat = con_high_mat + np.load(fn_con1)
            fn_con2 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[1], fmin, fmax)
            con_high_mat = con_high_mat + np.load(fn_con2)
            fn_con3 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[0], fmin, fmax)
            incon_high_mat = incon_high_mat + np.load(fn_con3)
            fn_con4 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[1], fmin, fmax)
            incon_high_mat = incon_high_mat + np.load(fn_con4)
            
        for ifreq in lfreqs:
            fmin, fmax = ifreq[0], ifreq[1]
            fn_con1 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[0], fmin, fmax)
            con_low_mat = con_low_mat + np.load(fn_con1)
            fn_con2 = mat_dir + '/%s_%d_%dHz.npy' %(con_event[1], fmin, fmax)
            incon_low_mat = incon_low_mat + np.load(fn_con2)    
            fn_con3 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[0], fmin, fmax)
            incon_low_mat = incon_low_mat + np.load(fn_con3)
            fn_con4 = mat_dir + '/%s_%d_%dHz.npy' %(incon_event[1], fmin, fmax)
            incon_low_mat = incon_low_mat + np.load(fn_con4)
        con_high_mat[con_high_mat > 0] = 2
        incon_high_mat[incon_high_mat > 0] = 2
        #incon_high_mat[incon_high_mat > 0] = 1
        con_low_mat[con_low_mat > 0] = 1
        incon_low_mat[incon_low_mat > 0] = 1
        #incon_low_mat[incon_low_mat > 0] = 1
        diff_con_mat = con_high_mat + con_low_mat
        #diff_con_mat = con_high_mat - con_low_mat
        diff_incon_mat = incon_high_mat + incon_low_mat
        #diff_incon_mat = incon_high_mat - incon_low_mat 
        #diff_con_mat[diff_con_mat < 0] = 0 
        #diff_incon_mat[diff_incon_mat < 0] = 0
        import pdb
        pdb.set_trace()
        _save_csv(diff_con_mat, fn_con_csv) 
        _save_csv(diff_incon_mat, fn_incon_csv)   
    print '>>>>>>>>>>>comparisons between incongruent and congruent conditions'
    print np.argwhere(dif_mat).shape
    print '>>>>>>>>>>>comparisons between high and low freqs of congruent conditions'
    print np.argwhere(diff_con_mat).shape
    print '>>>>>>>>>>>comparisons between high and low freqs of incongruent conditions'
    print np.argwhere(diff_incon_mat).shape
   
    return
        
        