
import sys
import glob
from apply_causality import causal_analysis, group_causality, diff_mat


if __name__ == '__main__':
    print '>>> Starting causality_analysis...'
    sfreq = 678.17 # Sampling rate
    morder = 21# Fixed model order
    per = 99.99 # Percentile for causal surrogates
    ifre = int(sfreq / (2 * morder))
    freqs = [(1, ifre), (ifre, 2*ifre), (2*ifre, 3*ifre), (3*ifre, 4*ifre), (4*ifre, 5*ifre)]
    st_list = ['LLst', 'RRst', 'RLst',  'LRst']


    #fn_monorm = glob.glob('./*[0-9]/*_labels_ts,norm,morder_*.npz')
    fn_monorm = glob.glob('./*[0-9]/*_labels_ts,norm.npy')
    assert len(fn_monorm) == 56, 'The number of files are not 56. Check.'
   
    ###############################
    # Individual causality analysis
    # -----------------------------
    do_cau = True
    if do_cau:
        print '>>> Make the causality analysis....'
        causal_analysis(fn_monorm, repeats=1000, morder=morder, method='GPDC', 
                         per=per, sfreq=sfreq, freqs=freqs)
        print '>>> FINISHED with causal matrices and surr-causal matrices generation.'
        print ''

    ##########################
    # Group causality analysis
    # ------------------------
    do_group = True
    if do_group:
        print '>>> Generate the group causal matrices....'
        for evt_st in st_list:
            fnsig_list = glob.glob('./*[0-9]/sig_cau_21/%s_sig_con_band.npy' %evt_st)
            print len(fnsig_list)
            if len(fnsig_list) >= 14:
                group_causality(fnsig_list, evt_st, freqs=freqs, out_path='causality', submount=14)
        print '>>> FINISHED with group causal matrices generation.'
        print ''

    print '>>> Stopping causality analysis.'
    
    mat_pro =True
    if mat_pro:
    # Difference between incongruent and congruent tasks
        mat_dir = './causality'
        lfreqs = freqs[0:2]
        hfreqs = freqs[2:4]
        freqs1 = lfreqs + hfreqs
        diff_mat(lfreqs, hfreqs, freqs1, mat_dir, hig_low=True, comp_con=True)
