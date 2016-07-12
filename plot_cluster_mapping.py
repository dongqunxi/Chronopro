import mne, os
subjects_dir = os.environ['SUBJECTS_DIR'] 
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
fn_stc = stcs_path + 'right_ttest_mv10_conf_per-lh.stc' 
fn_fig = fn_stc[:fn_stc.rfind('-lh.stc')] + '.tif'
stc = mne.read_source_estimate(fn_stc)
brain = stc.plot(subject='fsaverage', hemi='split', subjects_dir=subjects_dir,
                                        time_label='Duration significant (ms)')
brain.set_data_time_index(0)
brain.show_view('lateral')
#brain.save_image(fn_fig)