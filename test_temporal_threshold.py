import os, mne
from stat_cluster import clu2STC
import numpy as np
subjects_dir = os.environ['SUBJECTS_DIR'] 
# Set the path for storing STCs of conflicts processing
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
# Four contrasts
conditions = [('left', 'conf_per'), ('right', 'conf_per'), ('left', 'conf_res'), ('right', 'conf_res')]
p_v = 0.01# temporal threshold
tmin = 0. # parameter for MNE function 
t_thr = 10# tmin parameter for our function
c = 0 #The index for the Four contrasts
side = conditions[c][0]
c_type = conditions[c][1]
fn_cluster = stcs_path + '%s_permu16384_pthr0.0500_ttest_%s.npz' %(side, c_type)
#Transfer clusters into STC
clu2STC(fn_cluster, p_thre=p_v, tmin=tmin)
fn_stc = stcs_path + '%s_permu16384_pthr0.0500_ttest_%s,temp_%.3f,tmin_0.000-lh.stc' %(side, c_type, p_v)
fn_fig = fn_stc[:fn_stc.rfind('-lh.stc')] + '.tif'

stc = mne.read_source_estimate(fn_stc)
stc_sub = stc.copy().mean()
data = np.zeros(stc_sub.data.shape)
data[:, 0] = stc.data[:, 0]
abs_data = abs(data)
#data[abs_data<np.percentile(abs_data, 40)] = 0
#abs_data[abs_data<=20] = random(abs_data[abs_data<=20].shape) * 10
abs_data[abs_data<t_thr] = 0
stc_sub.data.setfield(abs_data, np.float32)
#stc_sub.times[0] = stc.times[0]
brain = stc_sub.plot(subject='fsaverage', hemi='lh', subjects_dir=subjects_dir,
                                        time_label='Duration significant (ms)')
brain.set_data_time_index(0)
brain.show_view('lateral')
#brain.save_image(fn_fig)