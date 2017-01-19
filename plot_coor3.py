import mne, os, random
import numpy as np
import glob
from surfer import Brain

subjects_dir = os.environ['SUBJECTS_DIR']
#subjects_dir = os.environ['SUBJECTS_DIR'] + '/CAU/'
#fn_avg = subjects_dir+'/fsaverage/dSPM_ROIs/109077/109077_Chrono01_110518_1415_1_c,rfDC,nr,ocarta,fibp1-45,evtW_LLrt_bc-lh.st-lh.stc'
#fn_avg = '/home/uais/data/freesurfer/subjects/fsaverage/stcs/203867/RLst/trial9_fsaverage-lh.stc'
#stc_avg = mne.read_source_estimate(fn_avg)
stcs_path = subjects_dir + '/fsaverage/conf_stc/' 
conditions = [('sti', 'LLst'), ('sti', 'RRst'), ('sti', 'RLst'), ('sti', 'LRst'),
             ('res', 'LLrt'), ('res', 'RRrt'), ('res', 'RLrt'), ('res', 'LRrt')]
stcs_path = subjects_dir + '/fsaverage/conf_stc/'
stc = 0
for cond in conditions:
    side = cond[0]
    conf_type = cond[1]
    fn_stc = stcs_path + 'Ttestpermu16384_pthr0.0001_%s_%s,temp_0.005-rh.stc' %(side, conf_type)
    stc += mne.read_source_estimate(fn_stc).mean()

stc_avg = stc / len(conditions)
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
hemi = "lh"
subject_id = 'fsaverage'
surf = 'inflated'
brain = Brain(subject_id, hemi, surf, cortex = 'bone', background='white')

# Get common labels
#list_file = subjects_dir+'/fsaverage/dSPM_ROIs/anno_ROIs/func_label_list.txt'
#list_file = subjects_dir+'fsaverage/dSPM_ROIs/func_label_list.txt'
list_file = subjects_dir+'/fsaverage/conf_stc/STC_ROI/func_list.txt'
with open(list_file, 'r') as fl:
            file_list = [line.rstrip('\n') for line in fl]
fl.close()
#stc_path = os.path.split(fn_avg)[0]
#stan_path = stc_path + '/standard/'
#reset_directory(stan_path)
color = ['#990033', '#9900CC', '#FF6600', '#FF3333', '#00CC33']
i = 0
for f in file_list:
    label = mne.read_label(f)
    stc_label = stc_avg.in_label(label)
    stc_label.data[stc_label.data < 0] = 1
    if np.all(stc_label.data) == 0:
        print label.name
    
    #brain.add_label(label, color='red', alpha=0.5)
    if label.hemi == 'lh':
        vtx, _, _ = stc_label.center_of_mass(subject_id, hemi=0)
        brain.add_foci(vtx, coords_as_verts=True, hemi='lh', color='red',
                        scale_factor=0.8)
    #elif label.hemi == 'rh':
    #       
    #    vtx, _, _ = stc_label.center_of_mass(subject_id, hemi=1)
    #    brain.add_foci(vtx, coords_as_verts=True, hemi='rh', color='blue',
    #                    scale_factor=0.8)             
        ind = i % 5
        brain.add_label(label, color=color[ind], alpha=0.8)
    i = i + 1
brain.add_annotation(annot='aparc', borders=True)
#brain.save_imageset('lhcoor_', views=['lateral', 'ventral', 'medial'], filetype='jpg', row=0, col=0)
