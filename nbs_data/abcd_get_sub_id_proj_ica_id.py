import os
from tqdm import tqdm
from scipy.io import loadmat
import pickle

def find_fmri_abcd(root_dir):
    """Find all fMRI files in ABCD dataset."""
    suffix = "SmNpdmcf_prest.nii"
    fMRI_file_paths = []
    metadata = []

    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        
        for session in os.listdir(subj_path):
            sess_path = os.path.join(subj_path, session)
            if not os.path.isdir(sess_path):
                continue

            for func in reversed(sorted(os.listdir(sess_path))):
                if func.startswith('func_'):
                    func_path = os.path.join(sess_path, func)
                    if not os.path.isdir(func_path):
                        continue
                    
                    target_fmri_file = [f for f in os.listdir(func_path) if f.endswith(suffix)]
                    
                    assert len(target_fmri_file) <= 1
                    if not target_fmri_file:
                        continue

                    file_path = os.path.join(func_path, target_fmri_file[0])
                    fMRI_file_paths.append(file_path)
                    metadata.append({'subject': subject, 'session': session})
                    break  # only process the latest func_*
                    
    return fMRI_file_paths, metadata

valid_fmri_paths, valid_metadata = find_fmri_abcd('/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/')
valid_dict = {p: (m['subject'], m['session']) for p, m in zip(valid_fmri_paths, valid_metadata)}

file = loadmat('/data/neuromark2/Results/Subject_selection/ABCD/R5/ABCD_sub_info_hd1000.mat')
all_fmri_paths = []

for i in range(len(file['analysis_file_list'])):
    all_fmri_paths.append(file['analysis_file_list'][i][0][0])
all_fmri_paths = [p + '/SmNpdmcf_prest.nii' for p in all_fmri_paths]

# get a mapping from subject_id/session_id to the id in all_fmri_paths
path_to_id = {}
for i, path in tqdm(enumerate(all_fmri_paths)):
    if path in valid_dict:
        subject_id, session_id = valid_dict[path]
        path_to_id[(subject_id, session_id)] = i

print(1)
# save
with open('data/ABCD/abcd_get_sub_id_proj_ica_id.pkl', 'wb') as f:
    pickle.dump(path_to_id, f)