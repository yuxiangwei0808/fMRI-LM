"""
Get the valid ADNI samples with correct fMRI paths, save with labels into a csv file
"""

import os
import pandas as pd


meta = pd.read_csv('/data/qneuromark/Data/ADNI/Updated/demos/ADNIMERGE_14May2025.csv')
root_path = '/data/qneuromark/Data/ADNI/Updated/fMRI/ADNI'

# transform some columns to str
meta['COLPROT'] = meta['COLPROT'].astype(str)
meta['PTID'] = meta['PTID'].astype(str)
meta['VISCODE'] = meta['VISCODE'].astype(str)

# transform date columns to datetime
meta['EXAMDATE'] = pd.to_datetime(meta['EXAMDATE'], errors='coerce')

valid_path = []
for idx, row in meta.iterrows():
    if row['COLPROT'] in ['ADNI4', 'ADNI3']:
        sub_id = row['PTID']
        time_point = row['EXAMDATE']

        # try to find if the path exsists
        fmri_dir = os.path.join(root_path, sub_id)
        if os.path.exists(fmri_dir):
            fmri_sessions = os.listdir(fmri_dir)
            fmri_sessions_path = [os.path.join(fmri_dir, s) for s in fmri_sessions]
            
            for sess_path in fmri_sessions_path:
                time_paths = os.listdir(sess_path)
                for t_path in time_paths:
                    t_date = pd.to_datetime(t_path[:10], format='%Y-%m-%d')
                    if abs((t_date - time_point).days) <= 30:
                        next_dir = os.listdir(os.path.join(sess_path, t_path))
                        full_path = os.path.join(sess_path, t_path, next_dir[0], 'rest', 'swarest1.nii')
                        if os.path.exists(full_path):
                            # now get some other features we will need
                            age = row['AGE']
                            sex = row['PTGENDER']
                            race = row['PTRACCAT']
                            diag = row['DX_bl']
                            apoe4 = row['APOE4']
                            av45 = row['AV45']
                            cdrsb = row['CDRSB']
                            mmse = row['MMSE']

                            if not diag == 'nan':
                                valid_path.append((idx, sub_id, sess_path, t_path, full_path,
                                                age, sex, race, diag, apoe4, av45, cdrsb, mmse))


valid_df = pd.DataFrame(valid_path, columns=['idx', 'subject_id', 'session_path', 'time_path', 'fmri_path',
                                             'age', 'sex', 'race', 'diagnosis', 'apoe4', 'av45', 'cdrsb', 'mmse'])
# encode categorical variables
valid_df['sex'] = valid_df['sex'].map({'Male': 1, 'Female': 0})
valid_df['race'] = valid_df['race'].astype('category').cat.codes
valid_df['diagnosis'] = valid_df['diagnosis'].map({'CN': 0, 'MCI': 1, 'AD': 2, 'EMCI': 1, 'SMC': 1})

valid_df.to_csv('data/ADNI/fmri/metadata.csv', index=False)