import os
import json
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


def get_targets(data_path):
    csv_path = '/data/qneuromark/Data/UKBiobank/Data_info/Basket/B4033904/ukb674036.csv'
    
    columns = {
        'age': ('21003-2.0', '21003-2.0'),
        'sex': ('31-0.0', '31-0.0'),
        'height': ('12144-2.0', '12144-3.0'),  # cm  Mean = 169.566, Std.dev = 9.47034
        'fluid intelligence': ('20016-2.0', '20016-3.0'),  # discrete, Mean = 5.60236, Std.dev = 2.01301
        'blood pressure level': ('4079-2.0', '4079-3.0'),  # mmHg, Mean = 81.6273, Std.dev = 10.5641
        'cholesterol level': ('26037-2.0', '26037-3.0'),  # mg/dL, Mean = 246.313, Std.dev = 192.974
        'BMI': ('23104-2.0', '23104-3.0'),
    }

    # Get all unique column names we need
    all_columns = ['eid'] + [col for col_tuple in columns.values() for col in col_tuple]
    all_columns = list(set(all_columns))  # Remove duplicates
    
    print("Loading CSV file in chunks to build EID lookup...")
    
    chunk_size = 10000
    
    # Now load the HDF5 file to get subject IDs we need
    file_handle = h5py.File(data_path, 'r')
    num_samples = len(file_handle['time_series'])
    
    # Get all subject IDs we need
    needed_subject_ids = set()
    for i in range(num_samples):
        subject_id = int(file_handle['metadata']['subjects'][i].decode('utf-8'))
        needed_subject_ids.add(subject_id)
    
    print(f"Need data for {len(needed_subject_ids)} unique subjects")
    
    # Second pass: Load only the rows we need
    needed_rows = {}
    row_offset = 0

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size, usecols=all_columns), desc="Loading CSV chunks"):
        # Find which rows in this chunk we need
        chunk_mask = chunk['eid'].isin(needed_subject_ids)
        if chunk_mask.any():
            needed_chunk = chunk[chunk_mask].copy()
            for _, row in needed_chunk.iterrows():
                needed_rows[row['eid']] = row.to_dict()
        row_offset += len(chunk)
    
    print(f"Loaded data for {len(needed_rows)} subjects")
    
    # List to store the results
    results = []

    for i in range(num_samples):
        subject_id = int(file_handle['metadata']['subjects'][i].decode('utf-8'))
        session_id = file_handle['metadata']['sessions'][i].decode('utf-8')

        if subject_id not in needed_rows:
            print(f"Warning: Subject {subject_id} not found in CSV data")
            continue
            
        row = needed_rows[subject_id]
        
        # Extract features for this subject
        subject_features = {'subject_id': subject_id, 'session_id': session_id}
        
        for feature_name, col_tuple in columns.items():
            col1, col2 = col_tuple

            if session_id == 'ses_01':
                value = row.get(col1)
            else:  # ses_02
                value = row.get(col2)

            if pd.notna(value) and value != '':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = np.nan
            else:
                value = np.nan

            subject_features[feature_name] = value
        
        results.append(subject_features)
    
    # Close the file handle
    file_handle.close()
    
    # Create the new dataframe
    result_df = pd.DataFrame(results)
    
    return result_df

class fMRITextGenerator:
    def __init__(self):
        self.templates = {
            'base_template': (
                "Subject demographics and clinical characteristics: "
                "{age_desc} {sex_desc} {height_desc} {cognitive_desc} "
                "{bp_desc} {cholesterol_desc} {bmi_desc}"
            ),
            
            'medical_template': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- Physical characteristics: {height_desc}. {bmi_desc}\n"
                "- Cognitive assessment: {cognitive_desc}\n"
                "- Cardiovascular profile: {bp_desc}. {cholesterol_desc}"
            ),
        }
        
        self.field_templates = {
            'age': {
                'present': lambda x: f"{int(x)}-year-old",
                'missing': "age unspecified"
            },
            'sex': {
                'present': lambda x: 'Male' if x == 1 else 'Female',
                'missing': "sex unspecified"
            },
            'height': {
                'present': lambda x: f"height {x:.1f} cm",
                'missing': "height not measured"
            },
            'fluid_intelligence': {
                'present': lambda x: self._format_fluid(x),
                'missing': "cognitive assessment not available"
            },
            'blood_pressure_level': {
                'present': lambda x: self._format_bp(x),
                'missing': "blood pressure not recorded"
            },
            'cholesterol_level': {
                'present': lambda x: f"cholesterol level: {x:.1f} mg/dL" + self._cholesterol_category(x),
                'missing': "cholesterol level not measured"
            },
            'bmi': {
                'present': lambda x: f"BMI: {x:.1f} kg/m²" + self._bmi_category(x),
                'missing': "BMI not calculated"
            }
        }
    
    def _format_bp(self, bp_value: float) -> str:
        """Format blood pressure with clinical interpretation"""
        if bp_value < 90:
            return f"blood pressure {bp_value:.1f} mmHg (hypotensive)"
        elif bp_value < 120:
            return f"blood pressure {bp_value:.1f} mmHg (normal)"
        elif bp_value < 140:
            return f"blood pressure {bp_value:.1f} mmHg (elevated)"
        else:
            return f"blood pressure {bp_value:.1f} mmHg (hypertensive)"
    
    def _bmi_category(self, bmi: float) -> str:
        """Add BMI category for clinical context"""
        if bmi < 18.5:
            return " (underweight)"
        elif bmi < 25:
            return " (normal weight)"
        elif bmi < 30:
            return " (overweight)"
        else:
            return " (obese)"

    def _cholesterol_category(self, cholesterol: float) -> str:
        """Add cholesterol category for clinical context"""
        if cholesterol < 200:
            return " (desirable)"
        elif cholesterol < 240:
            return " (borderline high)"
        else:
            return " (high)"

    def _format_fluid(self, x) -> str:
        z_score, raw_score = x
        text = f"fluid intelligence score is {raw_score}"
        if z_score < -1.5:
            text += " (below average for adults over 30)"
        elif z_score > 1.5:
            text += " (above average for adults over 30)"
        else:
            text += " (average for adults over 30)"
        return text

    def _format_field(self, field_name: str, value: Any) -> str:
        """Format individual field with missing value handling"""
        if pd.isna(value) or value is None:
            return self.field_templates[field_name]['missing']
        else:
            return self.field_templates[field_name]['present'](value)
    
    def generate_description(self, row: pd.Series, template_type: str = 'base_template') -> str:
        """Generate text description for a single sample"""
        field_descriptions = {
            'age_desc': self._format_field('age', row.get('age')),
            'sex_desc': self._format_field('sex', row.get('sex')),
            'height_desc': self._format_field('height', row.get('height')),
            'cognitive_desc': self._format_field('fluid_intelligence', (row.get('fluid_intelligence_z'), row.get('fluid intelligence'))),
            'bp_desc': self._format_field('blood_pressure_level', row.get('blood pressure level')),
            'cholesterol_desc': self._format_field('cholesterol_level', row.get('cholesterol level')),
            'bmi_desc': self._format_field('bmi', row.get('BMI'))
        }
        
        return self.templates[template_type].format(**field_descriptions)
    
    def generate_dataset(self, df: pd.DataFrame, template_type: str = 'base_template') -> pd.DataFrame:
        """Generate text descriptions for entire dataset"""
        
        df_copy = df.copy()
        df_copy['text_description'] = df_copy.apply(
            lambda row: self.generate_description(row, template_type), axis=1
        )
        return df_copy


if __name__ == '__main__':
    # df = get_targets('data/UKB/fmri/TianS3/data_resampled.h5')
    # df.to_csv('data/UKB/fmri/metadata.csv', index=False)
    df = pd.read_csv('data/UKB/fmri/metadata.csv')

    # get z-score transform of fluid intelligence
    df['fluid_intelligence_z'] = (df['fluid intelligence'] - 5.60236) / 2.01301

    text_gen = fMRITextGenerator()

    df_with_text = text_gen.generate_dataset(df, 'medical_template')
    df_with_text.to_csv('data/UKB/fmri/metadata_with_text_medical.csv', index=False)