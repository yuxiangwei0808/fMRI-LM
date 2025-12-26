import pandas as pd
import polars as pl
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional


def get_targets(data_path):
    df1 = pl.read_csv('/data/qneuromark/Data/HCP/Data_info/HCP_demo.csv')
    df2 = pl.read_csv('/data/qneuromark/Data/HCP/Data_info/RESTRICTED_12_2_2020_5_36_9.csv', ignore_errors=True)

    df = df1.join(df2, left_on='Subject', right_on='Subject', how='inner')
    
    columns = {
        'age': 'Age_in_Yrs',
        'BMI': 'BMI',
        'sex': 'Gender',
        'fluid_composite': 'CogFluidComp_AgeAdj',
    }

    # Get all unique column names we need
    all_columns = ['eid'] + [col for col_tuple in columns.values() for col in col_tuple]
    all_columns = list(set(all_columns))  # Remove duplicates
    
    # Now load the HDF5 file to get subject IDs we need
    file_handle = h5py.File(data_path, 'r')
    num_samples = len(file_handle['time_series'])
    
    # List to store the results
    results = []

    for i in range(num_samples):
        subject_id = int(file_handle['metadata']['subjects'][i].decode('utf-8'))
        session_id = file_handle['metadata']['sessions'][i].decode('utf-8')

        row = df.filter(subject_id == df['Subject']).to_dicts()[0]
        
        # Extract features for this subject
        subject_features = {'subject_id': subject_id, 'session_id': session_id}
        
        for feature_name, col in columns.items():
            value = row.get(col)
            if feature_name == 'sex':
                value = 1 if value == 'M' else 0
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
            'medical_template': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}, \n"
                "- Physical characteristics: {bmi_desc}\n"
                "- Cognitive assessment: {cognitive_desc}\n"
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
            'fluid_composite': {
                'present': lambda x: self._format_fluid_composite(x),
                'missing': "cognitive assessment not available"
            },
            'bmi': {
                'present': lambda x: f"BMI: {x:.1f} kg/m²" + self._bmi_category(x),
                'missing': "BMI not calculated"
            }
        }

    def _format_fluid_composite(self, x) -> str:
        raw_score, z_score = x
        if pd.isna(raw_score) or pd.isna(z_score):
            return "cognitive assessment not available"
        text = "fluid composite score: {:.1f}".format(raw_score)
        if z_score < -1.5:
            text += " (below average of 22-35 young adults)"
        elif z_score > 1.5:
            text += " (above average of 22-35 young adults)"
        else:
            text += " (average of 22-35 young adults)"
        return text

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
            'cognitive_desc': self._format_field('fluid_composite', (row.get('fluid_composite'), row.get('fluid_composite_z'))),
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
    # df = get_targets('data/HCP/fmri/TianS3/data_resampled_split180*2.h5')

    df = pd.read_csv('data/HCP/fmri/metadata_with_text_medical.csv')
    df['fluid_composite_z'] = (df['fluid_composite'] - df['fluid_composite'].mean()) / df['fluid_composite'].std()
    text_gen = fMRITextGenerator()

    df_with_text = text_gen.generate_dataset(df, 'medical_template')
    df_with_text.to_csv('data/HCP/fmri/metadata_with_text_medical.csv', index=False)