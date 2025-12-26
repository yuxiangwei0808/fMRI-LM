import pandas as pd
import polars as pl
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional


def get_targets(data_path):
    df = pd.read_table('/data/neuromark2/Data/HCP_Aging/Data_info/cogcomp01.txt', sep='\s+')
    df = df.drop(0)
    df = pl.from_pandas(df)
    
    df_col = pd.read_table('/data/neuromark2/Data/HCP_Aging/Data_info/bsc01.txt', sep='\s+')
    df_col = df_col.drop(0)
    df_col = pl.from_pandas(df_col)
    df_col = df_col[['src_subject_id', 'rsptc_no']]
    df_col = df_col.with_columns([df_col['rsptc_no'].cast(pl.Float64)])

    df_flanker = pd.read_table('/data/neuromark2/Data/HCP_Aging/Data_info/flanker01.txt', sep='\s+')
    df_flanker = df_flanker.drop(0)
    df_flanker = pl.from_pandas(df_flanker)
    df_flanker = df_flanker[['src_subject_id', 'nih_flanker_ageadjusted']]
    df_flanker = df_flanker.with_columns([df_flanker['nih_flanker_ageadjusted'].cast(pl.Float64)])
    
    df_physical = pd.read_table('/data/neuromark2/Data/HCP_Aging/Data_info/vitals01.txt', sep='\s+')
    df_physical = df_physical.drop(0)
    df_physical = pl.from_pandas(df_physical)
    df_physical = df_physical[['src_subject_id', 'bp', 'weight_std', 'vtl007']]  # pound, inch
    # seperate bp as systolic/diastolic
    df_physical = df_physical.with_columns([
        df_physical['bp'].str.split('/').list.get(0).cast(pl.Float64).alias('systolic_bp'),
        df_physical['bp'].str.split('/').list.get(1).cast(pl.Float64).alias('diastolic_bp'),
    ])
    df_physical = df_physical.drop('bp')
    df_physical = df_physical.with_columns([
        df_physical['weight_std'].cast(pl.Float64),
        df_physical['vtl007'].cast(pl.Float64),
    ])
    df_physical = df_physical.with_columns([
        (df_physical['weight_std'] * 0.453592).alias('weight_kg'),
        (df_physical['vtl007'] * 0.0254).alias('height_m'),
    ])
    df_physical = df_physical.drop(['weight_std', 'vtl007'])
    df_physical = df_physical.with_columns([(df_physical['weight_kg'] / (df_physical['height_m'] ** 2)).alias('bmi')])

    df = df.join(df_col, on='src_subject_id', how='left')
    df = df.join(df_flanker, on='src_subject_id', how='left')
    df = df.join(df_physical, on='src_subject_id', how='left')

    columns = {
        'age': 'interview_age',
        'sex': 'sex',
        'fluid_composite': 'nih_fluidcogcomp_ageadjusted',
        'flanker_score': 'nih_flanker_ageadjusted',
        'blood_pressure': 'diastolic_bp',
        'bmi': 'bmi',
        'cholesterol': 'rsptc_no',
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
        subject_id = str(file_handle['metadata']['subjects'][i].decode('utf-8'))
        session_id = file_handle['metadata']['sessions'][i].decode('utf-8')

        row = df.filter(subject_id.split('_')[0] == df['src_subject_id']).to_dicts()
        if not row:
            continue
        row = row[0]
        
        # Extract features for this subject
        subject_features = {'subject_id': subject_id, 'session_id': session_id}
        
        for feature_name, col in columns.items():
            value = row.get(col)
            if feature_name == 'sex':
                value = 1 if value == 'M' else 0
            elif feature_name == 'age':
                value = float(value) / 12

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
                "- Cognitive assessment: {cognitive_desc}. {flanker_desc}\n"
                "- Physical characteristics: {bp_desc}. {bmi_desc}. {cholesterol_desc}\n"
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
                'present': lambda x: self._format_fluid(x),
                'missing': "cognitive assessment not available"
            },
            'flanker_score': {
                'present': lambda x: self._format_flanker(x),
                'missing': "flanker score not available"
            },
            'blood_pressure': {
                'present': lambda x: self._format_bp(x),
                'missing': "blood pressure data unavailable"
            },
            'bmi': {
                'present': lambda x: f"BMI: {x:.1f}" + self._bmi_category(x),
                'missing': "BMI not calculated"
            },
            'cholesterol': {
                'present': lambda x: f"cholesterol level: {x:.1f} mg/dL" + self._cholesterol_category(x),
                'missing': "cholesterol level not measured"
            },
        }

    def _format_flanker(self, x) -> str:
        """Format flanker score with clinical interpretation"""
        z_score, raw_score = x
        text = f"flanker score (measures attentional control) is {raw_score}"
        if z_score < -1.5:
            text += " (below average for adults over 36)"
        elif z_score > 1.5:
            text += " (above average for adults over 36)"
        else:
            text += " (average for adults over 36)"
        return text

    def _format_fluid(self, x) -> str:
        z_score, raw_score = x
        text = f"fluid intelligence score is {raw_score}"
        if z_score < -1.5:
            text += " (below average for adults over 36)"
        elif z_score > 1.5:
            text += " (above average for adults over 36)"
        else:
            text += " (average for adults over 36)"
        return text

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
            'cognitive_desc': self._format_field('fluid_composite', (row.get('fluid_composite_z'), row.get('fluid_composite'))),
            'flanker_desc': self._format_field('flanker_score', (row.get('flanker_score_z'), row.get('flanker_score'))),
            'bp_desc': self._format_field('blood_pressure', row.get('blood_pressure')),
            'bmi_desc': self._format_field('bmi', row.get('bmi')),
            'cholesterol_desc': self._format_field('cholesterol', row.get('cholesterol')),
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
    # df = get_targets('data/HCP_Aging/fmri/TianS3/data_resampled.h5')
    # df.to_csv('data/HCP_Aging/fmri/metadata.csv', index=False)

    df = pd.read_csv('data/HCP_Aging/fmri/metadata.csv')
    df['fluid_composite_z'] = (df['fluid_composite'] - df['fluid_composite'].mean()) / df['fluid_composite'].std()
    df['flanker_score_z'] = (df['flanker_score'] - df['flanker_score'].mean()) / df['flanker_score'].std()

    text_gen = fMRITextGenerator()

    df_with_text = text_gen.generate_dataset(df, 'medical_template')
    df_with_text['text_description'] = df_with_text['text_description'].astype(str)
    df_with_text.to_csv('data/HCP_Aging/fmri/metadata_with_text_medical.csv', index=False)

    # def format_flanker(x):
    #     if x > 1.5:
    #         return 2
    #     elif x < -1.5:
    #         return 0
    #     elif x >= -1.5 and x <= 1.5:
    #         return 1
    #     else:
    #         return x

    # gpt['fluidcomp_enc'] = gpt['fluid_composite_z'].apply(format_flanker)
    # gpt['flanker_enc'] = gpt['flanker_score_z'].apply(format_flanker)