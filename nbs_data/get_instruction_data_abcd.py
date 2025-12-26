import pandas as pd
import polars as pl
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

def process_sesion(s: str) -> str:
    s = s.split('_')
    return '_'.join(s[:2])

def get_targets(data_path):
    df1 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/abcd-general/abcd_p_demo.csv', ignore_errors=True)
    df2 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/abcd-general/abcd_y_lt.csv', ignore_errors=True)
    df3 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/neurocognition/nc_y_nihtb.csv', ignore_errors=True)
    df4 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/physical-health/ph_y_bp.csv', ignore_errors=True)
    df5 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/physical-health/ph_y_anthro.csv', ignore_errors=True)
    df6 = pl.read_csv('/data/neuromark2/Data/ABCD/Data_info/Demo51/abcd-data-release-5.1/core/abcd-general/abcd_p_screen.csv', ignore_errors=True)

    df1 = df1.with_columns(pl.col('eventname').map_elements(process_sesion, return_dtype=pl.String).alias('session_id'))
    df2 = df2.with_columns(pl.col('eventname').map_elements(process_sesion, return_dtype=pl.String).alias('session_id'))
    df3 = df3.with_columns(pl.col('eventname').map_elements(process_sesion, return_dtype=pl.String).alias('session_id'))
    df4 = df4.with_columns(pl.col('eventname').map_elements(process_sesion, return_dtype=pl.String).alias('session_id'))
    df5 = df5.with_columns(pl.col('eventname').map_elements(process_sesion, return_dtype=pl.String).alias('session_id'))
    
    df6 = df6.with_columns(pl.col('eventname').map_elements(lambda s: 'baseline_year', return_dtype=pl.String).alias('session_id'))
    
    columns1 = {
        'sex': 'demo_sex_v2',
    }
    columns2 = {
        'age': 'interview_age',  # in months
    }
    columns3 = {
        'fluid_composite': 'nihtbx_fluidcomp_agecorrected',
    }
    columns4 = {
        'blood_pressure': 'blood_pressure_dia_mean',
    }
    columns5 = {
        'height': 'anthroheightcalc',  # inch
        'weight': 'anthroweightcalc',  # pound
    }
    columns6 = {
        'had_adhd': 'scrn_commondx',
        'had_schizophrenia': 'scrn_schiz',
        'had_asd': 'scrn_asd',
    }
    
    # Select only required columns from each dataframe
    df1_select = df1.select(['src_subject_id', 'session_id'] + list(columns1.values()))
    df2_select = df2.select(['src_subject_id', 'session_id'] + list(columns2.values()))
    df2_select = df2.with_columns((pl.col('interview_age') / 12).alias('interview_age'))  # convert age to years
    df3_select = df3.select(['src_subject_id', 'session_id'] + list(columns3.values()))
    df4_select = df4.select(['src_subject_id', 'session_id'] + list(columns4.values()))
    df5_select = df5.select(['src_subject_id', 'session_id'] + list(columns5.values()))
    df6_select = df6.select(['src_subject_id', 'session_id'] + list(columns6.values()))
    
    # Join dataframes on src_subject_id and session_id
    df = df1_select.join(df2_select, on=['src_subject_id', 'session_id'], how='outer')
    df = df.with_columns([
        pl.coalesce([pl.col("src_subject_id"), pl.col("src_subject_id_right")]).alias("src_subject_id"),
        pl.coalesce([pl.col("session_id"), pl.col("session_id_right")]).alias("session_id"),
        ]).drop(["src_subject_id_right", "session_id_right"])
    df = df.join(df3_select, on=['src_subject_id', 'session_id'], how='outer')
    df = df.with_columns([
        pl.coalesce([pl.col("src_subject_id"), pl.col("src_subject_id_right")]).alias("src_subject_id"),
        pl.coalesce([pl.col("session_id"), pl.col("session_id_right")]).alias("session_id"),
        ]).drop(["src_subject_id_right", "session_id_right"])
    df = df.join(df4_select, on=['src_subject_id', 'session_id'], how='outer')
    df = df.with_columns([
        pl.coalesce([pl.col("src_subject_id"), pl.col("src_subject_id_right")]).alias("src_subject_id"),
        pl.coalesce([pl.col("session_id"), pl.col("session_id_right")]).alias("session_id"),
        ]).drop(["src_subject_id_right", "session_id_right"])
    df = df.join(df5_select, on=['src_subject_id', 'session_id'], how='outer')
    df = df.with_columns([
        pl.coalesce([pl.col("src_subject_id"), pl.col("src_subject_id_right")]).alias("src_subject_id"),
        pl.coalesce([pl.col("session_id"), pl.col("session_id_right")]).alias("session_id"),
        ]).drop(["src_subject_id_right", "session_id_right"])
    df = df.join(df6_select, on=['src_subject_id'], how='left').drop('session_id_right')

    df = df.with_columns(pl.col('src_subject_id').map_elements(lambda x: ''.join(x.split('_')), return_dtype=pl.String).alias('src_subject_id'))

    session_dict = {'baseline_year': 'Baseline', '2_year': 'Twoyear', '4_year': 'Fouryear'}
    df = df.with_columns(pl.col('session_id').map_elements(lambda x: session_dict.get(x, x), return_dtype=pl.String).alias('session_id'))

    # remove rows that has nan or 3 (inter-sex) on the sex column
    df = df.filter((pl.col('demo_sex_v2').is_not_null()) & (pl.col('demo_sex_v2') != 3))
    # map 2 (female) to 0
    df = df.with_columns(pl.col('demo_sex_v2').map_elements(lambda x: 1 if x == 1 else 0, return_dtype=pl.Int64).alias('demo_sex_v2'))

    # compute BMI column
    df = df.with_columns(
        ((pl.col('anthroweightcalc') / (pl.col('anthroheightcalc') ** 2)) * 703).alias('BMI')
    )
    
    # Combine all column mappings
    columns = {**columns1, **columns2, **columns3, **columns4, **columns5, **columns6}

    # Now load the HDF5 file to get subject IDs we need
    file_handle = h5py.File(data_path, 'r')
    num_samples = len(file_handle['time_series'])
    
    # List to store the results
    results = []

    for i in range(num_samples):
        subject_id = str(file_handle['metadata']['subjects'][i].decode('utf-8'))
        session_id = file_handle['metadata']['sessions'][i].decode('utf-8')

        # find row by subject and session
        row = df.filter((pl.col('src_subject_id') == subject_id) & (pl.col('session_id') == session_id))
        if row.height == 0:
            print(f"Subject {subject_id} with session {session_id} not found in metadata.")
            continue
        row = row.to_dicts()[0]

        # Extract features for this subject
        subject_features = {'subject_id': subject_id, 'session_id': session_id}
        
        for feature_name, col in columns.items():
            value = row.get(col)
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
                "- Physical characteristics: {bmi_desc}; {blood_pressure_desc}\n"
                "- Cognitive assessment: {cognitive_desc}\n"
                "- Medical history: {adhd_desc}, {schizophrenia_desc}, {asd_desc}."
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
            },
            'blood_pressure': {
                'present': lambda x: f"diastolic blood pressure: {x:.1f} mmHg" + self._bp_category(x),
                'missing': "blood pressure not available"
            },
            'had_adhd': {
                'present': lambda x: "has a history of ADHD" if x == 1 else "no history of ADHD",
                'missing': "ADHD history not specified"
            },
            'had_schizophrenia': {
                'present': lambda x: "has a history of Schizophrenia" if x == 1 else "no history of Schizophrenia",
                'missing': "Schizophrenia history not specified"
            },
            'had_asd': {
                'present': lambda x: "has a history of ASD" if x == 1 else "no history of ASD",
                'missing': "ASD history not specified"
            },
        }

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

    def _bp_category(self, bp: float) -> str:
        if bp < 80:
            return " (normal)"
        elif bp < 90:
            return " (elevated)"
        elif bp < 120:
            return " (hypertension stage 1)"
        else:
            return " (hypertension stage 2)"
    
    def _format_fluid_composite(self, x) -> str:
        raw_score, z_score = x
        if pd.isna(raw_score) or pd.isna(z_score):
            return "cognitive assessment not available"
        text = "fluid composite score: {:.1f}".format(raw_score)
        if z_score < -1.5:
            text += " (below average of 10-year-olds)"
        elif z_score > 1.5:
            text += " (above average of 10-year-olds)"
        else:
            text += " (average of 10-year-olds)"
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
            'cognitive_desc': self._format_field('fluid_composite', (row.get('fluid_composite'), row.get('fluid_composite_z'))),
            'bmi_desc': self._format_field('bmi', row.get('BMI')),
            'blood_pressure_desc': self._format_field('blood_pressure', row.get('blood_pressure')),
            'adhd_desc': self._format_field('had_adhd', row.get('had_adhd')),
            'schizophrenia_desc': self._format_field('had_schizophrenia', row.get('had_schizophrenia')),
            'asd_desc': self._format_field('had_asd', row.get('had_asd')),
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
    df = get_targets('data/ABCD/fmri/TianS3/data_resampled.h5')
    df.to_csv('data/ABCD/fmri/metadata.csv', index=False)

    df = pd.read_csv('data/ABCD/fmri/metadata.csv')
    df['fluid_composite_z'] = (df['fluid_composite'] - df['fluid_composite'].mean()) / df['fluid_composite'].std()

    text_gen = fMRITextGenerator()

    df_with_text = text_gen.generate_dataset(df, 'medical_template')
    df_with_text.to_csv('data/ABCD/fmri/metadata_with_text_medical.csv', index=False)