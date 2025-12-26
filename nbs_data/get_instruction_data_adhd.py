import pandas as pd
import polars as pl
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

def get_targets():
    df = pd.read_excel('/data/qneuromark/Data/ADHD/ADHD200/Data_info/All_combined_phenotypic.xlsx')

    df['QC_Rest_1'] = df['QC_Rest_1'].fillna(0).astype(int)
    df['QC_Rest_2'] = df['QC_Rest_2'].fillna(0).astype(int)
    df['QC_Rest_3'] = df['QC_Rest_3'].fillna(0).astype(int)
    df['QC_Rest_4'] = df['QC_Rest_4'].fillna(0).astype(int)
    df['QC_Rest'] = df[['QC_Rest_1', 'QC_Rest_2', 'QC_Rest_3', 'QC_Rest_4']].max(axis=1)

    # remove rows with QC_Rest == 0
    df = df[df['QC_Rest'] != 0]

    df['DX'] = df['DX'].astype(str)
    # remove rows with DX == 'pending'
    df = df[df['DX'] != 'pending']
    df['DX'] = df['DX'].astype(int)

    df = df.rename(columns={'ScanDir ID': 'subject_id', 'Age': 'age', 'Gender': 'sex', 'DX': 'diagnosis'})

    # diagnosis: 0=Typical, 1=ADHD-Combined, 2=ADHD-Hyperactive, 3=ADHD-Inattentive
    df = df[['subject_id', 'diagnosis', 'age', 'sex', 'Verbal IQ', 'Performance IQ']]

    df.to_csv('data/ADHD200/fmri/metadata.csv', index=False)


class fMRITextGenerator:
    def __init__(self):
        self.templates = {            
            'medical_template': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- IQ Measures: {verbal_iq} {performance_iq}\n"
                "- ADHD Diagnosis: {diagnosis_desc}\n"
            ),
            'template_without_diagnosis': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- IQ Measures: {verbal_iq} {performance_iq}\n"
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
            'diagnosis': {
                'present': lambda x: {
                    0: "Typical development",
                    1: "ADHD-Combined",
                    2: "ADHD-Hyperactive",
                    3: "ADHD-Inattentive"
                }.get(x, "Unknown diagnosis"),
                'missing': "diagnosis unspecified"
            },
            'Verbal IQ': {
                'present': lambda x: self._format_viq(x),
                'missing': "Verbal IQ unspecified;"
            },
            'Performance IQ': {
                'present': lambda x: self._format_piq(x),
                'missing': "Performance IQ unspecified;"
            },
        }

    def _format_iq(self, x, iq_type="Verbal"):
        """Format IQ score with descriptive label and a short interpretive phrase."""
        if pd.isna(x):
            return f"{iq_type} IQ unspecified"

        score = int(x)
        if score < 70:
            label = "very low"
        elif score < 90:
            label = "below average"
        elif score < 120:
            label = "average"
        elif score < 130:
            label = "superior"
        else:
            label = "very superior"

        # Add short interpretive phrase by IQ type
        if iq_type == "Verbal":
            desc = "reflecting language-related cognitive abilities"
        else:
            desc = "reflecting nonverbal reasoning and spatial skills"

        return f"{iq_type} IQ ({desc}) is {score}, which is {label}."

    def _format_viq(self, x):
        return self._format_iq(x, iq_type="Verbal")

    def _format_piq(self, x):
        return self._format_iq(x, iq_type="Performance")
    
    def _format_field(self, field_name: str, value: Any) -> str:
        """Format individual field with missing value handling"""
        if pd.isna(value) or value is None:
            return self.field_templates[field_name]['missing']
        else:
            return self.field_templates[field_name]['present'](value)

    def generate_description(self, row: pd.Series, template_type: str = 'base_template') -> str:
        field_descriptions = {
            'age_desc': self._format_field('age', row.get('age')),
            'sex_desc': self._format_field('sex', row.get('sex')),
            'diagnosis_desc': self._format_field('diagnosis', row.get('diagnosis')),
            'verbal_iq': self._format_field('Verbal IQ', row.get('Verbal IQ')),
            'performance_iq': self._format_field('Performance IQ', row.get('Performance IQ')),
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
    # get_targets()

    df = pd.read_csv('data/ADHD200/fmri/metadata_with_text_medical.csv')
    df.loc[df['Verbal IQ'] < 0, 'Verbal IQ'] = np.nan
    df.loc[df['Performance IQ'] < 0, 'Performance IQ'] = np.nan
    df['ADHD'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)

    text_gen = fMRITextGenerator()
    df_with_text = text_gen.generate_dataset(df, template_type='template_without_diagnosis')
    df_with_text.to_csv('data/ADHD200/fmri/metadata_with_text_.csv', index=False)

    # def format_iq(x):
    #     if x < 90:
    #         return 0
    #     elif x < 120:
    #         return 1
    #     elif x >= 120:
    #         return 2
    #     else:
    #         return x
    # gpt['VIQ_enc'] = gpt['Verbal IQ'].apply(format_iq)
    # gpt['PIQ_enc'] = gpt['Performance IQ'].apply(format_iq)