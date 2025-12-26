import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional

def get_targets():
    df = pd.read_csv('/data/qneuromark/Data/Autism/ABIDE2/Data_info/ABIDEII_Composite_Phenotypic.csv', encoding='latin1')
    df['SITE_ID'] = df['SITE_ID'].fillna('').astype(str)
    df['SUB_ID'] = df['SUB_ID'].fillna('').astype(str)
    site_ids, sub_ids = df['SITE_ID'].values, df['SUB_ID'].values
    sub_ids = [f"{site_id}_{sub_id}" for site_id, sub_id in zip(site_ids, sub_ids)]

    df['subject_id'] = sub_ids
    df = df.rename(columns={'DX_GROUP': 'diagnosis', 'AGE_AT_SCAN ': 'age', 'SEX': 'sex'})

    df = df[['subject_id', 'diagnosis', 'age', 'sex', 'FIQ', 'VIQ', 'PIQ']]
    df['session_id'] = 'ses01'

    df.to_csv('data/ABIDE2/fmri/metadata.csv', index=False)


class fMRITextGenerator:
    def __init__(self):
        self.templates = {            
            'medical_template': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- IQ Measures: {verbal_iq} {performance_iq} {full_scale_iq}\n"
                "- Autism Diagnosis: {diagnosis_desc}\n"
            ),
            'template_without_diagnosis': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- IQ Measures: {verbal_iq} {performance_iq} {full_scale_iq}\n"
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
                    'present': lambda x: "diagnosed with autism spectrum disorder" if x == 1 else "neurotypical",
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
                'Full-Scale IQ': {
                    'present': lambda x: self._format_fiq(x),
                    'missing': "Full-Scale IQ unspecified;"
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
        elif iq_type == "Full-Scale":
            desc = "reflecting overall cognitive ability across multiple domains"
        else:
            desc = "reflecting nonverbal reasoning and spatial skills"

        return f"{iq_type} IQ ({desc}) is {score}, which is {label}."

    def _format_viq(self, x):
        return self._format_iq(x, iq_type="Verbal")

    def _format_piq(self, x):
        return self._format_iq(x, iq_type="Performance")

    def _format_fiq(self, x):
        return self._format_iq(x, iq_type="Full-Scale")
    
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
            'diagnosis_desc': self._format_field('diagnosis', row.get('ASD')),
            'verbal_iq': self._format_field('Verbal IQ', row.get('VIQ')),
            'performance_iq': self._format_field('Performance IQ', row.get('PIQ')),
            'full_scale_iq': self._format_field('Full-Scale IQ', row.get('FIQ')),
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

    df = pd.read_csv('data/ABIDE2/fmri/metadata_with_text_medical_gpt.csv')

    text_gen = fMRITextGenerator()
    df_with_text = text_gen.generate_dataset(df, template_type='template_without_diagnosis')
    df_with_text.to_csv('data/ABIDE2/fmri/metadata_with_text_.csv', index=False)
