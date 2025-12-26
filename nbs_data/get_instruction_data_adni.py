import pandas as pd
import polars as pl
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional


class fMRITextGenerator:
    def __init__(self):
        self.templates = {            
            'medical_template': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- Azheimer's Diagnosis: {diagnosis_desc}\n"
                "- Genetic risk: {apoe4_desc}\n"
                "- Biomarkers: {av45_desc}\n"
                "- Cognitive assessment: {cdrsb_desc}, {mmse_desc}\n"
            ),
            'template_without_diagnosis': (
                "Subject Information:\n"
                "- Demographics: {sex_desc} subject, {age_desc}\n"
                "- Genetic risk: {apoe4_desc}\n"
                "- Biomarkers: {av45_desc}\n"
                "- Cognitive assessment: {cdrsb_desc}, {mmse_desc}\n"
            )
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
                    0: "Cognitively Normal (CN)",
                    1: "Mild Cognitive Impairment (MCI)",
                    2: "Alzheimer's Disease (AD)"
                }.get(int(x), "diagnosis unknown"),
                'missing': "diagnosis unspecified"
            },
            'apoe4': {
                'present': self._format_apoe4,
                'missing': "APOE4 status unspecified"
            },
            'av45': {
                'present': self._format_av45,
                'missing': "AV45 PET data unavailable"
            },
            'cdrsb': {
                'present': self._format_csrsb,
                'missing': "CDR-SB score unavailable"
            },
            'mmse': {
                'present': self._format_mmse,
                'missing': "MMSE score unavailable"
            },
        }
    
    def _format_apoe4(self, value: Any) -> str:
        """Format APOE4 field with missing value handling"""
        if pd.isna(value) or value is None:
            return "APOE4 status unspecified"
        else:
            if value == 0:
                return "No APOE4 alleles (lowest genetic risk for Alzheimer's)"
            elif value == 1:
                return "One APOE4 allele (moderate genetic risk for Alzheimer's)"
            elif value == 2:
                return "Two APOE4 alleles (highest genetic risk for Alzheimer's)"
            else:
                return "APOE4 status unknown"
    
    def _format_av45(self, value: Any) -> str:
        """Format AV45 field with missing value handling"""
        if pd.isna(value) or value is None:
            return "AV45 PET data unavailable"
        else:
            if value < 1.1:
                return "AV45 PET shows low amyloid burden (typical of normal cognition)"
            elif 1.1 <= value < 1.3:
                return "AV45 PET shows moderate amyloid burden (possible MCI)"
            else:
                return "AV45 PET shows high amyloid burden (indicative of Alzheimer's disease)"

    def _format_csrsb(self, value: Any) -> str:
        """Format CDR-SB field with missing value handling"""
        if pd.isna(value) or value is None:
            return "CDR-SB score unavailable"
        else:
            if value < 0.5:
                return "CDR-SB score indicates normal cognition"
            elif 0.5 <= value < 4.0:
                return "CDR-SB score indicates very mild to mild cognitive impairment"    
            elif 4.0 <= value < 9.0:
                return "CDR-SB score indicates moderate cognitive impairment"
            else:
                return "CDR-SB score indicates severe cognitive impairment"

    def _format_mmse(self, value: Any) -> str:
        """Format MMSE field with missing value handling"""
        if pd.isna(value) or value is None:
            return "MMSE score unavailable"
        else:
            if value >= 24:
                return "MMSE score indicates normal cognition"
            elif 18 <= value < 24:
                return "MMSE score indicates mild cognitive impairment"
            elif 10 <= value < 18:
                return "MMSE score indicates moderate cognitive impairment"
            else:
                return "MMSE score indicates severe cognitive impairment"

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
            'diagnosis_desc': self._format_field('diagnosis', row.get('diagnosis')),
            'apoe4_desc': self._format_field('apoe4', row.get('apoe4')),
            'av45_desc': self._format_field('av45', row.get('av45')),
            'cdrsb_desc': self._format_field('cdrsb', row.get('cdrsb')),
            'mmse_desc': self._format_field('mmse', row.get('mmse')),
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
    df = pd.read_csv('data/ADNI/fmri/metadata.csv')
    # drop nan rows on diagnosis
    df = df.dropna(subset=['diagnosis'])
    df['AD'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)
    df['ADMCI'] = df['diagnosis']
    df['session_id'] = 'ses01'

    text_gen = fMRITextGenerator()

    df_with_text = text_gen.generate_dataset(df, 'template_without_diagnosis')
    df_with_text.to_csv('data/ADNI/fmri/metadata_with_text_medical.csv', index=False)

    # def format_av45(x):
    #     if x > 1.1:
    #         return 1
    #     elif x <= 1.1:
    #         return 0
    #     else:
    #         return x
    # def format_apoe(x):
    #     if x >= 1:
    #         return 1
    #     elif x == 0:
    #         return 0
    #     else:
    #         return x
    # gpt['av45_enc'] = gpt['av45'].apply(format_av45)
    # gpt['apoe4_enc'] = gpt['apoe4'].apply(format_apoe)