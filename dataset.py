import json
import os
import ast
import numpy as np
import random
import h5py
import torch
import pandas as pd
import polars as pl
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import AutoTokenizer

from utils import select_few_shot_indices

DATASET_INFO = {
    'UKB' : 'This subject is from UK Biobank dataset, where the cohort has 30-70 years old adults.',
    'ABCD': 'This subject is from ABCD dataset, where the cohort has 9-10 years old children.',
    'HCP': 'This subject is from HCP dataset, where the cohort has young adults aged 22-35.',
    'HCP_Aging': 'This subject is from HCP-Aging dataset, where the cohort has older adults aged 36-100.',
    'ADNI': 'This subject is from ADNI dataset, which focuses on Alzheimer\'s disease research and includes elderly participants.',
    'ADHD200': 'This subject is from ADHD200 dataset, which includes children and adolescents from 7-21.',
    'ABIDE2': 'This subject is from ABIDE2 dataset, which includes children and adults from 5-64.',
}

class fMRIDataSet(Dataset):
    def __init__(self, file, inds=None, norm='robust', GPT_training=False, patch_size=None, next_time_mask=False, clip_timepoints=160, **kwargs):
        self.h5_file = os.path.join(file, 'data_resampled.h5')
        with h5py.File(self.h5_file, 'r') as file_handle:
            self.keys = list(file_handle['time_series'].keys())
            self.subjs = [s.decode('utf-8') for s in file_handle['metadata']['subjects'][...]]
            self.sess = [s.decode('utf-8') for s in file_handle['metadata']['sessions'][...]]

        self.inds = inds if inds is not None else list(range(len(self.keys)))

        self.GPT_training = GPT_training
        self.next_time_mask = next_time_mask  # mask by predict all ROIs of next time step
        self.norm = norm
        self.patch_size = patch_size
        self.clip_timepoints = clip_timepoints

        norm_params = np.load(file + 'normalization_params.npz')
        if norm == 'robust':
            self.median, self.iqr = norm_params['medians'], norm_params['iqrs']
        elif norm == 'std':
            self.mean, self.std = norm_params['mean'], norm_params['std']

    def __len__(self):
        return len(self.inds)

    def interpolate_time_dimension(self, X, target_timepoints):
        """
        Interpolate the time dimension using nearest neighbor interpolation.
        
        Args:
            X: torch.Tensor of shape (N_rois, N_timepoints)
            target_timepoints: int, desired number of timepoints
            
        Returns:
            torch.Tensor of shape (N_rois, target_timepoints)
        """
        if X.shape[1] == target_timepoints:
            return X
        
        # Add batch and channel dimensions for interpolate: (1, N_rois, N_timepoints)
        X = X.unsqueeze(0)
        
        # Interpolate along the time dimension (last dimension)
        X_interp = torch.nn.functional.interpolate(X, size=target_timepoints, mode='nearest')
        
        # Remove batch dimension: (N_rois, target_timepoints)
        return X_interp.squeeze(0)

    def __getitem__(self, index):
        k = self.keys[self.inds[index]]
        with h5py.File(self.h5_file, 'r') as file_handle:
            X = file_handle['time_series'][k][...]
        
        N_rois, N_timepoints = X.shape
        X = torch.FloatTensor(X)
        
        # Clip or interpolate to match self.clip_timepoints
        if N_timepoints > self.clip_timepoints:
            X = X[:, :self.clip_timepoints]
        elif N_timepoints < self.clip_timepoints:
            X = self.interpolate_time_dimension(X, self.clip_timepoints)
        
        N_rois, N_timepoints = X.shape

        if self.norm is not None:
            if self.norm == 'std':
                X = (X - self.mean) / self.std
            elif self.norm == 'robust':
                X = (X - self.median) / self.iqr

        # TODO: implement GPT mask as in NeuroLM
        if self.GPT_training:
            N_time_segments = N_timepoints // self.patch_size
            if self.next_time_mask:
                gpt_mask = torch.tril(torch.ones(N_time_segments * N_rois, N_time_segments * N_rois))

                for t in range(N_time_segments):
                    start_idx = t * N_rois
                    end_idx = (t + 1) * N_rois
                    # All ROIs at timepoint t can attend to each other
                    gpt_mask[start_idx:end_idx, start_idx:end_idx] = 1
            else:
                gpt_mask = torch.ones(N_time_segments * N_rois, N_time_segments * N_rois)
            return X, gpt_mask.bool()
        
        return X, X

class fMRITextDataset(fMRIDataSet):
    def __init__(self, file, descriptor_types, lm_name, inds=None, norm='robust', GPT_training=False, patch_size=None, next_time_mask=False, max_len=768, is_val=False, **kwargs):
        super().__init__(file, inds, norm, GPT_training, patch_size, next_time_mask, **kwargs)

        self.lm_name = lm_name
        self.is_val = is_val
        self.add_fmri_delimiter = kwargs.get('add_fmri_delimiter', False)

        dataset_name = file.split('/')[1]
        if dataset_name == 'UKB':
            prefix = 'The subject is from UK Biobank dataset, where the cohort has 30-70 years old adults.\n'
        elif dataset_name == 'ABCD':
            prefix = 'The subject is from ABCD dataset, where the cohort has 9-10 years old children.\n'

        self.texts = {}
        if descriptor_types == ['semantic']:
            desc_file = pd.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv')
            for i in range(len(desc_file)):
                subj, sess = str(desc_file.iloc[i]['subject_id']), str(desc_file.iloc[i]['session_id'])
                desc = desc_file.iloc[i]['text_description']
                self.texts[(subj, sess)] = prefix + desc

            # filter for inds without text
            valid_inds = []
            for idx in self.inds:
                if (self.subjs[idx], self.sess[idx]) in self.texts:
                    valid_inds.append(idx)
            self.inds = valid_inds
        else:
            descriptor_dir = f'data/{dataset_name}/fmri/descriptors_rewritten'
            ds_type_dict = {'fc': 'fc_descriptors.csv', 'gradient': 'gradient_descriptors.csv', 'graph': 'graph_descriptors.csv', 'ica': 'ica_descriptors.csv'}
            desc_files = [pd.read_csv(os.path.join(descriptor_dir, ds_type_dict[dt])) for dt in descriptor_types]
            for i in range(len(desc_files[0])):
                subj, sess = str(desc_files[0].iloc[i]['subject_id']), str(desc_files[0].iloc[i]['session_id'])

                desc_list = []
                for j in range(len(desc_files)):
                    row = desc_files[j].iloc[i]
                    desc = row['summary']
                    if descriptor_types[j] == 'fc':
                        desc = "## Functional Connectivity:\n" + desc
                    elif descriptor_types[j] == 'gradient':
                        desc = "## Functional Gradient:\n" + desc
                    elif descriptor_types[j] == 'graph':
                        desc = "## Graph Metrics:\n" + desc
                    elif descriptor_types[j] == 'ica':
                        desc = "## Independent Component Analysis:\n" + desc
                    desc_list.append(desc)
                desc_all = '\n'.join(desc_list)
                self.texts[(subj, sess)] = prefix + desc_all

            valid_inds = np.load(f'data/{dataset_name}/fmri/inds_with_desc.npy').tolist()
            self.inds = [i for i in self.inds if i in valid_inds]

        with open('data/text_prompts/prompts/prompts_imaging.json', 'r') as f:
            data = json.load(f)
            self.prompts = data['paraphrases'] + [data['base_question']] if not is_val else [data['base_question']]

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.eos_token = self.tokenizer.eos_token_id
        self.bos_token = self.eos_token if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.eos_token
        self.fill_value = self.tokenizer.pad_token_id
        self.max_len = max_len

    def _encode_text(self, text):
        prompt  = "### Question: " + random.choice(self.prompts) + "\n### Answer: "
        if 'qwen' in self.lm_name.lower():
            messages = [{'role': 'user', 'content': prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = prompt + text if not self.is_val else prompt
        elif 'gpt2' in self.lm_name.lower():
            raise NotImplementedError
         
        # TODO Here we assume for evaluation the prompts are of the same length to simplify batching; consider dynamic prompt length
        if self.is_val:
            return self.tokenizer(text, return_tensors='pt', add_special_tokens=True, truncation=True, return_attention_mask=True)
        return self.tokenizer(text, return_tensors='pt', add_special_tokens=True, max_length=self.max_len, truncation=True, padding='max_length', return_attention_mask=True)

    def __getitem__(self, index):
        X, fmri_attn_mask = super().__getitem__(index)

        k = self.keys[self.inds[index]]
        idx = int(k[7:])  # remove prefix `sample_`
        with h5py.File(self.h5_file, 'r') as file:
            meta = file['metadata']
            subj_id = meta['subjects'][idx].decode('utf-8')
            sess_id = meta['sessions'][idx].decode('utf-8')

        text = self.texts[(subj_id, sess_id)]
        text_encoding = self._encode_text(text)
        
        text_input_ids = text_encoding['input_ids'].squeeze(0)
        text_attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        if self.is_val:
            return X, fmri_attn_mask, text_input_ids, text_attention_mask, text  # text will be the target for evaluation

        # Add an EOS token to the end of sequence
        if self.eos_token != self.fill_value:  # QWEN3 tokenizer
            pad_pos = (text_input_ids == self.fill_value).nonzero(as_tuple=True)[0]
            assert len(pad_pos) > 0, f'No pad token found in text: {text}'
            text_input_ids[pad_pos[0]] = self.eos_token  # replace the first pad token with eos token
            text_attention_mask[pad_pos[0]] = 1  # unmask the first pad token
        else:
            # unmask the first EOS token (since pad token may also be eos token id)
            eos_pos = (text_input_ids == self.eos_token).nonzero(as_tuple=True)[0]
            eos_pos = eos_pos[1:] if eos_pos[0] == 0 else eos_pos  # if eos is the first token (bos), ignore it (becuase we already added it in the attention mask)
            assert len(eos_pos) > 0, f'No eos token found in text: {text}'
            text_attention_mask[eos_pos[0]] = 1

        return X, fmri_attn_mask, text_input_ids, text_attention_mask
  

class InstrDataset(fMRIDataSet):
    def __init__(self, dataset_name, lm_name='gpt2', is_instruct=True, is_val=False, target_name=None, text_min_len=240,
                 add_source_info=False, add_desc=False, **kwargs):
        file = f'data/{dataset_name}/fmri/TianS3/'
        super().__init__(file, **kwargs)

        df = pl.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv')
        self.subject_mapping = {}
        for row in df.iter_rows(named=True):  # pre-load into a dict
            self.subject_mapping[(str(row['subject_id']), str(row['session_id']))] = {'Y': row[target_name], 'desc': row['text_description_refined']}

        # some subjects may not have label, filter them out
        if os.path.exists(f'data/{dataset_name}/fmri/inds_with_label_{target_name}.npy'):
            valid_inds = np.load(f'data/{dataset_name}/fmri/inds_with_label_{target_name}.npy').tolist() 
        else:
            valid_inds = np.load(f'data/{dataset_name}/fmri/inds_with_label.npy').tolist()
        self.inds = [i for i in self.inds if i in valid_inds]

        self.is_instruct = is_instruct
        self.lm_name = lm_name
        self.is_val = is_val
        self.text_min_len = text_min_len
        self.target_name = target_name
        self.add_source_info = add_source_info  # add the dataset source info to the prompt
        # add the text description (such as physical, cognitive, biomarker, etc) to the prompt.
        # should only be used for disease classification or biomarker positivity classification
        if target_name in ['AD', 'AsymAD', 'ADHD', 'ASD']:
            self.add_desc = add_desc
        else:
            self.add_desc = False
        self.source_info = DATASET_INFO.get(dataset_name, '')

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.eos_token = self.tokenizer.eos_token_id
        self.bos_token = self.eos_token if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
        self.fill_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token

        use_random_prompt = kwargs.get('use_random_prompt', False)
        
        if is_instruct:
            with open(f'data/text_prompts/prompts/prompts_{target_name}.json', 'r') as f:
                if use_random_prompt:
                    self.questions = json.load(f)['paraphrases']
                    self.questions = [s.strip("\"") for s in self.questions]
                else:
                    self.questions = [json.load(f)['base_question']]

            if target_name == 'sex':
                self.text = {
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' Male')},
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' Female')},
                }
            elif target_name == 'AD':
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' CN')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' AD')},
                    # 2: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' AD')},
                }
            elif target_name == 'AsymAD':
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' CN')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' AM')},
                }
            elif target_name == 'ADHD':
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' Control')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' ADHD')},
                }
            elif target_name == 'ASD':
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' Control')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' Autism')},
                }
            elif target_name in ['fluidintel_enc', 'fluidcomp_enc', 'flanker_enc', 'VIQ_enc', 'PIQ_enc']:
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' lower')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' mean')},
                    2: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' higher')},
                }
            elif target_name in ['av45_enc', 'apoe4_enc']:
                self.text = {
                    0: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' negative')},
                    1: {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': self.tokenizer.encode(' positive')},
                }
            elif target_name in ['age', 'fluidintel', 'fluidcomp', 'flanker']:
                # fluid intelligence: UKB, 0-15
                self.text = {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': None}
            self.prompt = '{src_info}{desc}### Question: {prompt}\n### Answer:'

    def get_text(self, target, desc, **kwargs):
        if self.target_name in ['age', 'fluidintel', 'fluidcomp', 'flanker']:
            qa_pair = self.text
        else:
            qa_pair = self.text[int(target)]
        
        # use random questions if self.questions has multiple entries
        question = random.choice(self.questions)
        question = self.tokenize_prompt(qa_pair['Q'], question, desc)

        if self.target_name in ['age', 'fluidintel', 'fluidcomp', 'flanker']:
            answer = torch.IntTensor(self.tokenizer.encode(str(target)))
        else:
            answer = torch.IntTensor(qa_pair['A'])
        return torch.cat((question, answer)), question.size(0)
    
    def tokenize_prompt(self, template, prompt, desc):
        template = template.replace('{prompt}', prompt)
        if self.add_source_info:
            template = template.replace('{src_info}', self.source_info + '\n')
        else:
            template = template.replace('{src_info}', '')
        if self.add_desc and desc is not None:
            template = template.replace('{desc}', '### Description: ' + desc + '\n')
        else:
            template = template.replace('{desc}', '')
        # tokens = [self.bos_token] + self.tokenizer.encode(template)

        # handle qwen which need special template
        if 'qwen' in self.lm_name.lower():
            messages = [{'role': 'user', 'content': template}]  # the last token will be `\n\n`
            template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.tokenizer.encode(template)

        return torch.IntTensor(tokens)
    
    def __getitem__(self, index):
        k = self.keys[self.inds[index]]
        with h5py.File(self.h5_file, 'r') as file:
            X = file['time_series'][k][...]
        
        N_rois, N_timepoints = X.shape[:2]
        X = torch.FloatTensor(X)
        
        if N_timepoints > self.clip_timepoints:
            X = X[:, :self.clip_timepoints]
        elif N_timepoints < self.clip_timepoints:
            X = self.interpolate_time_dimension(X, self.clip_timepoints)

        if self.norm is not None:
            if self.norm == 'std':
                X = (X - self.mean) / self.std
            elif self.norm == 'robust':
                X = (X - self.median) / self.iqr
        if X.ndim == 3: X = X.permute(2, 0, 1)  # C V T

        N_time_segments = self.clip_timepoints // self.patch_size
        num_fmri_tokens = N_time_segments * N_rois

        if self.next_time_mask:
            gpt_mask = torch.tril(torch.ones(num_fmri_tokens, num_fmri_tokens))
            for t in range(N_time_segments):
                start_idx = t * N_rois
                end_idx = (t + 1) * N_rois
                # All ROIs at timepoint t can attend to each other
                gpt_mask[start_idx:end_idx, start_idx:end_idx] = 1
        else:
            gpt_mask = torch.ones(num_fmri_tokens, num_fmri_tokens)

        if not self.is_instruct:
            return X, gpt_mask.bool()

        # get label from df
        idx = int(k[7:])  # remove prefix `sample_`
        with h5py.File(self.h5_file, 'r') as file:
            subj_id = file['metadata']['subjects'][idx].decode('utf-8')
            sess_id = file['metadata']['sessions'][idx].decode('utf-8')
        Y = self.subject_mapping[(subj_id, sess_id)]['Y']
        desc = self.subject_mapping[(subj_id, sess_id)]['desc']

        if self.is_val:
            # question = random.choice(self.questions)
            question = self.questions[0]

            text = self.tokenize_prompt(self.prompt, question, desc)
            valid_text_len = text.size(0)

            # text_gpt_mask is as usual even if padding is applied
            text_gpt_mask = torch.tril(torch.ones(valid_text_len, valid_text_len))
            pad_len = 0

            if self.add_desc and self.text_min_len > valid_text_len:  # manually padding (padding will be removed later; for batch processing purpose only)
                pad_len = self.text_min_len - valid_text_len
                text = torch.cat([text, torch.full((pad_len,), self.fill_value, dtype=torch.long)], dim=0)

                # enlarge gpt_mask since padding will be applied to the left of fMRI tokens
                # gpt_mask = torch.block_diag(torch.zeros(pad_len, pad_len), gpt_mask)
                text_gpt_mask = torch.block_diag(text_gpt_mask, torch.zeros(pad_len, pad_len))

            gpt_mask = torch.block_diag(gpt_mask, text_gpt_mask)
            gpt_mask[pad_len + num_fmri_tokens:pad_len + num_fmri_tokens + valid_text_len, pad_len:pad_len + num_fmri_tokens] = 1  # text can attend to all fMRI tokens

            return X, text, Y, gpt_mask.bool()
        else:
            # if any(y is None for y in Y):
            #     raise ValueError(f'Label {self.target_name} is missing for subject {subj_id}, session {sess_id}.')
            text, prompt_len = self.get_text(Y, desc, subject_id=subj_id, session_id=sess_id)
            # pad to text_min_len
            valid_text_len = text.size(0)
            
            # append EOS token
            text = torch.cat([text, torch.full((1,), self.eos_token, dtype=torch.long)], dim=0)

            if self.text_min_len > valid_text_len:  # manually padding (right padding)
                pad_len = self.text_min_len - valid_text_len
                text = torch.cat([text, torch.full((pad_len,), self.fill_value, dtype=torch.long)], dim=0)  # fill_value=50256 for GPT2

        text_gpt_mask = torch.zeros(text.size(0), text.size(0))
        text_gpt_mask[:valid_text_len, :valid_text_len] = torch.tril(torch.ones(valid_text_len, valid_text_len))
        gpt_mask = torch.block_diag(gpt_mask, text_gpt_mask)
        gpt_mask[num_fmri_tokens:num_fmri_tokens + valid_text_len, :num_fmri_tokens] = 1  # text can attend to all fMRI tokens
        
        Y_text = torch.full_like(text, fill_value=-1)
        # only preserve the answer part
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        # add a eos to the last token
        Y_text[valid_text_len - 1] = self.eos_token

        return X, text, Y_text, gpt_mask.bool(), Y

    @property
    def dataset_type(self):
        return 'instruction'

class MultiQuestionInstrDataset(InstrDataset):
    """Multi-question (multi-task) instruction dataset for classificiation tasks. target_name should be a list"""
    def __init__(self, dataset_name, lm_name='gpt2', is_instruct=True, is_val=False, target_name=None, text_min_len=200,
                 add_source_info=False, add_desc=False, **kwargs):
        super().__init__(dataset_name, lm_name, is_instruct, is_val, target_name[0], text_min_len,
                         add_source_info, add_desc, **kwargs)

        self.target_name = target_name  # list of target names
        df = pl.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv')
        self.subject_mapping = {}
        for row in df.iter_rows(named=True):  # pre-load into a dict
            targets = tuple(row[tn] for tn in target_name)
            self.subject_mapping[(str(row['subject_id']), str(row['session_id']))] = {'Y': targets, 'desc': row['text_description_refined']}

        # TODO make this more flexible; currently harcode the targets that will be used for each dataset
        # fluid intelligence is used to generate valid inds for UKB; flanker for HCP Aging; AD/Apeo4 for ADNI; VIQ/PIQ for ADHD200
        valid_inds = np.load(f'data/{dataset_name}/fmri/inds_with_label_mq.npy').tolist() 
        self.inds = [i for i in self.inds if i in valid_inds]

        self.target_dict = {
            'sex': {1: ' Male', 0: ' Female'},
            'AD': {0: ' CN', 1: ' AD'},
            'ADHD': {0: ' Control', 1: ' ADHD'},
            'fluidintel_enc': {0: ' lower', 1: ' mean', 2: ' higher'},
            'fluidcomp_enc': {0: ' lower', 1: ' mean', 2: ' higher'},
            'flanker_enc': {0: ' lower', 1: ' mean', 2: ' higher'},
            'VIQ_enc': {0: ' lower', 1: ' mean', 2: ' higher'},
            'PIQ_enc': {0: ' lower', 1: ' mean', 2: ' higher'},
            'av45_enc': {0: ' negative', 1: ' positive'},
            'apoe4_enc': {0: ' negative', 1: ' positive'},
        }

        self.questions = {}
        use_random_prompt = kwargs.get('use_random_prompt', False)
        # TODO make this more flexible so that different targets can be used for one dataset
        with open(f'data/text_prompts/prompts/prompts_mq_{dataset_name}.json', 'r') as f:
            if use_random_prompt:
                try:
                    paraphrases = json.load(f)['paraphrases']
                    self.questions = [s.strip("\"") for s in paraphrases]
                except:
                    self.questions = [json.load(f)['base_question']]
            else:
                self.questions = [json.load(f)['base_question']]

        self.text = {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': None}

        # if 'sex' not in target_name:
        #     self.add_desc = add_desc
        # else:
        #     self.add_desc = False

    def get_text(self, targets: tuple, desc: str, **kwargs):
        qa_pair = self.text
        
        # use random questions if self.questions has multiple entries
        question = random.choice(self.questions)
        question = self.tokenize_prompt(qa_pair['Q'], question, desc)

        answer_strs = []
        for i, tn in enumerate(self.target_name):
            if tn in ['age', 'fluidintel', 'fluidcomp', 'flanker', 'VIQ', 'PIQ']:
                answer_strs.append(' ' + str(targets[i]))
            else:
                answer_strs.append(self.target_dict[tn][targets[i]])
            
        answer_full = ''.join(answer_strs)
        answer = torch.IntTensor(self.tokenizer.encode(answer_full))
        return torch.cat((question, answer)), question.size(0)
    
    def __getitem__(self, index):
        result = super().__getitem__(index)
        
        # Convert Y from tuple to tensor for multi-task case
        if self.is_val:
            X, text, Y, gpt_mask = result
            # Use long for classification (discrete labels), float for regression
            Y = torch.tensor(Y, dtype=torch.long) if isinstance(Y, tuple) else Y
            return X, text, Y, gpt_mask
        else:
            X, text, Y_text, gpt_mask, Y = result
            # Use long for classification (discrete labels), float for regression
            Y = torch.tensor(Y, dtype=torch.long) if isinstance(Y, tuple) else Y
            return X, text, Y_text, gpt_mask, Y


class OpenEndedInstrDataset(InstrDataset):
    """Instruction dataset for open-ended generation tasks."""
    def __init__(self, dataset_name, lm_name='gpt2', is_instruct=True, is_val=False, target_name=None, text_min_len=200,
                 add_source_info=False, add_desc=False, **kwargs):
        super().__init__(dataset_name, lm_name, is_instruct, is_val, target_name, text_min_len,
                         add_source_info, add_desc, **kwargs)
        
        valid_inds = np.load(f'data/{dataset_name}/fmri/inds_with_label_open.npy').tolist() 
        self.inds = [i for i in self.inds if i in valid_inds]

        target_name_dict = {
            'UKB': ['sex', 'age_group', 'fluidintel_enc'],
            'HCP_Aging': ['sex', 'age_group', 'fluidcomp_enc', 'flanker_enc'],
            'ADNI': ['sex', 'age_group', 'diagnosis', 'apoe4', 'av45_enc'],
        }
        sex_map = {1: 'Male', 0: 'Female'}
        cog_map = {0: 'lower than cohort', 1: 'average within cohort', 2: 'higher than cohort'}
        AD_map = {0: 'Cognitively Normal', 1: 'mild cognitive impairment', 2: 'Alzheimer\'s Disease'}
        apoe_map = {0: 'no APOE4 alleles', 1: 'at least one APOE4 allele', 2: 'two APOE4 alleles'}
        av45_map = {0: 'amyloid negative', 1: 'amyloid positive'}

        target_name = target_name_dict[dataset_name]
        df = pl.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv')
        self.subject_mapping = {}
        for row in df.iter_rows(named=True):  # pre-load into a dict
            targets = {}
            for tn in target_name:
                if row[tn] is None:
                    continue
                if tn == 'sex':
                    targets[tn] = sex_map[row[tn]]
                elif tn in ['fluidintel_enc', 'fluidcomp_enc', 'flanker_enc']:
                    targets[tn] = cog_map[row[tn]]
                elif tn == 'diagnosis':
                    targets[tn] = AD_map[row[tn]]
                elif tn == 'apoe4':
                    targets[tn] = apoe_map[row[tn]]
                elif tn == 'av45_enc':
                    targets[tn] = av45_map[row[tn]]
                else:
                    targets[tn] = str(row[tn])
            self.subject_mapping[(str(row['subject_id']), str(row['session_id']))] = {'Y': targets, 'desc': row['text_description_refined']}

        self.questions = {}
        use_random_prompt = kwargs.get('use_random_prompt', False)
        with open(f'data/text_prompts/prompts/prompts_open.json', 'r') as f:
            if use_random_prompt:
                try:
                    paraphrases = json.load(f)['paraphrases']
                    self.questions = [s.strip("\"") for s in paraphrases]
                except:
                    self.questions = [json.load(f)['base_question']]
            else:
                self.questions = [json.load(f)['base_question']]
        self.text = {'Q': '{src_info}{desc}### Question: {prompt}\n### Answer:', 'A': None}

        # TODO currently hardcode target set for each dataset
        # UKB: sex/age_group/fluidintel_enc; HCP_Aging: sex/age_gorup/fluidcomp_enc/flanker_enc; ADNI: sex/age_group/AD/apoe4
        answers = pd.read_csv(f'data/{dataset_name}/fmri/answers_open.csv')
        self.answer_mapping = {}
        for row in answers.itertuples(index=False):
            self.answer_mapping[(str(row.subject_id), str(row.session_id))] = ast.literal_eval(row.answers)

    def get_text(self, target, desc, subject_id, session_id, **kwargs):
        qa_pair = self.text
        
        # use random questions if self.questions has multiple entries
        question = random.choice(self.questions)
        question = self.tokenize_prompt(qa_pair['Q'], question, desc)

        answer_str = random.choice(self.answer_mapping[(str(subject_id), str(session_id))])
        answer = torch.IntTensor(self.tokenizer.encode(' ' + answer_str))
        return torch.cat((question, answer)), question.size(0)

    def __getitem__(self, index):
        result = super().__getitem__(index)

        if self.is_val:
            X, text, Y, gpt_mask = result
            # Y is already a dict for validation - keep it as is
            return X, text, Y, gpt_mask
        else:
            X, text, Y_text, gpt_mask, Y = result
            # For training, Y is a dict - make it to 0 (not used during training anyway)
            # The actual training target is Y_text (tokenized answer)
            return X, text, Y_text, gpt_mask, 0
    

def get_fmri_data(file, data_cls=fMRIDataSet, train_ratio=1, val_ratio=0.2, **kwargs):
    """
    Create train and validation datasets from fMRI data files.
    
    Args:
        file: Single file path or list of file paths
        data_cls: Dataset class to instantiate
        train_ratio: Proportion of data for training (if < 1, use sequential split)
        val_ratio: Proportion of data for validation (if train_ratio == 1, use random split)
    """
    # Normalize file input to list
    files = file if isinstance(file, list) else [file]
    
    # Create initial dataset to get total length
    temp_dataset = ConcatDataset([fMRIDataSet(f, **kwargs) for f in files]) if len(files) > 1 else fMRIDataSet(files[0], **kwargs)
    total_samples = len(temp_dataset)
    
    # Determine train/val indices
    if train_ratio == 1:
        # Random validation split
        train_inds = list(range(total_samples))
        val_inds = np.random.choice(total_samples, size=int(total_samples * val_ratio), replace=False).tolist()
    else:
        # Sequential split
        split_point = int(total_samples * train_ratio)
        train_inds = list(range(split_point))
        val_inds = list(range(split_point, total_samples))
    
    # Create datasets with indices
    train_sets = [data_cls(f, inds=train_inds, **kwargs) for f in files]
    val_sets = [data_cls(f, inds=val_inds, is_val=True, **kwargs) for f in files]
    
    # Concatenate if multiple files
    train_set = ConcatDataset(train_sets) if len(files) > 1 else train_sets[0]
    val_set = ConcatDataset(val_sets) if len(files) > 1 else val_sets[0]
    
    return train_set, val_set

def get_data_info(target):
    if target == 'sex':
        return {'label_dic': {'Male': 1, 'Female': 0}, 'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                'is_binary': True, 'is_regression': False, 'target_name': 'sex', 'num_classes': 2}
    elif target == 'AD':
        return {'label_dic': {'CN': 0, 'AD': 1}, 'metrics': ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"], 
                'is_binary': True, 'is_regression': False, 'target_name': 'AD', 'num_classes': 2}
    elif target == 'AsymAD':
        return {'label_dic': {'CN': 0, 'AM': 1}, 'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                'is_binary': True, 'is_regression': False, 'target_name': 'AsymAD', 'num_classes': 2}
    elif target == 'ADHD':
        return {'label_dic': {'Control': 0, 'ADHD': 1}, 'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                'is_binary': True, 'is_regression': False, 'target_name': 'ADHD', 'num_classes': 2}
    elif target == 'ASD':
        return {'label_dic': {'Control': 0, 'Autism': 1}, 'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                'is_binary': True, 'is_regression': False, 'target_name': 'ASD', 'num_classes': 2}
    elif target in ['fluidintel_enc', 'fluidcomp_enc', 'flanker_enc', 'VIQ_enc', 'PIQ_enc']:
        return {'label_dic': {'lower': 0, 'mean': 1, 'higher': 2}, 'metrics': ["accuracy", "balanced_accuracy", "f1_macro"], 
                'is_binary': False, 'is_regression': False, 'target_name':  target, 'num_classes': 3}
    elif target in ['av45_enc', 'apoe4_enc']:
        return {'label_dic': {'negative': 0, 'positive': 1}, 'metrics': ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"], 
                'is_binary': True, 'is_regression': False, 'target_name': target, 'num_classes': 2}
    elif target in ['age', 'fluidintel', 'fluidcomp', 'flanker']:
        return {'label_dic': None, 'metrics': ["mae", "mse", "rmse", "r2"], 
                'is_binary': False, 'is_regression': True, 'target_name': target, 'num_classes': None}
    elif target == 'age_group':
        return {'label_dic': {'adolescent': 0, 'young adult': 1, 'middle-aged adult': 2, 'senior': 3, 'elderly': 4}, 'metrics': ["accuracy", "balanced_accuracy", "f1_macro"], 
                'is_binary': False, 'is_regression': False, 'target_name': target, 'num_classes': 5}
    else:
        raise ValueError(f'Target {target} not supported')

def get_fmri_data_inst(batch_size, val_batch_size, datasets=['UKB'], train_val_test_ratio=[0.7, 0.1, 0.2], dataset_target_mapping=None, dataset_config_dict=None, separate_multi_task_loaders=False, fewshot_samples=0, **kwargs):
    """
    Get fMRI instruction datasets and dataloaders.
    
    Args:
        batch_size: Batch size for training
        val_batch_size: Batch size for validation/test
        datasets: List of dataset names
        train_val_test_ratio: Split ratios for train/val/test
        dataset_target_mapping: Mapping of dataset names to target names
        dataset_config_dict: Configuration dict for datasets (e.g., is_multi flag)
        separate_multi_task_loaders: If True, return separate train loaders for multi-task datasets
                                      instead of concatenating them. This avoids batching issues
                                      when different datasets have different numbers of targets.
        fewshot_samples: Number of few-shot samples to include in training sets
        **kwargs: Additional arguments passed to dataset constructors
    
    Returns:
        If separate_multi_task_loaders=False:
            train_loader: Single concatenated DataLoader
            val_test_loaders: Dict of validation/test loaders per dataset-target
        If separate_multi_task_loaders=True:
            train_loaders: Dict of separate DataLoaders per dataset-target
            val_test_loaders: Dict of validation/test loaders per dataset-target
    """
    dataset_cls = InstrDataset
    
    # Default mapping if not provided
    default_target_names = {'UKB': ['fluidintel_enc'], 'HCP': ['sex'], 'ABCD': ['sex'], 'HCP_Aging': ['sex'], 'ADNI': ['AD'], 'ABIDE2': ['ASD'], 'ADHD200': ['ADHD'], 'EHBS': ['AsymAD']}
    
    # Use provided mapping or default
    all_target_names = dataset_target_mapping if dataset_target_mapping is not None else default_target_names
    
    train_sets = []
    train_loaders_dict = {}  # For separate loaders mode
    val_test_loaders = {}

    for name in datasets:
        # Get dataset configuration
        dataset_config = dataset_config_dict.get(name, {}) if dataset_config_dict else {}
        is_multi = dataset_config.get('is_multi', False)
        
        # Choose appropriate dataset class based on is_multi flag
        if is_multi and len(all_target_names[name]) > 1:
            # Use MultiQuestionInstrDataset for multi-task learning
            target_names = all_target_names[name]  # Pass all targets at once
            
            dataset = MultiQuestionInstrDataset(dataset_name=name, target_name=target_names, **kwargs)
            num_samples = len(dataset)

            # Load or create train/val/test splits
            # Use the first target name for the label_inds file
            primary_target = target_names[0]
            if os.path.exists(f'data/{name}/fmri/label_inds_{primary_target}.npy'):
                label_inds = np.load(f'data/{name}/fmri/label_inds_{primary_target}.npy', allow_pickle=True).item()
                train_inds = label_inds['train_inds']
                val_inds = label_inds['val_inds']
                test_inds = label_inds['test_inds']
            else:
                train_inds = list(range(int(num_samples * train_val_test_ratio[0])))
                val_inds = list(range(int(num_samples * train_val_test_ratio[0]), int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1]))))
                test_inds = list(range(int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1])), num_samples))

            if fewshot_samples > 0:
                raise NotImplementedError("Few-shot samples for multi-task datasets not implemented yet.")

            train_set = MultiQuestionInstrDataset(dataset_name=name, inds=train_inds, target_name=target_names, is_val=False, **kwargs)
            val_set = MultiQuestionInstrDataset(dataset_name=name, inds=val_inds, target_name=target_names, is_val=True, **kwargs)
            test_set = MultiQuestionInstrDataset(dataset_name=name, inds=test_inds, target_name=target_names, is_val=True, **kwargs)

            # Create key for multi-target dataset
            dataset_key = f'{name}-{"-".join(target_names)}'
            
            if separate_multi_task_loaders:
                # Create separate train loader for this dataset
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
                train_loaders_dict[dataset_key] = train_loader
            else:
                # Add to list for concatenation
                train_sets.append(train_set)

            val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
            test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

            val_test_loaders[dataset_key] = {'val': val_loader, 'test': test_loader, 'info': {t: get_data_info(t) for t in target_names}}
        else:
            # Use standard InstrDataset for single-task learning
            for target_name in all_target_names[name]:
                dataset = dataset_cls(dataset_name=name, target_name=target_name, **kwargs)
                num_samples = len(dataset)

                if os.path.exists(f'data/{name}/fmri/label_inds_{target_name}.npy'):
                    label_inds = np.load(f'data/{name}/fmri/label_inds_{target_name}.npy', allow_pickle=True).item()
                    train_inds = label_inds['train_inds']
                    val_inds = label_inds['val_inds']
                    test_inds = label_inds['test_inds']
                else:
                    train_inds = list(range(int(num_samples * train_val_test_ratio[0])))
                    val_inds = list(range(int(num_samples * train_val_test_ratio[0]), int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1]))))
                    test_inds = list(range(int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1])), num_samples))

                if fewshot_samples > 0:
                    # randomly and balancedely sample fewshot_samples from train_inds
                    train_inds = select_few_shot_indices(dataset, train_inds, fewshot_samples)
                    val_batch_size = 64

                train_set = dataset_cls(dataset_name=name, inds=train_inds, target_name=target_name, is_val=False, **kwargs)
                val_set = dataset_cls(dataset_name=name, inds=val_inds, target_name=target_name, is_val=True, **kwargs)
                test_set = dataset_cls(dataset_name=name, inds=test_inds, target_name=target_name, is_val=True, **kwargs)

                dataset_key = f'{name}-{target_name}'
                
                if separate_multi_task_loaders:
                    # Create separate train loader for this dataset-target combination
                    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
                    train_loaders_dict[dataset_key] = train_loader
                else:
                    # Add to list for concatenation
                    train_sets.append(train_set)

                val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
                test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

                data_info = get_data_info(target_name)
                val_test_loaders[dataset_key] = {'val': val_loader, 'test': test_loader, 'info': data_info}

    if separate_multi_task_loaders:
        # Return separate loaders
        return train_loaders_dict, val_test_loaders
    else:
        # Return concatenated loader
        train_set_merge = ConcatDataset(train_sets)
        train_loader = DataLoader(train_set_merge, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        return train_loader, val_test_loaders


def get_fmri_data_open(batch_size, val_batch_size, datasets=['UKB'], train_val_test_ratio=[0.7, 0.1, 0.2], dataset_config_dict=None, **kwargs):
    """Get fMRI open-ended instruction datasets and dataloaders."""

    train_sets = []
    val_test_loaders = {}
    for name in datasets:        
        dataset = OpenEndedInstrDataset(dataset_name=name, target_name='sex', **kwargs)
        num_samples = len(dataset)

        # Load or create train/val/test splits
        if os.path.exists(f'data/{name}/fmri/label_inds_open.npy'):
            label_inds = np.load(f'data/{name}/fmri/label_inds_open.npy', allow_pickle=True).item()
            train_inds = label_inds['train_inds']
            val_inds = label_inds['val_inds']
            test_inds = label_inds['test_inds']
        else:
            train_inds = list(range(int(num_samples * train_val_test_ratio[0])))
            val_inds = list(range(int(num_samples * train_val_test_ratio[0]), int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1]))))
            test_inds = list(range(int(num_samples * (train_val_test_ratio[0] + train_val_test_ratio[1])), num_samples))

        train_set = OpenEndedInstrDataset(dataset_name=name, target_name='sex', inds=train_inds, is_val=False, **kwargs)
        val_set = OpenEndedInstrDataset(dataset_name=name, target_name='sex', inds=val_inds, is_val=True, **kwargs)
        test_set = OpenEndedInstrDataset(dataset_name=name, target_name='sex', inds=test_inds, is_val=True, **kwargs)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

        data_info = {'is_regression': False, 'is_binary': False, 'target_name': 'open-ended', 'num_classes': None, 'metrics': []}   # data info is useless

        val_test_loaders[f'{name}-open'] = {'val': val_loader, 'test': test_loader, 'info': data_info}
        train_sets.append(train_set)

    train_set_merge = ConcatDataset(train_sets)
    train_loader = DataLoader(train_set_merge, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    return train_loader, val_test_loaders