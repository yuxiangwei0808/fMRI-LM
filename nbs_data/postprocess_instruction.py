import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from openai import OpenAI
from multiprocessing import Pool, Manager
from multiprocessing.managers import DictProxy
from functools import partial    

class LLMDescriptionRefiner:
    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.7,
        max_tokens: int = 200,
        cache_path: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        num_workers: int = 1
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_path = cache_path
        self.num_workers = num_workers
        self.system_prompt = system_prompt or (
            "You are a professional medical writer. Rewrite the medical description below so that it remains "
            "factually accurate but reads naturally and fluently.\n"
            "\n"
            "Guidelines:\n"
            "- Preserve all medical facts and terminology from the original text; do not add, remove, or infer new information.\n"
            "- Ensure consistency with any provided structured medical data.\n"
            "- Keep the rewritten text roughly the same length as the original (no more than 20% longer).\n"
            "- Write as a cohesive narrative paragraph (no bullet points or lists).\n"
            "- Use varied and professional language suitable for a clinical or research context.\n"
        )

        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError('DEEPSEEK_API_KEY environment variable is not set.')

        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.cache: Dict[str, str] = {}

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as cache_file:
                    self.cache = json.load(cache_file)
            except json.JSONDecodeError:
                print(f"Warning: could not decode cache file at {cache_path}, starting fresh cache.")
                self.cache = {}

    @staticmethod
    def _sanitize_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in facts.items():
            if isinstance(value, np.generic):
                sanitized[key] = value.item()
            elif isinstance(value, (np.ndarray, list, tuple)):
                sanitized[key] = list(np.asarray(value).flatten()[:10])
            else:
                sanitized[key] = value
        return sanitized

    def _build_prompt(self, original_text: str, facts: Dict[str, Any]) -> str:
        facts_json = json.dumps(self._sanitize_facts(facts), ensure_ascii=False, indent=2)
        return (
            "Original description:\n"
            f"{original_text}\n\n"
            "Return only the rewritten texts."
        )

    def _call_openai(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            presence_penalty=0.1,
            frequency_penalty=0.2
        )

        message = response.choices[0].message
        return (message.content or '').strip()

    def _maybe_save_cache(self) -> None:
        if not self.cache_path:
            return
        with open(self.cache_path, 'w', encoding='utf-8') as cache_file:
            json.dump(self.cache, cache_file, ensure_ascii=False, indent=2)

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    def revise_text(self, original_text: str, facts: Dict[str, Any]) -> str:
        if not original_text:
            return original_text

        cache_key = original_text
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = self._build_prompt(original_text, facts)

        revised_text = self._call_openai(prompt)

        if not revised_text:
            raise Exception("Failed to get a response from the LLM.")

        self.cache[cache_key] = revised_text
        self._maybe_save_cache()
        return revised_text

    def revise_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        metadata_columns: Optional[List[str]] = None,
        output_column: str = 'text_description_refined'
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        metadata_columns = metadata_columns or []
        df_copy = df.copy()
        
        if self.num_workers > 1:
            refined_texts = self._revise_dataset_parallel(df_copy, text_column, metadata_columns)
        else:
            refined_texts = self._revise_dataset_sequential(df_copy, text_column, metadata_columns)

        df_copy[output_column] = refined_texts
        return df_copy

    def _revise_dataset_sequential(
        self,
        df: pd.DataFrame,
        text_column: str,
        metadata_columns: List[str]
    ) -> List[str]:
        refined_texts = []
        iterator = tqdm(df.iterrows(), total=len(df), desc='Refining text descriptions')
        for _, row in iterator:
            original_text = row[text_column]
            facts = row[metadata_columns].to_dict() if metadata_columns else row.to_dict()
            refined_texts.append(self.revise_text(original_text, facts))
        return refined_texts

    def _revise_dataset_parallel(
        self,
        df: pd.DataFrame,
        text_column: str,
        metadata_columns: List[str]
    ) -> List[str]:
        manager = Manager()
        shared_cache = manager.dict(self.cache)
        cache_lock = manager.Lock()

        rows_data = []
        for _, row in df.iterrows():
            original_text = row[text_column]
            facts = row[metadata_columns].to_dict() if metadata_columns else row.to_dict()
            rows_data.append((original_text, facts))

        worker_fn = partial(
            _worker_revise_text,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            api_key=self.api_key,
            shared_cache=shared_cache,
            cache_lock=cache_lock
        )

        with Pool(processes=self.num_workers) as pool:
            refined_texts = list(tqdm(
                pool.imap(worker_fn, rows_data),
                total=len(rows_data),
                desc='Refining text descriptions (parallel)'
            ))

        self.cache.update(dict(shared_cache))
        self._maybe_save_cache()
        
        return refined_texts


def _worker_revise_text(
    row_data: tuple,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    api_key: str,
    shared_cache: Any,
    cache_lock: Any
) -> str:
    original_text, facts = row_data
    
    if not original_text:
        return original_text

    cache_key = original_text
    with cache_lock:
        if cache_key in shared_cache:
            return shared_cache[cache_key]

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def sanitize_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in facts.items():
            if isinstance(value, np.generic):
                sanitized[key] = value.item()
            elif isinstance(value, (np.ndarray, list, tuple)):
                sanitized[key] = list(np.asarray(value).flatten()[:10])
            else:
                sanitized[key] = value
        return sanitized

    def build_prompt(original_text: str, facts: Dict[str, Any]) -> str:
        return (
            "Original description:\n"
            f"{original_text}\n\n"
            "Return only the rewritten texts."
        )

    def call_openai(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            presence_penalty=0.1,
            frequency_penalty=0.2
        )
        message = response.choices[0].message
        return (message.content or '').strip()

    def word_count(text: str) -> int:
        return len(text.split())

    original_words = max(word_count(original_text), 1)
    prompt = build_prompt(original_text, facts)

    revised_text = call_openai(prompt)

    with cache_lock:
        shared_cache[cache_key] = revised_text
    
    return revised_text


if __name__ == "__main__":
    dataset_name = 'ADNI'
    df = pd.read_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical.csv')

    refiner = LLMDescriptionRefiner(
        model='deepseek-chat',
        cache_path=f'data/{dataset_name}/fmri/metadata_with_text_medical_cache.json',
        num_workers=64
    )
    df_with_refined_text = refiner.revise_dataset(
        df,
        text_column='text_description',
        metadata_columns=['subject_id', 'session_id']
    )
    df_with_refined_text.to_csv(f'data/{dataset_name}/fmri/metadata_with_text_medical_gpt.csv', index=False)