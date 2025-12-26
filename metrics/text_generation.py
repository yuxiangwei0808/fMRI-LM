"""
Text generation metrics for evaluating fMRI-to-text models.
Includes BLEU-4, CIDEr, BERTScore, Sentence-BERT cosine similarity, and ROUGE-L.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
from rouge_score import rouge_scorer

class TextGenerationMetrics:
    """
    Computes text generation metrics including BLEU-4, CIDEr, BERTScore, 
    Sentence-BERT cosine similarity, and ROUGE-L.
    """

    def __init__(self, sbert_model_name='all-MiniLM-L6-v2', device='cuda'):
        """
        Args:
            sbert_model_name: Name of the Sentence-BERT model to use for semantic similarity
            device: Device to run models on
        """
        self.device = device
        self.sbert_model = None
        self.sbert_model_name = sbert_model_name
        
    def _init_sbert(self):
        """Lazy initialization of Sentence-BERT model."""
        if self.sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(self.sbert_model_name, device=self.device)
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
    
    def compute_bleu(self, 
                     hypotheses: List[str], 
                     references: List[List[str]]) -> Dict[str, float]:
        """
        Compute BLEU scores (BLEU-1 to BLEU-4).
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings (each can have multiple references)
        
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        if Bleu is None:
            raise ImportError("pycocoevalcap not installed")
        
        # Format for pycocoevalcap: dict of {id: [text]}
        hyp_dict = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        ref_dict = {i: refs if isinstance(refs, list) else [refs] 
                   for i, refs in enumerate(references)}
        
        scorer = Bleu(4)
        scores, _ = scorer.compute_score(ref_dict, hyp_dict)
        
        return {
            'BLEU-1': scores[0],
            'BLEU-2': scores[1],
            'BLEU-3': scores[2],
            'BLEU-4': scores[3],
        }
    
    def compute_cider(self, 
                      hypotheses: List[str], 
                      references: List[List[str]]) -> float:
        """
        Compute CIDEr score.
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings (each can have multiple references)
        
        Returns:
            CIDEr score
        """
        if Cider is None:
            raise ImportError("pycocoevalcap not installed")
        
        # Format for pycocoevalcap: dict of {id: [text]}
        hyp_dict = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        ref_dict = {i: refs if isinstance(refs, list) else [refs] 
                   for i, refs in enumerate(references)}
        
        scorer = Cider()
        score, _ = scorer.compute_score(ref_dict, hyp_dict)
        
        return score
    
    def compute_bert_score(self,
                           hypotheses: List[str],
                           references: List[str],
                           lang: str = 'en',
                           model_type: str = None) -> Dict[str, float]:
        """
        Compute BERTScore between generated and reference texts.
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings (single reference per hypothesis)
            lang: Language of the texts
            model_type: Specific BERT model to use (e.g., 'microsoft/deberta-xlarge-mnli')
        
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if isinstance(references[0], list):
            # If references is a list of lists, take the first reference
            references = [refs[0] if isinstance(refs, list) else refs for refs in references]
        
        # Compute BERTScore
        P, R, F1 = bert_score(hypotheses, references, lang=lang, model_type=model_type, 
                              device=self.device, verbose=False)
        
        return {
            'BERTScore_precision': P.mean().item(),
            'BERTScore_recall': R.mean().item(),
            'BERTScore_f1': F1.mean().item(),
        }
    
    @torch.no_grad()
    def compute_sbert_similarity(self, 
                                 hypotheses: List[str], 
                                 references: List[str],
                                 batch_size: int = 32) -> Tuple[float, List[float]]:
        """
        Compute Sentence-BERT cosine similarity between generated and reference texts.
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings (single reference per hypothesis)
            batch_size: Batch size for processing
        
        Returns:
            Tuple of (average similarity, list of per-sample similarities)
        """
        self._init_sbert()
        
        if isinstance(references[0], list):
            # If references is a list of lists, take the first reference
            references = [refs[0] if isinstance(refs, list) else refs for refs in references]
        
        # Encode all texts
        hyp_embeddings = self.sbert_model.encode(hypotheses, batch_size=batch_size, 
                                                  convert_to_tensor=True, show_progress_bar=False)
        ref_embeddings = self.sbert_model.encode(references, batch_size=batch_size,
                                                  convert_to_tensor=True, show_progress_bar=False)
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(hyp_embeddings, ref_embeddings, dim=-1)
        similarities = similarities.cpu().numpy().tolist()
        
        avg_similarity = np.mean(similarities)
        return avg_similarity, similarities
    
    def compute_rouge_l(self,
                        hypotheses: List[str],
                        references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE-L score between generated and reference texts.
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings (single reference per hypothesis)
        
        Returns:
            Dictionary with ROUGE-L precision, recall, and F1 scores
        """
        if isinstance(references[0], list):
            # If references is a list of lists, take the first reference
            references = [refs[0] if isinstance(refs, list) else refs for refs in references]
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        precisions = []
        recalls = []
        fmeasures = []
        
        for hyp, ref in zip(hypotheses, references):
            scores = scorer.score(ref, hyp)
            precisions.append(scores['rougeL'].precision)
            recalls.append(scores['rougeL'].recall)
            fmeasures.append(scores['rougeL'].fmeasure)
        
        return {
            'ROUGE-L_precision': np.mean(precisions),
            'ROUGE-L_recall': np.mean(recalls),
            'ROUGE-L_f1': np.mean(fmeasures),
        }
    
    def compute_all_metrics(self, 
                           hypotheses: List[str], 
                           references: List[str],
                           sbert_batch_size: int = 32,
                           bert_score_model: str = None) -> Dict[str, float]:
        """
        Compute all metrics (BLEU-4, CIDEr, BERTScore, Sentence-BERT similarity, ROUGE-L).
        
        Args:
            hypotheses: List of generated text strings
            references: List of reference text strings
            sbert_batch_size: Batch size for Sentence-BERT processing
            bert_score_model: Specific model to use for BERTScore (optional)
        
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        # Ensure references is in the right format
        if not isinstance(references[0], list):
            references = [[ref] for ref in references]
        
        # Compute BLEU scores
        try:
            bleu_scores = self.compute_bleu(hypotheses, references)
            metrics.update(bleu_scores)
        except Exception as e:
            print(f"Warning: Failed to compute BLEU scores: {e}")
            metrics.update({'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0})
        
        # Compute CIDEr score
        try:
            cider_score = self.compute_cider(hypotheses, references)
            metrics['CIDEr'] = cider_score
        except Exception as e:
            print(f"Warning: Failed to compute CIDEr score: {e}")
            metrics['CIDEr'] = 0.0
        
        # Extract single reference for other metrics (take first if multiple)
        single_refs = [refs[0] if isinstance(refs, list) else refs for refs in references]
        
        # Compute BERTScore
        try:
            bert_scores = self.compute_bert_score(hypotheses, single_refs, model_type=bert_score_model)
            metrics.update(bert_scores)
        except Exception as e:
            print(f"Warning: Failed to compute BERTScore: {e}")
            metrics.update({'BERTScore_precision': 0.0, 'BERTScore_recall': 0.0, 'BERTScore_f1': 0.0})
        
        # Compute Sentence-BERT similarity
        try:
            sbert_sim, _ = self.compute_sbert_similarity(hypotheses, single_refs, sbert_batch_size)
            metrics['SBERT_similarity'] = sbert_sim
        except Exception as e:
            print(f"Warning: Failed to compute Sentence-BERT similarity: {e}")
            metrics['SBERT_similarity'] = 0.0
        
        # Compute ROUGE-L score
        try:
            rouge_scores = self.compute_rouge_l(hypotheses, single_refs)
            metrics.update(rouge_scores)
        except Exception as e:
            print(f"Warning: Failed to compute ROUGE-L: {e}")
            metrics.update({'ROUGE-L_precision': 0.0, 'ROUGE-L_recall': 0.0, 'ROUGE-L_f1': 0.0})
        
        return metrics
