import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from difflib import SequenceMatcher
from matplotlib import gridspec, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# For package data access
try:
    from importlib.resources import files
    _HAS_IMPORTLIB_RESOURCES = True
except ImportError:
    try:
        from importlib_resources import files
        _HAS_IMPORTLIB_RESOURCES = True
    except ImportError:
        import pkg_resources
        _HAS_IMPORTLIB_RESOURCES = False

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility. Call explicitly when needed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SemanticMatcher:
    """
    Handles sequence matching operations for entity attributes and embeddings.
    """
    
    def __init__(self, ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on model output using attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, model, tokenizer, texts, model_name=None):
        """Get embeddings for a list of texts using the specified model."""

        model = model.to(self.device)
        
        encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask']) if model_name == 'FremyCompany/BioLORD-2023' \
                    else model_output.last_hidden_state[:, 0, :]
        
        return F.normalize(embeddings, p=2, dim=1)

    def matrix_process_sequence_pair(self, gt_text, target_text,
                        model1, tokenizer1, model_name=None):
        """Process and compare two sequences of text using embeddings."""
        gt_flat = [item for item in gt_text]
        target_flat = [item for item in target_text]
        
        gt_embeddings = self.get_embeddings(model1, tokenizer1, gt_flat, model_name)
        target_embeddings = self.get_embeddings(model1, tokenizer1, target_flat, model_name)
        
        similarities = torch.mm(gt_embeddings, target_embeddings.t())
        return similarities
       
class DataConverter:
    """Handles conversion of dataframes to structured dictionaries."""
    
    def __init__(self, STRUCTURE_WEIGHTS, mode='rexval'):
        self.rel_weights = STRUCTURE_WEIGHTS[1]
        self.temporal_weights = STRUCTURE_WEIGHTS[0]
        self.mode = mode
    
    def convert_to_structured_dict(self, df_data):
        """Convert dataframe to structured dictionary format by study_id"""
        structured_data = {}
        if 'sequential' in self.mode:
            for subject_id in df_data['subject_id'].unique():
                subject_data = df_data[df_data['subject_id'] == subject_id].sort_values(by='sequence')
                subject_dict = {}
                for index, row in subject_data.iterrows():

                    # Skip if entity is NaN
                    entity_name = 'entity' if 'entity' in row else 'ent'
                    if pd.isna(row[entity_name]):
                        continue
                        
                    entity = row[entity_name]
                    study_id = row['study_id']
                    if study_id not in subject_dict:
                        subject_dict[study_id] = {}
                    if  entity not in subject_dict[study_id]:
                        subject_dict[study_id][entity] = []
                    
                    entry = {}
                    for key in self.rel_weights.keys():
                        if key in row and pd.notna(row[key]):
                            entry[key] = row[key]
                            
                    for key in self.temporal_weights.keys():
                        if key in row and pd.notna(row[key]):
                            entry[key] = row[key]
                    
                    # Add episode info
                    if 'episode' not in row:
                        entry['episode'] = 1  # Default episode value
                        
                    if 'entity_group' in row:
                        entry['entity_group'] = row['entity_group']
                    
                    if 'sequence' not in row:
                        entry['sequence'] = row['study_id']
                        
                    # Only append if entry contains meaningful data
                    subject_dict[study_id][entity].append(entry)

                # Only add study_dict if it contains data
                if subject_dict:
                    if type(subject_id) == str:
                        structured_data[str(subject_id)] = subject_dict
                    else:
                        structured_data[int(subject_id)] = subject_dict

        else:
            # Group by study_id
            for study_id in df_data['study_id'].unique():
                study_data = df_data[df_data['study_id'] == study_id]
                study_dict = {}
                
                for index, row in study_data.iterrows():
                    # Skip if entity is NaN
                    entity_name = 'entity' if 'entity' in row else 'ent'
                    if pd.isna(row[entity_name]):
                        continue
                        
                    entity = row[entity_name]
                    if entity not in study_dict:
                        study_dict[entity] = []
                    
                    entry = {}
                    for key in self.rel_weights.keys():
                        if key in row and pd.notna(row[key]):
                            entry[key] = row[key]
                    
                    for key in self.temporal_weights.keys():
                        if key in row and pd.notna(row[key]):
                            entry[key] = row[key]

                    entry['entity_group'] = 'single-eval'
                    entry['episode'] = 1  # Default episode value                    
                    entry['sequence'] = row['study_id']
                        
                    # Only append if entry contains meaningful data
                    if len(entry) > 1:  # More than just episode
                        study_dict[entity].append(entry)
                
                # Only add study_dict if it contains data
                if study_dict:
                    if type(study_id) == str:
                        structured_data[str(study_id)] = study_dict
                    else:
                        structured_data[int(study_id)] = study_dict

                    
        return structured_data

class SubjectMatcher:
    """Handles matching between subjects."""
    
    def __init__(self, mode):
        self.mode = mode

    def aggregate_entity_phrases(self, entity_dict):
        aggregate_keys = ['location', 'morphology', 'distribution', 'measurement', 'severity', 
                          'onset', 'improved', 'worsened', 'no change', 'placement']
        aggregated_phrases = []
        aggregated_row_info = []
        aggregated_study_ids = []
        aggregated_group = []
        aggregated_episode = []
        aggregated_dx_status = []
        if self.mode.startswith('sequential'):
            for study_id, entries in entity_dict.items():           
                phrases = []
                row_info = []
                for entity, instances in entries.items():
                    # Handle case where instances is a list of dictionaries
                    for instance in instances:
                        values = [instance.get(key, '') for key in aggregate_keys 
                                if key in instance and instance[key]]
                        phrase = ' '.join([entity] + values)
                        phrases.append(phrase.strip())
                        row_info.append(instance)
                        aggregated_study_ids.append(study_id)
                        aggregated_group.append(instance.get('entity_group', 'unknown'))
                        aggregated_episode.append(instance.get('episode', 'unknown'))
                        aggregated_dx_status.append(instance.get('dx_status', 'unknown'))
                aggregated_phrases.extend(phrases)
                aggregated_row_info.extend(row_info)
        
        else:
            for entity, entries in entity_dict.items():
                phrases = []
                row_info = []
                for instance in entries:                    
                    values = [instance[key] for key in aggregate_keys if key in instance and instance[key]]
                    phrase = ' '.join([entity] + values)
                    phrases.append(phrase.strip())
                    row_info.append(instance)
                    aggregated_group.append(instance.get('entity_group', 'unknown'))
                    aggregated_episode.append(instance.get('episode', 'unknown'))    
                    aggregated_dx_status.append(instance.get('dx_status', 'unknown'))
                aggregated_phrases.extend(phrases)
                aggregated_row_info.extend(row_info)
        
        return aggregated_phrases, aggregated_row_info, aggregated_study_ids, aggregated_group, aggregated_episode, aggregated_dx_status

    def compute_semantic_row_similarity(self, gt_row, pred_row, weights, matcher, model, tokenizer):
        """
        Compute semantic similarity between two rows based on their attributes.
        
        Args:
            gt_row: Ground truth row dictionary
            pred_row: Prediction row dictionary
            weights: Dictionary of weights
            matcher: SemanticMatcher instance
            model: Language model for semantic matching
            tokenizer: Tokenizer for the language model
            
        Returns:
            float: Normalized similarity score between the rows
        """
        score = 0.0
        norm = 0.0
        
        # Define binary/categorical fields that should use exact matching
        binary_fields = ['episode', 'dx_status', 'dx_certainty', 'sequence']
        
        if not self.mode.startswith('sequential'):
            binary_fields += ['entity_group']
        
        # Collect text fields for batch processing
        text_fields = []
        text_weights = []
        gt_texts = []
        pred_texts = []
        
        for rel, weight in weights.items():            
            gt_val = gt_row.get(rel)
            pred_val = pred_row.get(rel)
                        
            # Skip if both values are None/empty
            if (gt_val is None or str(gt_val).strip() == '') and (pred_val is None or str(pred_val).strip() == ''):
                continue
            
            # If either value exists, count this relation in normalization
            norm += weight
            
            # Handle binary/categorical fields with exact matching
            if rel in binary_fields:
                if gt_val is not None and pred_val is not None:
                    # For dx_status: positive/negative
                    if rel == 'dx_status':
                        gt_norm = str(gt_val).lower().strip()
                        pred_norm = str(pred_val).lower().strip()
                        
                        # Check for exact match in normalized values
                        if gt_norm == pred_norm:
                            score += weight
                    
                    # For dx_certainty: definitive/tentative
                    elif rel == 'dx_certainty':
                        gt_norm = str(gt_val).lower().strip()
                        pred_norm = str(pred_val).lower().strip()
                        
                        # Check for exact match in normalized values
                        if gt_norm == pred_norm:
                            score += weight
                    
                    # For episode and sequence: numeric comparison
                    elif rel in ['episode', 'sequence']:
                        try:
                            gt_num = int(gt_val)
                            pred_num = int(pred_val)
                            if gt_num == pred_num:
                                score += weight
                        except (ValueError, TypeError):
                            # If conversion fails, try string comparison
                            if str(gt_val) == str(pred_val):
                                score += weight

                    if not self.mode.startswith('sequential'):
                        if rel == 'entity_group':
                            if gt_val is not None and pred_val is not None:
                                if gt_val == pred_val:
                                    score += weight
                continue  # Skip semantic comparison for binary fields
            
            # For text fields, collect for batch processing
            if gt_val is not None and pred_val is not None and str(gt_val).strip() != '' and str(pred_val).strip() != '':
                text_fields.append(rel)
                text_weights.append(weight)
                gt_texts.append(str(gt_val))
                pred_texts.append(str(pred_val))
        
        
        # Batch process text fields if any exist
        if text_fields:
            try:
                # Process all text fields in a single batch
                batch_sim_matrix = matcher.matrix_process_sequence_pair(
                    gt_texts, pred_texts, model, tokenizer)
                
                # Extract diagonal elements (matching pairs)
                for i in range(min(len(gt_texts), len(pred_texts))):
                    if i < batch_sim_matrix.shape[0] and i < batch_sim_matrix.shape[1]:
                        sim_value = batch_sim_matrix[i, i].item()
                        score += text_weights[i] * sim_value
            except Exception as e:
                print(f"Error in batch similarity computation: {e}")
                # Fall back to individual processing
                for i, (rel, weight) in enumerate(zip(text_fields, text_weights)):
                    if i < len(gt_texts) and i < len(pred_texts):
                        try:
                            # Process individual text pair
                            column_sim = matcher.matrix_process_sequence_pair(
                                [gt_texts[i]], [pred_texts[i]], model, tokenizer)
                            
                            if torch.is_tensor(column_sim) and column_sim.numel() > 0:
                                sim_value = column_sim[0, 0].item()
                                score += weight * sim_value
                        except Exception as e:
                            print(f"Error computing similarity for {rel}: {e}")
                            # If semantic comparison fails, fall back to exact matching
                            if gt_texts[i] == pred_texts[i]:
                                score += weight
        
        # Normalize the score
        final_score = score / norm if norm > 0 else 0.0
        return final_score
            
    def semantic_match(self, gt_data, pred_data, rel_weights=None, temporal_weights=None,
                       save_plots=False, plot_dir="matrix_images"):
        gt_phrases, gt_row_info, gt_study_ids, gt_group, gt_episode, gt_dx_status = self.aggregate_entity_phrases(gt_data)
        pred_phrases, pred_row_info, pred_study_ids, pred_group, pred_episode, pred_dx_status = self.aggregate_entity_phrases(pred_data)

        if len(gt_study_ids) == 0 or len(pred_study_ids) == 0:
            gt_study_ids = [0]*len(gt_phrases)
            pred_study_ids = [0]*len(pred_phrases)

        # 2. similarity matrix
        matcher = SemanticMatcher()
        models = [
            'FremyCompany/BioLORD-2023',
            'ncbi/MedCPT-Query-Encoder',
            # 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            # 'emilyalsentzer/Bio_ClinicalBERT',
            # 'medicalai/ClinicalBERT',
            # 'dmis-lab/biobert-base-cased-v1.2'
        ]
        combined_sim_matrix = None
        combined_row_sim_matrix = None

        for model_idx, model_name in enumerate(models):
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
             # Calculate group similarity matrix
            if self.mode.startswith('sequential'):
                sim_matrix = matcher.matrix_process_sequence_pair(gt_group, pred_group, model, tokenizer, model_name)
            else:
                sim_matrix = matcher.matrix_process_sequence_pair(gt_phrases, pred_phrases, model, tokenizer, model_name)
                
            # Calculate row-level similarity scores
            row_similarity_scores = torch.zeros((len(gt_row_info), len(pred_row_info)))
            if torch.is_tensor(sim_matrix):
                row_similarity_scores = row_similarity_scores.to(sim_matrix.device)
            
            temporal_similarity_scores = torch.zeros((len(gt_row_info), len(pred_row_info)))
            if torch.is_tensor(sim_matrix):
                temporal_similarity_scores = temporal_similarity_scores.to(sim_matrix.device)

            for i, gt_row in enumerate(gt_row_info):
                for j, pred_row in enumerate(pred_row_info):
                    # Compute row similarity
                    similarity = self.compute_semantic_row_similarity(
                        gt_row, pred_row, rel_weights, matcher, model, tokenizer)
                    row_similarity_scores[i, j] = similarity

                    temporal_similarity = self.compute_semantic_row_similarity(
                        gt_row, pred_row, temporal_weights, matcher, model, tokenizer)
                    temporal_similarity_scores[i, j] = temporal_similarity

            # Initialize or add to combined matrices
            if combined_sim_matrix is None:
                # First model - initialize the combined matrices
                combined_sim_matrix = sim_matrix.clone()
                combined_temporal_sim_matrix = temporal_similarity_scores.clone()
                combined_row_sim_matrix = row_similarity_scores.clone()

            else:                
                # Add current model's matrices to combined matrices
                combined_sim_matrix += sim_matrix
                combined_temporal_sim_matrix += temporal_similarity_scores
                combined_row_sim_matrix += row_similarity_scores
        
        # Average the combined matrices
        num_models = len(models)
        combined_sim_matrix /= num_models
        combined_temporal_sim_matrix /= num_models
        combined_row_sim_matrix /= num_models

        # Calculate final similarity matrix
        final_sim_matrix = combined_sim_matrix * combined_row_sim_matrix * combined_temporal_sim_matrix
        
        if self.mode == 'sequential_mask_maira':
            mask_matrix = torch.zeros((len(gt_study_ids), len(pred_study_ids)), device=final_sim_matrix.device)
            for i in range(len(gt_study_ids)):
                for j in range(len(pred_study_ids)):
                    if gt_study_ids[i] == pred_study_ids[j]:
                        mask_matrix[i, j] = 1.0
                        
            final_sim_matrix = final_sim_matrix * mask_matrix
            
        # 3. Bipartite matching
        if torch.is_tensor(final_sim_matrix):
            try:
                cost_matrix = 1 - final_sim_matrix.detach().cpu().numpy()
            except (RuntimeError, ImportError):
                # Handle case where numpy conversion fails
                n, m = final_sim_matrix.shape
                cost_matrix = np.ones((n, m))
                for i in range(n):
                    for j in range(m):
                        cost_matrix[i, j] = 1 - final_sim_matrix[i, j].item()
        else:
            cost_matrix = 1 - final_sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        #########################################################
        unique_gt_ids = sorted(list(set(gt_study_ids)))
        unique_pred_ids = sorted(list(set(pred_study_ids)))
        gt_id_to_indices = {id: [] for id in unique_gt_ids}
        pred_id_to_indices = {id: [] for id in unique_pred_ids}

        for i, id in enumerate(gt_study_ids):
            gt_id_to_indices[id].append(i)
            
        for j, id in enumerate(pred_study_ids):
            pred_id_to_indices[id].append(j)

        gt_id_to_count = {id: len(indices) for id, indices in gt_id_to_indices.items()}
        pred_id_to_count = {id: len(indices) for id, indices in pred_id_to_indices.items()}

        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)

            # --- Plot 1: side-by-side similarity matrix + optimal matching ---
            try:
                final_np = final_sim_matrix.cpu().numpy()
            except (RuntimeError, ImportError):
                n, m = final_sim_matrix.shape
                final_np = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        final_np[i, j] = final_sim_matrix[i, j].item()

            plt.figure(figsize=(18, 9))
            gs = gridspec.GridSpec(1, 4, width_ratios=[10, 1, 10, 1])

            ax1 = plt.subplot(gs[0])
            im1 = ax1.imshow(final_np, cmap='viridis', aspect='auto')
            ax1.set_title('Final Similarity Matrix', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Predicted Reports', fontsize=14)
            ax1.set_ylabel('Ground Truth Reports', fontsize=14)

            y_tick_positions, y_tick_labels = [], []
            for id in unique_gt_ids:
                indices = gt_id_to_indices[id]
                if indices:
                    min_idx, max_idx = min(indices), max(indices)
                    if min_idx > 0:
                        ax1.axhline(y=min_idx - 0.5, color='#FF5555', linestyle='-', linewidth=1.5)
                    y_tick_positions.append((min_idx + max_idx) / 2)
                    y_tick_labels.append(f'{id} ({gt_id_to_count[id]})')

            x_tick_positions, x_tick_labels = [], []
            for id in unique_pred_ids:
                indices = pred_id_to_indices[id]
                if indices:
                    min_idx, max_idx = min(indices), max(indices)
                    if min_idx > 0:
                        ax1.axvline(x=min_idx - 0.5, color='#55AA55', linestyle='-', linewidth=1.5)
                    x_tick_positions.append((min_idx + max_idx) / 2)
                    x_tick_labels.append(f'{id} ({pred_id_to_count[id]})')

            ax1_top = ax1.twiny()
            ax1_right = ax1.twinx()
            ax1_top.set_xlim(ax1.get_xlim())
            ax1_right.set_ylim(ax1.get_ylim())
            ax1_top.set_xticks(x_tick_positions)
            ax1_top.set_xticklabels(x_tick_labels, rotation=45, ha='left', fontsize=9)
            ax1_right.set_yticks(y_tick_positions)
            ax1_right.set_yticklabels(y_tick_labels, fontsize=9)
            ax1.set_xticks([])
            ax1.set_yticks([])

            cax1 = plt.subplot(gs[1])
            plt.colorbar(im1, cax=cax1).set_label('Similarity Score', fontsize=12)

            ax2 = plt.subplot(gs[2])
            matching_matrix = np.zeros_like(final_np)
            for i, j in zip(row_ind, col_ind):
                if i < matching_matrix.shape[0] and j < matching_matrix.shape[1]:
                    matching_matrix[i, j] = 1

            colors = [(1, 1, 1), (1, 0.7, 0.7), (1, 0.5, 0.5), (1, 0.3, 0.3), (1, 0, 0)]
            cmap_match = LinearSegmentedColormap.from_list('custom_red', colors, N=100)
            im2 = ax2.imshow(matching_matrix, cmap=cmap_match, aspect='auto')
            ax2.set_title('Optimal Matching Result', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Predicted Reports', fontsize=14)
            ax2.set_ylabel('Ground Truth Reports', fontsize=14)

            for id in unique_gt_ids:
                indices = gt_id_to_indices[id]
                if indices and min(indices) > 0:
                    ax2.axhline(y=min(indices) - 0.5, color='#FF5555', linestyle='-', linewidth=1.5)
            for id in unique_pred_ids:
                indices = pred_id_to_indices[id]
                if indices and min(indices) > 0:
                    ax2.axvline(x=min(indices) - 0.5, color='#55AA55', linestyle='-', linewidth=1.5)

            ax2_top = ax2.twiny()
            ax2_right = ax2.twinx()
            ax2_top.set_xlim(ax2.get_xlim())
            ax2_right.set_ylim(ax2.get_ylim())
            ax2_top.set_xticks(x_tick_positions)
            ax2_top.set_xticklabels(x_tick_labels, rotation=45, ha='left', fontsize=9)
            ax2_right.set_yticks(y_tick_positions)
            ax2_right.set_yticklabels(y_tick_labels, fontsize=9)
            ax2.set_xticks([])
            ax2.set_yticks([])

            cax2 = plt.subplot(gs[3])
            plt.colorbar(im2, cax=cax2).set_label('Match (1) / No Match (0)', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'matching_result.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, 'matching_result.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

            # --- Plot 2: weighted matching with similarity scores ---
            weighted_matching = np.zeros_like(final_np)
            for i, j in zip(row_ind, col_ind):
                if i < weighted_matching.shape[0] and j < weighted_matching.shape[1]:
                    weighted_matching[i, j] = final_np[i, j]

            plt.figure(figsize=(10, 8))
            plt.imshow(weighted_matching, cmap='viridis', aspect='auto')
            plt.colorbar(label='Similarity Score of Matched Pairs')
            plt.title('Optimal Matching with Similarity Scores', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Reports', fontsize=14)
            plt.ylabel('Ground Truth Reports', fontsize=14)

            for id in unique_gt_ids:
                indices = gt_id_to_indices[id]
                if indices and min(indices) > 0:
                    plt.axhline(y=min(indices) - 0.5, color='#FF5555', linestyle='-', linewidth=1.5)
            for id in unique_pred_ids:
                indices = pred_id_to_indices[id]
                if indices and min(indices) > 0:
                    plt.axvline(x=min(indices) - 0.5, color='#55AA55', linestyle='-', linewidth=1.5)
            for i, j in zip(row_ind, col_ind):
                if i < weighted_matching.shape[0] and j < weighted_matching.shape[1]:
                    plt.plot(j, i, 'rx', markersize=5)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'weighted_matching.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, 'weighted_matching.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        #########################################################

        matched_pairs = []
        matched_scores = []
        matched_indices = set()
        
        # Store all matches and record similarity scores
        for i, j in zip(row_ind, col_ind):
            if i < len(gt_study_ids) and j < len(pred_study_ids):
                sim_score = final_sim_matrix[i, j].item() if torch.is_tensor(final_sim_matrix[i, j]) else final_sim_matrix[i, j]
                matched_pairs.append((gt_study_ids[i], pred_study_ids[j], gt_episode[i], pred_episode[j], gt_phrases[i], pred_phrases[j], sim_score))
                matched_scores.append(sim_score)
                matched_indices.add((i, j))

        # Identify unmatched items
        gt_matched_indices = {i for i, _ in matched_indices}
        pred_matched_indices = {j for _, j in matched_indices}
        
        gt_unmatched_indices = [i for i in range(len(gt_study_ids)) if i not in gt_matched_indices]
        pred_unmatched_indices = [j for j in range(len(pred_study_ids)) if j not in pred_matched_indices]
        
        # Collect info for unmatched GT items
        gt_unmatched = []
        gt_best_similarities = []
        for i in gt_unmatched_indices:
            if i < final_sim_matrix.shape[0] and final_sim_matrix.shape[1] > 0:
                # Find max similarity
                best_sim = torch.max(final_sim_matrix[i, :]).item() if torch.is_tensor(final_sim_matrix) else np.max(final_sim_matrix[i, :])
                gt_unmatched.append((gt_study_ids[i], gt_phrases[i]))
                gt_best_similarities.append(best_sim)
            else:
                gt_unmatched.append((gt_study_ids[i], gt_phrases[i]))
                gt_best_similarities.append(0.0)
        
        # Collect info for unmatched Pred items
        pred_unmatched = []
        pred_best_similarities = []
        for j in pred_unmatched_indices:
            if j < final_sim_matrix.shape[1] and final_sim_matrix.shape[0] > 0:
                # Find max similarity
                best_sim = torch.max(final_sim_matrix[:, j]).item() if torch.is_tensor(final_sim_matrix) else np.max(final_sim_matrix[:, j])
                pred_unmatched.append((pred_study_ids[j], pred_phrases[j]))
                pred_best_similarities.append(best_sim)
            else:
                pred_unmatched.append((pred_study_ids[j], pred_phrases[j]))
                pred_best_similarities.append(0.0)

        # F1 with similarity-weighted matching
        # Each match is proportional to similarity to TP
        # Unmatched items are proportional to (1-similarity) to FP/FN
        
        # 1. TP from matched results
        TP = sum(matched_scores)
        
        # 2. FP/FN from matched results (1-sim)
        # Imperfect matching should be reflect to FN and FP.
        FP_from_matches = sum(1 - score for score in matched_scores)
        FN_from_matches = sum(1 - score for score in matched_scores)
        
        # 3. Real unmatcned FP/FN
        # Max sim of FP/FN
        FP_from_unmatched = len(pred_unmatched) - sum(pred_best_similarities)
        FN_from_unmatched = len(gt_unmatched) - sum(gt_best_similarities)
        
        if self.mode == 'sequential_mask_maira':
            # Create a mapping of study IDs to track which ones are present
            gt_study_id_set = set()
            pred_study_id_set = set()
            
            # Extract study IDs from matched pairs
            for gt_id, pred_id, *_ in matched_pairs:
                gt_study_id_set.add(gt_id)
                pred_study_id_set.add(pred_id)
                
            # Extract study IDs from unmatched items
            for gt_id, _ in gt_unmatched:
                gt_study_id_set.add(gt_id)
            
            for pred_id, _ in pred_unmatched:
                pred_study_id_set.add(pred_id)
            
            # Filter unmatched items based on study ID presence
            filtered_gt_unmatched = []
            filtered_gt_best_similarities = []
            
            for i, (gt_id, phrase) in enumerate(gt_unmatched):
                # Only count as FN if the study ID exists in predictions
                if gt_id in pred_study_id_set:
                    filtered_gt_unmatched.append((gt_id, phrase))
                    if i < len(gt_best_similarities):
                        filtered_gt_best_similarities.append(gt_best_similarities[i])
            
            filtered_pred_unmatched = []
            filtered_pred_best_similarities = []
            
            for i, (pred_id, phrase) in enumerate(pred_unmatched):
                # Only count as FP if the study ID exists in ground truth
                if pred_id in gt_study_id_set:
                    filtered_pred_unmatched.append((pred_id, phrase))
                    if i < len(pred_best_similarities):
                        filtered_pred_best_similarities.append(pred_best_similarities[i])
            
            # Recalculate FP/FN with filtered unmatched items
            FP_from_unmatched = len(filtered_pred_unmatched) - sum(filtered_pred_best_similarities)
            FN_from_unmatched = len(filtered_gt_unmatched) - sum(filtered_gt_best_similarities)
            
        
        FP = FP_from_matches + FP_from_unmatched
        FN = FN_from_matches + FN_from_unmatched
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        

        return matched_pairs, f1_score, precision, recall, gt_unmatched, pred_unmatched
        
class TrendAnalyzer:
    """Analyzes trends in sequential data."""
    
    def __init__(self, severity_scale=None):
        self.severity_scale = severity_scale or {}
    
    def extract_trend(self, sequence_rows, target_rel):
        """Extract trend values for a specific relation from sequence rows."""
        trend = []
        last_val = None
        for row in sequence_rows:
            if row.get('no changing', False) and last_val is not None:
                trend.append(last_val)
                continue
            val = row.get(target_rel)
            if target_rel == 'severity':
                score = self.severity_scale.get(val, val)
                if score:
                    trend.append(score)
                    last_val = score
            else:
                # Extensible for other relations
                trend.append(val)
                last_val = val
        return trend
    
    def detect_direction(self, trend_scores):
        """Detect the direction of a trend (increasing, decreasing, stable, or mixed)."""
        if not trend_scores or len(trend_scores) < 2:
            return 'stable'
            
        diffs = [b - a for a, b in zip(trend_scores, trend_scores[1:])]

        if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
            return 'increasing'
        elif all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
            return 'decreasing'
        elif all(d == 0 for d in diffs):
            return 'stable'
        else:
            return 'mixed'
    
    def compare_trends_weighted(self, gt_rows, pred_rows, target_rel, rel_weights):
        """Compare trends between ground truth and predicted rows with weighting."""
        if len(gt_rows) < 2 or len(pred_rows) < 2:
            return 1.0  # Skip temporal evaluation for single-item sequences

        gt_trend = self.extract_trend(gt_rows, target_rel)
        pred_trend = self.extract_trend(pred_rows, target_rel)

        gt_dir = self.detect_direction(gt_trend)
        pred_dir = self.detect_direction(pred_trend)

        weight = rel_weights.get(target_rel, 0)

        if gt_dir == pred_dir:
            return 1.0
        elif 'stable' in [gt_dir, pred_dir]:
            return 1.0 - weight * 0.5   # Partial penalty
        else:
            return 1.0 - weight         # Critical error (opposite direction)

class MetricEvaluator:
    """Main class for evaluating metrics between ground truth and predictions."""

    def __init__(self, weights, mode='rexval', subject_matching_mode="semantic",
                 save_plots=False, plot_dir="matrix_images"):
        self.temporal_weights = weights[0]
        self.rel_weights = weights[1]
        self.mode = mode
        self.data_converter = DataConverter(weights, mode)
        self.subject_matcher = SubjectMatcher(mode)
        self.trend_analyzer = TrendAnalyzer()
        self.subject_matching_mode = subject_matching_mode
        self.save_plots = save_plots
        self.plot_dir = plot_dir

    def evaluate_subject(self, gt_data, pred_data, mode='semantic',
                         temporal_eval=True):
        """Evaluate subjects between ground truth and predictions."""

        matched_pairs, f1_score, prec, recall, gt_unmatched, pred_unmatched = self.subject_matcher.semantic_match(
            gt_data, pred_data, self.rel_weights, self.temporal_weights,
            save_plots=self.save_plots, plot_dir=self.plot_dir)

        return matched_pairs, f1_score, prec, recall, gt_unmatched, pred_unmatched

    def evaluate_dataset(self, gt_data_dict, pred_data_dict):
        """Evaluate an entire dataset of subjects."""
        scores = {}
        prec_scores = {}
        recall_scores = {}
        for study_id in gt_data_dict:
            if study_id in pred_data_dict:
                matched_pairs, f1_score, prec, recall, gt_unmatched, pred_unmatched = self.evaluate_subject(gt_data_dict[study_id], pred_data_dict[study_id], mode=self.mode, temporal_eval=False)
                scores[study_id] = f1_score
                prec_scores[study_id] = prec
                recall_scores[study_id] = recall
            else:
                print(f"Study ID {study_id}: Missing in predictions")
        return scores, prec_scores, recall_scores


def _attach_sequence_from_gt(df_pred: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate missing sequence column in predictions by borrowing from GT rows
    that share the same subject_id (and study_id when available).
    """
    if 'sequence' in df_pred.columns:
        return df_pred

    required_pred_cols = {'subject_id'}
    missing_pred = required_pred_cols - set(df_pred.columns)
    if missing_pred:
        raise KeyError(f"Prediction dataframe missing columns for sequence recovery: {missing_pred}")

    required_gt_cols = {'subject_id', 'sequence'}
    merge_keys = ['subject_id']
    use_study = 'study_id' in df_pred.columns and 'study_id' in gt_df.columns
    if use_study:
        required_gt_cols.add('study_id')
        merge_keys.append('study_id')
    missing_gt = required_gt_cols - set(gt_df.columns)
    if missing_gt:
        raise KeyError(f"GT dataframe missing columns for sequence recovery: {missing_gt}")

    df_pred = df_pred.copy()
    df_pred['subject_id'] = df_pred['subject_id'].astype(str)
    if use_study:
        df_pred['study_id'] = df_pred['study_id'].astype(str)

    lookup_df = (
        gt_df[list(required_gt_cols)]
        .dropna(subset=['sequence'])
        .assign(subject_id=lambda d: d['subject_id'].astype(str))
    )
    if use_study:
        lookup_df = lookup_df.assign(study_id=lambda d: d['study_id'].astype(str))
    else:
        dup_subjects = lookup_df['subject_id'].duplicated(keep=False)
        if dup_subjects.any():
            duplicated = lookup_df.loc[dup_subjects, 'subject_id'].unique().tolist()
            raise ValueError(
                "Cannot recover sequence using only subject_id because multiple "
                f"sequence values exist. Duplicated subject_ids: {duplicated[:5]}"
            )

    lookup_df = lookup_df.drop_duplicates(subset=merge_keys)
    lookup_df = lookup_df.rename(columns={'sequence': '_sequence_from_gt'})

    df_pred = df_pred.merge(lookup_df, on=merge_keys, how='left')

    if df_pred['_sequence_from_gt'].isna().any():
        unresolved = df_pred[df_pred['_sequence_from_gt'].isna()][merge_keys].drop_duplicates()
        raise ValueError(
            "Unable to recover sequence for some predictions. "
            f"Missing keys: {unresolved.to_dict(orient='records')[:5]}"
        )

    df_pred['sequence'] = df_pred['_sequence_from_gt']
    df_pred = df_pred.drop(columns=['_sequence_from_gt'])
    return df_pred


def _resolve_file_path(file_path, file_description="file", check_multiple_locations=True):
    """
    Resolve a file path, checking multiple possible locations if needed.
    """
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path

    if os.path.exists(file_path):
        return os.path.abspath(file_path)

    if check_multiple_locations:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(package_dir)
        project_root = os.path.dirname(src_dir)

        potential_bases = []
        cwd = os.getcwd()
        for d in [cwd, src_dir, project_root]:
            if d not in potential_bases:
                potential_bases.append(d)

        for base in potential_bases:
            potential_path = os.path.join(base, file_path)
            if os.path.exists(potential_path):
                return os.path.abspath(potential_path)

        if 'dataset' in file_path or 'Lunguage' in file_path:
            filename = os.path.basename(file_path)
            for base in [project_root, cwd]:
                alt_path = os.path.join(base, 'src', 'dataset', filename)
                if os.path.exists(alt_path):
                    return os.path.abspath(alt_path)
                alt_path2 = os.path.join(base, 'dataset', filename)
                if os.path.exists(alt_path2):
                    return os.path.abspath(alt_path2)

        error_msg = f"{file_description} not found: '{file_path}'\n"
        error_msg += f"Current working directory: {os.getcwd()}\n"
        error_msg += f"Please provide an absolute path.\n"
        raise FileNotFoundError(error_msg)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{file_description} not found: '{file_path}'\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Please provide the correct path."
            )
        return os.path.abspath(file_path)


def _load_metric_weights():
    """Load metric weights from package data."""
    try:
        if _HAS_IMPORTLIB_RESOURCES:
            weights_file = files("lunguage_score") / "metric_weights.yaml"
            with weights_file.open('r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            weights_path = pkg_resources.resource_filename("lunguage_score", "metric_weights.yaml")
            with open(weights_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
    except (FileNotFoundError, ModuleNotFoundError):
        weights_path = os.path.join(os.path.dirname(__file__), "metric_weights.yaml")
        if os.path.exists(weights_path):
            with open(weights_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        else:
            raise FileNotFoundError(
                "metric_weights.yaml not found. Please ensure the package is properly installed."
            )


def calculate_lunguagescore(mode: str, model_name: str | None, gt_path: str, benchmark_sr_dir: str, rexval_path: str | None = None) -> pd.DataFrame:
    """
    Compute LunguageScore (our F1) using structured candidate reports vs gold structured reports.

    - mode: "single" or "sequential"
    - model_name: one of the structured benchmark identifiers
    - gt_path: path to gold structured CSV (dataset/Lunguage.csv)
    - benchmark_sr_dir: directory containing <model_name>_SR.csv

    Returns a DataFrame with columns:
      single:  study_id, f1, precision, recall
      sequential: subject_id, f1, precision, recall
    """
    metric_weights = _load_metric_weights()
        
    temporal_weight = sum(metric_weights['TEMPORAL_WEIGHTS'].values())
    temporal_weights = {k: v / temporal_weight for k, v in metric_weights['TEMPORAL_WEIGHTS'].items()}
    rel_weight = sum(metric_weights['REL_WEIGHTS'].values())
    rel_weights = {k: v / rel_weight for k, v in metric_weights['REL_WEIGHTS'].items()}
    structure_weights = [temporal_weights, rel_weights]

    evaluator = MetricEvaluator(structure_weights, mode, subject_matching_mode="semantic")

    # ReXVal mode (handled here to avoid a separate function)
    if mode == 'rexval':
        if rexval_path is None:
            raise ValueError("rexval_path is required when mode='rexval'")
        rexval_path = _resolve_file_path(rexval_path, "ReXVal CSV file")
        rexval_struct = pd.read_csv(rexval_path)
        gt_report_data = rexval_struct[rexval_struct['report_type'] == 'gt_report']
        radgraph_data = rexval_struct[rexval_struct['report_type'] == 'radgraph']
        bertscore_data = rexval_struct[rexval_struct['report_type'] == 'bertscore']
        s_emb_data = rexval_struct[rexval_struct['report_type'] == 's_emb']
        bleu_data = rexval_struct[rexval_struct['report_type'] == 'bleu']

        data = {
            'gt': evaluator.data_converter.convert_to_structured_dict(gt_report_data),
            'radgraph': evaluator.data_converter.convert_to_structured_dict(radgraph_data),
            'bertscore': evaluator.data_converter.convert_to_structured_dict(bertscore_data),
            's_emb': evaluator.data_converter.convert_to_structured_dict(s_emb_data),
            'bleu': evaluator.data_converter.convert_to_structured_dict(bleu_data)
        }

        rows = []
        for candidate_type in ["bertscore", "s_emb", "bleu", "radgraph"]:
            for id_value in data["gt"].keys():
                if id_value in data[candidate_type].keys():
                    gt_struct = data["gt"][id_value]
                    cand_struct = data[candidate_type][id_value]
                    _, f1_score, prec, rec, _, _ = evaluator.evaluate_subject(gt_struct, cand_struct)
                    rows.append({"study_id": id_value, "f1": f1_score, "precision": prec, "recall": rec, "candidate_type": candidate_type})
        return pd.DataFrame(rows)

    # Single / Sequential modes
    elif mode in ('single', 'sequential'):
        if not model_name:
            raise ValueError("model_name is required for modes 'single' and 'sequential'")

        # Resolve and load GT
        gt_path = _resolve_file_path(gt_path, "Ground truth CSV file (Lunguage.csv)")
        sequential_gt = pd.read_csv(gt_path)
        # Harmonize GT column names to evaluator expectations
        if 'gt_entity_group' in sequential_gt.columns:
            sequential_gt = sequential_gt[sequential_gt['gt_entity_group'].notna()]
            sequential_gt = sequential_gt.rename(columns={'gt_entity_group': 'entity_group', 'gt_temporal_group': 'temporal_group'})

        if mode == "single":
            # Exclude history section for single report setting
            if 'section' in sequential_gt.columns:
                sequential_gt = sequential_gt[sequential_gt['section'] != 'hist']

        # Resolve prediction file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(package_dir)
        potential_bases = [os.getcwd(), src_dir, os.path.dirname(src_dir)]

        pred_path = None
        if mode == "single":
            pred_file_name = f"{model_name}_SR.csv"
        else:
            pred_file_name = f"{model_name}_SR_seq.csv"

        # Try provided path first, then fallback to _SR.csv for sequential
        for fname in ([pred_file_name, f"{model_name}_SR.csv"] if mode == "sequential" else [pred_file_name]):
            if pred_path:
                break
            candidate = os.path.join(benchmark_sr_dir, fname)
            if os.path.isabs(benchmark_sr_dir):
                if os.path.exists(candidate):
                    pred_path = os.path.abspath(candidate)
                    break
            else:
                if os.path.exists(candidate):
                    pred_path = os.path.abspath(candidate)
                    break
                for base in potential_bases:
                    p = os.path.join(base, benchmark_sr_dir, fname)
                    if os.path.exists(p):
                        pred_path = os.path.abspath(p)
                        break
                    for common_dir in ['benchmark_SR', 'src/benchmark_SR']:
                        p2 = os.path.join(base, common_dir, fname)
                        if os.path.exists(p2):
                            pred_path = os.path.abspath(p2)
                            break
                    if pred_path:
                        break

        if pred_path is None:
            error_msg = f"Prediction CSV not found for model '{model_name}' in '{benchmark_sr_dir}'.\n"
            error_msg += f"Current working directory: {os.getcwd()}\n"
            error_msg += f"Please provide the correct --benchmark_sr_dir path.\n"
            raise FileNotFoundError(error_msg)

        df_pred = pd.read_csv(pred_path)
        # Keep rows with valid clusters and temporal grouping; align column names
        if 'LLM_cluster' in df_pred.columns:
            df_pred = df_pred[(df_pred['LLM_cluster'].notna()) & (df_pred['temporal_group'].notna())]
            df_pred = df_pred.rename(columns={'LLM_cluster': 'entity_group'})

        if mode == "sequential" and 'sequence' not in df_pred.columns:
            df_pred = _attach_sequence_from_gt(df_pred, sequential_gt)

        # Select matching GT subset
        study_ids = df_pred['study_id'].unique()
        df_gt = sequential_gt[sequential_gt['study_id'].isin(study_ids)]

        # Convert to structured dicts
        data = {
            'pred': evaluator.data_converter.convert_to_structured_dict(df_pred),
            'gt': evaluator.data_converter.convert_to_structured_dict(df_gt)
        }

        scores = []
        for id_value in data['gt'].keys():
            if id_value in data['pred'].keys():
                gt_struct = data['gt'][id_value]
                cand_struct = data['pred'][id_value]
                _, f1_score, prec, rec, _, _ = evaluator.evaluate_subject(gt_struct, cand_struct)
                scores.append({"id": id_value, "f1": f1_score, "precision": prec, "recall": rec})

        df_scores = pd.DataFrame(scores)
        if mode == "single":
            df_scores = df_scores.rename(columns={"id": "study_id"})
        else:
            df_scores = df_scores.rename(columns={"id": "subject_id"})

        return df_scores

    raise ValueError("Unsupported mode. Choose from 'single', 'sequential', 'rexval'.")


def main():
    parser = argparse.ArgumentParser(description="Compute only LunguageScore (single / sequential / rexval)")
    parser.add_argument('--mode', choices=['single', 'sequential', 'rexval'], required=True, help='Evaluation mode')
    parser.add_argument('--model_name', required=False,
                        choices=['maira_standard', 'maira_cascade', 'rgrg', 'cvt2distilgpt2', 'medversa', 'medgemma', 'lingshu', 'libra', 'chexagent'],
                        help='Which model structured outputs to evaluate (expects <name>_SR.csv)')
    parser.add_argument('--gt_path', default='dataset/Lunguage.csv', help='Path to gold structured CSV')
    parser.add_argument('--benchmark_sr_dir', default='benchmark_SR', help='Directory of structured candidate CSVs')
    parser.add_argument('--rexval_path', default='benchmark/rexval.csv', help='Path to structured ReXVal CSV (with report_type column)')
    parser.add_argument('--output', default=None, help='Optional output CSV path. If not provided, saves under scores/.')
    parser.add_argument('--print_head', action='store_true', help='Print head of result to stdout')
    args = parser.parse_args()

    df_scores = calculate_lunguagescore(args.mode, args.model_name, args.gt_path, args.benchmark_sr_dir, rexval_path=args.rexval_path)

    out_path = args.output
    if out_path is None:
        os.makedirs('scores', exist_ok=True)
        if args.mode == 'single':
            out_path = os.path.join('scores', f'single_report_lunguagescore_{args.model_name}.csv')
        elif args.mode == 'sequential':
            out_path = os.path.join('scores', f'seq_report_lunguagescore_{args.model_name}.csv')
        else:
            out_path = os.path.join('scores', 'rexval_lunguagescore.csv')

    df_scores.to_csv(out_path, index=False)
    if args.print_head:
        print(df_scores.head())
    print(f"Saved LunguageScore to {out_path}")


if __name__ == '__main__':
    main()