import pandas as pd
import os
import numpy as np
import json
import re

# Optional imports
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    print("Warning: 'evaluate' module not found. BLEU and BERTScore will be unavailable.")

try:
    from .FineRadScore.gpt4_generations import generate_gpt4_response
    FINERADSCORE_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    FINERADSCORE_AVAILABLE = False
    print(f"Warning: FineRadScore module unavailable: {e}")

try:
    from RaTEScore import RaTEScore
    RATESCORE_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    RATESCORE_AVAILABLE = False
    print(f"Warning: RaTEScore module unavailable: {e}")

try:
    from .green_score import GREEN
    GREEN_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    GREEN_AVAILABLE = False
    print(f"Warning: green_score module unavailable: {e}")

try:
    from radgraph import F1RadGraph
    RADGRAPH_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    RADGRAPH_AVAILABLE = False
    print(f"Warning: radgraph module unavailable: {e}")

def calculate_rate_score(paired_reports, path, use_gpu=True):
    """
    Calculate RaTEScore
    - paired_reports: DataFrame with the following columns:
        - study_id: study_id for which to calculate metric
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    - path: where to store structured NER results
    - use_gpu: whether to use GPU (default: True, auto-detected)
    """
    if not RATESCORE_AVAILABLE:
        raise ImportError("RaTEScore module is not available.")

    print("----- RUN RATESCORE CALCULATION -----")

    # Check GPU availability
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU: True (CUDA available, device: {torch.cuda.get_device_name(0)})")
                use_gpu = True
            else:
                print("Warning: GPU requested but CUDA unavailable. Falling back to CPU.")
                use_gpu = False
        except ImportError:
            print("Warning: torch not found. Running on CPU.")
            use_gpu = False
    else:
        print("GPU: False (CPU mode)")

    # calculate RATEScore
    pred_report = list(paired_reports["report_cand"])
    gt_report = list(paired_reports["report_ref"])

    print(f"Processing {len(pred_report)} report pairs... (GPU: {use_gpu})")
    print("RaTEScore calculation may take a while. Please wait...")

    ratescore = RaTEScore(visualization_path = path, use_gpu=use_gpu)
    scores = ratescore.compute_score(pred_report, gt_report)

    print(f"RaTEScore complete! Mean score: {sum(scores)/len(scores):.4f}")

    # add scores to dataframe
    paired_reports["RATEScore"] = scores

    return paired_reports

def calculate_green_score(paired_reports, use_gpu=True):
    """
    Calculate GREEN
    - paired_reports: DataFrame with the following columns:
        - study_id: study_id for which to calculate metric
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    - use_gpu: whether to use GPU (default: True)
    """
    if not GREEN_AVAILABLE:
        raise ImportError("GREEN Score module is not available.")

    print("----- RUN GREEN SCORE CALCULATION -----")

    # Check GPU availability
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU: True (CUDA available, device: {torch.cuda.get_device_name(0)})")
                use_cpu = False
            else:
                print("Warning: GPU requested but CUDA unavailable. Falling back to CPU.")
                use_cpu = True
        except ImportError:
            print("Warning: torch not found. Running on CPU.")
            use_cpu = True
    else:
        print("GPU: False (CPU mode)")
        use_cpu = True

    # calculate GREEN Score
    pred_report = list(paired_reports["report_cand"])
    gt_report = list(paired_reports["report_ref"])
    model_name = "StanfordAIMI/GREEN-radllama2-7b"

    print(f"Processing {len(pred_report)} report pairs... (CPU: {use_cpu})")
    print("GREEN Score calculation may take a while. Please wait...")

    green_scorer = GREEN(model_name, output_dir=".", cpu=use_cpu)
    mean, std, green_score_list, summary, result_df = green_scorer(gt_report, pred_report)
    result_df["study_id"] = list(paired_reports["study_id"])

    print(f"GREEN Score complete! Mean score: {mean:.4f}")

    # we can calculate mean, std and green_score_list from "green" column in result_df
    # we don't need the summary
    return result_df

def normalize_fineradscore_response(response):
    """
    Normalize GPT-4 response keys to the format expected by FineRadScore.
    Post-processes the response without modifying FineRadScore source code.

    Expected key formats:
    - digits: "0", "1", "2", ...
    - or exactly "None"

    Possible variant keys from GPT-4:
    - "None_2", "None1", "None-1", etc. → converted to "None"
    - "[delete] ..." → removed
    - "Generated Text" etc. (invalid keys) → removed
    """
    if not isinstance(response, dict):
        return response

    normalized_response = {}
    none_count = 0  # counter to deduplicate None variants

    for key, value in response.items():
        # Exact digit keys are kept as-is
        if key.isdigit():
            normalized_response[key] = value
        # Exact "None"
        elif key == "None":
            normalized_response["None"] = value
        # None variants (None_1, None-1, None1, None_2, etc.)
        elif re.match(r'^none[-_\d]', key.lower()):
            # Only first None variant maps to "None" (FineRadScore expects at most one)
            if none_count == 0:
                normalized_response["None"] = value
                none_count += 1
        # "[delete] ..." keys are deletion markers — skip
        elif key.strip().startswith("[delete]"):
            continue
        # Other invalid keys (Generated Text, etc.) — skip
        elif key.strip() in ["Generated Text", "Ground Truth Text", "Failed"]:
            continue
        else:
            # If key starts with digits, extract the numeric prefix
            numeric_match = re.search(r'^\d+', key)
            if numeric_match:
                numeric_key = numeric_match.group()
                if numeric_key not in normalized_response:  # avoid duplicates
                    normalized_response[numeric_key] = value

    return normalized_response


def normalize_report_for_fineradscore(report):
    """
    Convert a report to the format expected by FineRadScore ([0] ... [1] ...).
    Pre-processes input without modifying FineRadScore source code.
    """
    if not report or pd.isna(report):
        return report

    # Already in [0] [1] format — return as-is
    if re.search(r'\[\d+\]', report):
        return report

    sentences = []

    # Look for [Category: Subcategory] pattern (CheXagent format)
    category_pattern = r'\[([^\]]+)\]\s*([^[]+)'
    matches = re.findall(category_pattern, report)

    if matches:
        # Extract sentences from each category block
        for category, content in matches:
            content = content.strip()
            if content:
                content_sentences = re.split(r'\.\s+', content)
                for sent in content_sentences:
                    sent = sent.strip()
                    if sent:
                        if not sent.endswith('.'):
                            sent += '.'
                        sentences.append(sent)
    else:
        # Plain sentence format (Libra, MAIRA, Medversa, etc.)
        # Split by newline first
        lines = report.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                line_sentences = re.split(r'\.\s+', line)
                for sent in line_sentences:
                    sent = sent.strip()
                    if sent:
                        if not sent.endswith('.'):
                            sent += '.'
                        sentences.append(sent)

    # Assign sentence IDs: [0] ... [1] ... (format expected by FineRadScore)
    if sentences:
        numbered_sentences = []
        for i, sent in enumerate(sentences):
            numbered_sentences.append(f"[{i}] {sent}")
        return ' '.join(numbered_sentences)

    return report


def get_GPT4_response(row, max_retries):
    """
    Prompt GPT4 to generate FineRadScore output
    - row: Series for which to get FineRadScore response, has the following columns
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    - max_retries: maximum number of times to try reprompting when an error occurs

    Note: Only pre-processes input reports; FineRadScore source code is not modified.
    """
    pred_target = row["report_cand"]
    gt_target = row["report_ref"]

    # Convert to [0] [1] ... format expected by FineRadScore
    pred_target_original = pred_target
    gt_target_original = gt_target

    pred_target = normalize_report_for_fineradscore(pred_target)
    gt_target = normalize_report_for_fineradscore(gt_target)

    study_id = row.get("study_id", "unknown")
    if pred_target_original != pred_target or gt_target_original != gt_target:
        print(f"[Preprocessed] Study ID: {study_id} - format conversion complete")
        print("-" * 80)
    
    total_cost = 0
    done = False
    retry_count = 0

    while not done:
        done = True

        try:
            response, cost = generate_gpt4_response(pred_target, gt_target)
        except Exception as e:
            print("something's wrong!")
            print(e)
            done = False
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Skipping this report.")
                done = True
                response = {}
                break
            continue

        total_cost += cost
        
        # Normalize response keys without modifying FineRadScore source code
        response = normalize_fineradscore_response(response)

        # Validate response format
        if not isinstance(response, dict):
            done = False
            print(f"Error: response is not a dict, got {type(response)}")
            print(f"Response content (first 200 chars): {str(response)[:200]}")
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Skipping this report.")
                done = True
                response = {}
                break
            continue
        
        # "Failed" key indicates JSON parsing failure
        if "Failed" in response:
            done = False
            print(f"Error: GPT-4 response parsing failed. Retrying...")
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Using empty response.")
                done = True
                response = {}
                break
            continue
            
        for sentence_id in response:
            # bad generation: regenerate
            if not sentence_id.isdigit() and sentence_id != "None":
                done = False
                print(f"Error: key '{sentence_id}' is not a sentence id (expected digit or 'None')")
                print(f"Available keys: {list(response.keys())[:5]}")  # show first 5 keys only
                break

            # bad generation: regenerate
            corrected_line = response[sentence_id]
            if not isinstance(corrected_line, dict):
                done = False
                print(f"Error: value for key '{sentence_id}' is not a dict")
                break
                
            if "corrections" not in corrected_line or "clinical severity" not in corrected_line or "comments" not in corrected_line or "error category" not in corrected_line:
                done = False
                print(f"Error: json object not formatted correctly for key '{sentence_id}'")
                print(f"Available keys in entry: {list(corrected_line.keys())}")
                break

        # bad generation: regenerate
        if len(response) == 0: 
            done = False
            print("Error: empty response")
            
        retry_count += 1
        if retry_count > max_retries:
            print(f"Max retries ({max_retries}) reached. Using empty response.")
            done = True
            if not isinstance(response, dict):
                response = {}
            break

    result = {"pred": pred_target, "ground_truth": gt_target, "response": response}
    
    return result, total_cost


SEVERITY_SCORE = {"no error": 0, "invalid comparison": 0, 
                 "not actionable": 1, "actionable nonurgent error": 2, 
                 "urgent error": 3, "emergent error": 4}

def calculate_fineradscore(paired_reports, path):
    """
    Calculate FineRadScore
    - paired_reports: DataFrame with the following columns:
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    - path: where to store full response for each study
    """
    if not FINERADSCORE_AVAILABLE:
        raise ImportError("FineRadScore module is not available.")

    print("----- RUN FINERADSCORE CALCULATION -----")

    total_cost = 0
    i = 0
    
    with open(path, "a") as f:
        for idx, row in paired_reports.iterrows():

            # get FineRadScore analysis
            result, cost = get_GPT4_response(row, 5) 
            total_cost += cost

            # extract severity scores from response
            scores = []
            for sent_id, entry in result["response"].items(): 
                try: 
                    severity = entry["clinical severity"].lower()
                    score = SEVERITY_SCORE[severity]
                    scores.append(score)
                except:
                    scores.append(0)

            # store sum of severity scores (FineRadScore orig. paper) and max of severity scores (ReXrank benchmark)
            if len(scores) != 0:
                paired_reports.at[idx, "sum_score"] = sum(scores)
                paired_reports.at[idx, "max_score"] = max(scores)
                result["sum_score"] = sum(scores)
                result["max_score"] = max(scores)
            else: 
                paired_reports.at[idx, "sum_score"] = float("nan")
                paired_reports.at[idx, "max_score"] = float("nan")
                result["sum_score"] = float("nan")
                result["max_score"] = float("nan")
                print("score is nan!")

            # store response for later use
            f.write(json.dumps(result) + "\n") 
            f.flush()

            i += 1
            if i % 10 == 0:
                print(f"total cost after {i} reports: {total_cost}")

    print(f"Total cost after FineRadScore calculation: {total_cost}")

    return paired_reports, total_cost

def calculate_bleu(paired_reports):
    """
    Calculate BLEU (use hugginface/evaluate library)
    - paired_reports: DataFrame with the following columns:
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    """
    if not EVALUATE_AVAILABLE:
        raise ImportError("'evaluate' module is not available. Cannot compute BLEU.")

    print("----- RUN BLEU SCORE CALCULATION -----")

    result_df = paired_reports.copy()
    result_df["bleu"] = 0
    bleu = evaluate.load("bleu")
    for i, row in paired_reports.iterrows():
        if not isinstance(row["report_cand"], float) and not isinstance(row["report_ref"], float):
            results = bleu.compute(predictions=[row["report_cand"]], references=[[row["report_ref"]]])
            result_df.at[i, "bleu"] = results["bleu"]
        else: 
            result_df.at[i, "bleu"] = np.nan      
    return result_df

def calculate_bertscore(paired_reports): 
    """
    Calculate BERTScore (use hugginface/evaluate library)
    - paired_reports: DataFrame with the following columns:
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    """
    if not EVALUATE_AVAILABLE:
        raise ImportError("'evaluate' module is not available. Cannot compute BERTScore.")

    print("----- RUN BERTSCORE CALCULATION -----")

    result_df = paired_reports.copy()
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=paired_reports["report_cand"].to_list(), references=paired_reports["report_ref"].to_list(), model_type="distilroberta-base")
    result_df["bertscore"] = results["f1"]
    return result_df     
        
def calculate_radgraphF1(paired_reports):
    """
    Calculate RadGraph F1
    - paired_reports: DataFrame with the following columns:
        - report_ref: reference ground truth report
        - report_cand: candidate predicted report to compare with reference
    - model_ver: Version of radgraph ("radgraph" or "radgraph-xl")
    """
    if not RADGRAPH_AVAILABLE:
        raise ImportError("radgraph module is not available.")

    print("----- RUN RADGRAPH CALCULATION -----")

    result_df = paired_reports.copy()
    hyps = list(paired_reports["report_cand"])
    refs = list(paired_reports["report_ref"])

    # run old radgraph
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph")
    _, reward_list, _, _ = f1radgraph(hyps=hyps, refs=refs)
    result_df["RadgraphF1"] = reward_list[1] # second value is RG_ER

    # run new radgraph-xl
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    _, reward_list, _, _ = f1radgraph(hyps=hyps, refs=refs)
    result_df["RadgraphF1-xl"] = reward_list[1] # second value is RG_ER

    return result_df