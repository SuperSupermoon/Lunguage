import pandas as pd
import json
import os
import argparse
import datetime
import time
import glob
import os
from .llm_utils import *
import torch



def load_and_process_data(input_df, args=None):
    if 'ent' not in input_df.columns and 'entity' in input_df.columns:
        input_df['ent'] = input_df['entity']

    input_df.loc[:, 'location'] = input_df['location'].str.replace(r'loc:\s*', '', regex=True).str.replace(r'det:\s*', '', regex=True)
    
    if 'ELA_cur_ent' not in input_df.columns:
        # Define the columns to concatenate
        RELATIONS = [
            'location', 'morphology', 'distribution', 'measurement', 'severity', 
            'comparison', 'onset', 'no change', 'improved', 'worsened', 
            'placement', 'past hx', 'other source', 'assessment limitations'
        ]
        
        if 'ent' in input_df.columns:        
            # Create a new column that concatenates the entity with all attribute values
            input_df['ELA_cur_ent'] = input_df.apply(
                lambda row: ' '.join([row['ent']] + 
                                    [str(row[col]) for col in RELATIONS 
                                    if pd.notna(row[col]) and row[col] != '']), 
                axis=1
            )
        else:
            input_df['ELA_cur_ent'] = input_df.apply(
                lambda row: ' '.join([row['entity']] + 
                                    [str(row[col]) for col in RELATIONS 
                                    if pd.notna(row[col]) and row[col] != '']), 
                axis=1
            )
    if 'sequence' not in input_df.columns:
        gold_path = (getattr(args, 'gold_path', None) if args is not None else None) or './dataset/Lunguage.csv'
        if not os.path.exists(gold_path):
            raise FileNotFoundError(
                f"Lunguage.csv not found at '{gold_path}'. "
                "Set SequentialSRConfig(gold_path='/path/to/Lunguage.csv') or ensure "
                "the file exists at ./dataset/Lunguage.csv relative to your work_dir."
            )
        gold = pd.read_csv(gold_path)
        sequence_mapping = gold[['study_id', 'sequence']].drop_duplicates().set_index('study_id')['sequence'].to_dict()
        input_df['sequence'] = input_df['study_id'].map(sequence_mapping)
                    
    if 'section' not in input_df.columns and args is not None and args.eval_section:
        input_df['section'] = args.eval_section
    
    # Later for long context handling. We use all observations at once to generate the response.
    if 'cluster_name' not in input_df.columns:
        input_df['cluster_name'] = 'ANALYZE THE FOLLOWING OBSERVATIONS'
        
    def format_study_time(time_str):
        """Format study time string to timedelta"""
        if pd.isna(time_str):
            return pd.Timedelta(0)
        
        try:
            time_str = str(time_str).strip()
            main_part = time_str.split('.')[0].zfill(6)
            hours = int(main_part[:2])
            minutes = int(main_part[2:4])
            seconds = int(main_part[4:6])
            return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse time value: {time_str}, Error: {e}")
            return pd.Timedelta(0)
        
    def calculate_time_from_first(df):
        """Calculate time differences from the first sequence (lowest sequence number) for each subject"""
        # Ensure sequence is integer type and StudyDateTime is datetime type
        df = df.copy()
        df['sequence'] = df['sequence'].astype(int)
        
        # Check if StudyDateTime column exists
        if 'StudyDateTime' not in df.columns:
            print("Error: StudyDateTime column not found in dataframe")
            return pd.DataFrame(columns=['subject_id', 'sequence', 'ent_idx', 'time_from_first', 
                                        'StudyDateTime', 'FirstStudyDateTime'])
        
        # Convert StudyDateTime to datetime if it's not already
        if df['StudyDateTime'].dtype == 'object':
            df['StudyDateTime'] = pd.to_datetime(df['StudyDateTime'])
        
        # Get first (lowest) sequence dates for each subject
        first_sequences = (df[['subject_id', 'StudyDateTime']]
                        .dropna(subset=['StudyDateTime'])
                        .sort_values('StudyDateTime')
                        .groupby('subject_id').first()
                        .reset_index()
                        .rename(columns={'StudyDateTime': 'FirstStudyDateTime'}))
        
        # Calculate differences from first sequence
        return (df[['subject_id', 'ent_idx',  'sequence', 'StudyDateTime']]
                # .drop_duplicates(['subject_id', 'sequence'])
                .sort_values(['subject_id', 'sequence'])
                .merge(first_sequences, on='subject_id', how='left')
                .assign(time_from_first=lambda x: x['StudyDateTime'] - x['FirstStudyDateTime'])
                [['subject_id', 'sequence', 'ent_idx', 'time_from_first', 'StudyDateTime', 'FirstStudyDateTime']])
                        
    if not all(col in input_df.columns for col in ['StudyDateTime']):
        # raise ValueError("StudyDateTime column not found in dataframe, please put mimic-cxr-2.0.0-metadata.csv in the same directory as the input file")
        metadata_path = './mimic-cxr-2.0.0-metadata.csv'
        
        # If not found in current directory, check alternative path
        if not os.path.exists(metadata_path):
            alt_base_path = os.environ.get('MIMIC_CXR_DIR', '/data/physionet.org/files/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org')
            
            # Check for compressed file first
            alt_gz_path = os.path.join(alt_base_path, 'mimic-cxr-2.0.0-metadata.csv.gz')
            alt_csv_path = os.path.join(alt_base_path, 'mimic-cxr-2.0.0-metadata.csv')
            
            if os.path.exists(alt_gz_path):
                metadata_path = alt_gz_path
            elif os.path.exists(alt_csv_path):
                metadata_path = alt_csv_path
            else:
                raise FileNotFoundError(
                    "MIMIC-CXR metadata file not found. Please download 'mimic-cxr-2.0.0-metadata.csv' from PhysioNet "
                    "(https://physionet.org/content/mimic-cxr/2.0.0/) and place it in the 'sequentialSR' directory, "
                    f"or ensure it exists at {alt_base_path}"
                )
            
        # pandas.read_csv automatically handles .gz files
        mimic_metadata = pd.read_csv(metadata_path)
        mimic_metadata['StudyDate'] = pd.to_datetime(mimic_metadata['StudyDate'].astype(str).str.zfill(8), format='%Y%m%d')
        mimic_metadata['StudyTime'] = mimic_metadata['StudyTime'].apply(format_study_time)
        mimic_metadata['StudyDateTime'] = mimic_metadata['StudyDate'] + mimic_metadata['StudyTime']
        mimic_metadata['study_id'] = mimic_metadata['study_id'].astype(str).apply(lambda x: 's' + x if not x.startswith('s') else x)
        
        datetime_mapping = (mimic_metadata[['study_id', 'StudyDate', 'StudyTime', 'StudyDateTime']]
                        .dropna()
                        .sort_values('StudyDateTime')
                        .groupby('study_id').first()
                        .reset_index())
        input_df = input_df.merge(datetime_mapping, on=['study_id'], how='left')    


    if 'time_from_first' not in input_df.columns:
        time_diffs = calculate_time_from_first(input_df)
        input_df = input_df.merge(time_diffs, on=['subject_id', 'sequence', 'ent_idx'], how='left')
    
    if 'day_from_first' not in input_df.columns:
        input_df['day_from_first'] = input_df['time_from_first'].apply(lambda x: '0 days' if pd.isna(x) or pd.Timedelta(x).days == 0 else f"{pd.Timedelta(x).days} days")
    
    return input_df
                
def run_llm(args, client, total_cost, clustered_df, missing_data=None, is_missing_process=False, existing_output_df=None):
    if is_missing_process and existing_output_df is not None:
        # Missing process: update based on existing output_df
        output_df = existing_output_df.copy()
        print(f"Loaded existing output_df with {len(output_df)} rows for updating")
    else:
        output_df = pd.DataFrame()
    
    if not is_missing_process:
        if not args.all_eval:
            clustered_df = clustered_df[clustered_df['subject_id'].isin(args.subset)]
        
        print("number of subject_id: ", clustered_df.subject_id.nunique())

        for subject_id in clustered_df.subject_id.unique():
            print(f'Patient: "{subject_id}"')
            # we fix this to use all observations at once to generate the response.
            print("number of groups: ", len(clustered_df[clustered_df['subject_id'] == subject_id].cluster_name.unique()))
            
            for group_name in clustered_df[clustered_df['subject_id'] == subject_id].cluster_name.unique():
                print(f'  Group: "{group_name}"')
                
                cur_group_df = clustered_df[(clustered_df['subject_id'] == subject_id) &
                                (clustered_df['cluster_name'] == group_name)]
                
                # Exclude groups with only one observation
                if len(cur_group_df["ELA_cur_ent"]) <= 1:
                    print("Skipping group with only one observation", group_name)
                    continue
                
                dict_obs, ent_idx_info, seq_info, sent_idx_info, sent_info, ent_info, study_id_info, section_info = {}, {}, {}, {}, {}, {}, {}, {}
                idx_counter = 0
                
                has_sent_idx_col = 'sent_idx' in cur_group_df.columns
                has_sent_col = 'sent' in cur_group_df.columns
                has_section_col = 'section' in cur_group_df.columns
                
                for idx, obs in enumerate(cur_group_df['ELA_cur_ent'].to_list()):
                    day = cur_group_df['day_from_first'].to_list()[idx]
                    status = cur_group_df['dx_status'].to_list()[idx]
                    day_num = int(day.split()[0])  # Extract the number from "X days"
                    dict_obs[f"IDX:{idx_counter}, DAY: {day_num}, status: {status}"] = [obs]  # Create single-item list for each observation
                    ent_idx_info[f"IDX:{idx_counter}, DAY: {day_num}"] = cur_group_df['ent_idx'].to_list()[idx]
                    seq_info[f"IDX:{idx_counter}, DAY: {day_num}"] = cur_group_df['sequence'].to_list()[idx]
                    # Get sent_idx and sent if they exist in cur_group_df
                    idx_day_key = f"IDX:{idx_counter}, DAY: {day_num}"
                    if has_sent_idx_col:
                        sent_idx_val = cur_group_df['sent_idx'].to_list()[idx]
                        sent_idx_info[idx_day_key] = sent_idx_val if pd.notna(sent_idx_val) else None
                    else:
                        sent_idx_info[idx_day_key] = None
                    if has_sent_col:
                        sent_val = cur_group_df['sent'].to_list()[idx]
                        sent_info[idx_day_key] = sent_val if pd.notna(sent_val) else None
                    else:
                        sent_info[idx_day_key] = None

                    # Track ent/entity and study_id for downstream saving
                    ent_value = None
                    if 'ent' in cur_group_df.columns:
                        ent_value = cur_group_df['ent'].to_list()[idx]
                    elif 'entity' in cur_group_df.columns:
                        ent_value = cur_group_df['entity'].to_list()[idx]
                    ent_info[idx_day_key] = ent_value

                    study_value = cur_group_df['study_id'].to_list()[idx] if 'study_id' in cur_group_df.columns else None
                    study_id_info[idx_day_key] = study_value
                    
                    # Track section for downstream saving
                    section_value = cur_group_df['section'].to_list()[idx] if has_section_col else None
                    section_info[idx_day_key] = section_value if pd.notna(section_value) else None
                    
                    idx_counter += 1

                # Should be:
                input_json = {
                    "cluster_name": group_name,
                    "observations": dict_obs
                }
                Input = json.dumps(input_json, ensure_ascii=False)
                
                conversation = generate_few_shot(Input, args)
                conversation = convert_sets_to_lists(conversation)
                
                # Ensure each message has the correct format
                formatted_conversation = []
                for msg in conversation:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # If content is a dict or list, convert it to a string
                        if isinstance(msg['content'], (dict, list)):
                            msg['content'] = str(msg['content'])
                        formatted_conversation.append(msg)
                    else:
                        print(f"Skipping invalid message format: {msg}")
            
                try:
                    if args.LLM_name.startswith('claude'):
                        response = client.messages.create(
                            model=args.LLM_name,
                            max_tokens=8096,
                            messages=conversation,
                            response_model=RadiologyOutput
                        )
                    elif args.LLM_name.startswith('gpt-5'):
                        response = client.chat.completions.create(
                            model="gpt-5",
                            messages=conversation,
                            response_model=RadiologyOutput
                        )
                    elif args.LLM_name.startswith('baichuan') or args.LLM_name.startswith('medgemma') or args.LLM_name.startswith('gpt-oss-120b') or args.LLM_name.startswith('gpt-oss-20b'):
                        response = client.chat.completions.create(
                            model= "baichuan-inc/Baichuan-M1-14B-Instruct" if args.LLM_name.startswith('baichuan') else "medgemma-27b-text-it" if args.LLM_name.startswith('medgemma') else "gpt-oss-20b" if args.LLM_name.startswith('gpt-oss-20b') else "gpt-oss-120b",
                            messages=conversation,
                            response_model=RadiologyOutput
                            )
                    else:
                        response = client.chat.completions.create(
                        model=f'accounts/fireworks/models/{args.LLM_name}',
                        max_tokens= 8092,
                        temperature=0.0,
                        messages=formatted_conversation,
                        response_model=RadiologyOutput
                    )
                except Exception as e:
                    print(f"Error processing patient {subject_id}: {e}")
                    continue

                input_data = {
                    "batch_idx": "non-batch",
                    "subject_id": subject_id,
                    "Input": Input,
                    "Input_ent_idx": ent_idx_info,
                    "Input_seq": seq_info,
                    "Input_sent_idx": sent_idx_info,
                    "Input_sent": sent_info,
                    "Input_ent": ent_info,
                    "Input_study_id": study_id_info,
                    "Input_section": section_info,
                    "Group_name": group_name
                }
                llm_output_df, stats = process_radiology_output(response, input_data, clustered_df=clustered_df)
                output_df = pd.concat([output_df, llm_output_df])
        output_path = f"{args.output_path}/output_df.csv"
        
    else:
        for subject_id, subject_clusters in missing_data.items():            
            for group_name, cluster_content in subject_clusters.items():
                conversation = generate_few_shot(cluster_content, args)
                
                # Convert any sets to lists and ensure proper message format
                conversation = convert_sets_to_lists(conversation)

                # Ensure each message has the correct format
                formatted_conversation = []
                for msg in conversation:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # If content is a dict or list, convert it to a string
                        if isinstance(msg['content'], (dict, list)):
                            msg['content'] = str(msg['content'])
                        formatted_conversation.append(msg)
                    else:
                        print(f"Skipping invalid message format: {msg}")

        
                try:
                    if args.LLM_name.startswith('claude'):
                        response = client.messages.create(
                            model=args.LLM_name,
                            max_tokens=8096,
                            messages=conversation,
                            response_model=RadiologyOutput
                        )
                    elif args.LLM_name.startswith('baichuan') or args.LLM_name.startswith('medgemma') or args.LLM_name.startswith('gpt-oss-120b') or args.LLM_name.startswith('gpt-oss-20b'):
                        response = client.chat.completions.create(
                            model="baichuan-inc/Baichuan-M1-14B-Instruct" if args.LLM_name.startswith('baichuan') else "medgemma-27b-text-it" if args.LLM_name.startswith('medgemma') else "gpt-oss-20b" if args.LLM_name.startswith('gpt-oss-20b') else "gpt-oss-120b",
                            messages=conversation,
                            response_model=RadiologyOutput
                            )
                    else:
                        response = client.chat.completions.create(
                            model=f'accounts/fireworks/models/{args.LLM_name}',
                            max_tokens= 8092,
                            temperature=0.0,
                            messages=formatted_conversation,
                            response_model=RadiologyOutput
                        )
                except Exception as e:
                    print(f"Error processing patient {subject_id}: {e}")
                    continue

                input_data = {
                    "batch_idx": "non-batch-missing",
                    "Group_name": group_name,
                    "subject_id": subject_id
                }
                # Add sent_idx and sent information if available in result_data
                if subject_id in missing_data and group_name in missing_data[subject_id]:
                    cluster_data = missing_data[subject_id][group_name]
                    if "Input_sent_idx" in cluster_data:
                        input_data["Input_sent_idx"] = cluster_data["Input_sent_idx"]
                    if "Input_sent" in cluster_data:
                        input_data["Input_sent"] = cluster_data["Input_sent"]
                    if "Input_ent" in cluster_data:
                        input_data["Input_ent"] = cluster_data["Input_ent"]
                    if "Input_study_id" in cluster_data:
                        input_data["Input_study_id"] = cluster_data["Input_study_id"]
                    if "Input_section" in cluster_data:
                        input_data["Input_section"] = cluster_data["Input_section"]
                llm_output_df, stats = process_radiology_output(response, input_data, is_missing_process=True, clustered_df=clustered_df)
                output_df = pd.concat([output_df, llm_output_df])
        output_path = f"{args.output_path}/missing_outputs.csv"
    
    try:
        for col in ['ent', 'study_id', 'section']:
            if col not in output_df.columns:
                output_df[col] = None
        dup_key = ['subject_id', 'study_id', 'sequence', 'ent_idx', 'ent', 'sent_idx', 'sent', 'ELA_cur_ent']
        output_df = output_df.drop_duplicates(subset=dup_key, keep='last')
        output_df.to_csv(output_path, index=False)
        print(f"Successfully saved output_df to {output_path}")
    except Exception as e:
        print(f"Error saving output_df: {e}")
    return output_df

def run_batch(args, client, batch_file, is_missing_process=False, iteration=0):
    print(f"\nRun batch processing with {batch_file.split('/')[-1]}")
    file_path = os.path.join(batch_file)
    
    if not os.path.isfile(batch_file):
        raise FileNotFoundError(f"File not found: {file_path}")

    batch_job = create_batch_job(file_path, client, args)
        
    print(f"Successfully created batch job: {batch_job.id}\n")
    batch_job_id = batch_job.id
    check_interval = 2  # 2sec interval
    previous_status = None


    print("-" * 50)
    try:
        while True:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            batch = client.batches.retrieve(batch_job_id)
            batch_status = batch.status
            completed_count = batch.request_counts.completed
            total_count = batch.request_counts.total
                            
            if batch_status != previous_status:
                print(f"\n[{current_time}] Status changed: {previous_status} → {batch_status}")
                print(f"Completed: {completed_count}/{total_count}")
                
                if batch_status == "in_progress":
                    print(f"In progress: {completed_count}/{total_count}")
                
                if batch_status in ["completed", "succeeded", "ended"]:
                    print(f"Completed! Result file ID: {batch.output_file_id}")
                    break
                    
                if batch_status in ["failed", "errored"]:
                    print("Failed!")
                    break
                    
                previous_status = batch_status
            else:
                if batch_status == "in_progress" and total_count > 0:
                    completion_percent = (completed_count / total_count) * 100
                    print(f"\r[{current_time}] Progress: {completion_percent:.1f}% ({completed_count}/{total_count})", end="")
                else:
                    print(f"\r[{current_time}] Current status: {batch_status}", end="")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        cancel_batch = client.batches.cancel(batch_job_id)
        print("Batch cancelled", cancel_batch)
    
    batch = client.batches.retrieve(batch_job_id)
    completed_count = batch.request_counts.completed
    total_count = batch.request_counts.total
        
    print("\n\nFinal batch status:")
    print(f"Status: {batch_status}")
    print(f"Completed: {completed_count}/{total_count}")

    if batch_status in ["completed", "succeeded", "ended"]:
        print("\nBatch job completed!")
        if batch.output_file_id:
            print(f"Result file ID: {batch.output_file_id}")
            try:
                result_content = client.files.content(batch.output_file_id).text
                if not is_missing_process:
                    with open(f'{args.output_path}/batch_results.jsonl', 'w') as f:
                        f.write(result_content)
                else:
                    with open(f'{args.output_path}/missing_outputs{iteration}.jsonl', 'w') as f:
                        f.write(result_content)
                print("-----")
            except Exception as e:
                print(f"Error retrieving batch results: {e}")
                print("Batch completed but could not retrieve results file.")
        else:
            print("Warning: Batch completed but output_file_id is None. Results may not be available.")
            print("-----")

def export_structured_benchmark_file(args):
    """
    Merge the sequential SR output with the original benchmark file and
    save it as <model_name>_SR.csv next to the input file (e.g. chexagent_SR.csv).
    """
    output_df_path = os.path.join(args.output_path, "output_df.csv")
    if not os.path.exists(output_df_path):
        print(f"[SR Export] Skipping export because {output_df_path} does not exist.")
        return None

    try:
        seq_df = pd.read_csv(output_df_path)
    except Exception as e:
        print(f"[SR Export] Failed to read sequential output ({output_df_path}): {e}")
        return None

    if seq_df.empty:
        print("[SR Export] Sequential output is empty. Skipping export.")
        return None

    try:
        main_df = pd.read_csv(args.input_path)
    except Exception as e:
        print(f"[SR Export] Failed to read input benchmark file ({args.input_path}): {e}")
        return None

    if 'ent' not in main_df.columns and 'entity' in main_df.columns:
        main_df = main_df.rename(columns={'entity': 'ent'})

    # Check if section column exists in both dataframes
    has_section_in_seq = 'section' in seq_df.columns
    has_section_in_main = 'section' in main_df.columns
    
    dedupe_cols = ['subject_id', 'sequence', 'sent_idx', 'ent_idx']
    if has_section_in_seq:
        dedupe_cols.append('section')
    
    missing_dedupe_cols = [col for col in dedupe_cols if col not in seq_df.columns]
    if missing_dedupe_cols:
        print(f"[SR Export] Required columns missing in output_df: {missing_dedupe_cols}")
        return None

    seq_df = seq_df.drop_duplicates(subset=dedupe_cols, keep='last')
    dup_mask = seq_df.duplicated(subset=dedupe_cols, keep=False)
    dedupe_cols_str = ', '.join(dedupe_cols)
    print(f"Duplicate ({dedupe_cols_str}) combinations count:", int(dup_mask.sum()))
    if dup_mask.any():
        print("Duplicate indices (head):", seq_df[dup_mask].index.tolist()[:10])
    else:
        print("Duplicate indices (head): []")
    print("Total row count:", len(seq_df))

    keep_cols = ['subject_id', 'sequence', 'sent_idx', 'ent_idx', 'temporal_group', 'LLM_cluster']
    if has_section_in_seq:
        keep_cols.append('section')
    keep_cols = [col for col in keep_cols if col in seq_df.columns]
    seq_small = seq_df[keep_cols].copy()
    
    # Deduplicate with section if available
    dedupe_cols_small = ['subject_id', 'sequence', 'sent_idx', 'ent_idx']
    if has_section_in_seq:
        dedupe_cols_small.append('section')
    seq_small = seq_small.drop_duplicates(subset=dedupe_cols_small)

    merge_keys = ['subject_id', 'sequence', 'sent_idx', 'ent_idx']
    # Include section in merge keys if it exists in both dataframes
    if has_section_in_main and has_section_in_seq:
        merge_keys.append('section')
        print(f"[SR Export] Using merge keys with section: {merge_keys}")
    else:
        print(f"[SR Export] Using merge keys without section: {merge_keys}")
        if has_section_in_seq and not has_section_in_main:
            print(f"[SR Export] Warning: section exists in output_df but not in input file. section will be added to merged result.")
        elif has_section_in_main and not has_section_in_seq:
            print(f"[SR Export] Warning: section exists in input file but not in output_df. Merge will use input file's section.")
    
    missing_merge_keys = [col for col in merge_keys if col not in main_df.columns]
    if missing_merge_keys:
        print(f"[SR Export] Input benchmark file missing merge keys: {missing_merge_keys}")
        return None

    merged_df = pd.merge(main_df, seq_small, on=merge_keys, how='left')

    target_dir = os.path.dirname(os.path.abspath('./benchmark_SR'))
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{args.LLM_name}_SR.csv")
    merged_df.to_csv(target_path, index=False)
    print(f"[SR Export] Saved structured benchmark file to: {target_path}")
    return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LLM_name', type=str, 
                        default='qwen3-235b-a22b', 
                        choices=['baichuan',
                                'medgemma-27b-text-it',
                                'gpt-4.1', 
                                'gpt-5', 
                                'qwen3-235b-a22b', 
                                'deepseek-v3-0324',
                                'llama4-maverick-instruct-basic',
                                'gpt-oss-120b',
                                'gpt-oss-20b'])

    parser.add_argument('--model_run', action='store_true')
    parser.add_argument('--batch_path', type=str, default='./sequentialSR/batch_files')
    parser.add_argument('--all_eval', action='store_true')
    parser.add_argument('--subset', type=list, default= 
                                                        # Default evaluation subset (30 subjects)
                                                        ['p10046166', 'p10532326', 'p10885696', 'p11540283', 'p11607628',
                                                        'p11879886', 'p12966004', 'p15094735', 'p15109122', 'p15207316',
                                                        'p15272972', 'p16059470', 'p17270742', 'p17288844', 'p17396677',
                                                        'p17962324', 'p18417750', 'p18517718', 'p18570152', 'p19150427',
                                                        'p10274145', 'p10523725', 'p10886362', 'p10959054', 'p12433421',
                                                        'p15321868', 'p15446959', 'p15881535', 'p17720924', 'p18079481'])

    parser.add_argument('--process_missing', action='store_true')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--eval_section', type=str, default=None)

    parser.add_argument('--input_path', type=str, default='./dataset/Lunguage.csv') ## change file name here
    parser.add_argument('--output_path', type=str, default='./sequentialSR/results') ## change output path here
    # Add API key arguments
    parser.add_argument('--api_key', type=str, default='local_LLM',
                        help='API key for the selected model for third party usage like OpenAI, Fireworks, Anthropic, etc.')

    args = parser.parse_args()

    # Set API keys from arguments if provided
    if args.api_key:
        os.environ["API_KEY"] = args.api_key

    if args.all_eval:
        if args.few_shot:
            args.output_path = f'{args.output_path}/{args.LLM_name}/all_eval/few_shot/{args.input_path.split("/")[-1].split(".")[0]}'
            args.batch_path = f'{args.batch_path}/{args.LLM_name}/all_eval/few_shot/{args.input_path.split("/")[-1].split(".")[0]}'
        else:
            args.output_path = f'{args.output_path}/{args.LLM_name}/all_eval/zero_shot/{args.input_path.split("/")[-1].split(".")[0]}'
            args.batch_path = f'{args.batch_path}/{args.LLM_name}/all_eval/zero_shot/{args.input_path.split("/")[-1].split(".")[0]}'
    else:
        if args.few_shot:
            args.output_path = f'{args.output_path}/{args.LLM_name}/subset{len(args.subset)}/few_shot/{args.input_path.split("/")[-1].split(".")[0]}'
            args.batch_path = f'{args.batch_path}/{args.LLM_name}/subset{len(args.subset)}/few_shot/{args.input_path.split("/")[-1].split(".")[0]}'
        else:
            args.output_path = f'{args.output_path}/{args.LLM_name}/subset{len(args.subset)}/zero_shot/{args.input_path.split("/")[-1].split(".")[0]}'
            args.batch_path = f'{args.batch_path}/{args.LLM_name}/subset{len(args.subset)}/zero_shot/{args.input_path.split("/")[-1].split(".")[0]}'


    os.makedirs(args.output_path, exist_ok=True)

    client, tokenizer = initialize_llm_client(args.LLM_name)
    clustered_df = pd.read_csv(args.input_path)
    clustered_df = load_and_process_data(clustered_df, args)
    
    # Save processed data with ELA_cur_ent column
    processed_input_path = f"{args.output_path}/processed_input_with_ELA_cur_ent.csv"
    os.makedirs(args.output_path, exist_ok=True)
    clustered_df.to_csv(processed_input_path, index=False)
    print(f"Saved processed input data with ELA_cur_ent to: {processed_input_path}")
           
    if not args.all_eval:
        clustered_df = clustered_df[clustered_df['subject_id'].isin(args.subset)]    
    
    if (args.LLM_name.startswith('gpt-4.1') ) and os.getenv('API_KEY') != 'local_LLM':
        print(f"{args.LLM_name} Batch processing start!")
        os.makedirs(args.batch_path, exist_ok=True)
        all_inputs = creating_batch_file(clustered_df, args)
        iteration = 0
        
        if args.model_run:
            run_batch(args, client, f"{args.batch_path}/llm_batch.jsonl", is_missing_process=False)
            batch_result = f"{args.output_path}/batch_results.jsonl"
            llm_output = read_batch_results_to_csv(args, batch_result, clustered_df, all_inputs, is_missing_process=False)
            clustered_df = post_process(llm_output, clustered_df, args.output_path, is_missing_process=False)

            unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
            matched_count = (clustered_df['llm_processed'] == 'matched').sum()
            print(f"\n Iteration 0 completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
            clustered_df.drop_duplicates(inplace=True)
            clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)

        else:
            iteration = 0
            processed_files = glob.glob(f"{args.output_path}/final_processed*_0missing.csv")
            if processed_files:
                # Sort files to get the most recent one
                latest_file = sorted(processed_files)[-1]
                output_df = pd.read_csv(latest_file)
                if 'ent' not in output_df.columns and 'entity' in output_df.columns:
                    output_df['ent'] = output_df['entity']
                print(f"Loaded processed data from {latest_file}")
            else:
                # Try to load output_df.csv as fallback
                output_df_path = f"{args.output_path}/output_df.csv"
                if os.path.exists(output_df_path):
                    latest_file = output_df_path
                    output_df = pd.read_csv(latest_file)
                    if 'ent' not in output_df.columns and 'entity' in output_df.columns:
                        output_df['ent'] = output_df['entity']
                    print(f"Loaded processed data from {latest_file}")
                else:
                    target_path = os.path.join(args.input_path)
                    if os.path.exists(target_path):
                        output_df = pd.read_csv(target_path)
                        if 'ent' not in output_df.columns and 'entity' in output_df.columns:
                            output_df['ent'] = output_df['entity']
                        print(f"Loaded processed data from {target_path}")
                    else:
                        raise FileNotFoundError(f"No processed data found in {target_path}")
            
                if 'gt_temporal_group' not in output_df.columns:
                    merge_cols = ['subject_id', 'study_id', 'ent', 'ent_idx', 'section', 'sent_idx']
                    # Only merge columns from clustered_df that are missing in output_df
                    missing_col_set = set(['gt_temporal_group', 'gt_entity_group']) & set(clustered_df.columns)
                    clustered_df = output_df.merge(
                        clustered_df[merge_cols + list(missing_col_set)],
                        on=merge_cols,
                        how='left',
                        suffixes=('', '_cl')
                    )
                else:
                    clustered_df = output_df

        if args.process_missing:
            # Load existing output_df.csv if it exists
            existing_output_df = pd.DataFrame()
            existing_output_path = f"{args.output_path}/output_df.csv"
            if os.path.exists(existing_output_path):
                try:
                    existing_output_df = pd.read_csv(existing_output_path)
                    print(f"Loaded existing output_df.csv with {len(existing_output_df)} rows for batch processing")
                except Exception as e:
                    print(f"Warning: Could not load existing output_df.csv: {e}")
                    existing_output_df = pd.DataFrame()
            
            os.makedirs(args.batch_path, exist_ok=True)
            while (clustered_df['llm_processed'] == 'unmatched').any():
                iteration += 1
                print(f"{args.LLM_name} Missing process iteration {iteration} start!")
                result_data = prepare_missing_inputs(clustered_df)
                missing_dict = {}
            
                for subject_id, subject_clusters in result_data.items():
                    print(f"\n=== Subject ID: {subject_id} ===")
                    missing_dict[subject_id] = {}
            
                    for cluster_name in subject_clusters.keys():
                        missing_input = create_missing_input(subject_id, cluster_name, result_data)
                        missing_dict[subject_id][cluster_name] = missing_input
            
                _ = creating_batch_file(clustered_df, args, missing_data=missing_dict, all_inputs=all_inputs, iteration=iteration)
                run_batch(args, client, f"{args.batch_path}/missing_batch{iteration}.jsonl", is_missing_process=True, iteration=iteration)
            
                llm_output = read_batch_results_to_csv(args, f"{args.output_path}/missing_outputs{iteration}.jsonl", clustered_df, all_inputs, is_missing_process=True, existing_output_df=existing_output_df)
                clustered_df = post_process(llm_output, clustered_df, args.output_path, is_missing_process=True, iteration=iteration)
                
                # Save updated output_df for the next iteration
                existing_output_df = llm_output.copy()

                # Break the loop if no progress is being made (to prevent infinite loops)
                unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
                matched_count = (clustered_df['llm_processed'] == 'matched').sum()
                print(f"Iteration {iteration} completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
                clustered_df.drop_duplicates(inplace=True)
                clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
                
                if (clustered_df['llm_processed'] == 'unmatched').sum() == 0 or iteration > 1:
                    break
        
        print("clustered_df.columns:", clustered_df.columns)
        if 'gt_temporal_group' in clustered_df.columns:
            eval_df = clustered_df[clustered_df['gt_temporal_group'].notna() & clustered_df['gt_entity_group'].notna()]
            if 'ent' not in eval_df.columns and 'entity' in eval_df.columns:
                eval_df['ent'] = eval_df['entity']
                clustered_df['ent'] = clustered_df['entity']
            dup_key = ['subject_id', 'study_id', 'section', 'ent_idx', 'ent', 'sent_idx', 'sent']
            eval_df = eval_df.drop_duplicates(subset=dup_key, keep='last')
            try:
                print("Evaluation start!")
                eval_func(args, eval_df)
            except Exception as e:
                print(f"Warning: Evaluation function failed: {e}\nContinuing without evaluation...")
        print(f"Process completed! Path: ({args.output_path}/final_processed{iteration}.csv)")
    else:
        print(f"{args.LLM_name} Single processing start!\n")
        total_cost = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "gpt-45-cost": [],
            "gpt-4o-batch-cost": [],
            "gpt-4o-cost": [],
            "gpt-4o-mini-cost": [],
        }
        iteration = 0
        # Initialize existing_output_df for missing process
        existing_output_df = None
        
        if args.model_run:
            llm_output = run_llm(args, client, total_cost, clustered_df, is_missing_process=False)
            clustered_df = post_process(llm_output, clustered_df, args.output_path, is_missing_process=False)
            unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
            matched_count = (clustered_df['llm_processed'] == 'matched').sum()
            print(f"Iteration 0 completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
            clustered_df.drop_duplicates(inplace=True)
            clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
            # Set existing_output_df from first iteration if process_missing will be used
            if args.process_missing:
                existing_output_df = llm_output.copy() if llm_output is not None and not llm_output.empty else None
        else:
            processed_files = glob.glob(f"{args.output_path}/final_processed*_0missing.csv")
            if processed_files:
                # Sort files to get the most recent one
                latest_file = sorted(processed_files)[-1]
                clustered_df = pd.read_csv(latest_file)
                if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                    clustered_df['ent'] = clustered_df['entity']
                print(f"Loaded processed data from {latest_file}")
            else:
                # Try to load output_df.csv as fallback
                output_df_path = f"{args.output_path}/output_df.csv"
                if os.path.exists(output_df_path):
                    latest_file = output_df_path
                    clustered_df = pd.read_csv(latest_file)
                    if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                        clustered_df['ent'] = clustered_df['entity']
                    print(f"Loaded processed data from {latest_file}")
                else:
                    target_path = os.path.join(args.input_path)
                    if os.path.exists(target_path):
                        clustered_df = pd.read_csv(target_path)
                        if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                            clustered_df['ent'] = clustered_df['entity']
                        print(f"Loaded processed data from {target_path}")
                    else:
                        raise FileNotFoundError(f"No processed data found in {target_path}")
        
        if args.process_missing:
            while (clustered_df['llm_processed'] == 'unmatched').any():
                iteration += 1
                print(f"{args.LLM_name} Missing process iteration {iteration} start!")
                result_data = prepare_missing_inputs(clustered_df)
                
                missing_dict = {}
                for subject_id, subject_clusters in result_data.items():
                    print(f"\n=== Subject ID: {subject_id} ===")
                    missing_dict[subject_id] = {}
                    for cluster_name in subject_clusters.keys():
                        missing_input = create_missing_input(subject_id, cluster_name, result_data)
                        missing_dict[subject_id][cluster_name] = missing_input
                
                # Pass existing output_df to update
                missing_llm_output = run_llm(args, client, total_cost, clustered_df, missing_data=missing_dict, is_missing_process=True, existing_output_df=existing_output_df)
                clustered_df = post_process(missing_llm_output, clustered_df, args.output_path, is_missing_process=True, iteration=iteration)
                
                # Save updated output_df for the next iteration
                existing_output_df = missing_llm_output.copy()
                # Break the loop if no progress is being made (to prevent infinite loops)
                unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
                matched_count = (clustered_df['llm_processed'] == 'matched').sum()  
                print(f"Iteration {iteration} completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
                clustered_df.drop_duplicates(inplace=True)
                clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
                
                if (clustered_df['llm_processed'] == 'unmatched').sum() == 0 or iteration > 1:
                    break
    
        print("clustered_df.columns:", clustered_df.columns)
        if 'gt_temporal_group' in clustered_df.columns:

            eval_df = clustered_df[clustered_df['gt_temporal_group'].notna() & clustered_df['gt_entity_group'].notna()]
            if 'ent' not in eval_df.columns and 'entity' in eval_df.columns:
                eval_df['ent'] = eval_df['entity']
                clustered_df['ent'] = clustered_df['entity']
            dup_key = ['subject_id', 'study_id', 'section', 'ent_idx', 'ent', 'sent_idx', 'sent']
            eval_df = eval_df.drop_duplicates(subset=dup_key, keep='last')
            try:
                print("Evaluation start!")
                eval_func(args, eval_df)
            except Exception as e:
                print(f"Warning: Evaluation function failed: {e}\nContinuing without evaluation...")
        print(f"Process completed! Path: ({args.output_path}/final_processed{iteration}.csv)")
    
    export_structured_benchmark_file(args)


# ─── Package-level entry points ────────────────────────────────────────────────

def process_with_llm(args, client, clustered_df):
    """Single (non-batch) LLM processing for local/vLLM models."""
    print(f"{args.LLM_name} Single processing start!\n")
    total_cost = {
        "prompt_tokens": [],
        "completion_tokens": [],
        "gpt-45-cost": [],
        "gpt-4o-batch-cost": [],
        "gpt-4o-cost": [],
        "gpt-4o-mini-cost": [],
    }
    iteration = 0
    existing_output_df = None

    if args.model_run:
        llm_output = run_llm(args, client, total_cost, clustered_df, is_missing_process=False)
        clustered_df = post_process(llm_output, clustered_df, args.output_path, is_missing_process=False)
        if 'llm_processed' in clustered_df.columns:
            unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
            matched_count = (clustered_df['llm_processed'] == 'matched').sum()
        else:
            unmatched_count = 0
            matched_count = 0
        print(f"Iteration 0 completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
        clustered_df.drop_duplicates(inplace=True)
        clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
        if args.process_missing:
            existing_output_df = llm_output.copy() if llm_output is not None and not llm_output.empty else None
    else:
        processed_files = glob.glob(f"{args.output_path}/final_processed*_0missing.csv")
        if processed_files:
            latest_file = sorted(processed_files)[-1]
            clustered_df = pd.read_csv(latest_file)
            if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                clustered_df['ent'] = clustered_df['entity']
            print(f"Loaded processed data from {latest_file}")
        else:
            output_df_path = f"{args.output_path}/output_df.csv"
            if os.path.exists(output_df_path):
                clustered_df = pd.read_csv(output_df_path)
                if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                    clustered_df['ent'] = clustered_df['entity']
                print(f"Loaded processed data from {output_df_path}")
            else:
                target_path = args.input_path
                if os.path.exists(target_path):
                    clustered_df = pd.read_csv(target_path)
                    if 'ent' not in clustered_df.columns and 'entity' in clustered_df.columns:
                        clustered_df['ent'] = clustered_df['entity']
                    print(f"Loaded processed data from {target_path}")
                else:
                    raise FileNotFoundError(f"No processed data found in {target_path}")

    if args.process_missing and 'llm_processed' in clustered_df.columns:
        while (clustered_df['llm_processed'] == 'unmatched').any():
            iteration += 1
            print(f"{args.LLM_name} Missing process iteration {iteration} start!")
            result_data = prepare_missing_inputs(clustered_df)
            missing_dict = {}
            for subject_id, subject_clusters in result_data.items():
                print(f"\n=== Subject ID: {subject_id} ===")
                missing_dict[subject_id] = {}
                for cluster_name in subject_clusters.keys():
                    missing_input = create_missing_input(subject_id, cluster_name, result_data)
                    missing_dict[subject_id][cluster_name] = missing_input

            missing_llm_output = run_llm(args, client, total_cost, clustered_df, missing_data=missing_dict,
                                         is_missing_process=True, existing_output_df=existing_output_df)
            clustered_df = post_process(missing_llm_output, clustered_df, args.output_path,
                                        is_missing_process=True, iteration=iteration)
            existing_output_df = missing_llm_output.copy()
            if 'llm_processed' in clustered_df.columns:
                unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
                matched_count = (clustered_df['llm_processed'] == 'matched').sum()
            else:
                unmatched_count = 0
                matched_count = 0
            print(f"Iteration {iteration} completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
            clustered_df.drop_duplicates(inplace=True)
            clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
            if unmatched_count == 0 or iteration > 1:
                break

    print("clustered_df.columns:", clustered_df.columns)
    if 'gt_temporal_group' in clustered_df.columns:
        eval_df = clustered_df[clustered_df['gt_temporal_group'].notna() & clustered_df['gt_entity_group'].notna()]
        if 'ent' not in eval_df.columns and 'entity' in eval_df.columns:
            eval_df['ent'] = eval_df['entity']
            clustered_df['ent'] = clustered_df['entity']
        dup_key = ['subject_id', 'study_id', 'section', 'ent_idx', 'ent', 'sent_idx', 'sent']
        eval_df = eval_df.drop_duplicates(subset=dup_key, keep='last')
        try:
            print("Evaluation start!")
            eval_func(args, eval_df)
        except Exception as e:
            print(f"Warning: Evaluation function failed: {e}\nContinuing without evaluation...")
    print(f"Process completed! Path: ({args.output_path}/final_processed{iteration}.csv)")


def _run_pipeline(args, client, clustered_df: pd.DataFrame):
    """Core pipeline logic shared by run() and run_with_dataframe()."""
    clustered_df = load_and_process_data(clustered_df, args)

    processed_input_path = f"{args.output_path}/processed_input_with_ELA_cur_ent.csv"
    os.makedirs(args.output_path, exist_ok=True)
    clustered_df.to_csv(processed_input_path, index=False)
    print(f"Saved processed input data: {processed_input_path}")

    if not args.all_eval:
        clustered_df = clustered_df[clustered_df['subject_id'].isin(args.subset)]

    if (args.LLM_name.startswith('gpt-4.1')) and os.getenv('API_KEY') != 'local_LLM':
        print(f"{args.LLM_name} Batch processing start!")
        os.makedirs(args.batch_path, exist_ok=True)
        all_inputs = creating_batch_file(clustered_df, args)
        iteration = 0

        if args.model_run:
            run_batch(args, client, f"{args.batch_path}/llm_batch.jsonl", is_missing_process=False)
            batch_result = f"{args.output_path}/batch_results.jsonl"
            llm_output = read_batch_results_to_csv(args, batch_result, clustered_df, all_inputs, is_missing_process=False)
            clustered_df = post_process(llm_output, clustered_df, args.output_path, is_missing_process=False)

            unmatched_count = (clustered_df['llm_processed'] == 'unmatched').sum()
            matched_count = (clustered_df['llm_processed'] == 'matched').sum()
            print(f"Iteration 0 completed. Remaining unmatched: {unmatched_count}, Matched: {matched_count}")
            clustered_df.drop_duplicates(inplace=True)
            clustered_df.to_csv(f"{args.output_path}/final_processed{iteration}_{unmatched_count}missing.csv", index=False)
    else:
        process_with_llm(args, client, clustered_df)

    export_structured_benchmark_file(args)


def run(args, client):
    """Load input from args.input_path and run the sequential SR pipeline."""
    clustered_df = pd.read_csv(args.input_path)
    _run_pipeline(args, client, clustered_df)


def run_with_dataframe(args, client, input_df: pd.DataFrame):
    """Run the sequential SR pipeline on a pre-loaded DataFrame."""
    _run_pipeline(args, client, input_df.copy())