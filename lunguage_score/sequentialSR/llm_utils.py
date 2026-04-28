import pandas as pd
from typing import List, Dict, Any
import instructor
from pydantic import BaseModel, Field, RootModel
import os
import json
import numpy as np
import re
from .normalize_prompt import Sequential_findings_review_prompt, Result_Review_w_time_gap_Ex1, Result_Review_w_time_gap_Ex2, Result_Review_w_time_gap_Ex3, Result_Review_w_time_gap_Ex4, Result_Review_w_time_gap_Ex5
from collections import defaultdict
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from openai import OpenAI
from ..config import CLAUDE_SONNET_MODEL
END_POINT = "/v1/chat/completions"


def create_batch_job(file_name, client, args):
    if args.LLM_name.startswith('claude'):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request            
        requests = []
        schema_format = [
            {
                "name": "sequential_matching",
                "description": "build the sequential matching object",
                "input_schema": RadiologyOutput.model_json_schema()
            }
        ]

        with open(file_name, 'r') as file:
            for line in file:
                task = json.loads(line)
                custom_id = task.get('custom_id', '')
                messages = task.get('messages', [])
                
                request = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=CLAUDE_SONNET_MODEL,
                        max_tokens=4096 if 'haiku' in args.LLM_name.lower() else 8192,
                        system=[
                                {"type": "text",
                                "text": Sequential_findings_review_prompt,
                                "cache_control": {"type": "ephemeral"}}
                            ],
                        messages = messages,
                        tools = schema_format,
                        tool_choice={"type": "tool", "name": "sequential_matching"}
                    )
                )
                requests.append(request)

        
        batch_job = client.beta.messages.batches.create(requests=requests)
    else:
        batch_file = client.files.create(
                        file=open(file_name, "rb"),
                        purpose="batch"
                        )       
        batch_job = client.batches.create(
                        input_file_id=batch_file.id,
                        endpoint=END_POINT,
                        completion_window="24h",
                        
                        )
    return batch_job

def _clean_malformed_json(json_str):
    """Fix common errors in a JSON string."""
    import re
    
    # 1. Strip markdown code fences (```json...``` or ```...```)
    if json_str.strip().startswith('```'):
        # Remove opening/closing code fence markers
        lines = json_str.strip().split('\n')
        
        # Remove first line if it is ```json or ```
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
            
        # Remove last line if it is ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
            
        json_str = '\n'.join(lines)
    
    # 2. Strip leading/trailing whitespace
    json_str = json_str.strip()
    
    # 3. Handle duplicate rationale fields
    # Fix pattern: "rationale": "...", "episodes": [...], "rationale": "..."
    pattern = r'"rationale":\s*"([^"]*(?:\\.[^"]*)*)",\s*"episodes":\s*(\[[^\]]*\]),\s*"rationale":\s*"([^"]*(?:\\.[^"]*)*)"'
    
    def replace_duplicate_rationale(match):
        rationale1 = match.group(1)
        episodes = match.group(2) 
        rationale2 = match.group(3)
        
        # Use the second rationale (usually more concise and accurate)
        return f'"rationale": "{rationale2}", "episodes": {episodes}'
    
    cleaned = re.sub(pattern, replace_duplicate_rationale, json_str, flags=re.DOTALL)
    
    # 4. Remove erroneous ], inside rationale
    pattern2 = r'"rationale":\s*"([^"]*)\s*\],([^"]*)"'
    cleaned = re.sub(pattern2, r'"rationale": "\1\2"', cleaned)
    
    # 5. Detect and warn on truncated JSON
    if not cleaned.strip().endswith('}') and not cleaned.strip().endswith(']'):
        print("Warning: JSON appears to be truncated")
        # Attempt basic closing (simple cases only)
        if cleaned.count('{') > cleaned.count('}'):
            # More opening braces than closing
            missing_braces = cleaned.count('{') - cleaned.count('}')
            cleaned += '}' * missing_braces
        if cleaned.count('[') > cleaned.count(']'):
            # More opening brackets than closing
            missing_brackets = cleaned.count('[') - cleaned.count(']')
            cleaned += ']' * missing_brackets
    
    return cleaned

def _attempt_json_recovery(json_str):
    """Attempt additional recovery for malformed JSON."""
    import re
    import json
    
    # 1. Retry basic cleaning
    cleaned = _clean_malformed_json(json_str)
    
    # 2. Detect and recover incomplete JSON structure
    
    # 2-1. results array started but not completed
    if '"results"' in cleaned and cleaned.count('[') > cleaned.count(']'):
        # Locate results array and process remaining content
        results_match = re.search(r'"results":\s*\[', cleaned)
        if results_match:
            # Check content after start of results array
            start_pos = results_match.end()
            content_after_results = cleaned[start_pos:]
            
            # Attempt to extract only complete groups
            groups = []
            brace_count = 0
            current_group = ""
            in_string = False
            escape_next = False
            
            for char in content_after_results:
                if escape_next:
                    current_group += char
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    current_group += char
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    
                current_group += char
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        # Found a complete group
                        if brace_count == 0 and current_group.strip():
                            # Remove leading comma
                            group_content = current_group.strip()
                            if group_content.startswith(','):
                                group_content = group_content[1:].strip()
                            
                            try:
                                # Attempt to parse as single group
                                group_obj = json.loads(group_content)
                                groups.append(group_obj)
                                current_group = ""
                            except:
                                pass
                    elif char == ',' and brace_count == 0:
                        # Group separator
                        current_group = ""
            
            if groups:
                # Build new JSON from recovered groups
                recovered = json.dumps({"results": groups})
                print(f"Recovered {len(groups)} complete groups from truncated JSON")
                return recovered
    
    # 3. Attempt simple structural recovery
    try:
        # Attempt basic JSON closing
        if cleaned.strip():
            # JSON starts with { but does not end with }
            if cleaned.strip().startswith('{') and not cleaned.strip().endswith('}'):
                # Simple closing attempt
                test_json = cleaned.strip() + '}'
                json.loads(test_json)  # validation
                return test_json
            
            # Array starts with [ but does not end with ]  
            if cleaned.strip().startswith('[') and not cleaned.strip().endswith(']'):
                test_json = cleaned.strip() + ']'
                json.loads(test_json)  # validation
                return test_json
                
    except:
        pass
    
    # 4. Attempt to extract results from partial JSON
    results_pattern = r'"results":\s*\[(.*?)(?:\]|}|$)'
    match = re.search(results_pattern, cleaned, re.DOTALL)
    if match:
        try:
            results_content = match.group(1).strip()
            if results_content:
                # Remove last incomplete object
                if results_content.endswith(','):
                    results_content = results_content[:-1]
                
                test_json = '{"results": [' + results_content + ']}'
                json.loads(test_json)  # validation
                return test_json
        except:
            pass
    
    print("Could not recover JSON structure")
    return None

def _validate_content_structure(content_obj):
    """Validate that content_obj has the expected structure."""
    if not isinstance(content_obj, dict):
        return False
        
    # Check that 'results' key exists and is a list
    if 'results' not in content_obj:
        return False
        
    results = content_obj['results']
    if not isinstance(results, list):
        return False
        
    # Verify each result has the required structure
    for result in results:
        if not isinstance(result, dict):
            continue
            
        required_fields = ['group_name', 'findings', 'episodes', 'rationale']
        if not all(field in result for field in required_fields):
            return False
            
        # Verify findings and episodes are lists
        if not isinstance(result['findings'], list) or not isinstance(result['episodes'], list):
            return False
            
    return True

def _fix_content_structure(content_obj):
    """Repair the structure of content_obj."""
    if not isinstance(content_obj, dict):
        return content_obj
        
    # If no results key, wrap entire object as results
    if 'results' not in content_obj:
        # Object is a dictionary of finding groups
        results = []
        for key, value in content_obj.items():
            if isinstance(value, dict):
                # Use key as group_name if not present
                if 'group_name' not in value:
                    value['group_name'] = key
                results.append(value)
            elif isinstance(value, list):
                # If list, treat each item as a separate group
                for item in value:
                    if isinstance(item, dict):
                        if 'group_name' not in item:
                            item['group_name'] = key
                        results.append(item)
        content_obj = {'results': results}
    
    # Fix structure of each result
    results = content_obj.get('results', [])
    fixed_results = []
    
    for result in results:
        if not isinstance(result, dict):
            continue
            
        fixed_result = {}
        
        # Set required fields
        fixed_result['group_name'] = result.get('group_name', 'unnamed_group')
        fixed_result['findings'] = result.get('findings', [])
        fixed_result['episodes'] = result.get('episodes', [])
        fixed_result['rationale'] = result.get('rationale', '')
        
        # Fix findings/episodes if not lists
        if not isinstance(fixed_result['findings'], list):
            fixed_result['findings'] = []
            
        if not isinstance(fixed_result['episodes'], list):
            fixed_result['episodes'] = []
            
        fixed_results.append(fixed_result)
    
    content_obj['results'] = fixed_results
    return content_obj

def _fix_episodes_structure(episodes, group_name):
    """Validate and repair the structure of an episodes list."""
    if not episodes:
        return []
    
    fixed_episodes = []
    
    for i, episode in enumerate(episodes):
        try:
            if isinstance(episode, dict):
                # Check required fields
                episode_num = episode.get('episode', i + 1)
                days = episode.get('days', [])
                
                # Fix days if not a list
                if not isinstance(days, list):
                    if isinstance(days, (int, float)):
                        days = [int(days)]
                    else:
                        print(f"Warning: Invalid days format in episode {episode_num} for group {group_name}")
                        days = []
                
                # Fix episode number if not numeric
                if not isinstance(episode_num, (int, float)):
                    episode_num = i + 1
                
                fixed_episodes.append({
                    'episode': int(episode_num),
                    'days': [int(day) for day in days if isinstance(day, (int, float))]
                })
            else:
                # Attempt to handle string or other format
                parsed_episode = None
                episode_str = str(episode)

                # Parse episode number
                ep_match = re.search(r'episode\s*=\s*(\d+)', episode_str, re.IGNORECASE)
                if ep_match:
                    try:
                        episode_num = int(ep_match.group(1))
                    except ValueError:
                        episode_num = i + 1

                # Parse days
                days_match = re.search(r'days\s*=\s*\[([^\]]*)\]', episode_str, re.IGNORECASE)
                days_list: List[int] = []
                if days_match:
                    raw_days = days_match.group(1)
                    days_list = [
                        int(day_str)
                        for day_str in re.findall(r'-?\d+', raw_days)
                    ]
                else:
                    # Handle case where days has no brackets
                    days_list = [int(day_str) for day_str in re.findall(r'-?\d+', episode_str)]

                if days_list:
                    parsed_episode = {
                        'episode': int(episode_num),
                        'days': days_list
                    }

                if parsed_episode:
                    fixed_episodes.append(parsed_episode)
                else:
                    print(f"Warning: Episode is not a dict for group {group_name}: {episode}")
                    fixed_episodes.append({
                        'episode': i + 1,
                        'days': []
                    })
        except Exception as e:
            print(f"Error fixing episode structure for group {group_name}: {e}")
            fixed_episodes.append({
                'episode': i + 1,
                'days': []
            })
    
    return fixed_episodes

def process_radiology_output(response, input_data=None, is_missing_process=False, clustered_df=None):
    flattened_data = []
    
    stats = {
        'total_input_observations': 0,
        'matched_observations': 0,
        'unmatched_observations': 0
    }
    
    subject_id = input_data.get("subject_id", "") if input_data else ""
    content_obj = None
    try:
        if isinstance(response, str):
            try:
                # Fix potential errors before JSON parsing
                cleaned_response = _clean_malformed_json(response)
                content_obj = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"Error: Could not parse response as JSON after cleaning: {e}")
                
                # Additional recovery attempt
                try:
                    recovered_json = _attempt_json_recovery(response)
                    if recovered_json:
                        content_obj = json.loads(recovered_json)
                        print("Successfully recovered JSON after additional processing")
                    else:
                        return pd.DataFrame(), stats
                except Exception as recovery_error:
                    print(f"JSON recovery also failed: {recovery_error}")
                    return pd.DataFrame(), stats
        elif hasattr(response, 'root'):
            content_obj = response.root
        elif isinstance(response, dict):
            content_obj = response
        elif isinstance(response, RadiologyOutput):
            # Handle RadiologyOutput type
            content_obj = {"results": response.results}
        else:
            print(f"Error: Unsupported response type: {type(response)}")
            return pd.DataFrame(), stats
            
        if not content_obj:
            print("Error: No content object found in response")
            return pd.DataFrame(), stats
            
        # Additional validation: check content_obj has expected structure
        if not _validate_content_structure(content_obj):
            print("Warning: Content object has unexpected structure, attempting to fix...")
            content_obj = _fix_content_structure(content_obj)
            
    except Exception as e:
        print(f"Error processing response: {e}")
        return pd.DataFrame(), stats

    if not is_missing_process:
        all_input_observations = {}
        subject_unmatched = []
        
        if input_data:
            try:
                if isinstance(input_data["Input"], str):
                    input_json = json.loads(input_data["Input"])
                else:
                    input_json = input_data["Input"]
                    
                observations = input_json.get("observations", {})
                ent_idx_dict = input_data.get("Input_ent_idx", {})
                seq_dict = input_data.get("Input_seq", {})
                sent_idx_dict = input_data.get("Input_sent_idx", {})
                sent_dict = input_data.get("Input_sent", {})
                ent_dict = input_data.get("Input_ent", {})
                study_id_dict = input_data.get("Input_study_id", {})
                section_dict = input_data.get("Input_section", {})
                
                for idx_day_str, findings in observations.items():
                    idx_day_str = idx_day_str.split(", status")[0]
                    
                    for finding in findings:
                        
                        obs_key = f"{idx_day_str}|{finding}"
                        
                        ent_idx_val = ent_idx_dict.get(idx_day_str, -1)
                        seq_val = seq_dict.get(idx_day_str, -1)
                        sent_idx_val = sent_idx_dict.get(idx_day_str, None)
                        sent_val = sent_dict.get(idx_day_str, None)
                        ent_val = ent_dict.get(idx_day_str, None)
                        study_id_val = study_id_dict.get(idx_day_str, None)
                        section_val = section_dict.get(idx_day_str, None)
                        
                        all_input_observations[obs_key] = {
                            "idx_day_str": idx_day_str,
                            "finding": finding,
                            "ent_idx": ent_idx_val,
                            "seq": seq_val,
                            "sent_idx": sent_idx_val,
                            "sent": sent_val,
                            "ent": ent_val,
                            "study_id": study_id_val,
                            "section": section_val,
                            "matched": False,
                            "batch_idx": input_data.get("batch_idx", 0),
                            "group_name": input_data.get("Group_name", "")
                        }
                        
                        stats['total_input_observations'] += 1
            except Exception as e:
                print(f"Error parsing input data: {e}")
        
        if 'results' in content_obj:
            findings_list = content_obj.get('results', [])
            
            for finding_group in findings_list:
                # Extract data from the finding group with improved error handling
                try:
                    if isinstance(finding_group, dict):
                        group_name = finding_group.get('group_name', 'unnamed_group')
                        findings = finding_group.get('findings', [])
                        episodes = finding_group.get('episodes', [])
                        rationale = finding_group.get('rationale', '')
                    else:  # Pydantic model
                        group_name = getattr(finding_group, 'group_name', 'unnamed_group')
                        findings = getattr(finding_group, 'findings', [])
                        episodes = getattr(finding_group, 'episodes', [])
                        rationale = getattr(finding_group, 'rationale', '')
                    
                    # Additional validation and repair
                    if not isinstance(findings, list):
                        print(f"Warning: findings is not a list for group {group_name}, converting...")
                        findings = [] if findings is None else [findings]
                    
                    if not isinstance(episodes, list):
                        print(f"Warning: episodes is not a list for group {group_name}, converting...")
                        episodes = [] if episodes is None else [episodes]
                    
                    # Validate and repair episodes structure
                    episodes = _fix_episodes_structure(episodes, group_name)
                    
                except Exception as e:
                    print(f"Error processing finding group: {e}")
                    continue
                
                for finding in findings:
                    temporal_group = 1  # Default value
                    
                    # Safe attribute access with improved validation
                    try:
                        if isinstance(finding, dict):
                            finding_IDX = finding.get('IDX', -1)
                            finding_day = finding.get('DAY', -1)
                            finding_text = finding.get('finding', '')
                        else:  # Pydantic model
                            finding_IDX = getattr(finding, 'IDX', -1)
                            finding_day = getattr(finding, 'DAY', -1)
                            finding_text = getattr(finding, 'finding', '')
                        
                        # Validate and fix data types
                        finding_IDX = int(finding_IDX) if isinstance(finding_IDX, (int, float, str)) and str(finding_IDX).isdigit() else -1
                        finding_day = int(finding_day) if isinstance(finding_day, (int, float, str)) and str(finding_day).isdigit() else -1
                        finding_text = str(finding_text) if finding_text is not None else ''
                        
                    except Exception as e:
                        print(f"Error processing finding in group {group_name}: {e}")
                        continue
                    
                    # Find matching date in episodes
                    for ep in episodes:
                        if isinstance(ep, dict):
                            episode_days = ep.get('days', [])
                            if finding_day in episode_days:
                                temporal_group = ep.get('episode', -1)
                                break
                        else:  # Pydantic model
                            episode_days = getattr(ep, 'days', [])
                            if finding_day in episode_days:
                                temporal_group = getattr(ep, 'episode', -1)
                                break
                    
                    idx_day_str = f"IDX:{finding_IDX}, DAY: {finding_day}"
                    obs_key = f"{idx_day_str}|{finding_text}"
                    
                    if obs_key in all_input_observations:
                        all_input_observations[obs_key]["matched"] = True
                        stats['matched_observations'] += 1
                        
                        matched_obs = all_input_observations[obs_key]
                        sent_idx_val = matched_obs.get("sent_idx", None)
                        sent_val = matched_obs.get("sent", None)
                        section_val = matched_obs.get("section", None)

                        data_row = {
                            'subject_id': subject_id,
                            'study_id': matched_obs.get("study_id"),
                            'batch_idx': matched_obs["batch_idx"],
                            'LLM_cluster': group_name,
                            'episodes': str(episodes),
                            'rationale': rationale,
                            'ent_idx': matched_obs["ent_idx"],
                            'IDX': finding_IDX,
                            'sequence': matched_obs["seq"],
                            'sent_idx': sent_idx_val,
                            'sent': sent_val,
                            'ent': matched_obs.get("ent"),
                            'temporal_group': temporal_group,
                            'DAY': finding_day,
                            'finding': finding_text,
                            'status': 'matched',
                            'section': section_val
                        }
                        
                        if input_data and input_data.get("Group_name"):
                            data_row["1st_cluster"] = input_data["Group_name"]
                        
                        flattened_data.append(data_row)
        else:
            raise ValueError("No SCHEMA or PYDANTIC ERROR found in content_obj")
        
        for obs_key, obs_data in all_input_observations.items():
            if not obs_data["matched"]:
                stats['unmatched_observations'] += 1
                
                idx_day_parts = obs_data["idx_day_str"].split(", DAY: ")
                idx_day_val = idx_day_parts[1] if len(idx_day_parts) > 1 else ""
                
                IDX_parts = idx_day_parts[0].split(":")
                llm_IDX_val = IDX_parts[1] if len(IDX_parts) > 1 else -1
                
                # Get section from input_data
                section_val = obs_data.get("section", None)
                
                unmatched_info = {
                    'subject_id': subject_id,
                    'batch_idx': obs_data["batch_idx"],
                    'idx_day_str': obs_data["idx_day_str"],
                    'DAY': idx_day_val,
                    'finding': obs_data["finding"],
                    'ent_idx': obs_data["ent_idx"],
                    'seq': obs_data["seq"],
                    'sent_idx': obs_data.get("sent_idx", None),
                    'sent': obs_data.get("sent", None),
                    'status': 'not_in_output'
                }
                
                subject_unmatched.append(unmatched_info)
                
                unmatched_row = {
                    'subject_id': subject_id,
                    'study_id': obs_data.get("study_id"),
                    'batch_idx': obs_data["batch_idx"],
                    'LLM_cluster': 'UNMATCHED',
                    'episodes': '[]',
                    'rationale': '',
                    'IDX': llm_IDX_val,
                    'ent_idx': obs_data["ent_idx"],
                    'sequence': obs_data["seq"],
                    'sent_idx': obs_data.get("sent_idx", None),
                    'sent': obs_data.get("sent", None),
                    'ent': obs_data.get("ent"),
                    'temporal_group': -1,
                    'DAY': idx_day_val,
                    'finding': obs_data["finding"],
                    'status': 'unmatched',
                    'section': section_val
                }
                
                if input_data and input_data.get("Group_name"):
                    unmatched_row["1st_cluster"] = input_data["Group_name"]
                
                flattened_data.append(unmatched_row)
        
        if subject_unmatched:
            pass
    
    else:
        missing_tracking = {
            'total': 0,
            'matched': 0,
            'unmatched': 0
        }
        
        processed_findings = set()
        cluster_name = input_data.get("Group_name", "") if input_data else ""

        if 'results' in content_obj:
            findings_list = content_obj['results']
            
            for finding_group in findings_list:
                # Safe attribute access with improved error handling
                try:
                    if isinstance(finding_group, dict):
                        group_name = finding_group.get('group_name', 'unnamed_group')
                        findings = finding_group.get('findings', [])
                        episodes = finding_group.get('episodes', [])
                        rationale = finding_group.get('rationale', '')
                    else:  # Pydantic model
                        group_name = getattr(finding_group, 'group_name', 'unnamed_group')
                        findings = getattr(finding_group, 'findings', [])
                        episodes = getattr(finding_group, 'episodes', [])
                        rationale = getattr(finding_group, 'rationale', '')
                    
                    # Additional validation and repair (same logic as the main process)
                    if not isinstance(findings, list):
                        print(f"Warning: findings is not a list for group {group_name}, converting...")
                        findings = [] if findings is None else [findings]
                    
                    if not isinstance(episodes, list):
                        print(f"Warning: episodes is not a list for group {group_name}, converting...")
                        episodes = [] if episodes is None else [episodes]
                    
                    # Validate and repair episodes structure
                    episodes = _fix_episodes_structure(episodes, group_name)
                    
                except Exception as e:
                    print(f"Error processing finding group in missing process: {e}")
                    continue
                
                # Process findings
                for finding in findings:
                    missing_tracking['total'] += 1
                    stats['total_input_observations'] += 1
                    
                    # Safe attribute access with improved validation
                    try:
                        if isinstance(finding, dict):
                            finding_idx = finding.get('IDX', -1)
                            finding_day = finding.get('DAY', -1)
                            finding_text = finding.get('finding', '')
                        else:  # Pydantic model
                            finding_idx = getattr(finding, 'IDX', -1)
                            finding_day = getattr(finding, 'DAY', -1)
                            finding_text = getattr(finding, 'finding', '')
                        
                        # Validate and fix data types
                        finding_idx = int(finding_idx) if isinstance(finding_idx, (int, float, str)) and str(finding_idx).isdigit() else -1
                        finding_day = int(finding_day) if isinstance(finding_day, (int, float, str)) and str(finding_day).isdigit() else -1
                        finding_text = str(finding_text) if finding_text is not None else ''
                        
                    except Exception as e:
                        print(f"Error processing finding in missing process for group {group_name}: {e}")
                        print(f"Finding data: {finding}")
                        continue
                    
                    # Find episode information
                    temporal_group = 1
                    for ep in episodes:
                        if isinstance(ep, dict):
                            episode_days = ep.get('days', [])
                            if finding_day in episode_days:
                                temporal_group = ep.get('episode', -1)
                                break
                        else:  # Pydantic model
                            episode_days = getattr(ep, 'days', [])
                            if finding_day in episode_days:
                                temporal_group = getattr(ep, 'episode', -1)
                                break
                    
                    idx_day_str = f"IDX:{finding_idx}, DAY: {finding_day}"
                    finding_key = f"{idx_day_str}|{finding_text}"
                    
                    if finding_key in processed_findings:
                        continue
                    
                    processed_findings.add(finding_key)
                    missing_tracking['matched'] += 1
                    
                    # Try to find sent_idx and sent from clustered_df
                    sent_idx_val = None
                    sent_val = None
                    ent_val = None
                    study_id_val = None
                    section_val = None
                    if clustered_df is not None:
                        filter_mask = (
                            (clustered_df['subject_id'] == subject_id) &
                            (clustered_df['cluster_name'] == cluster_name) &
                            (clustered_df['ELA_cur_ent'] == finding_text)
                        )
                        matched_rows = clustered_df[filter_mask]
                        if len(matched_rows) > 0:
                            first_row = matched_rows.iloc[0]
                            sent_idx_val = first_row['sent_idx'] if 'sent_idx' in matched_rows.columns else None
                            sent_val = first_row['sent'] if 'sent' in matched_rows.columns else None
                            ent_val = first_row['ent'] if 'ent' in matched_rows.columns else first_row['entity'] if 'entity' in matched_rows.columns else None
                            study_id_val = first_row['study_id'] if 'study_id' in matched_rows.columns else None
                            section_val = first_row['section'] if 'section' in matched_rows.columns else None
                    
                    data_row = {
                        'subject_id': subject_id,
                        'study_id': study_id_val,
                        'batch_idx': input_data.get('batch_idx', ''),
                        'LLM_cluster': group_name,
                        'episodes': str(episodes),
                        'rationale': rationale,
                        'ent_idx': -1,
                        'IDX': finding_idx,
                        'sequence': -1,
                        'sent_idx': sent_idx_val,
                        'sent': sent_val,
                        'ent': ent_val,
                        'temporal_group': temporal_group,
                        'DAY': finding_day,
                        'finding': finding_text,
                        'status': 'matched',
                        'section': section_val
                    }
                    
                    data_row["1st_cluster"] = cluster_name
                    
                    flattened_data.append(data_row)
        else:            
            for group_name, group_data in content_obj.items():
                if isinstance(group_data, str):
                    print(f"Warning: Group data is a string: {group_data[:100]}...")
                    continue
                    
                if hasattr(group_data, 'timeframe'):
                    timeframe = group_data.timeframe
                    rationale = group_data.rationale
                    episodes = group_data.episodes
                    findings = group_data.findings
                else:
                    if isinstance(group_data, list):
                        print(f"Warning: group_data is a list with {len(group_data)} items, using first item")
                        if group_data and isinstance(group_data[0], dict):
                            group_data = group_data[0]
                        else:
                            print(f"Warning: Cannot process group_data list: {group_data[:2]}...")
                            continue

                    timeframe = group_data.get('timeframe', '')
                    rationale = group_data.get('rationale', '')
                    episodes = group_data.get('episodes', [])
                    findings = group_data.get('findings', [])
                
                for finding in findings:
                    missing_tracking['total'] += 1
                    stats['total_input_observations'] += 1
                    
                    try:
                        if hasattr(finding, 'DAY'):
                            finding_idx = finding.IDX
                            finding_day = finding.DAY
                            finding_text = finding.finding
                        else:
                            finding_idx = finding.get('IDX', -1)
                            finding_day = finding.get('DAY', -1)
                            finding_text = finding.get('finding', '')
                        
                        # Validate and fix data types
                        finding_idx = int(finding_idx) if isinstance(finding_idx, (int, float, str)) and str(finding_idx).isdigit() else -1
                        finding_day = int(finding_day) if isinstance(finding_day, (int, float, str)) and str(finding_day).isdigit() else -1
                        finding_text = str(finding_text) if finding_text is not None else ''
                        
                    except Exception as e:
                        print(f"Error processing finding in alternative structure for group {group_name}: {e}")
                        print(f"Finding data: {finding}")
                        continue
                        
                    temporal_group = 1
                    for ep in episodes:
                        if hasattr(ep, 'days'):
                            episode_days = ep.days
                            if finding_day in episode_days:
                                temporal_group = ep.episode
                                break
                        else:
                            episode_days = ep.get('days', [])
                            if finding_day in episode_days:
                                temporal_group = ep.get('episode', -1)
                                break
                    
                    idx_day_str = f"IDX:{finding_idx}, DAY: {finding_day}"
                    finding_key = f"{idx_day_str}|{finding_text}"
                    
                    if finding_key in processed_findings:
                        continue
                    
                    processed_findings.add(finding_key)
                    missing_tracking['matched'] += 1
                    
                    # Try to find sent_idx and sent from clustered_df
                    sent_idx_val = None
                    sent_val = None
                    ent_val = None
                    study_id_val = None
                    section_val = None
                    if clustered_df is not None:
                        filter_mask = (
                            (clustered_df['subject_id'] == subject_id) &
                            (clustered_df['cluster_name'] == cluster_name) &
                            (clustered_df['ELA_cur_ent'] == finding_text)
                        )
                        matched_rows = clustered_df[filter_mask]
                        if len(matched_rows) > 0:
                            first_row = matched_rows.iloc[0]
                            sent_idx_val = first_row['sent_idx'] if 'sent_idx' in matched_rows.columns else None
                            sent_val = first_row['sent'] if 'sent' in matched_rows.columns else None
                            ent_val = first_row['ent'] if 'ent' in matched_rows.columns else first_row['entity'] if 'entity' in matched_rows.columns else None
                            study_id_val = first_row['study_id'] if 'study_id' in matched_rows.columns else None
                            section_val = first_row['section'] if 'section' in matched_rows.columns else None
                    
                    data_row = {
                        'subject_id': subject_id,
                        'study_id': study_id_val,
                        'batch_idx': input_data.get('batch_idx', ''),
                        'LLM_cluster': group_name,
                        'episodes': str(episodes),
                        'rationale': rationale,
                        'ent_idx': -1,
                        'IDX': finding_idx,
                        'sequence': -1,
                        'sent_idx': sent_idx_val,
                        'sent': sent_val,
                        'ent': ent_val,
                        'temporal_group': temporal_group,
                        'DAY': finding_day,
                        'finding': finding_text,
                        'status': 'matched',
                        'section': section_val
                    }
                    
                    data_row["1st_cluster"] = cluster_name
                    
                    flattened_data.append(data_row)
            
        # Print missing statistics
        print(f"  Missing statistics:")
        print(f"  - Total observations: {missing_tracking['total']}")
        print(f"  - Matched: {missing_tracking['matched']}")
        print(f"  - Unmatched: {missing_tracking['total'] - missing_tracking['matched']} \n")
        
        # Important: recalculate full statistics after missing processing
        stats['matched_observations'] = missing_tracking['matched']
        stats['unmatched_observations'] = missing_tracking['total'] - missing_tracking['matched']
    
    # Print statistics
    print("=== Statistics ===")
    print(f"Total input observations: {stats['total_input_observations']}")
    print(f"Matched observations: {stats['matched_observations']}")
    print(f"Unmatched observations: {stats['unmatched_observations']}")
    
    # Create and sort dataframe
    df = pd.DataFrame(flattened_data)
    if not df.empty:
        # Add missing required columns
        if '1st_cluster' not in df.columns:
            df['1st_cluster'] = input_data.get("Group_name", "") if input_data else ""
        sort_columns = ['subject_id', 'LLM_cluster', 'temporal_group', 'DAY', 'IDX']
        sort_columns = [col for col in sort_columns if col in df.columns]
        if sort_columns:
            df = df.sort_values(sort_columns)
    else:
        # Add required columns if dataframe is empty
        df = pd.DataFrame(columns=[
            'subject_id', 'batch_idx', '1st_cluster', 'LLM_cluster',
            'episodes', 'rationale', 'ent_idx', 'IDX', 'sequence',
            'sent_idx', 'sent', 'temporal_group', 'DAY', 'finding', 'status', 'section'
        ])
    
    return df, stats

def convert_sets_to_lists(obj):
    """Recursively convert sets to lists in nested structures"""
    if isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj

def read_batch_results_to_csv(args, batch_results_file, clustered_df, all_model_outputs, is_missing_process=False, existing_output_df=None):
    print("\nTemporal mapping of elements within each cluster:")        
    line_count = 0
    success_count = 0
    
    # For missing process: load or receive existing output_df
    if is_missing_process and existing_output_df is not None and not existing_output_df.empty:
        output_df = existing_output_df.copy()
        print(f"Loaded existing output_df with {len(output_df)} rows for updating in batch processing")
    else:
        output_df = pd.DataFrame()
    
    print("batch_results_file", batch_results_file)

    with open(batch_results_file, 'r') as file:
        for line in file:
            line_count += 1
            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSONL line {line_count}: {e}")
                print(f"Skipping problematic line: {line[:100]}...")
                continue
            
            custom_id = result.get('custom_id', '')
            parts = custom_id.split('_')
            if len(parts) >= 2:
                subject_id = parts[0]
                batch_idx = parts[1]
                print(f"\nsubject_id {subject_id}, batch_idx {batch_idx}")
            else:
                print(f"Warning: Unexpected custom_id format: {custom_id}")
                continue
            
            try:
                try:
                    batch_idx_key = int(batch_idx)
                except ValueError:
                    batch_idx_key = batch_idx
                
                if subject_id not in all_model_outputs:
                    print(f"Warning: Subject ID {subject_id} not found in model outputs")
                    continue
                    
                if batch_idx_key not in all_model_outputs[subject_id]:
                    if isinstance(batch_idx_key, int) and str(batch_idx_key) in all_model_outputs[subject_id]:
                        batch_idx_key = str(batch_idx_key)
                    else:
                        print(f"Warning: Batch index {batch_idx} not found for subject {subject_id}")
                        print(f"Available keys: {list(all_model_outputs[subject_id].keys())}")
                        continue
                
                input_data = {
                    "batch_idx": batch_idx,
                    "subject_id": subject_id,
                    "Input": all_model_outputs[subject_id][batch_idx_key]['Input'],
                    "Input_ent_idx": all_model_outputs[subject_id][batch_idx_key]['Input_ent_idx'],
                    "Input_seq": all_model_outputs[subject_id][batch_idx_key]['Input_seq'],
                    "Input_sent_idx": all_model_outputs[subject_id][batch_idx_key].get('Input_sent_idx', {}),
                    "Input_sent": all_model_outputs[subject_id][batch_idx_key].get('Input_sent', {}),
                    "Input_ent": all_model_outputs[subject_id][batch_idx_key].get('Input_ent', {}),
                    "Input_study_id": all_model_outputs[subject_id][batch_idx_key].get('Input_study_id', {}),
                    "Input_section": all_model_outputs[subject_id][batch_idx_key].get('Input_section', {}),
                    "Group_name": all_model_outputs[subject_id][batch_idx_key]['Group_name'],
                }                

                content = None
                if "content" in result:
                    content = result.get("content", "")
                else:

                    response = result.get('response', {})
                    body = response.get('body', {})
                    
                    if isinstance(body, str):

                        try:
                            body = json.loads(body)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse body as JSON for custom_id {custom_id}")
                            continue
                    
                    choices = body.get('choices', [])
                    
                    if not choices or len(choices) == 0:
                        print(f"Warning: No choices found for custom_id {custom_id}")
                        continue
                    
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                
                if not content:
                    print(f"Warning: No content found for custom_id {custom_id}")
                    continue

                try:
                    llm_output_df, stats = process_radiology_output(content, input_data, is_missing_process, clustered_df=clustered_df)
                    # Missing process: merge old and new results, keeping the latest on duplicates
                    output_df = pd.concat([output_df, llm_output_df])
                    success_count += 1
                except Exception as e:
                    print(f"Error processing content: {e}")
                    import traceback
                    print(f"Detailed error: {traceback.format_exc()}")
                    continue
                
            except Exception as e:
                print(f"Error extracting content: {e}")
                print(f"Result structure: {list(result.keys())}")
                continue
                
        if is_missing_process:
            output_path = f"{args.output_path}/missing_outputs.csv"
        else:
            output_path = f"{args.output_path}/output_df.csv"
        
        if not output_df.empty:
            try:
                output_df = output_df.drop_duplicates(keep='last')
                output_df.to_csv(output_path, index=False)
                print(f"Successfully saved output to {output_path}")
            except Exception as e:
                print(f"Error saving output: {e}")
                
    print(f"\nProcessed {line_count} lines, successfully parsed {success_count}")
    
    return output_df

def creating_batch_file(clustered_df, args, missing_data=None, all_inputs=None, iteration=0):   
    print(f"\n Creating batch file for {args.LLM_name}, number of subject_id: {clustered_df.subject_id.nunique()}")
    if all_inputs is None:
        all_inputs = {}

    tasks = []
    if missing_data is None:
        for subject_id in clustered_df.subject_id.unique():
            print(f'Patient: "{subject_id}"')    
            all_inputs[subject_id] = {}  # Initialize dict for this subject

            print("number of groups: ", len(clustered_df[clustered_df['subject_id']==subject_id].cluster_name.unique()))
            
            for group_idx, group_name in enumerate(clustered_df[clustered_df['subject_id']==subject_id].cluster_name.unique()):
                print(f'  Group: "{group_name}"')
                
                # Get the subset of data for this subject and group
                cur_group_df = clustered_df[(clustered_df['subject_id']==subject_id)&
                            (clustered_df['cluster_name']==group_name)]
                
                if group_idx not in all_inputs[subject_id]:
                    all_inputs[subject_id][group_idx] = {}
                # Debug print
                print(f"Number of observations in group: {len(cur_group_df)}, std_len: {len(cur_group_df['study_id'].unique())}")
                        
                # Exclude groups with only one observation
                if len(cur_group_df["ELA_cur_ent"]) <= 1:
                    print(f"Skipping group {group_name} - only has {len(cur_group_df)} observation(s)")
                    continue
                
                # Create dictionary of observations
                dict_obs, ent_idx_info, seq_info, sent_idx_info, sent_info, ent_info, study_id_info, section_info = {}, {}, {}, {}, {}, {}, {}, {}
                idx_counter = 0
                
                # Check if sent_idx and sent columns exist
                has_sent_idx_col = 'sent_idx' in cur_group_df.columns
                has_sent_col = 'sent' in cur_group_df.columns
                has_section_col = 'section' in cur_group_df.columns
                
                for idx, obs in enumerate(cur_group_df['ELA_cur_ent'].to_list()):
                    day = cur_group_df['day_from_first'].to_list()[idx]
                    status = cur_group_df['dx_status'].to_list()[idx]
                    day_num = int(day.split()[0])  # Extract the number from "X days"
                    dict_obs[f"IDX:{idx_counter}, DAY: {day_num}, status: {status}"] = [obs]  # Create single-item list for each observation
                    idx_day_key = f"IDX:{idx_counter}, DAY: {day_num}"
                    ent_idx_info[idx_day_key] = cur_group_df['ent_idx'].to_list()[idx]
                    seq_info[idx_day_key] = cur_group_df['sequence'].to_list()[idx]
                    
                    # Get sent_idx and sent if they exist in cur_group_df
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

                    if 'ent' in cur_group_df.columns:
                        ent_info[idx_day_key] = cur_group_df['ent'].to_list()[idx]
                    elif 'entity' in cur_group_df.columns:
                        ent_info[idx_day_key] = cur_group_df['entity'].to_list()[idx]
                    else:
                        ent_info[idx_day_key] = None

                    study_id_info[idx_day_key] = cur_group_df['study_id'].to_list()[idx] if 'study_id' in cur_group_df.columns else None
                    
                    # Get section if it exists in cur_group_df
                    section_value = cur_group_df['section'].to_list()[idx] if has_section_col else None
                    section_info[idx_day_key] = section_value if pd.notna(section_value) else None
                    
                    idx_counter += 1

                input_json = {
                    "cluster_name": group_name,
                    "observations": dict_obs
                }
                Input = json.dumps(input_json, ensure_ascii=False)

                
                all_inputs[subject_id][group_idx]["Input"] = Input
                all_inputs[subject_id][group_idx]["Input_ent_idx"] = ent_idx_info
                all_inputs[subject_id][group_idx]["Input_seq"] = seq_info
                all_inputs[subject_id][group_idx]["Input_sent_idx"] = sent_idx_info
                all_inputs[subject_id][group_idx]["Input_sent"] = sent_info
                all_inputs[subject_id][group_idx]["Input_ent"] = ent_info
                all_inputs[subject_id][group_idx]["Input_study_id"] = study_id_info
                all_inputs[subject_id][group_idx]["Input_section"] = section_info
                all_inputs[subject_id][group_idx]["Group_name"] = group_name
                
                conversation = generate_few_shot(Input, args)

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

                custom_id = f"{subject_id}_{group_idx}"
                schema_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_json_schema.json")
                with open(schema_file_path, 'r') as f:
                    gpt_json_schema = json.load(f)
                if args.LLM_name.startswith('gpt'):
                    task = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": args.LLM_name,
                            "temperature": 0.0,
                            "messages": formatted_conversation,
                            "response_format": { 
                                "type": "json_schema",
                                "json_schema": gpt_json_schema
                            }
                        }
                    }
                else:
                    task = {
                        "custom_id": custom_id,
                        "messages": formatted_conversation
                    }
                    
                # Verify task is properly formed
                try:
                    # Test JSON serialization
                    json.dumps(task)
                    tasks.append(task)
                    # print(f"Successfully added task for {subject_id}, group {group_name}")
                except TypeError as e:
                    print(f"Error creating task for {subject_id}, group {group_name}: {e}")
                    print(f"Problematic task structure: {task}")
        file_name = f"{args.batch_path}/llm_batch.jsonl"
        
    else:
        for subject_id, subject_clusters in missing_data.items():
            print(f"\n=== Processing missing observations for Subject ID: {subject_id} ===")
            print("missing group number of subject_id: ", len(subject_clusters))

            for group_idx, (cluster_name, cluster_content) in enumerate(subject_clusters.items()):
                print("Processing cluster:", cluster_name)

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

                custom_id = f"{subject_id}_{group_idx}"
                if args.LLM_name.startswith('gpt'):
                    task = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": args.LLM_name,
                            "temperature": 0.1,
                            "messages": formatted_conversation
                        }
                    }
                else:
                    task = {
                        "custom_id": custom_id,
                        "messages": formatted_conversation
                    }
                    
                # Verify task is properly formed
                try:
                    # Test JSON serialization
                    json.dumps(task)
                    tasks.append(task)
                except TypeError as e:
                    print(f"Error creating task for {subject_id}, group {cluster_name}: {e}")
                    print(f"Problematic task structure: {task}")
        
        file_name = f"{args.batch_path}/missing_batch{iteration}.jsonl"
    
    # Write tasks to file, ensuring all data is JSON serializable
    with open(file_name, 'w') as file:
        for task in tasks:
            try:
                json_str = json.dumps(task)
                file.write(json_str + '\n')
                # print(f"Successfully wrote task for {task['custom_id']}")
            except TypeError as e:
                print(f"Error serializing task: {e}")
                print(f"Problematic task: {task}")
                continue
    
    print("Batch file created!")
    print(f"\nSummary:")
    print(f"Total tasks created: {len(tasks)}")
    print(f"Batch file: {file_name} \n")
    return all_inputs

def prepare_missing_inputs(result_df):
    unmatched_dict = defaultdict(lambda: defaultdict(list))
    matched_dict = defaultdict(lambda: defaultdict(dict))
    
    cluster_sizes = result_df.groupby(['subject_id', 'cluster_name']).size()
    multi_obs_clusters = {(subject_id, cluster_name) 
                         for (subject_id, cluster_name), size in cluster_sizes.items() 
                         if size > 1}
    
    for _, row in result_df.iterrows():
        subject_id, cluster_name, status = row['subject_id'], row['cluster_name'], row['llm_processed']
        idx_value = row.get('IDX', -1)
        
        # Include both explicit 'unmatched' and NaN status for clusters with multiple observations
        if status == 'unmatched' or (pd.isna(status) and (subject_id, cluster_name) in multi_obs_clusters):
            unmatched_dict[subject_id][cluster_name].append({
                'DAY': row['DAY'],
                'finding': row['ELA_cur_ent'],
                'IDX': idx_value
            })

        else:
            llm_cluster = row['LLM_cluster']
            
            if llm_cluster not in matched_dict[subject_id][cluster_name]:
                try:
                    episodes_data = row['episodes']
                    processed_episodes = []
                    
                    if isinstance(episodes_data, list) and all(hasattr(ep, 'episode') for ep in episodes_data if hasattr(ep, '__dict__')):
                        for ep in episodes_data:
                            if hasattr(ep, 'episode') and hasattr(ep, 'days'):
                                processed_episodes.append({
                                    'episode': ep.episode,
                                    'days': ep.days
                                })

                    elif isinstance(episodes_data, str) and 'Episode(' in episodes_data:
                        import re
                        episode_matches = re.finditer(r'Episode\(episode=(\d+), days=\[([\d, ]+)\]\)', episodes_data)
                        for match in episode_matches:
                            episode_num = int(match.group(1))
                            days = [int(d.strip()) for d in match.group(2).split(',') if d.strip()]
                            processed_episodes.append({
                                'episode': episode_num,
                                'days': days
                            })

                    elif isinstance(episodes_data, list) and all(isinstance(ep, dict) for ep in episodes_data if ep):
                        processed_episodes = episodes_data

                    elif isinstance(episodes_data, str):
                        try:
                            parsed_data = json.loads(episodes_data)
                            if isinstance(parsed_data, list):
                                processed_episodes = parsed_data
                        except json.JSONDecodeError:
                            import ast
                            try:
                                parsed_data = ast.literal_eval(episodes_data)
                                if isinstance(parsed_data, list):
                                    processed_episodes = parsed_data
                            except (SyntaxError, ValueError):
                                processed_episodes = []
                                print(f"Warning: Could not parse episodes: {episodes_data}")
                    else:
                        processed_episodes = episodes_data
                except Exception as e: 
                    processed_episodes = []
                    print(f"Error parsing episodes: {e}")
      
                matched_dict[subject_id][cluster_name][llm_cluster] = {
                    'findings': [],
                    'episodes': processed_episodes,
                    'rationale': row['rationale']
                }
            
            matched_dict[subject_id][cluster_name][llm_cluster]['findings'].append({
                'IDX': idx_value,
                'DAY': row['DAY'],
                'finding': row['ELA_cur_ent']
            })
    
    final_inputs = {}
    
    for subject_id, clusters in unmatched_dict.items():
        subject_inputs = {}
        
        for cluster_name, unmatched_items in clusters.items():
            if not unmatched_items:
                continue
            
            unprocessed_observations = {}
            for item in unmatched_items:
                idx_day_str = f"IDX:{item['IDX']}, DAY: {item['DAY']}"
                if idx_day_str not in unprocessed_observations:
                    unprocessed_observations[idx_day_str] = []
                unprocessed_observations[idx_day_str].append(item["finding"])
            
            subject_inputs[cluster_name] = {
                "existing_results": matched_dict[subject_id].get(cluster_name, {}),
                "unprocessed_observations": unprocessed_observations,
                "cluster_name": cluster_name
            }
            
        if subject_inputs:
            final_inputs[subject_id] = subject_inputs
            
    return final_inputs

def create_missing_input(subject_id, cluster_name, result_data):
    if subject_id not in result_data or cluster_name not in result_data[subject_id]:
        return None
    
    cluster_data = result_data[subject_id][cluster_name]
    existing_results = json.dumps(cluster_data["existing_results"], indent=2, cls=NumpyEncoder)
    unprocessed_groups = json.dumps(cluster_data["unprocessed_observations"], indent=2, cls=NumpyEncoder)
    
    missing_prompt = f'''The following observations were previously missed. Your task is to:
    1. Review each missing observation
    2. If there are existing results, assign each observation to the appropriate existing group and episode when applicable
    3. If existing results are empty or not suitable for some observations, create new groups as needed
    4. Return the COMPLETE JSON for ALL groups (both existing and newly created)

    Missing_observations: {unprocessed_groups}
    Existing_results: {existing_results}
    
    Note: 
    - If existing_results is empty, you should create completely new groups for the missing observations
    - Your output should follow the original format directly, with Group Names as top-level keys
    - DO NOT wrap your output in "Missing_observations" or "Existing_results" structure
    - Return a direct JSON object with all groups in the same format as the examples you saw earlier
    '''
    return missing_prompt
 
def post_process(llm_output, clustered_df, output_path, is_missing_process=False, iteration=0):
    
    llm_output = llm_output.rename(columns={
        'finding': 'ELA_cur_ent',  # Changed to match clustered_df's column name
        'status': 'llm_processed'
    })

    # Guard: if llm_output is empty or missing required columns, skip updates
    if llm_output is None or llm_output.empty:
        print("Warning: llm_output is empty; skipping post_process updates.")
        return clustered_df

    # Ensure expected columns exist before building match keys
    if 'subject_id' not in llm_output.columns or 'ELA_cur_ent' not in llm_output.columns:
        print(f"Warning: llm_output missing required columns. Available columns: {list(llm_output.columns)}")
        return clustered_df

    if not is_missing_process:
        # Add new columns to clustered_df with default values only if they do not exist
        default_columns = {
            'LLM_cluster': None,
            'episodes': None,
            'rationale': None,
            'llm_processed': None,
            'temporal_group': None,
            'DAY': None,
            'IDX': None
        }

        for col, default_val in default_columns.items():
            if col not in clustered_df.columns:
                clustered_df[col] = default_val

        # Create a unique identifier for matching
        clustered_df['match_key'] = clustered_df['subject_id'] + '_' + clustered_df['sequence'].fillna(0).astype(int).astype(str) + '_' + clustered_df['ent_idx'].fillna(0).astype(int).astype(str) + '_' + clustered_df['ELA_cur_ent']
        llm_output['match_key'] = llm_output['subject_id'] + '_' + llm_output['sequence'].fillna(0).astype(int).astype(str) + '_' + llm_output['ent_idx'].fillna(0).astype(int).astype(str) + '_' + llm_output['ELA_cur_ent']
        
    else:
        # Create a unique identifier for matching
        clustered_df['match_key'] = clustered_df['subject_id'] + '_' + clustered_df['IDX'].fillna(0).astype(float).astype(int).astype(str) + '_' + clustered_df['ELA_cur_ent']
        if 'IDX' not in llm_output.columns:
            print(f"Warning: llm_output missing 'IDX' column in missing process. Available columns: {list(llm_output.columns)}")
            return clustered_df
        llm_output['match_key'] = llm_output['subject_id'] + '_' + llm_output['IDX'].fillna(0).astype(float).astype(int).astype(str) + '_' + llm_output['ELA_cur_ent']
    
    # Find matching records
    matching_keys = set(clustered_df['match_key']) & set(llm_output['match_key'])

    # Update matching records
    match_count = 0
    for key in matching_keys:
        try:
            llm_data = llm_output[llm_output['match_key'] == key].iloc[0]
            clustered_df.loc[clustered_df['match_key'] == key, 'llm_processed'] = llm_data['llm_processed']
            clustered_df.loc[clustered_df['match_key'] == key, 'LLM_cluster'] = llm_data['LLM_cluster']
            clustered_df.loc[clustered_df['match_key'] == key, 'episodes'] = llm_data['episodes']
            clustered_df.loc[clustered_df['match_key'] == key, 'rationale'] = llm_data['rationale']
            
            temporal_group_val = llm_data['temporal_group']
            clustered_df.loc[clustered_df['match_key'] == key, 'temporal_group'] = temporal_group_val
            clustered_df.loc[clustered_df['match_key'] == key, 'DAY'] = llm_data['DAY']
            clustered_df.loc[clustered_df['match_key'] == key, 'IDX'] = llm_data['IDX']
            
            # Update sent_idx and sent if they exist in llm_output
            if 'sent_idx' in llm_output.columns and 'sent_idx' in clustered_df.columns:
                clustered_df.loc[clustered_df['match_key'] == key, 'sent_idx'] = llm_data['sent_idx']
            if 'sent' in llm_output.columns and 'sent' in clustered_df.columns:
                clustered_df.loc[clustered_df['match_key'] == key, 'sent'] = llm_data['sent']
            
            match_count += 1
        except Exception as e:
            print(f"Error updating match for key {key}: {e}")
    
    print(f"Updated {match_count} records out of {len(matching_keys)} matching keys")
    
    # Remove temporary match_key column
    clustered_df = clustered_df.drop('match_key', axis=1)
    
    return clustered_df


def initialize_llm_client(llm_name, api_key=None, port=None):
    """
    Initialize an LLM client.

    Args:
        llm_name: Model deployment name.
        api_key:  API key, or 'local_LLM' for a local vLLM server.
                  Falls back to the API_KEY env var if not provided.
        port:     vLLM server port. Falls back to the PORT env var if not provided.

    Supported backends:
        - Local vLLM:       api_key='local_LLM', port=<port>
        - OpenAI:           llm_name starts with 'gpt'
        - Fireworks AI:     llm_name starts with 'deepseek', 'llama4', or 'qwen3'
        - Anthropic Claude: llm_name starts with 'claude'  (requires: pip install anthropic)
          Returns a raw anthropic.Anthropic client (used directly by batch job functions).
        - MedGemma API:     llm_name starts with 'medgemma' (Google Gemini-compatible endpoint)
        - Baichuan:         llm_name starts with 'baichuan'
    """
    resolved_api_key = api_key or os.getenv('API_KEY')
    if resolved_api_key == 'local_LLM':
        resolved_port = port or os.getenv('PORT')
        if not resolved_port:
            raise EnvironmentError(
                "vLLM server port is not specified. "
                "Pass port= to initialize_llm_client(), set SequentialSRConfig(port=8100), "
                "or export PORT=8100 before running."
            )
        return instructor.from_openai(
            OpenAI(api_key=resolved_api_key, base_url=f"http://localhost:{resolved_port}/v1"),
            mode=instructor.Mode.JSON,
        ), None
    elif llm_name.startswith('gpt'):
        return instructor.from_openai(OpenAI(api_key=resolved_api_key), mode=instructor.Mode.JSON), None
    elif llm_name.startswith('deepseek') or llm_name.startswith('llama4') or llm_name.startswith('qwen3'):
        return instructor.from_openai(
            OpenAI(api_key=resolved_api_key, base_url="https://api.fireworks.ai/inference/v1"),
            mode=instructor.Mode.JSON,
        ), None
    elif llm_name.startswith('claude'):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for Claude models. "
                "Install it with: pip install anthropic  or  pip install -e '.[claude]'"
            )
        # Wrap with instructor so runner.py can call client.messages.create(response_model=...)
        return instructor.from_anthropic(anthropic.Anthropic(api_key=resolved_api_key)), None
    elif llm_name.startswith('medgemma'):
        # MedGemma via Google Gemini OpenAI-compatible endpoint
        return instructor.from_openai(
            OpenAI(api_key=resolved_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            mode=instructor.Mode.JSON,
        ), None
    elif llm_name.startswith('baichuan'):
        return instructor.from_openai(
            OpenAI(api_key=resolved_api_key, base_url="https://api.baichuan-ai.com/v1"),
            mode=instructor.Mode.JSON,
        ), None
    else:
        raise ValueError(
            f"Unsupported LLM name: '{llm_name}'. "
            "Supported prefixes: 'gpt', 'deepseek', 'llama4', 'qwen3', 'claude', 'medgemma', 'baichuan'. "
            "For local vLLM, set api_key='local_LLM' and provide a port."
        )

def generate_few_shot(Input, args):
    if args.LLM_name.startswith('claude'):
        return [
            {"role": "user",
             "content": [{"type": "text", "text": Result_Review_w_time_gap_Ex1['Input']}]},
            {"role": "assistant",
             "content": json.dumps(Result_Review_w_time_gap_Ex1['Output'])},
            {"role": "user",
             "content": [{"type": "text", "text": Result_Review_w_time_gap_Ex2['Input']}]},
            {"role": "assistant",
             "content": json.dumps(Result_Review_w_time_gap_Ex2['Output'])},
            {"role": "user",
             "content": [{"type": "text", "text": Result_Review_w_time_gap_Ex3['Input']}]},
            {"role": "assistant",
             "content": json.dumps(Result_Review_w_time_gap_Ex3['Output'])},
            {"role": "user",
             "content": [{"type": "text", "text": Result_Review_w_time_gap_Ex4['Input']}]},
            
            {"role": "assistant", 
             "content": json.dumps(Result_Review_w_time_gap_Ex4['Output'])},
            
            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": Result_Review_w_time_gap_Ex5['Input'],
                 }
             ]},

            {"role": "assistant",
             "content": [
                 {
                     "type": "text",
                     "text": json.dumps(Result_Review_w_time_gap_Ex5['Output']),
                     "cache_control": {"type": "ephemeral"},
                 }
             ]},

            {"role": "user",
             "content": [
                 {
                     "type": "text",
                     "text": Input
                 }
             ]}
        ]
    else:
        if args.few_shot:
            return [
                {"role": "system", "content": Sequential_findings_review_prompt},
                    
                {"role": "user", "content": Result_Review_w_time_gap_Ex1['Input']},
                {"role": "assistant", "content": json.dumps(Result_Review_w_time_gap_Ex1['Output'])},
                
                {"role": "user", "content": Result_Review_w_time_gap_Ex2['Input']},
                {"role": "assistant", "content": json.dumps(Result_Review_w_time_gap_Ex2['Output'])},
                
                {"role": "user", "content": Result_Review_w_time_gap_Ex3['Input']},
                {"role": "assistant", "content": json.dumps(Result_Review_w_time_gap_Ex3['Output'])},
                
                {"role": "user", "content": Result_Review_w_time_gap_Ex4['Input']},
                {"role": "assistant", "content": json.dumps(Result_Review_w_time_gap_Ex4['Output'])},
                            
                {"role": "user", "content": Result_Review_w_time_gap_Ex5['Input']},
                {"role": "assistant", "content": json.dumps(Result_Review_w_time_gap_Ex5['Output'])},
                            
                {"role": "user", "content": Input}
                ]
        else:
            return [
                {"role": "system", "content": Sequential_findings_review_prompt},
                {"role": "user", "content": Input}
            ]

def combined_precision_recall(true_clusters, pred_clusters, fuzzy_weight=0.6, jaccard_weight=0.4, threshold=0.6):
    tp_combined = 0
    fp = 0
    fn = 0
    
    # For each predicted cluster
    for pred_cluster in pred_clusters:
        pred_tokens = set(pred_cluster.lower().split())
        best_match = None
        best_score = 0
        
        # Find the most similar true cluster
        for true_cluster in true_clusters:
            true_tokens = set(true_cluster.lower().split())
            
            # Calculate fuzzy similarity
            fuzzy_similarity = fuzz.token_sort_ratio(pred_cluster, true_cluster) / 100.0
            
            # Calculate Jaccard similarity
            if not pred_tokens or not true_tokens:
                jaccard_similarity = 0
            else:
                intersection = len(pred_tokens.intersection(true_tokens))
                union = len(pred_tokens.union(true_tokens))
                jaccard_similarity = intersection / union if union > 0 else 0
            
            # Calculate combined score with weighted average
            combined_similarity = fuzzy_weight * fuzzy_similarity + jaccard_weight * jaccard_similarity
            
            if combined_similarity > best_score:
                best_score = combined_similarity
                best_match = true_cluster
        
        # Count as TP if score exceeds threshold, otherwise FP
        if best_score >= threshold:
            tp_combined += best_score  # Use similarity as weight
        else:
            fp += 1
    
    # True clusters that are not matched are FN
    matched_true_clusters = set()
    for pred_cluster in pred_clusters:
        best_match = None
        best_score = 0
        
        for true_cluster in true_clusters:
            # Calculate fuzzy and Jaccard similarity (same as above)
            fuzzy_similarity = fuzz.token_sort_ratio(pred_cluster, true_cluster) / 100.0
            
            pred_tokens = set(pred_cluster.lower().split())
            true_tokens = set(true_cluster.lower().split())
            
            if not pred_tokens or not true_tokens:
                jaccard_similarity = 0
            else:
                intersection = len(pred_tokens.intersection(true_tokens))
                union = len(pred_tokens.union(true_tokens))
                jaccard_similarity = intersection / union if union > 0 else 0
            
            combined_similarity = fuzzy_weight * fuzzy_similarity + jaccard_weight * jaccard_similarity
            
            if combined_similarity > best_score:
                best_score = combined_similarity
                best_match = true_cluster
        
        if best_score >= threshold:
            matched_true_clusters.add(best_match)
    
    fn = len(true_clusters) - len(matched_true_clusters)
    
    # Calculate precision and recall
    precision = tp_combined / (tp_combined + fp) if (tp_combined + fp) > 0 else 0
    recall = tp_combined / (tp_combined + fn) if (tp_combined + fn) > 0 else 0
    
    return precision, recall

def text_f1_score(pred_name, true_name):
    # Tokenization (simple whitespace split)
    precision, recall = combined_precision_recall(true_name, pred_name)
        
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def group_wise_accuracy(true_groups, pred_groups):
    """Compute accuracy for each group independently (handles negative values)"""
    unique_true_groups = np.unique(true_groups)
    group_accuracies = {}
    
    for group in unique_true_groups:
        # Indices of items belonging to this group
        group_indices = (true_groups == group)
        
        # Prediction accuracy for this group
        if np.sum(group_indices) > 0:
            # The most predicted class within this group
            pred_for_group = pred_groups[group_indices]
            
            # Use Counter as negative values may occur
            from collections import Counter
            pred_counts = Counter(pred_for_group)
            most_common_pred = pred_counts.most_common(1)[0][0]
            
            # Ratio of correctly clustered items in this group
            correct_clustering = np.sum(pred_for_group == most_common_pred)
            accuracy = correct_clustering / np.sum(group_indices)
            
            group_accuracies[float(group) if isinstance(group, (int, float, np.number)) else group] = {
                'accuracy': float(accuracy),
                'count': int(np.sum(group_indices)),
                'most_common_pred': float(most_common_pred) if isinstance(most_common_pred, (int, float, np.number)) else most_common_pred
            }
    
    return group_accuracies

def calculate_purity_fscore(df):
    """
    Calculate Purity and F-score for clustering results.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing clustering results
        
    Returns:
    --------
    dict
        Dictionary containing Purity and F-score values
    """
    import numpy as np
    from sklearn.metrics import f1_score
    
    # Purity calculation
    def calculate_purity(y_true, y_pred):
        """
        Calculate cluster purity
        """
        # Sum of the largest class in each cluster
        contingency_matrix = pd.crosstab(y_pred, y_true)
        return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix.values)
    
    # entity_group Purity
    entity_purity = calculate_purity(
        df['gt_entity_group'].fillna('nan').astype(str),
        df['LLM_cluster'].fillna('nan').astype(str)
    )
    
    # temporal_group Purity
    temporal_purity = calculate_purity(
        df['gt_temporal_group'].fillna('nan').astype(str),
        df['temporal_group'].fillna('nan').astype(str)
    )
    
    # Prepare for F-score calculation
    # For each GT class pair, set 1 if in the same cluster, 0 otherwise
    def create_pair_matrix(labels):
        n = len(labels)
        pairs = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] == labels[j]:
                    pairs[i, j] = pairs[j, i] = 1
        return pairs
    
    # entity_group F-score
    gt_entity_pairs = create_pair_matrix(df['gt_entity_group'].fillna('nan').astype(str).values)
    pred_entity_pairs = create_pair_matrix(df['LLM_cluster'].fillna('nan').astype(str).values)
    
    # Use only the upper triangle (remove duplicates)
    mask = np.triu_indices(len(gt_entity_pairs), k=1)
    gt_entity_pairs_flat = gt_entity_pairs[mask]
    pred_entity_pairs_flat = pred_entity_pairs[mask]
    
    entity_f1 = f1_score(gt_entity_pairs_flat, pred_entity_pairs_flat)
    entity_precision = np.sum(gt_entity_pairs_flat & pred_entity_pairs_flat) / np.sum(pred_entity_pairs_flat)
    entity_recall = np.sum(gt_entity_pairs_flat & pred_entity_pairs_flat) / np.sum(gt_entity_pairs_flat)
    
    # temporal_group F-score
    gt_temporal_pairs = create_pair_matrix(df['gt_temporal_group'].fillna('nan').astype(str).values)
    pred_temporal_pairs = create_pair_matrix(df['temporal_group'].fillna('nan').astype(str).values)
    
    # Use only the upper triangle (remove duplicates)
    gt_temporal_pairs_flat = gt_temporal_pairs[mask]
    pred_temporal_pairs_flat = pred_temporal_pairs[mask]
    
    temporal_f1 = f1_score(gt_temporal_pairs_flat, pred_temporal_pairs_flat)
    temporal_precision = np.sum(gt_temporal_pairs_flat & pred_temporal_pairs_flat) / np.sum(pred_temporal_pairs_flat)
    temporal_recall = np.sum(gt_temporal_pairs_flat & pred_temporal_pairs_flat) / np.sum(gt_temporal_pairs_flat)
    
    return {
        'entity_purity': entity_purity,
        'entity_f1': entity_f1,
        'entity_precision': entity_precision,
        'entity_recall': entity_recall,
        'temporal_purity': temporal_purity,
        'temporal_f1': temporal_f1,
        'temporal_precision': temporal_precision,
        'temporal_recall': temporal_recall
    }

def calculate_subject_purity_fscore(df):
    """
    Calculate purity and F-score for each subject.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing clustering results

    Returns
    -------
    dict
        Dictionary containing per-subject purity and F-score, plus their averages
    """
    subject_metrics = {}

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        # Only calculate if enough samples
        if len(subject_df) >= 2:
            metrics = calculate_purity_fscore(subject_df)
            subject_metrics[subject_id] = metrics

    # Compute averages
    avg_metrics = {
        'entity_purity_mean': np.mean([m['entity_purity'] for m in subject_metrics.values()]),
        'entity_f1_mean': np.mean([m['entity_f1'] for m in subject_metrics.values()]),
        'entity_precision_mean': np.mean([m['entity_precision'] for m in subject_metrics.values()]),
        'entity_recall_mean': np.mean([m['entity_recall'] for m in subject_metrics.values()]),
        'temporal_purity_mean': np.mean([m['temporal_purity'] for m in subject_metrics.values()]),
        'temporal_f1_mean': np.mean([m['temporal_f1'] for m in subject_metrics.values()]),
        'temporal_precision_mean': np.mean([m['temporal_precision'] for m in subject_metrics.values()]),
        'temporal_recall_mean': np.mean([m['temporal_recall'] for m in subject_metrics.values()])
    }

    return {
        'subject_purity_fscore': subject_metrics,
        **avg_metrics
    }

def analyze_low_performance_subjects(df, threshold=0.6):
    """
    Analyze low-performing subjects.
    """
    from sklearn.metrics import adjusted_rand_score

    low_performance = []

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        # Calculate ARI
        ari = adjusted_rand_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                  subject_df['LLM_cluster'].fillna('nan').astype(str))

        if ari < threshold:
            low_performance.append((subject_id, ari))

    # Detailed analysis of low-performing subjects
    for subject_id, ari in low_performance:
        print(f"\nAnalyzing subject: {subject_id}, ARI: {ari:.4f}")

        subject_df = df[df['subject_id'] == subject_id]

        # Mapping table between GT groups and predicted groups
        mapping_table = pd.crosstab(subject_df['gt_entity_group'], subject_df['LLM_cluster'])
        print("\nMapping table between GT groups and predicted groups:")
        print(mapping_table)

    return low_performance

def calculate_text_f1(df):
    """
    Compute F1 score evaluating whether identical texts belong to the same group.
    """
    # If 'concatenated' column doesn't exist, fallback to 'ELA_cur_ent' or return default value
    if 'concatenated' not in df.columns:
        print("Warning: 'concatenated' column not found. Using 'ELA_cur_ent' as fallback.")
        if 'ELA_cur_ent' in df.columns:
            df = df.copy()
            df['concatenated'] = df['ELA_cur_ent']
        else:
            print("Warning: Neither 'concatenated' nor 'ELA_cur_ent' columns found. Returning 0 for text F1.")
            return 0

    # Compute consistency for each unique text
    text_consistency = {}
    for text in df['concatenated'].unique():
        text_items = df[df['concatenated'] == text]

        # GT group consistency
        gt_groups = text_items['gt_entity_group'].value_counts()
        gt_consistency = gt_groups.max() / len(text_items) if len(gt_groups) > 0 else 0

        # Predicted group consistency
        pred_groups = text_items['LLM_cluster'].value_counts()
        pred_consistency = pred_groups.max() / len(text_items) if len(pred_groups) > 0 else 0

        text_consistency[text] = {
            'count': len(text_items),
            'gt_consistency': gt_consistency,
            'pred_consistency': pred_consistency
        }

    # Weighted average (by number of items)
    total_items = sum(info['count'] for info in text_consistency.values())
    weighted_gt_consistency = sum(info['count'] * info['gt_consistency'] for info in text_consistency.values()) / total_items
    weighted_pred_consistency = sum(info['count'] * info['pred_consistency'] for info in text_consistency.values()) / total_items

    # F1 score calculation
    if weighted_gt_consistency + weighted_pred_consistency == 0:
        return 0

    f1 = 2 * (weighted_gt_consistency * weighted_pred_consistency) / (weighted_gt_consistency + weighted_pred_consistency)
    return f1

def calculate_temporal_group_accuracies(df):
    """
    Calculate accuracy for each temporal group.
    """
    accuracies = {}

    for group in df['gt_temporal_group'].unique():
        group_items = df[df['gt_temporal_group'] == group]
        pred_groups = group_items['temporal_group'].value_counts()
        most_common_pred = pred_groups.idxmax() if len(pred_groups) > 0 else None
        accuracy = pred_groups.max() / len(group_items) if len(pred_groups) > 0 else 0

        accuracies[group] = {
            'accuracy': accuracy,
            'count': len(group_items),
            'most_common_pred': most_common_pred
        }

    return accuracies

def calculate_subject_metrics(df):
    """
    Calculate metrics for each subject.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

    subject_metrics = {}

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        # Only calculate if enough samples
        if len(subject_df) >= 2:
            # Metrics for temporal_group
            temporal_ari = adjusted_rand_score(subject_df['gt_temporal_group'].fillna('nan').astype(str),
                                               subject_df['temporal_group'].fillna('nan').astype(str))
            temporal_nmi = normalized_mutual_info_score(subject_df['gt_temporal_group'].fillna('nan').astype(str),
                                                        subject_df['temporal_group'].fillna('nan').astype(str))
            temporal_homogeneity = homogeneity_score(subject_df['gt_temporal_group'].fillna('nan').astype(str),
                                                     subject_df['temporal_group'].fillna('nan').astype(str))
            temporal_completeness = completeness_score(subject_df['gt_temporal_group'].fillna('nan').astype(str),
                                                       subject_df['temporal_group'].fillna('nan').astype(str))
            temporal_v_measure = v_measure_score(subject_df['gt_temporal_group'].fillna('nan').astype(str),
                                                 subject_df['temporal_group'].fillna('nan').astype(str))

            # Metrics for entity_group
            entity_ari = adjusted_rand_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                             subject_df['LLM_cluster'].fillna('nan').astype(str))
            entity_nmi = normalized_mutual_info_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                                      subject_df['LLM_cluster'].fillna('nan').astype(str))
            entity_homogeneity = homogeneity_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                                   subject_df['LLM_cluster'].fillna('nan').astype(str))
            entity_completeness = completeness_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                                     subject_df['LLM_cluster'].fillna('nan').astype(str))
            entity_v_measure = v_measure_score(subject_df['gt_entity_group'].fillna('nan').astype(str),
                                               subject_df['LLM_cluster'].fillna('nan').astype(str))

            # Text F1
            text_f1 = calculate_text_f1(subject_df)

            subject_metrics[subject_id] = {
                'temporal_ARI': temporal_ari,
                'temporal_NMI': temporal_nmi,
                'temporal_homogeneity': temporal_homogeneity,
                'temporal_completeness': temporal_completeness,
                'temporal_v_measure': temporal_v_measure,
                'entity_ARI': entity_ari,
                'entity_NMI': entity_nmi,
                'entity_homogeneity': entity_homogeneity,
                'entity_completeness': entity_completeness,
                'entity_v_measure': entity_v_measure,
                'text_f1': text_f1,
                'count': len(subject_df)
            }

    # Compute mean metrics
    metrics_sum = {
        'temporal_ARI_sum': 0,
        'temporal_NMI_sum': 0,
        'temporal_homogeneity_sum': 0,
        'temporal_completeness_sum': 0,
        'temporal_v_measure_sum': 0,
        'entity_ARI_sum': 0,
        'entity_NMI_sum': 0,
        'entity_homogeneity_sum': 0,
        'entity_completeness_sum': 0,
        'entity_v_measure_sum': 0,
        'text_f1_sum': 0,
        'total_count': 0
    }

    for metrics in subject_metrics.values():
        metrics_sum['temporal_ARI_sum'] += metrics['temporal_ARI']
        metrics_sum['temporal_NMI_sum'] += metrics['temporal_NMI']
        metrics_sum['temporal_homogeneity_sum'] += metrics['temporal_homogeneity']
        metrics_sum['temporal_completeness_sum'] += metrics['temporal_completeness']
        metrics_sum['temporal_v_measure_sum'] += metrics['temporal_v_measure']
        metrics_sum['entity_ARI_sum'] += metrics['entity_ARI']
        metrics_sum['entity_NMI_sum'] += metrics['entity_NMI']
        metrics_sum['entity_homogeneity_sum'] += metrics['entity_homogeneity']
        metrics_sum['entity_completeness_sum'] += metrics['entity_completeness']
        metrics_sum['entity_v_measure_sum'] += metrics['entity_v_measure']
        metrics_sum['text_f1_sum'] += metrics['text_f1']
        metrics_sum['total_count'] += 1

    # Calculate means
    count = metrics_sum['total_count']
    if count > 0:
        return {
            'temporal_ARI_mean': metrics_sum['temporal_ARI_sum'] / count,
            'temporal_NMI_mean': metrics_sum['temporal_NMI_sum'] / count,
            'temporal_homogeneity_mean': metrics_sum['temporal_homogeneity_sum'] / count,
            'temporal_completeness_mean': metrics_sum['temporal_completeness_sum'] / count,
            'temporal_v_measure_mean': metrics_sum['temporal_v_measure_sum'] / count,
            'entity_ARI_mean': metrics_sum['entity_ARI_sum'] / count,
            'entity_NMI_mean': metrics_sum['entity_NMI_sum'] / count,
            'entity_homogeneity_mean': metrics_sum['entity_homogeneity_sum'] / count,
            'entity_completeness_mean': metrics_sum['entity_completeness_sum'] / count,
            'entity_v_measure_mean': metrics_sum['entity_v_measure_sum'] / count,
            'text_f1_mean': metrics_sum['text_f1_sum'] / count,
            'subject_metrics': subject_metrics
        }
    else:
        return {
            'temporal_ARI_mean': 0,
            'temporal_NMI_mean': 0,
            'temporal_homogeneity_mean': 0,
            'temporal_completeness_mean': 0,
            'temporal_v_measure_mean': 0,
            'entity_ARI_mean': 0,
            'entity_NMI_mean': 0,
            'entity_homogeneity_mean': 0,
            'entity_completeness_mean': 0,
            'entity_v_measure_mean': 0,
            'text_f1_mean': 0,
            'subject_metrics': {}
        }

def eval_func(args, clustered_df):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    from tabulate import tabulate

    # Create results directory if it doesn't exist
    os.makedirs(f'{args.output_path}/eval', exist_ok=True)

    # Check if required columns exist before filtering
    required_columns = ['gt_temporal_group', 'gt_entity_group', 'temporal_group', 'LLM_cluster']
    missing_columns = [col for col in required_columns if col not in clustered_df.columns]

    if missing_columns:
        print(f"Warning: Missing required columns for evaluation: {missing_columns}")
        print("Skipping evaluation function.")
        return

    clustered_df = clustered_df[(clustered_df.gt_temporal_group.notna())&
                                (clustered_df.gt_entity_group.notna())&
                                (clustered_df.temporal_group.notna())&
                                (clustered_df.LLM_cluster.notna())]

    if len(clustered_df) == 0:
        raise ValueError("Warning: No valid data remaining after filtering. Skipping evaluation.")

    analysis_results = analyze_zero_ari_subjects(clustered_df)
    compare_zero_and_high_ari_subjects(clustered_df)

    clustered_df.to_csv(f'{args.output_path}/eval/eval_func_clustered_df.csv', index=True)

    temporal_ari = adjusted_rand_score(clustered_df['gt_temporal_group'].fillna('nan').astype(str), 
                                       clustered_df['temporal_group'].fillna('nan').astype(str))
    temporal_nmi = normalized_mutual_info_score(clustered_df['gt_temporal_group'].fillna('nan').astype(str), 
                                                clustered_df['temporal_group'].fillna('nan').astype(str))

    entity_ari = adjusted_rand_score(clustered_df['gt_entity_group'].fillna('nan').astype(str), 
                                     clustered_df['LLM_cluster'].fillna('nan').astype(str))
    entity_nmi = normalized_mutual_info_score(clustered_df['gt_entity_group'].fillna('nan').astype(str), 
                                              clustered_df['LLM_cluster'].fillna('nan').astype(str))
    # Calculate F-score and Purity
    purity_fscore = calculate_purity_fscore(clustered_df)
    text_f1 = calculate_text_f1(clustered_df)
    temporal_group_accuracies = calculate_temporal_group_accuracies(clustered_df)

    subject_metrics = calculate_subject_metrics(clustered_df)
    subject_purity_fscore = calculate_subject_purity_fscore(clustered_df)

    # Create metrics tables with metrics on columns and grouping types on rows
    # 1. Overall metrics table - Homogeneity, Completeness, V-measure excluded
    overall_metrics = [
        ["Grouping Type", "ARI", "NMI", "Purity", "F1-score", "Precision", "Recall"],
        ["Temporal Grouping", f"{temporal_ari:.4f}", f"{temporal_nmi:.4f}", f"{purity_fscore['temporal_purity']:.4f}", 
         f"{purity_fscore['temporal_f1']:.4f}", f"{purity_fscore['temporal_precision']:.4f}", f"{purity_fscore['temporal_recall']:.4f}"],
        ["Entity Grouping", f"{entity_ari:.4f}", f"{entity_nmi:.4f}", f"{purity_fscore['entity_purity']:.4f}", 
         f"{purity_fscore['entity_f1']:.4f}", f"{purity_fscore['entity_precision']:.4f}", f"{purity_fscore['entity_recall']:.4f}"]
    ]

    # 2. Subject-wise average metrics table - Homogeneity, Completeness, V-measure excluded
    subject_avg_metrics = [
        ["Grouping Type", "Avg. ARI", "Avg. NMI", "Avg. Purity", "Avg. F1-score", "Avg. Precision", "Avg. Recall"],
        ["Temporal Grouping", f"{subject_metrics.get('temporal_ARI_mean', 0):.4f}", f"{subject_metrics.get('temporal_NMI_mean', 0):.4f}", 
         f"{subject_purity_fscore.get('temporal_purity_mean', 0):.4f}", f"{subject_purity_fscore.get('temporal_f1_mean', 0):.4f}", 
         f"{subject_purity_fscore.get('temporal_precision_mean', 0):.4f}", f"{subject_purity_fscore.get('temporal_recall_mean', 0):.4f}"],
        ["Entity Grouping", f"{subject_metrics.get('entity_ARI_mean', 0):.4f}", f"{subject_metrics.get('entity_NMI_mean', 0):.4f}", 
         f"{subject_purity_fscore.get('entity_purity_mean', 0):.4f}", f"{subject_purity_fscore.get('entity_f1_mean', 0):.4f}", 
         f"{subject_purity_fscore.get('entity_precision_mean', 0):.4f}", f"{subject_purity_fscore.get('entity_recall_mean', 0):.4f}"]
    ]

    # 3. Top performing subjects table (top 5)
    temporal_performance = []
    for subject_id, metrics in subject_metrics.get('subject_metrics', {}).items():
        purity = subject_purity_fscore.get('subject_purity_fscore', {}).get(subject_id, {}).get('temporal_purity', 0)
        f1 = subject_purity_fscore.get('subject_purity_fscore', {}).get(subject_id, {}).get('temporal_f1', 0)
        temporal_performance.append((subject_id, metrics.get('temporal_ARI', 0), purity, f1))

    temporal_performance.sort(key=lambda x: x[1], reverse=True)
    top_temporal_subjects = [["Subject ID", "ARI", "Purity", "F1-score"]]
    for subject_id, ari, purity, f1 in temporal_performance[:5]:
        top_temporal_subjects.append([subject_id, f"{ari:.4f}", f"{purity:.4f}", f"{f1:.4f}"])

    entity_performance = []
    for subject_id, metrics in subject_metrics.get('subject_metrics', {}).items():
        purity = subject_purity_fscore.get('subject_purity_fscore', {}).get(subject_id, {}).get('entity_purity', 0)
        f1 = subject_purity_fscore.get('subject_purity_fscore', {}).get(subject_id, {}).get('entity_f1', 0)
        entity_performance.append((subject_id, metrics.get('entity_ARI', 0), purity, f1))

    entity_performance.sort(key=lambda x: x[1], reverse=True)
    top_entity_subjects = [["Subject ID", "ARI", "Purity", "F1-score"]]
    for subject_id, ari, purity, f1 in entity_performance[:5]:
        top_entity_subjects.append([subject_id, f"{ari:.4f}", f"{purity:.4f}", f"{f1:.4f}"])

    # Generate LaTeX tables
    def table_to_latex(table_data, caption, label):
        headers = table_data[0]
        rows = table_data[1:]

        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\caption{" + caption + "}\n"
        latex += "\\label{" + label + "}\n"

        # Create column format
        col_format = "|" + "|".join(["c"] * len(headers)) + "|"
        latex += "\\begin{tabular}{" + col_format + "}\n\\hline\n"

        # Add headers
        latex += " & ".join(headers) + " \\\\ \\hline\n"

        # Add rows
        for row in rows:
            latex += " & ".join([str(cell) for cell in row]) + " \\\\ \\hline\n"

        latex += "\\end{tabular}\n\\end{table}"
        return latex

    # Generate LaTeX tables
    overall_latex = table_to_latex(overall_metrics, "Overall Evaluation Metrics", "tab:overall_metrics")
    subject_avg_latex = table_to_latex(subject_avg_metrics, "Subject-wise Average Metrics", "tab:subject_avg_metrics")
    top_temporal_latex = table_to_latex(top_temporal_subjects, "Top 5 Subjects by Temporal Grouping Performance", "tab:top_temporal")
    top_entity_latex = table_to_latex(top_entity_subjects, "Top 5 Subjects by Entity Grouping Performance", "tab:top_entity")

    # Save LaTeX tables to files
    with open(f'{args.output_path}/eval/overall_metrics.tex', 'w') as f:
        f.write(overall_latex)

    with open(f'{args.output_path}/eval/subject_avg_metrics.tex', 'w') as f:
        f.write(subject_avg_latex)

    with open(f'{args.output_path}/eval/top_temporal_subjects.tex', 'w') as f:
        f.write(top_temporal_latex)

    with open(f'{args.output_path}/eval/top_entity_subjects.tex', 'w') as f:
        f.write(top_entity_latex)

    # Generate PNG tables using matplotlib
    def table_to_png(table_data, title, filename):
        fig, ax = plt.figure(figsize=(10, len(table_data) * 0.5 + 1)), plt.gca()
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{args.output_path}/eval/{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Generate PNG tables
    table_to_png(overall_metrics, "Overall Evaluation Metrics", "overall_metrics")
    table_to_png(subject_avg_metrics, "Subject-wise Average Metrics", "subject_avg_metrics")
    table_to_png(top_temporal_subjects, "Top 5 Subjects by Temporal Grouping Performance", "top_temporal_subjects")
    table_to_png(top_entity_subjects, "Top 5 Subjects by Entity Grouping Performance", "top_entity_subjects")

    # Visualization: Compare the top 10 subjects and the overall result
    # 1. Temporal Grouping
    top10_temporal = temporal_performance[:10]

    subjects = [subj[0] for subj in top10_temporal] + ['Overall']
    ari_values = [subj[1] for subj in top10_temporal] + [temporal_ari]
    purity_values = [subj[2] for subj in top10_temporal] + [purity_fscore['temporal_purity']]
    f1_values = [subj[3] for subj in top10_temporal] + [purity_fscore['temporal_f1']]

    fig, ax = plt.subplots(figsize=(15, 8))

    bar_width = 0.25
    r1 = np.arange(len(subjects))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, ari_values, width=bar_width, label='ARI', color='skyblue')
    ax.bar(r2, purity_values, width=bar_width, label='Purity', color='lightgreen')
    ax.bar(r3, f1_values, width=bar_width, label='F1-score', color='salmon')

    ax.set_xlabel('Subject ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Temporal Grouping Performance: Top 10 Subjects vs Overall', fontweight='bold', fontsize=14)
    ax.set_xticks([r + bar_width for r in range(len(subjects))])
    ax.set_xticklabels(subjects, rotation=45, ha='right')

    # Add a vertical line before 'Overall'
    ax.axvline(x=len(subjects)-1.5, color='gray', linestyle='--')

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f'{args.output_path}/eval/temporal_top10_vs_overall.png', dpi=300)
    plt.close()

    # 2. Entity Grouping
    top10_entity = entity_performance[:10]

    subjects = [subj[0] for subj in top10_entity] + ['Overall']
    ari_values = [subj[1] for subj in top10_entity] + [entity_ari]
    purity_values = [subj[2] for subj in top10_entity] + [purity_fscore['entity_purity']]
    f1_values = [subj[3] for subj in top10_entity] + [purity_fscore['entity_f1']]

    fig, ax = plt.subplots(figsize=(15, 8))

    bar_width = 0.25
    r1 = np.arange(len(subjects))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, ari_values, width=bar_width, label='ARI', color='skyblue')
    ax.bar(r2, purity_values, width=bar_width, label='Purity', color='lightgreen')
    ax.bar(r3, f1_values, width=bar_width, label='F1-score', color='salmon')

    ax.set_xlabel('Subject ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Entity Grouping Performance: Top 10 Subjects vs Overall', fontweight='bold', fontsize=14)
    ax.set_xticks([r + bar_width for r in range(len(subjects))])
    ax.set_xticklabels(subjects, rotation=45, ha='right')

    # Add a vertical line before 'Overall'
    ax.axvline(x=len(subjects)-1.5, color='gray', linestyle='--')

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f'{args.output_path}/eval/entity_top10_vs_overall.png', dpi=300)
    plt.close()

    # Mapping analysis and visualization (from earlier code)
    print("\n=== Mapping result analysis ===\n")

    # Create and visualize mapping tables for the entire dataset
    print("\n=== Mapping tables for the entire dataset ===\n")

    # Temporal Grouping overall mapping table
    overall_temporal_confusion = pd.crosstab(
        clustered_df['gt_temporal_group'], 
        clustered_df['temporal_group']
    )
    overall_temporal_confusion.to_csv(f'{args.output_path}/eval/overall_temporal_mapping.csv', index=True)

    # Visualization - size adjustment in case of many classes
    plt.figure(figsize=(max(12, len(overall_temporal_confusion.columns)//2), 
                        max(10, len(overall_temporal_confusion.index)//2)))
    sns.heatmap(overall_temporal_confusion, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Overall Temporal Grouping Mapping Table')
    plt.xlabel('Predicted Group')
    plt.ylabel('Actual Group')
    plt.tight_layout()
    plt.savefig(f'{args.output_path}/eval/overall_temporal_mapping.png')
    plt.close()

    # Entity Grouping overall mapping table
    overall_entity_confusion = pd.crosstab(
        clustered_df['gt_entity_group'], 
        clustered_df['LLM_cluster']
    )
    overall_entity_confusion.to_csv(f'{args.output_path}/eval/overall_entity_mapping.csv', index=True)

    # Visualization - size adjustment in case of many classes
    plt.figure(figsize=(max(15, len(overall_entity_confusion.columns)//2), 
                        max(12, len(overall_entity_confusion.index)//2)))
    sns.heatmap(overall_entity_confusion, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Overall Entity Grouping Mapping Table')
    plt.xlabel('Predicted Group')
    plt.ylabel('Actual Group')
    plt.tight_layout()
    plt.savefig(f'{args.output_path}/eval/overall_entity_mapping.png')
    plt.close()

    # Create and visualize mapping tables for each subject
    for subject in clustered_df['subject_id'].unique():
        subject_df = clustered_df[clustered_df['subject_id'] == subject]

        print(f"\n--- subject_id: {subject} ---")
        print(f"Number of items: {len(subject_df)}")

        gt_col, pred_col = 'gt_temporal_group', 'temporal_group'
        gt_groups = subject_df[gt_col].value_counts()
        pred_groups = subject_df[pred_col].value_counts()

        print(f"\nTemporal Grouping:")
        print(f"Actual Group Distribution: {dict(gt_groups)}")
        print(f"Predicted Group Distribution: {dict(pred_groups)}")

        # Confusion matrix
        confusion = pd.crosstab(subject_df[gt_col], subject_df[pred_col])
        confusion.to_csv(f'{args.output_path}/eval/{subject}_temporal_mapping.csv', index=True)

        # Visualize mapping table
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Subject ID: {subject} - Temporal Grouping Mapping Table')
        plt.xlabel('Predicted Group')
        plt.ylabel('Actual Group')
        plt.tight_layout()
        plt.savefig(f'{args.output_path}/eval/{subject}_temporal_mapping.png')
        plt.close()

        gt_col, pred_col = 'gt_entity_group', 'LLM_cluster'
        gt_groups = subject_df[gt_col].value_counts()
        pred_groups = subject_df[pred_col].value_counts()

        print(f"\nEntity Grouping:")
        print(f"Actual Group Distribution: {dict(gt_groups)}")
        print(f"Predicted Group Distribution: {dict(pred_groups)}")

        # Confusion matrix
        confusion = pd.crosstab(subject_df[gt_col], subject_df[pred_col])
        confusion.to_csv(f'{args.output_path}/eval/{subject}_entity_mapping.csv', index=True)

        # Visualize mapping table
        plt.figure(figsize=(max(12, len(pred_groups)//2), max(10, len(gt_groups)//2)))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Subject ID: {subject} - Entity Grouping Mapping Table')
        plt.xlabel('Predicted Group')
        plt.ylabel('Actual Group')
        plt.tight_layout()
        plt.savefig(f'{args.output_path}/eval/{subject}_entity_mapping.png')
        plt.close()

        print("\n=== Different group assignment for identical text analysis ===")

        # Only run if 'concatenated' field exists
        if 'concatenated' in subject_df.columns:
            text_groups = {}
            for idx, row in subject_df.iterrows():
                concat_text = row['concatenated']
                if concat_text not in text_groups:
                    text_groups[concat_text] = []
                text_groups[concat_text].append((idx, row['gt_entity_group'], row['LLM_cluster']))

            inconsistent_texts = []
            for text, items in text_groups.items():
                if len(items) > 1:
                    gt_groups = set([item[1] for item in items])
                    pred_groups = set([item[2] for item in items])

                    if len(gt_groups) > 1 or len(pred_groups) > 1:
                        inconsistent_texts.append((text, gt_groups, pred_groups, items))
                        print(f"\nIdentical text assigned to different groups:")
                        print(f"  Text: {text[:100]}..." if len(text) > 100 else f"  Text: {text}")
                        print(f"  GT Groups: {gt_groups}")
                        print(f"  Predicted Groups: {pred_groups}")
                        print("  Item details:")
                        for item in items:
                            print(f"    * Item ID: {item[0]}, GT: {item[1]}, Pred: {item[2]}")

            if inconsistent_texts:
                with open(f'{args.output_path}/eval/{subject}_inconsistent_texts.txt', 'w') as f:
                    f.write(f"Subject ID: {subject} - Analysis of identical text assigned to different groups\n\n")
                    for text, gt_groups, pred_groups, items in inconsistent_texts:
                        f.write(f"Text: {text}\n")
                        f.write(f"GT Groups: {gt_groups}\n")
                        f.write(f"Predicted Groups: {pred_groups}\n")
                        f.write("Item details:\n")
                        for item in items:
                            f.write(f"  * Item ID: {item[0]}, GT: {item[1]}, Pred: {item[2]}\n")
                        f.write("\n")
        else:
            print("  'concatenated' field not found in DataFrame.")

    print("\n=== Identical text assigned to different groups (overall dataset) ===\n")

    if 'concatenated' in clustered_df.columns:
        overall_text_groups = {}
        for idx, row in clustered_df.iterrows():
            concat_text = row['concatenated']
            subject_id = row['subject_id']
            if concat_text not in overall_text_groups:
                overall_text_groups[concat_text] = []
            overall_text_groups[concat_text].append((idx, subject_id, row['gt_entity_group'], row['LLM_cluster']))

        overall_inconsistent_texts = []
        for text, items in overall_text_groups.items():
            if len(items) > 1:
                gt_groups = set([item[2] for item in items])
                pred_groups = set([item[3] for item in items])

                if len(gt_groups) > 1 or len(pred_groups) > 1:
                    overall_inconsistent_texts.append((text, gt_groups, pred_groups, items))

        if overall_inconsistent_texts:
            with open(f'{args.output_path}/eval/overall_inconsistent_texts.txt', 'w') as f:
                f.write("Overall dataset - Identical text assigned to different groups\n\n")
                for text, gt_groups, pred_groups, items in overall_inconsistent_texts:
                    f.write(f"Text: {text}\n")
                    f.write(f"GT Groups: {gt_groups}\n")
                    f.write(f"Predicted Groups: {pred_groups}\n")
                    f.write("Item details:\n")
                    for item in items:
                        f.write(f"  * Item ID: {item[0]}, Subject ID: {item[1]}, GT: {item[2]}, Pred: {item[3]}\n")
                    f.write("\n")

            print(f"{len(overall_inconsistent_texts)} cases of identical text assigned to different groups in the overall dataset")
            print(f"For more details, see '{args.output_path}/eval/overall_inconsistent_texts.txt'")
    else:
        print("  'concatenated' field not found in DataFrame.")

    print("\n=== Full dataset evaluation results ===\n")
    print("temporal_group metrics:")
    print(f"Adjusted Rand Index: {temporal_ari:.4f}")
    print(f"Normalized Mutual Information: {temporal_nmi:.4f}")
    print(f"Purity: {purity_fscore['temporal_purity']:.4f}")
    print(f"F1-score: {purity_fscore['temporal_f1']:.4f}")
    print(f"Precision: {purity_fscore['temporal_precision']:.4f}")
    print(f"Recall: {purity_fscore['temporal_recall']:.4f}")
    print(f"Temporal Group Accuracies: {temporal_group_accuracies}")

    print("\nentity_group metrics:")
    print(f"Adjusted Rand Index: {entity_ari:.4f}")
    print(f"Normalized Mutual Information: {entity_nmi:.4f}")
    print(f"Purity: {purity_fscore['entity_purity']:.4f}")
    print(f"F1-score: {purity_fscore['entity_f1']:.4f}")
    print(f"Precision: {purity_fscore['entity_precision']:.4f}")
    print(f"Recall: {purity_fscore['entity_recall']:.4f}")
    print(f"Text F1: {text_f1:.4f}")
    print("\n=== case analysis ===\n")

    print("\nOverall Evaluation Metrics:")
    print(tabulate(overall_metrics, headers="firstrow", tablefmt="grid"))

    print("\nSubject-wise Average Metrics:")
    print(tabulate(subject_avg_metrics, headers="firstrow", tablefmt="grid"))

    print("\nTop 5 Subjects by Temporal Grouping Performance:")
    print(tabulate(top_temporal_subjects, headers="firstrow", tablefmt="grid"))

    print("\nTop 5 Subjects by Entity Grouping Performance:")
    print(tabulate(top_entity_subjects, headers="firstrow", tablefmt="grid"))

    for subject_id, ari, purity, f1 in temporal_performance:
        print(f"subject_id: {subject_id}, temporal_group ARI: {ari:.4f}, Purity: {purity:.4f}, F1: {f1:.4f}")

    for subject_id, ari, purity, f1 in entity_performance:
        print(f"subject_id: {subject_id}, entity_group ARI: {ari:.4f}, Purity: {purity:.4f}, F1: {f1:.4f}")

    print("\n=== Low performance subjects ===\n")
    low_perf_subjects = analyze_low_performance_subjects(clustered_df, threshold=0.6)

    print(f"\nLow performance subjects: {len(low_perf_subjects)}")
    print("Low performance subjects list:")
    for subject_id, ari in sorted(low_perf_subjects, key=lambda x: x[1]):
        print(f"  - {subject_id}: ARI = {ari:.4f}")

    metrics = {
        'temporal_ARI': temporal_ari,
        'temporal_NMI': temporal_nmi,
        'temporal_purity': purity_fscore['temporal_purity'],
        'temporal_f1': purity_fscore['temporal_f1'],
        'temporal_precision': purity_fscore['temporal_precision'],
        'temporal_recall': purity_fscore['temporal_recall'],
        'entity_ARI': entity_ari,
        'entity_NMI': entity_nmi,
        'entity_purity': purity_fscore['entity_purity'],
        'entity_f1': purity_fscore['entity_f1'],
        'entity_precision': purity_fscore['entity_precision'],
        'entity_recall': purity_fscore['entity_recall'],
        'text_f1': text_f1
    }

    return metrics

def analyze_zero_ari_subjects(df):
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    import numpy as np
    from collections import Counter

    # Find subjects with ARI of 0
    zero_ari_subjects = []

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        # Calculate temporal grouping ARI
        temporal_ari = adjusted_rand_score(
            subject_df['gt_temporal_group'].fillna('nan').astype(str),
            subject_df['temporal_group'].fillna('nan').astype(str)
        )

        if temporal_ari == 0:
            zero_ari_subjects.append(subject_id)

    print(f"\n=== Analysis of subjects with temporal grouping ARI == 0 (n={len(zero_ari_subjects)}) ===\n")

    analysis_results = {}

    for subject_id in zero_ari_subjects:
        subject_df = df[df['subject_id'] == subject_id]

        print(f"\n## Subject ID: {subject_id} ##")

        # 1. Basic Info
        n_items = len(subject_df)
        n_gt_groups = subject_df['gt_temporal_group'].nunique()
        n_pred_groups = subject_df['temporal_group'].nunique()

        print(f"Number of items: {n_items}")
        print(f"Number of GT temporal groups: {n_gt_groups}")
        print(f"Number of predicted temporal groups: {n_pred_groups}")

        # 2. GT temporal group distribution
        gt_group_counts = subject_df['gt_temporal_group'].value_counts().sort_index()
        print("\nGT temporal group distribution:")
        for group, count in gt_group_counts.items():
            print(f"  - Group {group}: {count} items ({count/n_items*100:.1f}%)")

        # 3. Predicted temporal group distribution
        pred_group_counts = subject_df['temporal_group'].value_counts().sort_index()
        print("\nPredicted temporal group distribution:")
        for group, count in pred_group_counts.items():
            print(f"  - Group {group}: {count} items ({count/n_items*100:.1f}%)")

        # 4. Crosstab analysis
        cross_tab = pd.crosstab(subject_df['gt_temporal_group'], subject_df['temporal_group'])
        print("\nGT vs Predicted Temporal Groups Crosstab:")
        print(cross_tab)

        # 5. Error Pattern Analysis
        print("\nError pattern analysis:")

        # 5.1 All items assigned to a single predicted group
        if n_pred_groups == 1:
            print("  - Issue: All items assigned to a single predicted group")
            print(f"    * Predicted group: {subject_df['temporal_group'].iloc[0]}")

        # 5.2 Different number of GT groups and predicted groups
        elif n_gt_groups != n_pred_groups:
            print(f"  - Issue: Number of GT groups ({n_gt_groups}) and predicted groups ({n_pred_groups}) differ")

        # 5.3 Groups do not match exactly
        else:
            # Find the most assigned predicted group for each GT group
            for gt_group in subject_df['gt_temporal_group'].unique():
                gt_items = subject_df[subject_df['gt_temporal_group'] == gt_group]
                pred_groups = gt_items['temporal_group'].value_counts()
                most_common_pred = pred_groups.index[0]

                print(f"  - GT group {gt_group}:")
                print(f"    * Number of items: {len(gt_items)}")
                print(f"    * Most assigned predicted group: {most_common_pred} ({pred_groups[most_common_pred]} items, {pred_groups[most_common_pred]/len(gt_items)*100:.1f}%)")

                if len(pred_groups) > 1:
                    print("    * Items assigned to other predicted groups:")
                    for pred_group, count in pred_groups.items():
                        if pred_group != most_common_pred:
                            print(f"      - Group {pred_group}: {count} items")
                            examples = gt_items[gt_items['temporal_group'] == pred_group].head(2)
                            for _, row in examples.iterrows():
                                print(f"        * Example: '{row['concatenated']}' (date: {row.get('date', 'N/A')})")

        # 6. Date analysis (if date column exists)
        if 'date' in subject_df.columns:
            print("\nDate analysis:")

            # Convert date to datetime
            subject_df['date_dt'] = pd.to_datetime(subject_df['date'], errors='coerce')

            # GT group-wise date range
            for gt_group in subject_df['gt_temporal_group'].unique():
                gt_items = subject_df[subject_df['gt_temporal_group'] == gt_group]

                if gt_items['date_dt'].notna().any():
                    min_date = gt_items['date_dt'].min()
                    max_date = gt_items['date_dt'].max()
                    date_range = (max_date - min_date).days

                    print(f"  - GT group {gt_group}:")
                    print(f"    * Date range: {min_date.date()} ~ {max_date.date()} ({date_range} days)")

                    # Predicted group-wise date range within GT group
                    for pred_group in gt_items['temporal_group'].unique():
                        pred_items = gt_items[gt_items['temporal_group'] == pred_group]

                        if pred_items['date_dt'].notna().any():
                            pred_min_date = pred_items['date_dt'].min()
                            pred_max_date = pred_items['date_dt'].max()
                            pred_date_range = (pred_max_date - pred_min_date).days

                            print(f"    * Predicted group {pred_group} date range: {pred_min_date.date()} ~ {pred_max_date.date()} ({pred_date_range} days)")

        # 7. Suggestions
        print("\nSuggestions:")

        if n_pred_groups == 1:
            print("  - The temporal grouping algorithm failed to discover multiple groups for this subject.")
            print("  - Consider leveraging date information more effectively, or extracting more temporal-related text features.")

        elif n_gt_groups != n_pred_groups:
            print(f"  - Consider adjusting the number of predicted groups ({n_pred_groups}) to match the number of GT groups ({n_gt_groups}).")
            print("  - Consider tuning clustering algorithm parameters or trying a different algorithm.")

        else:
            print("  - Improve the mapping between predicted and GT groups.")
            print("  - Consider more precise usage of the date information.")

        # Store results
        analysis_results[subject_id] = {
            'n_items': n_items,
            'n_gt_groups': n_gt_groups,
            'n_pred_groups': n_pred_groups,
            'gt_group_counts': gt_group_counts.to_dict(),
            'pred_group_counts': pred_group_counts.to_dict(),
            'cross_tab': cross_tab.to_dict()
        }

    return analysis_results

def compare_zero_and_high_ari_subjects(df):
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    import numpy as np

    # Calculate ARI for each subject
    subject_aris = {}

    for subject_id in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject_id]

        # Calculate temporal grouping ARI
        temporal_ari = adjusted_rand_score(
            subject_df['gt_temporal_group'].fillna('nan').astype(str),
            subject_df['temporal_group'].fillna('nan').astype(str)
        )

        subject_aris[subject_id] = temporal_ari

    # Find subjects with ARI=0 and the subject with maximum ARI
    zero_ari_subjects = [subject_id for subject_id, ari in subject_aris.items() if ari == 0]
    high_ari_subject = max(subject_aris.items(), key=lambda x: x[1])[0]

    print(f"\n=== Compare subject(s) with ARI=0 and subject with highest ARI (ARI={subject_aris[high_ari_subject]:.4f}) ===\n")

    # Analyze subject with highest ARI
    high_ari_df = df[df['subject_id'] == high_ari_subject]

    print(f"## Subject with highest ARI: {high_ari_subject} ##")
    print(f"Number of items: {len(high_ari_df)}")
    print(f"Number of GT temporal groups: {high_ari_df['gt_temporal_group'].nunique()}")
    print(f"Number of predicted temporal groups: {high_ari_df['temporal_group'].nunique()}")

    # Crosstab
    high_cross_tab = pd.crosstab(high_ari_df['gt_temporal_group'], high_ari_df['temporal_group'])
    print("\nGT vs Predicted Temporal Groups Crosstab (high ARI):")
    print(high_cross_tab)

    # Date analysis (if date column exists)
    if 'date' in high_ari_df.columns:
        print("\nDate analysis (high ARI):")

        high_ari_df['date_dt'] = pd.to_datetime(high_ari_df['date'], errors='coerce')

        for gt_group in high_ari_df['gt_temporal_group'].unique():
            gt_items = high_ari_df[high_ari_df['gt_temporal_group'] == gt_group]

            if gt_items['date_dt'].notna().any():
                min_date = gt_items['date_dt'].min()
                max_date = gt_items['date_dt'].max()
                date_range = (max_date - min_date).days

                print(f"  - GT group {gt_group}:")
                print(f"    * Date range: {min_date.date()} ~ {max_date.date()} ({date_range} days)")

    # Analyze and compare one zero-ARI subject if exists
    if zero_ari_subjects:
        zero_ari_subject = zero_ari_subjects[0]
        zero_ari_df = df[df['subject_id'] == zero_ari_subject]

        print(f"\n## Subject with ARI=0: {zero_ari_subject} ##")
        print(f"Number of items: {len(zero_ari_df)}")
        print(f"Number of GT temporal groups: {zero_ari_df['gt_temporal_group'].nunique()}")
        print(f"Number of predicted temporal groups: {zero_ari_df['temporal_group'].nunique()}")

        # Crosstab
        zero_cross_tab = pd.crosstab(zero_ari_df['gt_temporal_group'], zero_ari_df['temporal_group'])
        print("\nGT vs Predicted Temporal Groups Crosstab (ARI=0):")
        print(zero_cross_tab)

        # Date analysis (if date column exists)
        if 'date' in zero_ari_df.columns:
            print("\nDate analysis (ARI=0):")

            zero_ari_df['date_dt'] = pd.to_datetime(zero_ari_df['date'], errors='coerce')

            for gt_group in zero_ari_df['gt_temporal_group'].unique():
                gt_items = zero_ari_df[zero_ari_df['gt_temporal_group'] == gt_group]

                if gt_items['date_dt'].notna().any():
                    min_date = gt_items['date_dt'].min()
                    max_date = gt_items['date_dt'].max()
                    date_range = (max_date - min_date).days

                    print(f"  - GT group {gt_group}:")
                    print(f"    * Date range: {min_date.date()} ~ {max_date.date()} ({date_range} days)")

        # Main differences analysis
        print("\n## Main difference analysis ##")

        # 1. Difference in group counts
        print("1. Number of groups:")
        print(f"  - High ARI subject: GT {high_ari_df['gt_temporal_group'].nunique()}, Predicted {high_ari_df['temporal_group'].nunique()}")
        print(f"  - ARI=0 subject:   GT {zero_ari_df['gt_temporal_group'].nunique()}, Predicted {zero_ari_df['temporal_group'].nunique()}")

        # 2. Difference in group distributions
        print("\n2. Group distributions:")
        print("  - High ARI subject GT group distribution:")
        for group, count in high_ari_df['gt_temporal_group'].value_counts().sort_index().items():
            print(f"    * Group {group}: {count} ({count/len(high_ari_df)*100:.1f}%)")

        print("  - ARI=0 subject GT group distribution:")
        for group, count in zero_ari_df['gt_temporal_group'].value_counts().sort_index().items():
            print(f"    * Group {group}: {count} ({count/len(zero_ari_df)*100:.1f}%)")

        # 3. Difference in date patterns (if date info exists)
        if 'date' in df.columns:
            print("\n3. Date interval patterns:")

            # Date intervals for high ARI subject
            high_ari_df['date_dt'] = pd.to_datetime(high_ari_df['date'], errors='coerce')
            if high_ari_df['date_dt'].notna().any() and len(high_ari_df['date_dt'].unique()) > 1:
                high_dates = sorted(high_ari_df['date_dt'].unique())
                high_intervals = [(high_dates[i+1] - high_dates[i]).days for i in range(len(high_dates)-1)]

                print(f"  - High ARI subject date intervals: {high_intervals} (days)")
                print(f"    * Mean interval: {np.mean(high_intervals):.1f} days")
                print(f"    * Min interval: {min(high_intervals)} days")
                print(f"    * Max interval: {max(high_intervals)} days")

            # Date intervals for ARI=0 subject
            zero_ari_df['date_dt'] = pd.to_datetime(zero_ari_df['date'], errors='coerce')
            if zero_ari_df['date_dt'].notna().any() and len(zero_ari_df['date_dt'].unique()) > 1:
                zero_dates = sorted(zero_ari_df['date_dt'].unique())
                zero_intervals = [(zero_dates[i+1] - zero_dates[i]).days for i in range(len(zero_dates)-1)]

                print(f"  - ARI=0 subject date intervals: {zero_intervals} (days)")
                print(f"    * Mean interval: {np.mean(zero_intervals):.1f} days")
                print(f"    * Min interval: {min(zero_intervals)} days")
                print(f"    * Max interval: {max(zero_intervals)} days")
                
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Finding(BaseModel):
    IDX: int = Field(description="The index number of the finding from the input")
    DAY: int = Field(description="The day number when the finding was observed")
    finding: str = Field(description="The description of the finding")

class Episode(BaseModel):
    episode: int = Field(description="Sequential episode number")
    days: List[int] = Field(description="Array of day numbers that belong to this episode")

class FindingGroup(BaseModel):
    group_name: str = Field(description="Name of the finding group")
    findings: List[Finding] = Field(description="List of all findings in this group")
    episodes: List[Episode] = Field(description="Temporal groupings of these findings")
    rationale: str = Field(description="Explanation for the grouping decisions")

class RadiologyOutput(BaseModel):
    """Output schema for normalized and grouped radiological findings"""
    results: List[FindingGroup] = Field(description="List of finding groups including episodes and findings")