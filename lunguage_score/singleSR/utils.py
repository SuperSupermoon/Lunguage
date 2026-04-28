import re
import os
import glob
import time
import json
import torch
import datetime
import seaborn as sns
import pandas as pd
from rank_bm25 import BM25Okapi
from collections import defaultdict
from rapidfuzz import fuzz, process
from .generate_prompt import create_sys_msg
from ..config import CLAUDE_SONNET_MODEL

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Any
import instructor
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool, cpu_count, Process
from .eval import SR_EVAL, SRO_EVAL, cal_result_SR_EVAL, cal_result_SRO_EVAL, Gen_report_SR_EVAL, Gen_report_SRO_EVAL, cal_result_gen_report_SR_EVAL, cal_result_gen_report_SRO_EVAL


RELATIONS = ['cat', 'dx_status', 'dx_certainty', 'location', 'placement', 'associate', 'evidence',
'morphology', 'distribution', 'measurement', 'severity', 'comparison',
'onset', 'no change', 'improved', 'worsened', 'past hx', 'other source', 'assessment limitations']

def initialize_llm_client(llm_name, api_key, port=None):
    """
    Initialize an LLM client.

    Args:
        llm_name: Model deployment name.
        api_key:  API key, or 'local_LLM' for a local vLLM server.
        port:     vLLM server port. If None, falls back to the PORT env var.

    Supported backends:
        - Local vLLM:       api_key='local_LLM', port=<port>
        - OpenAI:           llm_name starts with 'gpt'
        - Fireworks AI:     llm_name starts with 'deepseek', 'llama4', or 'qwen3'
        - Anthropic Claude: llm_name starts with 'claude'  (requires: pip install anthropic)
        - MedGemma API:     llm_name starts with 'medgemma' (Google Gemini-compatible endpoint)
        - Baichuan:         llm_name starts with 'baichuan'
    """
    if api_key == 'local_LLM':
        resolved_port = port or os.getenv('PORT')
        if not resolved_port:
            raise EnvironmentError(
                "vLLM server port is not specified. "
                "Pass port= to initialize_llm_client(), set SingleSRConfig(port=8100), "
                "or export PORT=8100 before running."
            )
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=f"http://localhost:{resolved_port}/v1")
        return instructor.from_openai(client, mode=instructor.Mode.JSON), None
    elif llm_name.startswith('gpt'):
        from openai import OpenAI
        return instructor.from_openai(OpenAI(api_key=api_key), mode=instructor.Mode.JSON), None
    elif llm_name.startswith('deepseek') or llm_name.startswith('llama4') or llm_name.startswith('qwen3'):
        from openai import OpenAI
        return instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1"),
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
        return instructor.from_anthropic(anthropic.Anthropic(api_key=api_key)), None
    elif llm_name.startswith('medgemma'):
        # MedGemma via Google Gemini OpenAI-compatible endpoint
        from openai import OpenAI
        return instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            mode=instructor.Mode.JSON,
        ), None
    elif llm_name.startswith('baichuan'):
        from openai import OpenAI
        return instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://api.baichuan-ai.com/v1"),
            mode=instructor.Mode.JSON,
        ), None
    else:
        raise ValueError(
            f"Unsupported LLM name: '{llm_name}'. "
            "Supported prefixes: 'gpt', 'deepseek', 'llama4', 'qwen3', 'claude', 'medgemma', 'baichuan'. "
            "For local vLLM, set api_key='local_LLM' and provide a port."
        )

def process_single_item(args_tuple):
    """Process a single item for batch file creation"""
    custom_id, input_data, devset, args, system_message = args_tuple
    
    query = input_data[custom_id]['passage']
    query_subject_id = input_data[custom_id]['subject_id']
    if query is None or len(query) < 2 or pd.isna(query) or not isinstance(query, str):
        return None 
    
    json_vocab_input, user_history, assistant_history = retreive_query_related_fewshot(devset, query, query_subject_id, args)
    
    conversation = [{"role": "system", "content": system_message}]

    for i in range(len(user_history)):
        conversation.append({"role": "user", "content": user_history[i]})
        conversation.append({"role": "assistant", "content": assistant_history[i]})

    # Input for GT candidate or no-candidate mode
    if ('gt' in args.candidate_type) or (args.candidate_type == 'no_candidates'):
        json_gt_input = json.dumps(input_data[custom_id]['json_input'], indent=4)
        user_string = f"INPUT:\n{json_gt_input}\n"
    # Input for vocab candidate mode — uses json_vocab_input retrieved from vocab
    else:
        json_vocab_input = json.dumps(json_vocab_input, indent=4)
        user_string = f"INPUT:\n{json_vocab_input}\n"

    prompt = {"role": "user", "content": user_string}
    
    conversation.append(prompt)

    schema_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_json_schema.json")
    with open(schema_file_path, 'r') as f:
        gpt_json_schema = json.load(f)

    if args.deployment_name.startswith('gpt'):
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.deployment_name,
                "messages": conversation,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": gpt_json_schema
                }
            }
        }
    elif args.deployment_name.startswith('claude'):
        # Claude Batch API: system message must be separate from messages list
        task = {
            "custom_id": custom_id,
            "system_message": system_message,
            "messages": [m for m in conversation if m.get("role") != "system"],
        }
    else:
        task = {
            "custom_id": custom_id,
            "system_message": system_message,
            "messages": conversation
        }

    return task

def create_batch_file(args):
    input_data, devset = create_input(args)
    system_message = create_sys_msg(args)
    
    print(f"Starting multiprocessing with {min(cpu_count(), 64)} workers")
    
    # Prepare arguments for multiprocessing
    process_args = [(custom_id, input_data, devset, args, system_message) 
                   for custom_id in input_data.keys()]
    
    # Use multiprocessing to process items in parallel
    with Pool(processes=min(cpu_count(), 64)) as pool:
        tasks = list(tqdm(
            pool.imap(process_single_item, process_args),
            total=len(process_args),
            desc="Generating batch file"
        ))

    # Filter out None values
    tasks = [task for task in tasks if task is not None]
    
    if not args.dynamic_retrieval:
        data_dir = f'./singleSR/data/batch_files/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'
    else:
        data_dir = f'./singleSR/data/batch_files/dynamic_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'

    if not os.path.exists(data_dir):
        os.makedirs(f'{data_dir}', exist_ok=True)

    # Claude batch API limit: 100K requests per batch — split into chunks if needed
    CLAUDE_BATCH_LIMIT = 100_000
    if args.deployment_name.startswith('claude') and len(tasks) > CLAUDE_BATCH_LIMIT:
        n_chunks = (len(tasks) + CLAUDE_BATCH_LIMIT - 1) // CLAUDE_BATCH_LIMIT
        for i in range(n_chunks):
            chunk = tasks[i * CLAUDE_BATCH_LIMIT:(i + 1) * CLAUDE_BATCH_LIMIT]
            fname = os.path.join(data_dir, f'batch_file_{i:03d}.jsonl')
            with open(fname, 'w') as f:
                for obj in chunk:
                    f.write(json.dumps(obj) + '\n')
        print(f'Batch files created: {n_chunks} chunks ({len(tasks)} total requests, limit {CLAUDE_BATCH_LIMIT}/chunk)')
    else:
        with open(f'{data_dir}/batch_file.jsonl', 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')
        print('Batch file has been created')


def run_llm_batch(file_name, client, args):
    if args.deployment_name.startswith('claude'):
        try:
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request as AnthropicRequest
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for Claude batch processing. "
                "Install it with: pip install anthropic  or  pip install -e '.[claude]'"
            )

        # Build Claude tool schema from the shared gpt_json_schema.json
        schema_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpt_json_schema.json")
        with open(schema_file_path, 'r') as f:
            gpt_schema = json.load(f)
        claude_tool = [{
            "name": "structured_extraction",
            "description": "Extract structured entities and relations from a radiology report",
            "input_schema": gpt_schema["schema"],
        }]

        requests = []
        with open(file_name, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                system_text = task.get("system_message", "")
                messages = task.get("messages", [])
                request = AnthropicRequest(
                    custom_id=task["custom_id"],
                    params=MessageCreateParamsNonStreaming(
                        model=args.deployment_name,
                        max_tokens=8192,
                        system=[{
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=messages,
                        tools=claude_tool,
                        tool_choice={"type": "tool", "name": "structured_extraction"},
                    ),
                )
                requests.append(request)

        batch_job = client.beta.messages.batches.create(requests=requests)
        print(f"Claude batch job created: {batch_job.id}  ({len(requests)} requests)")
        return batch_job

    else:
        batch_file = client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        print(f"Batch file uploaded: {batch_file}")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(batch_job)
        return batch_job

def extract_passage_history(message):
    passage = None
    user_history = []
    assistant_history = []

    for idx, m in enumerate(message):
        if idx == len(message) - 1:
            passage = m['content']
        else:
            if m['role'] == 'user':
                user_history.append(m['content'])
            elif m['role'] == 'assistant':
                assistant_history.append(m['content'])

    return passage, user_history, assistant_history
        
def read_batch_results(batch_path, results_path, args):
    batch_file_path = batch_path + '/batch_file.jsonl'
    batch_results_file_path = results_path + '/batch_results.jsonl'
    all_model_outputs = {}
    all_results = []
    with open(batch_results_file_path, 'r') as file:
        for line in file:
            result = json.loads(line)            
            custom_id = result.get('custom_id', '')
            content = None
            try:
                if "content" in result:
                    content = result.get("content", "")
                else:
                    # This is GPT format (nested content)
                    response = result.get('response', {})
                    body = response.get('body', {})
                    
                    if isinstance(body, str):
                        # If body is a string, attempt to parse it
                        try:
                            body = json.loads(body)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse body as JSON for custom_id {custom_id}")
                            continue
                    
                    # Get choices array
                    choices = body.get('choices', [])
                    
                    if not choices or len(choices) == 0:
                        print(f"Warning: No choices found for custom_id {custom_id}")
                        continue
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
            except Exception as e:
                print(f"Error extracting content from result: {e}")
                continue
            
            if not content:
                print(f"Warning: No content found for custom_id {custom_id}")
                continue
            all_model_outputs[custom_id] = content

    # Process batch file
    with open(batch_file_path, 'r') as file:
        for line in file:
            result = json.loads(line)
            custom_id = result.get('custom_id', '')
            if 'body' in result:
                body = result.get('body', {})
                message = body.get('messages', {}) # list
            else:
                message = result.get('messages', {})
            
            passage, user_history, assistant_history = extract_passage_history(message)

            results = {
                'custom_id': custom_id,
                'passage': passage,
                'user_history': user_history,
                'assistant_history': assistant_history,
                'model_output': all_model_outputs[custom_id]
            }

            all_results.append(results)
            
    total_results = {
        "exp_name": args.mode,
        "exp_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployment_name": args.deployment_name,
        "entity_types": args.entity_types,
        "relation_types": args.relation_types,
        "attribute_types": args.attribute_types,
        "output_format": args.output_format,
        "input_unit": args.unit,
        "retreive_unit": args.unit,
        "n_retrieval": args.n_retrieval,
        "diverse_retrieval": args.diverse_retrieval,
        "annotations": all_results,
        }
    
    output_file = os.path.join(results_path, f'{args.deployment_name}_{args.output_format}_{args.mode}_{args.unit}.json')
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(total_results, outfile, indent=4, ensure_ascii=False)
    return total_results, output_file

def monitor_batch_job(batch_job_id, client, args, check_interval=2):
    """
    Poll a batch job until it finishes and return the final status string.

    OpenAI:  client.batches.retrieve()  →  batch.status / request_counts.completed / .total
    Claude:  client.messages.batches.retrieve()  →  batch.processing_status /
             request_counts.succeeded / .errored / .processing / .canceled
    """
    is_claude = args.deployment_name.startswith('claude')
    print("-" * 50)
    previous_status = None

    try:
        while True:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            if is_claude:
                _raw = getattr(client, 'client', client)
                batch = _raw.messages.batches.retrieve(batch_job_id)
                batch_status = batch.processing_status          # "in_progress" | "canceling" | "ended"
                completed_count = batch.request_counts.succeeded
                total_count = (batch.request_counts.succeeded +
                               batch.request_counts.errored +
                               batch.request_counts.processing +
                               batch.request_counts.canceled)
                terminal_statuses = {"ended", "canceling"}
            else:
                batch = client.batches.retrieve(batch_job_id)
                batch_status = batch.status
                completed_count = batch.request_counts.completed
                total_count = batch.request_counts.total
                terminal_statuses = {"completed", "succeeded", "ended", "failed", "errored"}

            if batch_status != previous_status:
                print(f"\n[{current_time}] Status changed: {previous_status} → {batch_status}")
                print(f"Completed: {completed_count}/{total_count}")

                if batch_status in terminal_statuses:
                    if not is_claude and hasattr(batch, 'output_file_id'):
                        print(f"Result file ID: {batch.output_file_id}")
                    print("Batch finished." if batch_status not in {"failed", "errored"} else "Batch failed!")
                    break

                previous_status = batch_status
            else:
                if total_count > 0:
                    pct = (completed_count / total_count) * 100
                    print(f"\r[{current_time}] Progress: {pct:.1f}% ({completed_count}/{total_count})", end="")
                else:
                    print(f"\r[{current_time}] Current status: {batch_status}", end="")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if is_claude:
            _raw = getattr(client, 'client', client)
            _raw.messages.batches.cancel(batch_job_id)
        else:
            client.batches.cancel(batch_job_id)
        print("Batch cancelled.")

    return batch_status

def generate_gt_output(triplet_list):
    
    unique_entities = list(set([entity[0] for entity in triplet_list]))
    entities = []
    
    for entity in unique_entities:
        relations = []
        for triplet in triplet_list:
            if triplet[0] == entity:
                if triplet[1] in ['associate', 'evidence']:
                    # Split value like "pneumonia, obj_ent_idx2, effusion, obj_ent_idx3"
                    tokens = [t.strip() for t in triplet[2].split(',')]
                    for i in range(0, len(tokens), 2):
                        value = tokens[i]
                        if i+1 < len(tokens) and tokens[i+1].startswith('obj_ent_idx'):
                            obj_ent_idx = int(tokens[i+1].replace('obj_ent_idx', ''))
                            relations.append({
                                "relation": triplet[1],
                                "value": value,
                                "obj_ent_idx": obj_ent_idx
                            })
                else:
                    value = triplet[2]
                    if triplet[1] == 'location' and isinstance(value, str) and value.lower().startswith("loc:"):
                        value = value[4:].strip()
                    if triplet[1] in ['cat', 'dx_status', 'dx_certainty']:
                        value = value.upper()
                    relations.append({
                        "relation": triplet[1],
                        "value": value
                    })
            
        entity_obj = {
                "name": entity,
                "sent_idx": 1,
                "ent_idx": 1,
                "relations": relations
            }
        entities.append(entity_obj)

    return "OUTPUT: " + json.dumps({"entities": entities}, indent=2)


def process_conversation(user_history, assistant_history, system_message, input_data, custom_id, args):
    
    conversation = [{"role": "system", "content": system_message}]
    
    for hist_idx, history in enumerate(user_history):
        conversation.append({"role": "user", "content": "----------- New Report Section -------------"})
        for i in range(len(history)):
            conversation.append({"role": "user", "content": history[i]})
            conversation.append({"role": "assistant", "content": assistant_history[hist_idx][i]})
    
    return conversation

def process_chunk(chunk_data, devset, results_path, args, chunk_id, client=None, tokenizer=None):
    """Process a chunk of data and save intermediate results"""
    # Use the provided client and tokenizer instances, or create new ones if None
    if client is None or tokenizer is None:
        client, tokenizer = initialize_llm_client(args.deployment_name, args.api_key, port=getattr(args, 'port', None))
    system_message = create_sys_msg(args)
    all_results = []
    error_cases = []
    
    # Create intermediate results directory
    intermediate_dir = os.path.join(results_path, 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Load existing results if any
    chunk_file = os.path.join(intermediate_dir, f'chunk_{chunk_id}.json')
    if os.path.exists(chunk_file):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            all_results = existing_data.get('annotations', [])
            error_cases = existing_data.get('error_cases', [])
            # Get processed file indices (only from successful results)
            processed_indices = {result['custom_id'] for result in all_results}
            # Get error case indices
            error_indices = {error['file_idx'] for error in error_cases}
    else:
        processed_indices = set()
        error_indices = set()
    
    # If we have error cases, only process those
    if error_cases:
        print(f"Reprocessing {len(error_cases)} error cases in chunk {chunk_id}")
        to_process = {idx: chunk_data[idx] for idx in error_indices if idx in chunk_data}
    else:
        to_process = {idx: data for idx, data in chunk_data.items() if idx not in processed_indices}
    
    progress_bar = tqdm(enumerate(to_process.keys()), total=len(to_process), desc=f"Processing chunk {chunk_id}")
    
    for itr, file_idx in progress_bar:
        try:
            query = chunk_data[file_idx]['passage']
            query_subject_id = chunk_data[file_idx]['subject_id']
            if query is None or len(query) < 2 or pd.isna(query) or not isinstance(query, str):
                continue
            else:
                json_vocab_input, user_history, assistant_history = retreive_query_related_fewshot(devset, query, query_subject_id, args)
                
                conversation = [{"role": "system", "content": system_message}]
                        
                if args.candidate_usage != 1:
                    target_length = int(len(query_words) * args.candidate_usage)
                    step = len(query_words) // target_length
                    query_words = query_words[::step][:target_length]

                for i in range(len(user_history)):
                    conversation.append({"role": "user", "content": user_history[i]})
                    conversation.append({"role": "assistant", "content": assistant_history[i]})

                if ('gt' in args.candidate_type) or (args.candidate_type == 'no_candidates'):
                    json_gt_input = json.dumps(chunk_data[file_idx]['json_input'], indent=4)
                    user_string = f"INPUT:\n{json_gt_input}\n"
                else:
                    json_vocab_input = json.dumps(json_vocab_input, indent=4)
                    user_string = f"INPUT:\n{json_vocab_input}\n"

                prompt = {"role": "user", "content": user_string}
                conversation.append(prompt)

                try:
                    # Add delay to avoid overwhelming the API
                    time.sleep(1.0)  # Increased delay for better stability
                    
                    # Retry mechanism for API calls
                    max_retries = 3
                    retry_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            if args.deployment_name.startswith('gemini'):
                                response = client.messages.create(
                                    messages=conversation,
                                    generation_config={
                                        "temperature": 0.0
                                    },
                                    response_model=StructuredOutput
                                )

                            elif getattr(args, 'api_key', 'local_LLM') == 'local_LLM':
                                # Local vLLM: use deployment_name directly as served model name
                                response = client.chat.completions.create(
                                    model=args.deployment_name,
                                    max_tokens=1024,
                                    temperature=0.0,
                                    messages=conversation,
                                    response_model=StructuredOutput)
                            elif args.deployment_name.startswith('baichuan') or args.deployment_name.startswith('medgemma') or args.deployment_name.startswith('gpt-oss-20b') or args.deployment_name.startswith('gpt-oss-120b'):
                                response = client.chat.completions.create(
                                    model= "baichuan-inc/Baichuan-M1-14B-Instruct" if args.deployment_name.startswith('baichuan') else "medgemma-27b-text-it" if args.deployment_name.startswith('medgemma') else "gpt-oss-20b" if args.deployment_name.startswith('gpt-oss-20b') else "gpt-oss-120b",
                                    messages=conversation,
                                    response_model=StructuredOutput
                                    )
                            else:
                                response = client.chat.completions.create(
                                    model=f'accounts/fireworks/models/{args.deployment_name}',
                                    max_tokens= 8092,
                                    temperature=0.0,
                                    messages=conversation,
                                    response_model=StructuredOutput)
                            
                            # If we get here, the API call was successful
                            break
                            
                        except Exception as api_error:
                            if attempt < max_retries - 1:
                                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                # Final attempt failed, re-raise the exception
                                raise api_error
                    
                    triplets = convert_to_sr_structure(response)
            
                    passage, user_history, assistant_history = extract_passage_history(conversation)

                    results = {
                        'custom_id': file_idx,
                        'passage': passage,
                        'user_history': user_history,
                        'assistant_history': assistant_history,
                        'model_output': triplets
                    }

                    all_results.append(results)
                    
                    # If this was an error case, remove it from error_cases
                    if file_idx in error_indices:
                        error_cases = [error for error in error_cases if error['file_idx'] != file_idx]
                        print(f"Successfully reprocessed error case {file_idx}")
                        
                except Exception as e:
                    error_info = {
                        'file_idx': file_idx,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'conversation': conversation
                    }

                    # Only add to error_cases if it's not already there
                    if not any(error['file_idx'] == file_idx for error in error_cases):
                        error_cases.append(error_info)
                    print(f"Error processing file {file_idx}: {str(e)}")
                
                # Save intermediate results after each file
                chunk_results = {
                    "chunk_id": chunk_id,
                    "exp_name": args.mode,
                    "exp_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "deployment_name": args.deployment_name,
                    "entity_types": args.entity_types,
                    "relation_types": args.relation_types,
                    "attribute_types": args.attribute_types,
                    "output_format": args.output_format,
                    "input_unit": args.unit,
                    "retreive_unit": args.unit,
                    "n_retrieval": args.n_retrieval,
                    "diverse_retrieval": args.diverse_retrieval,
                    "annotations": all_results,
                    "error_cases": error_cases
                }
                
                with open(chunk_file, "w", encoding="utf-8") as outfile:
                    json.dump(chunk_results, outfile, indent=4, ensure_ascii=False)
                continue

        except Exception as e:
            error_info = {
                'file_idx': file_idx,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            # Only add to error_cases if it's not already there
            if not any(error['file_idx'] == file_idx for error in error_cases):
                error_cases.append(error_info)
            print(f"Error in initial processing of file {file_idx}: {str(e)}")
            
            # Save intermediate results even when there's an error
            chunk_results = {
                "chunk_id": chunk_id,
                "exp_name": args.mode,
                "exp_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "deployment_name": args.deployment_name,
                "entity_types": args.entity_types,
                "relation_types": args.relation_types,
                "attribute_types": args.attribute_types,
                "output_format": args.output_format,
                "input_unit": args.unit,
                "retreive_unit": args.unit,
                "n_retrieval": args.n_retrieval,
                "diverse_retrieval": args.diverse_retrieval,
                "annotations": all_results,
                "error_cases": error_cases
            }
            
            with open(chunk_file, "w", encoding="utf-8") as outfile:
                json.dump(chunk_results, outfile, indent=4, ensure_ascii=False)
            continue
    
    return chunk_file

def merge_results(intermediate_dir, final_output_file):
    """Merge all intermediate results into a single file"""
    all_results = []
    all_error_cases = []
    
    # Get all chunk files
    chunk_files = sorted([f for f in os.listdir(intermediate_dir) if f.startswith('chunk_')])
    
    for chunk_file in chunk_files:
        with open(os.path.join(intermediate_dir, chunk_file), 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
            all_results.extend(chunk_data['annotations'])
            all_error_cases.extend(chunk_data['error_cases'])
    # Create final merged results
    merged_results = {
        "exp_name": chunk_data['exp_name'],
        "exp_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deployment_name": chunk_data['deployment_name'],
        "entity_types": chunk_data['entity_types'],
        "relation_types": chunk_data['relation_types'],
        "attribute_types": chunk_data['attribute_types'],
        "output_format": chunk_data['output_format'],
        "input_unit": chunk_data['input_unit'],
        "retreive_unit": chunk_data['retreive_unit'],
        "n_retrieval": chunk_data['n_retrieval'],
        "diverse_retrieval": chunk_data['diverse_retrieval'],
        "annotations": all_results,
        "error_cases": all_error_cases
    }
    
    # Save merged results
    with open(final_output_file, "w", encoding="utf-8") as outfile:
        json.dump(merged_results, outfile, indent=4, ensure_ascii=False)

def transform_triplets(triplets):
    """Transform triplets to a list of strings"""
    
    import random
    transformed_triplets = []
    for triplet in triplets:
        
        choice = random.randint(0, 2)
        
        if choice == 0:
            # Transform 1: Change dx_status p to n, dx_certainty d to t
            if triplet[1] == 'dx_status' and triplet[2] == 'positive':
                transformed_triplets.append((triplet[0], 'dx_status', 'negative'))
            elif triplet[1] == 'dx_certainty' and triplet[2] == 'definitive':
                transformed_triplets.append((triplet[0], 'dx_certainty', 'tentative'))
            elif triplet[0] and ' ' in triplet[0]:
                # Delete a random word from entity
                words = triplet[0].split()
                word_to_remove = random.choice(words)
                modified_entity = ' '.join([w for w in words if w != word_to_remove])
                transformed_triplets.append((modified_entity, triplet[1], triplet[2]))
            elif triplet[2] and ' ' in str(triplet[2]):
                # Delete a random word from value
                words = str(triplet[2]).split()
                word_to_remove = random.choice(words)
                modified_value = ' '.join([w for w in words if w != word_to_remove])
                # Remove commas from the beginning and end of modified_value
                modified_value = modified_value.strip(',')
                transformed_triplets.append((triplet[0], triplet[1], modified_value))
            else:
                transformed_triplets.append((triplet[0], triplet[1], triplet[2]))
        elif choice == 1:
            transformed_triplets.append((triplet[0], triplet[1], triplet[2]))
        elif choice == 2:
            continue
    
    return transformed_triplets

def get_fewshot(query, df, args, vocab, vocab_lookup, vocab_keys):

    ENTITY_CATEGORY = ['pf', 'cf', 'cof', 'cof/ncd', 'oth', 'ncd', 'pf/cf', 'patient info.']

    json_data = {
        'report_section': []
    }

    section_order = {'hist': 0, 'find': 1, 'impr': 2}
    
    df = df.sort_values(by='section', key=lambda x: x.map(section_order))

    df_idx_removed = df.copy()
    
    if args.candidate_type == 'no_candidates':
        max_sent_idx = df_idx_removed['sent_idx'].max()
        for sent_idx in range(1, max_sent_idx + 1):
            df_for_gt_sro = df_idx_removed[df_idx_removed['sent_idx'] == sent_idx]
            json_data['report_section'].append({
                'sent_idx': sent_idx,
                'sentence': df_for_gt_sro['sent'].iloc[0],
            })
        json_input = json.dumps(json_data, indent=4)
        user_string = f"INPUT:\n{json_input}\n"
    else:
        if args.unit == 'sent':
            if args.candidate_type == 'vocab_ent_rcg':
                candidates = list(set(get_words(query, vocab, vocab_lookup, vocab_keys, args)))
                word_cat_list = []
                for word in candidates:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    word_cat_list.extend([(word, cat_list)])
                json_data['report_section'].append({
                            'sent_idx': 1,
                            'sentence': query,
                            'candidates': word_cat_list})
                json_input = json.dumps(json_data, indent=4)
                user_string = f"INPUT:\n{json_input}\n"
            
        else:

            if 'gt' in args.candidate_type:
                max_sent_idx = df_idx_removed['sent_idx'].max()
                for sent_idx in range(1, max_sent_idx + 1):
                    df_for_gt_sro = df_idx_removed[df_idx_removed['sent_idx'] == sent_idx]

                    gt = []

                    for _, row in df_for_gt_sro.iterrows():
                        triplet_list, _, _ = extract_triplets(row)
                        if args.candidate_type == 'gt_sro_review':
                            gt = transform_triplets(triplet_list)
                        elif args.candidate_type == 'gt_sro':
                            # Create a set to track unique triplets
                            unique_triplets = []
                            seen = set()
                            for triplet in triplet_list:
                                # Create a hashable representation of the triplet
                                triplet_key = (triplet[0], triplet[1], triplet[2])
                                if triplet_key not in seen:
                                    seen.add(triplet_key)
                                    unique_triplets.append(triplet_key)

                            gt.extend(unique_triplets)

                        elif args.candidate_type == 'gt_so':
                            for triplet in triplet_list:
                                gt.extend([(triplet[0])])
                                if triplet[1] not in ['cat', 'dx_status', 'dx_certainty']:
                                    gt.extend([(triplet[2])])
                        elif args.candidate_type == 'gt_s':
                            for triplet in triplet_list:
                                gt.extend([(triplet[0])])
                            
                        elif args.candidate_type == 'gt_ent_rcg':
                            for triplet in triplet_list:
                                ent_cat_list = vocab[vocab['target_term'] == triplet[0]]['category'].tolist()
                                gt.extend([(triplet[0], ent_cat_list)])
                                if triplet[1] not in ['cat', 'dx_status', 'dx_certainty', 'associate', 'evidence']:
                                    gt.extend([(triplet[2], [triplet[1]])])

                    if args.candidate_type in ['gt_sro', 'gt_sro_review']:
                        gt = list(set(tuple(triplet) for triplet in gt))
                    elif args.candidate_type == 'gt_so' or args.candidate_type == 'gt_s':
                        gt = list(dict.fromkeys(gt))
                    elif args.candidate_type == 'gt_ent_rcg':
                        # Remove duplicates while preserving category information
                        unique_entities = {}
                        for entity, categories in gt:
                            if entity in unique_entities:
                                # If entity already exists, merge the categories
                                unique_entities[entity] = list(set(unique_entities[entity] + categories))
                            else:
                                unique_entities[entity] = categories
                        # Convert back to list of tuples
                        gt = [(entity, categories) for entity, categories in unique_entities.items()]

                    json_data['report_section'].append({
                        'sent_idx': sent_idx,
                        'sentence': df_for_gt_sro['sent'].iloc[0],
                        'candidates': gt
                    })

                json_input = json.dumps(json_data, indent=4)
                user_string = f"INPUT:\n{json_input}\n"

            else:
                max_sent_idx = df['sent_idx'].max()
                
                for sent_idx in range(1, max_sent_idx + 1):
                    sent = df[df['sent_idx'] == sent_idx]['sent'].iloc[0]
                    candidates = list(set(get_words(sent.strip(), vocab, vocab_lookup, vocab_keys, args)))

                    if args.candidate_usage != 1:
                        target_length = int(len(candidates) * args.candidate_usage)
                        step = len(candidates) // max(target_length, 1)
                        candidates = candidates[::step][:target_length]

                    if args.candidate_type == 'vocab_so':
                        json_candidates = candidates
                    elif args.candidate_type == 'vocab_s':
                        ent_words = []
                        for word in candidates:
                            cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                            if all(cat in ENTITY_CATEGORY for cat in cat_list):
                                ent_words.append(word)
                        json_candidates = ent_words
                    elif args.candidate_type == 'vocab_ent_rcg':
                        word_cat_list = []
                        for word in candidates:
                            cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                            new_cat_list = []
                            for cat in cat_list:
                                if cat in ENTITY_CATEGORY:
                                    new_cat_list.append('entity')
                                else:
                                    new_cat_list.append(cat)
                            word_cat_list.extend([(word, new_cat_list)])
                        json_candidates = word_cat_list

                    json_data['report_section'].append({
                        'sent_idx': sent_idx,
                        'sentence': sent,
                        'candidates': json_candidates
                    })

                json_input = json.dumps(json_data, indent=4)
                user_string = f"INPUT:\n{json_input}\n"

    entities = []
    current_study_id = df['study_id'].iloc[0]
    ent_idx_counter = 1

    for _, row in df.iterrows():
        if row['cat'].upper() not in args.entity_types:
            continue
        if row['study_id'] != current_study_id:
            continue

        relations = []
        for attr in ['Cat', 'Dx_Status', 'Dx_Certainty'] + args.relation_types + args.attribute_types:
            attr_key = attr.lower()
            if pd.notna(row[attr_key]):
                if attr in ['Associate', 'Evidence']:
                    # Split value like "pneumonia, obj_ent_idx2, effusion, obj_ent_idx3"
                    tokens = [t.strip() for t in row[attr_key].split(',')]
                    for i in range(0, len(tokens), 2):
                        value = tokens[i]
                        if i+1 < len(tokens) and tokens[i+1].startswith('obj_ent_idx'):
                            obj_ent_idx = int(tokens[i+1].replace('obj_ent_idx', ''))
                            relations.append({
                                "relation": attr,
                                "value": value,
                                "obj_ent_idx": obj_ent_idx
                            })
                else:
                    value = row[attr_key]
                    if attr == 'Location' and isinstance(value, str) and value.lower().startswith("loc:"):
                        value = value[4:].strip()
                    if attr in ['Cat', 'Dx_Status', 'Dx_Certainty']:
                        value = value.upper()
                    relations.append({
                        "relation": attr,
                        "value": value
                    })

        entity_obj = {
            "name": row['ent'],
            "sent_idx": int(row['sent_idx']),
            "ent_idx": int(row['ent_idx']),
            "relations": relations
        }
        entities.append(entity_obj)

    assistant_string = "OUTPUT: " + json.dumps({"entities": entities}, indent=2)

    return user_string, assistant_string

def get_fewshot_multi_turn(query, df, args, vocab, vocab_lookup, vocab_keys):

    ENTITY_CATEGORY = ['pf', 'cf', 'cof', 'cof/ncd', 'oth', 'ncd', 'pf/cf', 'patient info.']

    user_string_list = []
    assistant_string_list = []
    
    section_order = {'hist': 0, 'find': 1, 'impr': 2}
    
    df = df.sort_values(by='section', key=lambda x: x.map(section_order))

    df_idx_removed = df.copy()
    
    if args.candidate_type == 'no_candidates':
        max_sent_idx = df_idx_removed['sent_idx'].max()
        for sent_idx in range(1, max_sent_idx + 1):
            df_for_gt_sro = df_idx_removed[df_idx_removed['sent_idx'] == sent_idx]
            json_data['report_section'].append({
                'sent_idx': sent_idx,
                'sentence': df_for_gt_sro['sent'].iloc[0],
            })
        json_input = json.dumps(json_data, indent=4)
        user_string = f"INPUT:\n{json_input}\n"
    else:
        max_sent_idx = df['sent_idx'].max()
        
        for sent_idx in range(1, max_sent_idx + 1):
            
            json_data = {
                'report_section': []
            }
            sent = df[df['sent_idx'] == sent_idx]['sent'].iloc[0]
            candidates = list(set(get_words(sent.strip(), vocab, vocab_lookup, vocab_keys, args)))
            if args.candidate_usage != 1:
                target_length = int(len(candidates) * args.candidate_usage)
                step = len(candidates) // max(target_length, 1)
                candidates = candidates[::step][:target_length]
            if args.candidate_type == 'vocab_so':
                json_candidates = candidates
            elif args.candidate_type == 'vocab_s':
                ent_words = []
                for word in candidates:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    if all(cat in ENTITY_CATEGORY for cat in cat_list):
                        ent_words.append(word)
                json_candidates = ent_words
            elif args.candidate_type == 'vocab_ent_rcg':
                word_cat_list = []
                for word in candidates:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    word_cat_list.extend([(word, cat_list)])
                json_candidates = word_cat_list
            json_data['report_section'].append({
                'sent_idx': sent_idx,
                'sentence': sent,
                'candidates': json_candidates
            })
            json_input = json.dumps(json_data, indent=4)
            user_string = f"INPUT:\n{json_input}\n"
            user_string_list.append(user_string)


   
    current_study_id = df['study_id'].iloc[0]
    ent_idx_counter = 1

    for sent_idx in range(1, max_sent_idx + 1):
        
        df_idx = df[df['sent_idx'] == sent_idx]
        entities = []
        for _, row in df_idx.iterrows():
            if row['cat'].upper() not in args.entity_types:
                continue
            if row['study_id'] != current_study_id:
                continue

            relations = []
            for attr in ['Cat', 'Dx_Status', 'Dx_Certainty'] + args.relation_types + args.attribute_types:
                attr_key = attr.lower()
                if pd.notna(row[attr_key]):
                    if attr in ['Associate', 'Evidence']:
                        # Split value like "pneumonia, obj_ent_idx2, effusion, obj_ent_idx3"
                        tokens = [t.strip() for t in row[attr_key].split(',')]
                        for i in range(0, len(tokens), 2):
                            value = tokens[i]
                            if i+1 < len(tokens) and tokens[i+1].startswith('obj_ent_idx'):
                                obj_ent_idx = int(tokens[i+1].replace('obj_ent_idx', ''))
                                relations.append({
                                    "relation": attr,
                                    "value": value,
                                    "obj_ent_idx": obj_ent_idx
                                })
                    else:
                        value = row[attr_key]
                        if attr == 'Location' and isinstance(value, str) and value.lower().startswith("loc:"):
                            value = value[4:].strip()
                        if attr in ['Cat', 'Dx_Status', 'Dx_Certainty']:
                            value = value.upper()
                        relations.append({
                            "relation": attr,
                            "value": value
                        })

            entity_obj = {
                "name": row['ent'],
                "sent_idx": int(row['sent_idx']),
                "ent_idx": int(row['ent_idx']),
                "relations": relations
            }
            entities.append(entity_obj)

        assistant_string = "OUTPUT: " + json.dumps({"entities": entities}, indent=2)
        assistant_string_list.append(assistant_string)
        
    return user_string_list, assistant_string_list

def preprocess_text(text):
    """Lowercase text, replace certain punctuation with spaces, and strip extra whitespace."""
    text = str(text).lower()
    for p in ['.', ',', '-', '/', '!', '?', '(', ')', '"', "'"]:
        text = text.replace(p, ' ')
    return ' '.join(text.split()).strip()

def find_subsequence_indices(pattern_tokens, tokens, start=0):
    """
    Recursively return all index sequences (tuples) where pattern_tokens appear in order within tokens.
    e.g. pattern_tokens=["a","b"] and tokens=["a","x","b","a","b"] returns (0,2) and (3,4).
    """
    if not pattern_tokens:
        return [()]
    results = []
    first = pattern_tokens[0]
    for i in range(start, len(tokens)):
        if tokens[i] == first:
            for tail in find_subsequence_indices(pattern_tokens[1:], tokens, i+1):
                results.append((i,) + tail)
    return results

def find_subsequence_indices_with_window(pattern_tokens, tokens, start=0, window=10):
    """
    Recursively returns all index sequences (tuples) where pattern_tokens appear in order within tokens.
    The window parameter restricts the search range for each token to a specified size around the current token.
    Example: pattern_tokens=["a", "b"] and tokens=["a", "x", "b", "a", "b"], window=1 → returns (3,4).
    """
    if not pattern_tokens:
        return [()]
    
    results = []
    first = pattern_tokens[0]
    
    # Iterate through the token list from the start position to the end
    for i in range(start, len(tokens)):
        if tokens[i] == first:
            # If the current token matches the first pattern token,
            # search for the next pattern token(s) within the specified window size
            next_start = i + 1
            next_end = min(i + 1 + window, len(tokens))
            # Recursively search the rest of the pattern within the window range
            for tail in find_subsequence_indices(pattern_tokens[1:], tokens[next_start:next_end], 0):
                # Adjust the indices of tail to match the original position
                adjusted_tail = tuple(idx + next_start for idx in tail)
                results.append((i,) + adjusted_tail)
    
    return results

def map_tokens_to_original(original_sent, tokens):
    """
    Map preprocessed tokens to their original positions in the original sentence.
    Use re.search to find matches, and fallback to searching from the current position for the token length if no match is found.
    """
    original_tokens = []
    current_pos = 0
    for token, _, _ in tokens:
        pattern = re.escape(token)
        match = re.search(pattern, original_sent[current_pos:], flags=re.IGNORECASE)
        if match:
            real_start = current_pos + match.start()
            token_end  = current_pos + match.end()
            original_tokens.append((original_sent[real_start:token_end], real_start, token_end))
            current_pos = token_end
        else:
            original_tokens.append((token, current_pos, current_pos + len(token)))
            current_pos += len(token)
    return original_tokens

def tokenize_sentence(sent):
    """
    Tokenize a sentence: preprocess the text, extract token spans,
    map tokens back to their positions in the original sentence,
    and return both processed and original token information.
    """
    preprocessed = preprocess_text(sent)  # Convert punctuation (e.g., '.', ',') to spaces
    tokens = [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', preprocessed)]  # token, start, end (in preprocessed)
    original_tokens = map_tokens_to_original(sent, tokens)  # (token_text, start, end) for each token based on the original sentence
    token_texts = [t[0] for t in tokens]
    return preprocessed, tokens, original_tokens, token_texts

def get_vocab_lookup(vocab):
    """
    Initialize the vocabulary lookup for fuzzy continuous matching.
    Store a list of match info for each processed vocabulary term.
    """
    vocab_lookup = defaultdict(list)
    fields_to_use = ['target_term']

    # If 'raw_term' in vocab.columns, you may want to add it as well.
    # if 'raw_term' in vocab.columns:
    #    fields_to_use.append('raw_term')

    for idx, row in vocab.iterrows():
        for field in fields_to_use:
            if field in row and pd.notna(row[field]):
                term = str(row[field])
                processed_term = preprocess_text(term)
                if processed_term:
                    vocab_lookup[processed_term].append({
                        'matched_term': term,   # Will be used as matched_word
                        'source_field': field,
                        'target_term': row.get('target_term', None),
                        'category': row.get('category', None),
                        'normed_term': row.get('normed_term', None)
                    })

    vocab_keys = list(vocab_lookup.keys())

    return vocab_lookup, vocab_keys

def find_fuzzy_continuous_matches(text, vocab_lookup, vocab_keys, fuzzy_threshold=100):
    """
    For all continuous candidate spans of the original text,
    calculate the fuzzy similarity between the preprocessed candidate and the keys in vocab_lookup using RapidFuzz's fuzz.ratio.
    If the similarity is greater than or equal to fuzzy_threshold, add the match to the result.
      - 'word': original text span
      - 'matched_word': matched vocabulary word
    """

    if pd.isna(text):
        return []
    
    original_text = text # 
    _, tokens, original_tokens, _ = tokenize_sentence(text)
    matches = []
    seen_matches = set() 
    n = len(tokens)
    
    # Generate all possible continuous spans of original_tokens
    # text = "patient has known pulmonary fibrosis with interstitial abnormalities, larger in the lower lobes bilaterally."
    # original_tokens = [('patient', 0, 7), ('has', 8, 11), ('known', 12, 17), ('pulmonary', 18, 27), ('fibrosis', 28, 36), ('with', 37, 41), ('interstitial', 42, 54), ('abnormalities', 55, 68), ('larger', 70, 76), ('in', 77, 79), ('the', 80, 83), ('lower', 84, 89), ('lobes', 90, 95), ('bilaterally', 96, 107)]

    candidate_spans = []
    for length in range(n, 0, -1): 
        for i in range(n - length + 1): 
            start_pos = original_tokens[i][1]
            end_pos = original_tokens[i + length - 1][2]
            span_text = original_text[start_pos:end_pos].strip()
            processed_span = preprocess_text(span_text)
            if processed_span:
                candidate_spans.append((span_text, processed_span, start_pos, end_pos))
    
    for span_text, processed_span, start_pos, end_pos in candidate_spans:
        fuzzy_result = process.extractOne(processed_span, vocab_keys, scorer=fuzz.ratio)
        if fuzzy_result and fuzzy_result[1] >= fuzzy_threshold:
            best_key, score, _ = fuzzy_result
            for entry in vocab_lookup[best_key]:
                key = (start_pos, end_pos, entry['source_field'], entry['matched_term'].lower(), entry['category'], 'fuzzy_continuous')
                if key not in seen_matches:
                    matches.append({
                        'word': span_text,                           # original text span
                        'matched_word': entry['matched_term'],       # matched vocabulary word
                        'source_field': entry['source_field'],       # source field (raw_term or target_term)
                        'target_term': entry['target_term'],         # target term
                        'category': entry['category'],               # category
                        'normed_term': entry['normed_term'],         # normed term
                        'start': start_pos,                          # start position of the span in the original text
                        'end': end_pos,                              # end position of the span in the original text
                        'match_type': 'fuzzy_continuous',            # match type
                        'fuzzy_score': score                         # fuzzy score
                    })
                    seen_matches.add(key)

    return matches

def find_fuzzy_discontinuous_matches(sent, vocab_df, fuzzy_threshold=90):
    """
    For each vocabulary term (e.g., target_term), obtain its preprocessed token list.
    Then, find all discontinuous matches (candidate index sequences) in the sentence while preserving token order.
    Among the candidates, select the one with the smallest span length in the original text,
    and compute the fuzzy similarity between the preprocessed candidate and the vocabulary term.
    If the similarity is greater than or equal to fuzzy_threshold, include it in the results.
      - 'word': the exact span extracted from the original text
      - 'matched_word': the matched vocabulary term
    """
    if pd.isna(sent):
        return []
    
    original_sent = sent
    _, _, original_tokens, token_texts = tokenize_sentence(sent)
    matches = []
    seen_matches = set()
    
    # Sort vocab_df by the length of target_term in descending order
    sorted_vocab_df = vocab_df.copy().sort_values(by='target_term', key=lambda x: x.str.len(), ascending=False)
    
    for _, row in sorted_vocab_df.iterrows():

        term = row['target_term']
        
        if pd.isna(term):
            continue

        processed_vocab = preprocess_text(term)
        pattern_tokens = processed_vocab.split() # split terms in vocabulary, ex) left lung -> ['left', 'lung']
        #if len(pattern_tokens) < 2:
        #    continue
        
        # Find all subsequence indices of pattern_tokens in token_texts with a window of 10
        subseq_indices = find_subsequence_indices_with_window(pattern_tokens, token_texts, window=5)
        if not subseq_indices:
            continue
        
        # Select the candidate with the smallest span length
        best_seq, best_length = None, None
        for seq in subseq_indices:
            if len(seq) == 1:
                start_pos = original_tokens[seq[0]][1]
                end_pos = original_tokens[seq[0]][2]
            else:
                start_pos = original_tokens[seq[0]][1]
                end_pos = original_tokens[seq[-1]][2]
            length = end_pos - start_pos
            if best_seq is None or length < best_length:
                best_seq, best_length = seq, length

        # text = "left lower lung" and pattern_tokens = ['left', 'lung']
        # best_seq = (0, 2)

        if best_seq is not None:
            # preprocessed candidate (for fuzzy comparison)
            candidate_text = " ".join(token_texts[i] for i in best_seq)
            # actual original text span
            if len(best_seq) == 1:
                start_pos = original_tokens[best_seq[0]][1]
                end_pos = original_tokens[best_seq[0]][2]
            else:
                start_pos = original_tokens[best_seq[0]][1]
                end_pos = original_tokens[best_seq[-1]][2]
            candidate_original = original_sent[start_pos:end_pos].strip()
            fuzzy_score = fuzz.ratio(candidate_text, processed_vocab)
            if fuzzy_score >= fuzzy_threshold:
                key = (candidate_original.lower(), row.get('category', None), 'fuzzy_discontinuous')
                if key not in seen_matches:
                    matches.append({
                        'word': candidate_original,    # original text span
                        'matched_word': term,          # matched vocabulary term
                        'source_field': 'target_term',
                        'target_term': term,
                        'category': row.get('category', None),
                        'normed_term': row.get('normed_term', None),
                        'start': start_pos,
                        'end': end_pos,
                        'match_type': 'fuzzy_discontinuous',
                        'fuzzy_score': fuzzy_score
                    })
                    seen_matches.add(key)
    return matches

def process_with_all_fuzzy_matches(sent, vocab, vocab_lookup, vocab_keys, fuzzy_threshold=90, allow_overlap=True):
    """
    Performs both continuous (fuzzy_continuous) and discontinuous (fuzzy_discontinuous) matching on a sentence.
    For each result:
      - 'word' represents the actual span in the original text
      - 'matched_word' represents the matched target word
    If allow_overlap is False, overlapping spans are filtered with preference for longer spans.
    """
    continuous_matches = find_fuzzy_continuous_matches(sent, vocab_lookup, vocab_keys, fuzzy_threshold)
    discontinuous_matches = find_fuzzy_discontinuous_matches(sent, vocab, fuzzy_threshold)
    all_matches = continuous_matches + discontinuous_matches
    
    return all_matches

def get_target_term_positions(sent_term, target_term, sent_start, sent_end):
    # Convert to lowercase for case-insensitive matching
    sent_term_lower = sent_term.lower()
    target_term_lower = target_term.lower()
    
    # Find exact match position
    target_pos = sent_term_lower.find(target_term_lower)
    
    if target_pos != -1:
        # If exact match found, calculate absolute positions
        target_start = sent_start + target_pos
        target_end = target_start + len(target_term)
    else:
        # If no exact match, find longest matching substring
        longest_match = 0
        match_start = 0
        match_end = 0
        
        # Compare characters to find longest matching substring
        for i in range(len(sent_term_lower)):
            for j in range(i+1, len(sent_term_lower)+1):
                substring = sent_term_lower[i:j]
                if substring in target_term_lower:
                    if len(substring) > longest_match:
                        longest_match = len(substring)
                        match_start = i
                        match_end = j
        
        # Calculate absolute positions based on matching substring
        target_start = sent_start + match_start
        target_end = sent_start + match_end
        
    return target_start, target_end

def get_overlap_status(target_start, target_end, used_target_start, used_target_end):
    
    if target_start > used_target_end or target_end < used_target_start:
        return False
    else:
        return True

def get_words(text, vocab, vocab_lookup, vocab_keys, args):

    if args.candidate_discontinuous:
        vocab_match = find_fuzzy_discontinuous_matches(text, vocab, fuzzy_threshold=100)
    else:
        vocab_match = find_fuzzy_continuous_matches(text, vocab_lookup, vocab_keys, fuzzy_threshold=100)
        
    # Get all target terms from vocab values and convert to lowercase
    sent_terms = []

    for match in vocab_match:
        # Get sent term positions
        org_start = match['start']
        org_end = match['end']
        org_term = str(match['word']).lower()
        target_term = str(match['matched_word']).lower()
        
        if any(char.isdigit() for char in target_term):
            continue

        # Get target term positions
        target_start, target_end = get_target_term_positions(org_term, target_term, org_start, org_end)
        
        sent_terms.append([
            org_term, 
            target_term,
            org_start, 
            org_end,
            target_start,
            target_end
        ])

    # Filter out contained terms
    filtered_sent_terms = []
    used_positions = set()
    
    # Sort by length (descending) to check longer terms first
    if args.candidate_discontinuous:
        sent_terms_sorted = sorted(sent_terms, key=lambda x: x[3]-x[2], reverse=True) # x[3] - x[2] = org_end - org_start
    else:
        sent_terms_sorted = sorted(sent_terms, key=lambda x: x[5]-x[4], reverse=True) # x[5] - x[4] = target_end - target_start
    
    for sent_term in sent_terms_sorted:
        
        if args.candidate_discontinuous:
            org_start, org_end = sent_term[2], sent_term[3]
            
            if any((org_start, org_end) == (used_org_start, used_org_end) for used_org_start, used_org_end in used_positions):
                continue
            
            if not any(get_overlap_status(org_start, org_end, used_org_start, used_org_end) for used_org_start, used_org_end in used_positions):
                filtered_sent_terms.append(sent_term)
                used_positions.add((org_start, org_end))
        else:
            target_start, target_end = sent_term[4], sent_term[5]
        
            # Check if this term's position is already used
            if any((target_start, target_end) == (used_target_start, used_target_end) for used_target_start, used_target_end in used_positions):
                continue

            # Check if this term is contained within any other term
            if not any(get_overlap_status(target_start, target_end, used_target_start, used_target_end) # Changed from 'and' to 'or' to avoid re-sampling already selected spans (2025.03.31)
                      for used_target_start, used_target_end in used_positions):
                filtered_sent_terms.append(sent_term)
                used_positions.add((target_start, target_end))
    
    filtered_terms = [term[1] for term in filtered_sent_terms]

    return filtered_terms

def get_cols(query_words, vocab):
    
    cols = {'location', 'evidence', 'associate'}
    entity_cats = set()
    entities = set()

    clinical_categories = ['comparison', 'distribution', 'improved', 'location', 
                         'measurement', 'morphology', 'no change', 'onset', 
                         'other source', 'past hx', 'placement',
                         'severity', 'assessment limitations', 'worsened']

    for word in query_words:
        categories = set(vocab[vocab['target_term'] == word]['category'])
        for category in categories:
            if category in clinical_categories:
                cols.add(category)
            else:
                if category in ['lf', 'pf']:
                    entity_cats.add('pf')
                else:
                    entity_cats.add(category)
                entities.add(word)

    return list(cols), list(entity_cats), list(entities)


def get_n_retrieval(query, result, vocab, devset, args):

    devset = devset.copy()
    
    ENTITY_CATEGORY = ['pf', 'cf', 'cof', 'cof/ncd', 'oth', 'ncd', 'pf/cf', 'patient info.']
    
    section_key = 'section_report' if args.unit == 'section' else 'sent'
    diverse_result = pd.DataFrame()
    vocab_lookup, vocab_keys = get_vocab_lookup(vocab)
    query_words_dict = {}

    ###### Generate JSON INPUT for New User Input ######
    # Note: Candidates are extracted based on vocabulary
    json_vocab_input = {
        'report_section': []
    }

    query_words = []
    if section_key == 'sent':
        idx = re.findall(r'\((\d+)\)', query, re.DOTALL)
        query_words = list(set(get_words(query, vocab, vocab_lookup, vocab_keys, args)))
        
        if args.candidate_type == 'vocab_ent_rcg':
            word_cat_list = []
            for word in query_words:
                cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                word_cat_list.extend([(word, cat_list)])
            candidates = word_cat_list
            
        json_vocab_input['report_section'].append({
            'sent_idx': idx,
            'sentence': query,
            'candidates': candidates
        })
        
    else:
        # Extract numbered sentences from the query (format: "(1) text (2) text...")
        sentences = re.findall(r'\(\d+\)\s*(.*?)(?=\s*\(\d+\)|$)', query, re.DOTALL)
        sentences = [s.strip() for s in sentences if s.strip()]

        for idx, sentence in enumerate(sentences):
            
            all_words = list(set(get_words(sentence, vocab, vocab_lookup, vocab_keys, args)))
            
            query_words.extend(all_words)
            
            if ('gt' in args.candidate_type) or (args.candidate_type == 'no_candidates'):
                candidates = all_words
            
            elif args.candidate_type == 'vocab_so':
                candidates = all_words
            
            elif args.candidate_type == 'vocab_s': # entity name only (no category)
                ent_words = []
                for word in all_words:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    if all(cat in ENTITY_CATEGORY for cat in cat_list):
                        ent_words.append(word)
                candidates = ent_words
            
            elif args.candidate_type == 'vocab_ent_rcg':
                word_cat_list = []
                for word in all_words:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    new_cat_list = []
                    for cat in cat_list:
                        if cat in ENTITY_CATEGORY:
                            new_cat_list.append('entity')
                        else:
                            new_cat_list.append(cat)
                            
                    word_cat_list.extend([(word, new_cat_list)])
                    
                candidates = word_cat_list

            
            if args.candidate_usage != 1:
                target_length = int(len(candidates) * args.candidate_usage)
                step = len(candidates) // max(target_length, 1)
                candidates = candidates[::step][:target_length]
                        
            json_vocab_input['report_section'].append({
                'sent_idx': idx+1,
                'sentence': sentence,
                'candidates': candidates
            })
    
    query_words = list(set(query_words))
    query_cols, query_entity_cats, query_entities = get_cols(query_words, vocab)

    # Convert sets to sorted lists for deterministic order
    query_cols = sorted(query_cols)
    query_entity_cats = sorted(query_entity_cats)
    query_entities = sorted(query_entities)
    
    # Sort devset by sentence order from result and check for duplicate sentences
    ordered_sections = result['section'].unique().tolist()
    devset.loc[:, f'{section_key}_order'] = devset[f'{section_key}'].map({s: i for i, s in enumerate(ordered_sections)})
    # Calculate how many query_entities are contained in each section_report
    if len(query_entities) > 0:
        # Function to check if entities are contained in the section_report
        def count_entities(text):
            return sum(1 for entity in query_entities if entity in text)
        
        # Add new column with count of entities in each section_report
        devset.loc[:, 'entity_count'] = devset[section_key].apply(count_entities)
        
        # Sort first by entity count (descending) then by section_order
        devset = devset.sort_values(['entity_count', f'{section_key}_order'], ascending=[False, True])
    else:
        # If no entities, sort only by section_order
        devset = devset.sort_values(f'{section_key}_order')
    
    # Remove temporary columns after sorting
    devset = devset.drop(['entity_count', f'{section_key}_order'], axis=1, errors='ignore')
    seen_sections = set()
    shot_count = 0
    
    # Process columns first, then entity categories
    while len(query_cols) > 0 or len(query_entities) > 0 or shot_count < args.n_retrieval:
        section_list = []
        
        if len(query_entities) > 0:
            entity = query_entities[0]  # Take first element instead of random pop
            query_entities = query_entities[1:]  # Remove first element
            section_list = list(devset[devset['ent'] == entity][section_key].unique())
        elif len(query_cols) > 0:
            col = query_cols[0]  # Take first element instead of random pop
            query_cols = query_cols[1:]  # Remove first element
            section_list = list(devset[devset[col].notna()][section_key].unique())
        elif shot_count < args.n_retrieval:
            section_list = list(devset[section_key].unique())

        if len(section_list) == 0:
            continue

        # Sort section_list for deterministic selection
        selected_section = None
        for section in section_list:
            if section not in seen_sections:
                selected_section = section
                seen_sections.add(section)
                break

        if selected_section is None:
            continue
        
        if section_key == 'sent':
            first_group_keys = devset[devset[section_key] == selected_section][['subject_id', 'study_id', 'sequence', 'sent', 'section']].iloc[0]

            # Take the first group's all rows after grouping
            devset_col = devset[
                (devset['subject_id'] == first_group_keys['subject_id']) & 
                (devset['study_id'] == first_group_keys['study_id']) &
                (devset['sequence'] == first_group_keys['sequence']) &
                (devset['sent'] == first_group_keys['sent']) &
                (devset['section'] == first_group_keys['section'])
            ]
        elif section_key == 'report':
            first_group_keys = devset[devset[section_key] == selected_section][['subject_id', 'study_id', 'sequence', 'report', 'section']].iloc[0]
            devset_col = devset[
                (devset['subject_id'] == first_group_keys['subject_id']) & 
                (devset['study_id'] == first_group_keys['study_id']) &
                (devset['sequence'] == first_group_keys['sequence']) &
                (devset['report'] == first_group_keys['report']) &
                (devset['section'] == first_group_keys['section'])
            ]
        else:
            first_group_keys = devset[devset[section_key] == selected_section][['subject_id', 'study_id', 'section', 'section_report']].iloc[0]
            devset_col = devset[
                (devset['subject_id'] == first_group_keys['subject_id']) & 
                (devset['study_id'] == first_group_keys['study_id'])&
                (devset['section'] == first_group_keys['section'])&
                (devset['section_report'] == first_group_keys['section_report'])
            ]

        filled_cols = sorted([col for col in query_cols if not devset_col[col].isna().all()])
        filled_entities = sorted(list(devset_col['ent'].unique()))
        query_cols = sorted(list(set(query_cols) - set(filled_cols)))
        query_entities = sorted(list(set(query_entities) - set(filled_entities)))

        diverse_result = pd.concat([diverse_result, devset_col])
        shot_count += 1
        
    user_history = []
    assistant_history = []
    unique_sections = list(diverse_result[section_key].unique())

    for section in unique_sections[:args.n_retrieval]:
        if args.multi:
            user, assistant = get_fewshot_multi_turn(section, diverse_result[diverse_result[section_key] == section], args, vocab, vocab_lookup, vocab_keys)
        else:
            user, assistant = get_fewshot(section, diverse_result[diverse_result[section_key] == section], args, vocab, vocab_lookup, vocab_keys)
        user_history.append(user)
        assistant_history.append(assistant)

    return json_vocab_input, user_history, assistant_history

def get_dynamic_retrieval(query, result, vocab, devset, args):

    devset = devset.copy()
    
    ENTITY_CATEGORY = ['pf', 'cf', 'cof', 'cof/ncd', 'oth', 'ncd', 'pf/cf', 'patient info.']
    
    
    section_key = 'section_report' if args.unit == 'section' else 'sent'
    diverse_result = pd.DataFrame()
    vocab_lookup, vocab_keys = get_vocab_lookup(vocab)
    query_words_dict = {}
    ###### Generate JSON INPUT for New User Input ######
    # Note: candidates are extracted based on vocab

    json_vocab_input = {
        'report_section': []
    }

    query_words = []
    if section_key == 'sent':
        query_words = get_words(query, vocab, vocab_lookup, vocab_keys, args)
    else:
        # Extract numbered sentences from the query (format: "(1) text (2) text...")
        sentences = re.findall(r'\(\d+\)\s*(.*?)(?=\s*\(\d+\)|$)', query, re.DOTALL)
        sentences = [s.strip() for s in sentences if s.strip()]

        for idx, sentence in enumerate(sentences):      # idx = 0, 1, 2, ...
            
            all_words = list(set(get_words(sentence, vocab, vocab_lookup, vocab_keys, args)))
            
            query_words.extend(all_words)
            
            if ('gt' in args.candidate_type) or (args.candidate_type == 'no_candidates'):
                candidates = all_words
            
            elif args.candidate_type == 'vocab_so':
                candidates = all_words
            
            elif args.candidate_type == 'vocab_s': # entity category Only
                ent_words = []
                for word in all_words:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    if all(cat in ENTITY_CATEGORY for cat in cat_list):
                        ent_words.append(word)
                candidates = ent_words
            
            elif args.candidate_type == 'vocab_ent_rcg':
                word_cat_list = []
                for word in all_words:
                    cat_list = list(set(vocab[vocab['target_term'] == word]['category']))
                    word_cat_list.extend([(word, cat_list)])
                candidates = word_cat_list

            
            if args.candidate_usage != 1:
                target_length = int(len(candidates) * args.candidate_usage)
                step = len(candidates) // max(target_length, 1)
                candidates = candidates[::step][:target_length]
                        
            json_vocab_input['report_section'].append({
                'sent_idx': idx+1,
                'sentence': sentence,
                'candidates': candidates
            })
    
    query_words = list(set(query_words))
    query_cols, query_entity_cats, query_entities = get_cols(query_words, vocab)

    # Convert sets to sorted lists for deterministic order
    query_cols = sorted(query_cols)
    query_entity_cats = sorted(query_entity_cats)
    query_entities = sorted(query_entities)
    
    # Sort devset by sentence order in result and check for duplicate sentences
    ordered_sections = result['section'].unique().tolist()
    devset.loc[:, f'{section_key}_order'] = devset[f'{section_key}'].map({s: i for i, s in enumerate(ordered_sections)})
    
    # Calculate how many query_entities are included in each section_report
    if len(query_entities) > 0:
        # Function to check if entities are included in the section_report
        def count_entities(text):
            return sum(1 for entity in query_entities if entity in text)
        
        # Add new column with count of entities included in each section_report
        devset.loc[:, 'entity_count'] = devset[section_key].apply(count_entities)
        
        # Sort first by entity count (descending) then by section_order
        devset = devset.sort_values(['entity_count', f'{section_key}_order'], ascending=[False, True])
    else:
        # If no entities, sort only by section_order
        devset = devset.sort_values(f'{section_key}_order')
    
    # Remove temporary columns after sorting
    devset = devset.drop(['entity_count', f'{section_key}_order'], axis=1, errors='ignore')
    
    
    seen_sections = set()
    shot_count = 0
    
    # query = '(1) ..... (2) ..... (3) .....'
    # query_entities = [opacity, cardiomegaly, a, b, c]
    # attr = [no change, morphology, distribution, improvement, worsening, d, e, f]
    
    # shot 1 [opacity, cardiomegaly, a], [d, e]
    
    # Process columns first, then entity categories
    while len(query_cols) > 0 or len(query_entities) > 0 or shot_count < args.n_retrieval:
        section_list = []
        
        if len(query_entities) > 0:
            entity = query_entities[0]  # Take first element instead of random pop
            query_entities = query_entities[1:]  # Remove first element
            section_list = list(devset[devset['ent'] == entity][section_key].unique())
        elif len(query_cols) > 0:
            col = query_cols[0]  # Take first element instead of random pop
            query_cols = query_cols[1:]  # Remove first element
            section_list = list(devset[devset[col].notna()][section_key].unique())
        elif shot_count < args.n_retrieval:
            section_list = list(devset[section_key].unique())

        if len(section_list) == 0:
            continue

        # Sort section_list for deterministic selection
        selected_section = None
        for section in section_list:
            if section not in seen_sections:
                selected_section = section
                seen_sections.add(section)
                break

        if selected_section is None:
            continue
        

        first_group_keys = devset[devset[section_key] == selected_section][['subject_id', 'study_id', 'section', 'section_report']].iloc[0]
        devset_col = devset[
            (devset['subject_id'] == first_group_keys['subject_id']) & 
            (devset['study_id'] == first_group_keys['study_id'])&
            (devset['section'] == first_group_keys['section'])&
            (devset['section_report'] == first_group_keys['section_report'])
        ]

        filled_cols = sorted([col for col in query_cols if not devset_col[col].isna().all()])
        filled_entities = sorted(list(devset_col['ent'].unique()))
        query_cols = sorted(list(set(query_cols) - set(filled_cols)))
        query_entities = sorted(list(set(query_entities) - set(filled_entities)))

        diverse_result = pd.concat([diverse_result, devset_col])
        shot_count += 1
        
    user_history = []
    assistant_history = []
    unique_sections = list(diverse_result[section_key].unique())

    for section in unique_sections:
        user, assistant = get_fewshot(section, diverse_result[diverse_result[section_key] == section], args, vocab, vocab_lookup, vocab_keys)
        user_history.append(user)
        assistant_history.append(assistant)

    return json_vocab_input, user_history, assistant_history

def retreive_query_related_fewshot(devset, query, query_subject_id, args):
    """
    Search for similar sentences using BM25
    
    Args:
        query (str): Query string to search for
        n_retrieval (int): Number of results to retrieve
        
    Returns:
        list: List of matching sentences with their metadata from devset
    """

    # BM25
    if args.unit == 'sent':
        corpus_col = 'sent'
    elif args.unit == 'section':
        corpus_col = 'section_report'

    corpus = devset[devset['subject_id'] != query_subject_id][corpus_col].dropna().tolist()
    devset = devset[devset['subject_id'] != query_subject_id]

    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Get scores
    tokenized_query = query.split()
    corpus_scores = bm25.get_scores(tokenized_query)

    # Create dataframe
    corpus_ranked = pd.DataFrame({'section': corpus, 'score': corpus_scores})
    corpus_ranked = corpus_ranked.sort_values(by='score', ascending=False)

    vocab = pd.read_csv(args.vocab_path)
        
    # If diverse retrieval is True, get diverse retrieval
    if args.dynamic_retrieval:
        json_vocab_input, user_history, assistant_history = get_dynamic_retrieval(query, corpus_ranked, vocab, devset, args)
    else:
        json_vocab_input, user_history, assistant_history = get_n_retrieval(query, corpus_ranked, vocab, devset, args)

    return json_vocab_input, user_history, assistant_history

def parse_eval_format(idx, text):
    pred_triplet = defaultdict(list)
    if isinstance(text, dict):
        # Check if text is already in the expected format with relation keys
        if any(key.lower() in RELATIONS for key in text.keys()):
            # Already in the correct format, just ensure keys are lowercase
            for relation in text:
                relation_lower = relation.lower()
                if relation_lower in RELATIONS:
                    pred_triplet[relation_lower] = text[relation]
            return idx, pred_triplet
        else:
            # Try to convert from structured output format
            try:
                return idx, convert_to_sr_structure(text)
            except Exception as e:
                print(f"Error converting structured output: {e}")

    elif isinstance(text, list):
        raise ValueError(f"Unsupported text format: expected dict or str, got list.")

    elif isinstance(text, str):
        try:
            # Check if text is a JSON string
            if "{" in text and "}" in text:
                parsed_json = json.loads(text)
                
                for entity in parsed_json["entities"]:
                    entity_name = entity["name"].lower().strip()
                    sent_idx = entity.get("sent_idx", None)
                    ent_idx = entity.get("ent_idx", None)
                    
                    for relation in entity["relations"]:
                        relation_name = relation["relation"].lower()
                        relation_value = relation["value"].lower().strip()
                        obj_ent_idx = relation.get("obj_ent_idx", None)
                        
                        if relation_name in RELATIONS:
                            pred_triplet[relation_name].append((
                                entity_name, relation_name, relation_value,
                                sent_idx, ent_idx, obj_ent_idx
                            ))
                return idx, pred_triplet                
        except:
            pass
        
    else:
        raise ValueError(f"Unsupported text format: expected dict or str, got {type(text).__name__}.")
    
def generate_graph(eval_path, args):
    # Load results
    results = {}
    results_no_sent_idx = {}
    
    if args.mode == 'rexval':
        triplets_path = f'{eval_path}/SRO_result_by_model.json'
        if os.path.exists(triplets_path):
            results['triplets'] = json.load(open(triplets_path))

        subj_path = f'{eval_path}/SR_result_by_model.json'
        if os.path.exists(subj_path):
            results['subj'] = json.load(open(subj_path))
        
        # For rexval mode, create separate visualizations for each model
        if 'triplets' in results or 'subj' in results:
            # Get all model names from either triplets or subj results
            model_names = []
            if 'triplets' in results:
                model_names = list(results['triplets'].keys())
            elif 'subj' in results:
                model_names = list(results['subj'].keys())
            
            # Create a separate visualization for each model
            for model_name in model_names:
                create_model_visualization(eval_path, model_name, results, args)
    else:
        triplets_path = f'{eval_path}/SRO_result.json'
        if os.path.exists(triplets_path):
            results['triplets'] = json.load(open(triplets_path))

        subj_path = f'{eval_path}/SR_result.json'
        if os.path.exists(subj_path):
            results['subj'] = json.load(open(subj_path))
        
        # Create a single visualization for non-rexval mode
        create_standard_visualization(eval_path, results, args)
        
        triplets_path = f'{eval_path}/SRO_result_no_sent_idx.json'
        if os.path.exists(triplets_path):
            results_no_sent_idx['triplets'] = json.load(open(triplets_path))

        subj_path = f'{eval_path}/SR_result_no_sent_idx.json'
        if os.path.exists(subj_path):
            results_no_sent_idx['subj'] = json.load(open(subj_path))
        
        # Create a single visualization for non-rexval mode
        create_standard_visualization(eval_path, results_no_sent_idx, args, no_sent_idx=True)


def create_model_visualization(eval_path, model_name, results, args):
    """Create visualization for a specific model in rexval mode"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    x = np.arange(len(RELATIONS) + 1)  # Add 1 for average
    width = 0.2
    
    # Calculate F1 scores
    f1_scores = {}
    metrics = {}  # Store TP, TN, FP, FN for each relation
    
    for key in ['triplets', 'subj']:
        if key not in results or model_name not in results[key]:
            continue
            
        result = results[key][model_name]
        scores = []
        metrics[key] = []
        
        for r in RELATIONS:
            if r in result:
                scores.append(result[r]['f1'])
                # Store metrics for each relation
                metrics[key].append({
                    'tp': result[r].get('tp', 0),
                    'tn': result[r].get('tn', 0),
                    'fp': result[r].get('fp', 0),
                    'fn': result[r].get('all', 0) - result[r].get('tp', 0) if key == 'subj' else result[r].get('miss', 0)
                })
            else:
                scores.append(0.0)
                metrics[key].append({'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
        
        # Add average metrics
        avg_metrics = {
            'tp': sum(m['tp'] for m in metrics[key]) // len(RELATIONS),
            'tn': sum(m['tn'] for m in metrics[key]) // len(RELATIONS),
            'fp': sum(m['fp'] for m in metrics[key]) // len(RELATIONS),
            'fn': sum(m['fn'] for m in metrics[key]) // len(RELATIONS)
        }
        metrics[key].append(avg_metrics)
        
        # Add average of non-zero scores
        scores.append(np.mean([s for s in scores if s > 0]))
        f1_scores[key] = scores
        
        # Calculate overall F1 based on total TP, FP, FN
        total_tp = sum(m['tp'] for m in metrics[key][:-1])  # Exclude the average we just added
        total_fp = sum(m['fp'] for m in metrics[key][:-1])
        total_fn = sum(m['fn'] for m in metrics[key][:-1])
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[f'{key}_overall'] = [overall_f1] * len(RELATIONS) + [overall_f1]  # Same value for all relations + average
        metrics[f'{key}_overall'] = [{
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': 0  # Not typically used for F1
        }] * (len(RELATIONS) + 1)  # Same metrics for all relations + average
    
    # Plot configurations
    plot_configs = [
        (ax1, 'bar', f'{model_name} - Triplets F1 Scores (Bar Plot)'),
        (ax2, 'line', f'{model_name} - Triplets F1 Scores (Line Plot)'),
        (ax3, 'bar', f'{model_name} - Subject F1 Scores (Bar Plot)'),
        (ax4, 'line', f'{model_name} - Subject F1 Scores (Line Plot)')
    ]
    
    for ax, plot_type, title in plot_configs:
        result_type = 'triplets' if 'Triplets' in title else 'subj'
        
        if result_type in f1_scores:
            if plot_type == 'bar':
                # Plot per-relation F1 scores
                bars = ax.bar(x - width, f1_scores[result_type], width, label=f'{args.output_format} (Per Relation)', alpha=0.8)
                
                # Add metrics text on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    metric = metrics[result_type][i]
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
                # Plot overall F1 score
                if f'{result_type}_overall' in f1_scores:
                    bars = ax.bar(x, f1_scores[f'{result_type}_overall'], width, label=f'{args.output_format} (Overall)', alpha=0.8)
                    # Add overall metrics
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        metric = metrics[f'{result_type}_overall'][i]
                
                if f'{result_type}_with_gpt' in f1_scores:
                    bars = ax.bar(x + width, f1_scores[f'{result_type}_with_gpt'], width, label=f'{args.output_format} with GPT', alpha=0.8)
                    # Add metrics for second set if available
                    if f'{result_type}_with_gpt' in metrics:
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            metric = metrics[f'{result_type}_with_gpt'][i]
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                    ha='center', va='bottom', fontsize=8, rotation=0)
                
                if f'{result_type}_with_gpt2' in f1_scores:
                    bars = ax.bar(x + width*2, f1_scores[f'{result_type}_with_gpt2'], width, label=f'{args.output_format} with GPT2', alpha=0.8)
                    # Add metrics for third set if available
                    if f'{result_type}_with_gpt2' in metrics:
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            metric = metrics[f'{result_type}_with_gpt2'][i]
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                    ha='center', va='bottom', fontsize=8, rotation=0)
            else:  # line plot
                # Plot per-relation F1 scores
                line = ax.plot(x, f1_scores[result_type], marker='s', linestyle='-', label=f'{args.output_format} (Per Relation)', alpha=0.8)
                # Add metrics as annotations on line points
                for i, (xi, yi) in enumerate(zip(x, f1_scores[result_type])):
                    metric = metrics[result_type][i]
                    ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                # Plot overall F1 score
                if f'{result_type}_overall' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_overall'], marker='d', linestyle='--', 
                                  label=f'{args.output_format} (Overall)', alpha=0.8)
                    # Add overall metrics
                    for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_overall'])):
                        metric = metrics[f'{result_type}_overall'][i]
                        if i == len(x) // 2:  # Only annotate in the middle to avoid clutter
                            ax.annotate(f'Overall: TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, -30), textcoords='offset points',
                                        ha='center', va='top', fontsize=8, 
                                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
                
                if f'{result_type}_with_gpt' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_with_gpt'], marker='o', linestyle='-', label=f'{args.output_format} with GPT', alpha=0.8)
                    if f'{result_type}_with_gpt' in metrics:
                        for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_with_gpt'])):
                            metric = metrics[f'{result_type}_with_gpt'][i]
                            ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                if f'{result_type}_with_gpt2' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_with_gpt2'], marker='^', linestyle='-', label=f'{args.output_format} with GPT2', alpha=0.8)
                    if f'{result_type}_with_gpt2' in metrics:
                        for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_with_gpt2'])):
                            metric = metrics[f'{result_type}_with_gpt2'][i]
                            ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                ax.grid(True)
        
        ax.set_ylabel('F1 Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(RELATIONS + ['Average'], rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(f'{eval_path}/figures', exist_ok=True)
    plt.savefig(f'{eval_path}/figures/{model_name}_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation visualization for {model_name} saved to {eval_path}/figures/{model_name}_evaluation_results.png")

def create_standard_visualization(eval_path, results, args, no_sent_idx=False):
    """Create standard visualization for non-rexval mode"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    x = np.arange(len(RELATIONS) + 1)  # Add 1 for average
    width = 0.2
    
    # Calculate F1 scores
    f1_scores = {}
    metrics = {}  # Store TP, TN, FP, FN for each relation
    
    for key, result in results.items():
        scores = []
        metrics[key] = []
        
        for r in RELATIONS:
            if r in result:
                scores.append(result[r]['f1'])
                # Store metrics for each relation
                metrics[key].append({
                    'tp': result[r].get('tp', 0),
                    'tn': result[r].get('tn', 0),
                    'fp': result[r].get('fp', 0),
                    'fn': result[r].get('all', 0) - result[r].get('tp', 0) if key == 'subj' else result[r].get('miss', 0)
                })
            else:
                scores.append(0.0)
                metrics[key].append({'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
        
        # Add average metrics
        avg_metrics = {
            'tp': sum(m['tp'] for m in metrics[key]) // len(RELATIONS),
            'tn': sum(m['tn'] for m in metrics[key]) // len(RELATIONS),
            'fp': sum(m['fp'] for m in metrics[key]) // len(RELATIONS),
            'fn': sum(m['fn'] for m in metrics[key]) // len(RELATIONS)
        }
        metrics[key].append(avg_metrics)
        
        # Add average of non-zero scores
        scores.append(np.mean([s for s in scores if s > 0]))
        f1_scores[key] = scores
        
        # Calculate overall F1 based on total TP, FP, FN
        total_tp = sum(m['tp'] for m in metrics[key][:-1])  # Exclude the average we just added
        total_fp = sum(m['fp'] for m in metrics[key][:-1])
        total_fn = sum(m['fn'] for m in metrics[key][:-1])
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[f'{key}_overall'] = [overall_f1] * len(RELATIONS) + [overall_f1]  # Same value for all relations + average
        metrics[f'{key}_overall'] = [{
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': 0  # Not typically used for F1
        }] * (len(RELATIONS) + 1)  # Same metrics for all relations + average
    
    # Plot configurations
    plot_configs = [
        (ax1, 'bar', 'triplets F1 Scores (Bar Plot)'),
        (ax2, 'line', 'triplets F1 Scores (Line Plot)'),
        (ax3, 'bar', 'Subject F1 Scores (Bar Plot)'),
        (ax4, 'line', 'Subject F1 Scores (Line Plot)')
    ]
    
    for ax, plot_type, title in plot_configs:
        result_type = 'triplets' if 'triplets' in title else 'subj'
        
        if result_type in f1_scores:
            if plot_type == 'bar':
                # Plot per-relation F1 scores
                bars = ax.bar(x - width, f1_scores[result_type], width, label=f'{args.output_format} (Per Relation)', alpha=0.8)
                
                # Add metrics text on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    metric = metrics[result_type][i]
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
                # Plot overall F1 score
                if f'{result_type}_overall' in f1_scores:
                    bars = ax.bar(x, f1_scores[f'{result_type}_overall'], width, label=f'{args.output_format} (Overall)', alpha=0.8)
                    # Add overall metrics
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        metric = metrics[f'{result_type}_overall'][i]
                
                if f'{result_type}_with_gpt' in f1_scores:
                    bars = ax.bar(x + width, f1_scores[f'{result_type}_with_gpt'], width, label=f'{args.output_format} with GPT', alpha=0.8)
                    # Add metrics for second set if available
                    if f'{result_type}_with_gpt' in metrics:
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            metric = metrics[f'{result_type}_with_gpt'][i]
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                    ha='center', va='bottom', fontsize=8, rotation=0)
                
                if f'{result_type}_with_gpt2' in f1_scores:
                    bars = ax.bar(x + width*2, f1_scores[f'{result_type}_with_gpt2'], width, label=f'{args.output_format} with GPT2', alpha=0.8)
                    # Add metrics for third set if available
                    if f'{result_type}_with_gpt2' in metrics:
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            metric = metrics[f'{result_type}_with_gpt2'][i]
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                    f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                    ha='center', va='bottom', fontsize=8, rotation=0)
            else:  # line plot
                # Plot per-relation F1 scores
                line = ax.plot(x, f1_scores[result_type], marker='s', linestyle='-', label=f'{args.output_format} (Per Relation)', alpha=0.8)
                # Add metrics as annotations on line points
                for i, (xi, yi) in enumerate(zip(x, f1_scores[result_type])):
                    metric = metrics[result_type][i]
                    ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                # Plot overall F1 score
                if f'{result_type}_overall' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_overall'], marker='d', linestyle='--', 
                                  label=f'{args.output_format} (Overall)', alpha=0.8)
                    # Add overall metrics
                    for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_overall'])):
                        metric = metrics[f'{result_type}_overall'][i]
                        if i == len(x) // 2:  # Only annotate in the middle to avoid clutter
                            ax.annotate(f'Overall: TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, -30), textcoords='offset points',
                                        ha='center', va='top', fontsize=8, 
                                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
                
                if f'{result_type}_with_gpt' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_with_gpt'], marker='o', linestyle='-', label=f'{args.output_format} with GPT', alpha=0.8)
                    if f'{result_type}_with_gpt' in metrics:
                        for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_with_gpt'])):
                            metric = metrics[f'{result_type}_with_gpt'][i]
                            ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                if f'{result_type}_with_gpt2' in f1_scores:
                    line = ax.plot(x, f1_scores[f'{result_type}_with_gpt2'], marker='^', linestyle='-', label=f'{args.output_format} with GPT2', alpha=0.8)
                    if f'{result_type}_with_gpt2' in metrics:
                        for i, (xi, yi) in enumerate(zip(x, f1_scores[f'{result_type}_with_gpt2'])):
                            metric = metrics[f'{result_type}_with_gpt2'][i]
                            ax.annotate(f'TP:{metric["tp"]}\nFP:{metric["fp"]}\nFN:{metric["fn"]}',
                                        xy=(xi, yi), xytext=(0, 10), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                
                ax.grid(True)
        
        ax.set_ylabel('F1 Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(RELATIONS + ['Average'], rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    if no_sent_idx:
        os.makedirs(f'{eval_path}/figures', exist_ok=True)
        plt.savefig(f'{eval_path}/figures/evaluation_results_no_sent_idx.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Evaluation visualization saved to {eval_path}/figures/evaluation_results_no_sent_idx.png")
    else:
        os.makedirs(f'{eval_path}/figures', exist_ok=True)
        plt.savefig(f'{eval_path}/figures/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Evaluation visualization saved to {eval_path}/figures/evaluation_results.png")

def create_fp(eval_path, gold_file_path):
    
    sro_review_path = os.path.join(eval_path, 'all.json')
    #sr_review_path = os.path.join(eval_path, 'all_subject.json')
    gold_file = json.load(open(gold_file_path))
    json_data = json.load(open(sro_review_path))
    
    fp_data = []
    for data in json_data:
        wrong_triplet_list = data['wrong_triplet_list']
        for triplet in wrong_triplet_list:
            fp_data.append({
                'custom_id': data['custom_id'],
                'study_id': gold_file[data['custom_id']]['study_id'],
                'sentence': data['sentence'],
                'wrong_triplet': triplet,
            })
    
    # Convert to CSV
    fp_df = pd.DataFrame(fp_data)
    csv_path = os.path.join(eval_path, 'false_positives.csv')
    fp_df.to_csv(csv_path, index=False)
    print(f"False positives saved to {csv_path}")
    

def create_fn(eval_path, gold_file_path):
    
    sro_review_path = os.path.join(eval_path, 'all.json')
    #sr_review_path = os.path.join(eval_path, 'all_subject.json')
    gold_file = json.load(open(gold_file_path))
    json_data = json.load(open(sro_review_path))
    
    fn_data = []
    for data in json_data:
        wrong_triplet_list = data['miss_triplet_list']
        for triplet in wrong_triplet_list:
            fn_data.append({
                'custom_id': data['custom_id'],
                'study_id': gold_file[data['custom_id']]['study_id'],
                'sentence': data['sentence'],
                'miss_triplet': triplet,
            })
    
    # Convert to CSV
    fn_df = pd.DataFrame(fn_data)
    csv_path = os.path.join(eval_path, 'false_negatives.csv')
    fn_df.to_csv(csv_path, index=False)
    print(f"False negatives saved to {csv_path}")
    
# test reports
def extract_triplets(row):
    """Extract triplets from a row of grouped data"""
    triplet_list = []
    same_triplet_list = []
    sent_idx = row['sent_idx']
  
    for relation in RELATIONS:
        ent = row['ent']
        if str(row[relation]) != 'nan':
            
            if relation in ['evidence', 'associate']:
                # Split by comma and process pairs of values and obj_ent_idx
                relation_values = row[relation].split(', ')
                i = 0
                while i < len(relation_values):
                    value = relation_values[i].strip()
                    # Check if the next item is an obj_ent_idx reference
                    if i+1 < len(relation_values) and 'obj_ent_idx' in relation_values[i+1]:
                        # Extract the index number from obj_ent_idx
                        obj_ent_idx = int(relation_values[i+1].replace('obj_ent_idx', ''))
                        triplet = (ent, relation, value, row['sent_idx'], row['ent_idx'], obj_ent_idx)
                        i += 2  # Skip to the next pair
                    else:
                        print("row", row)
                        raise ValueError(f"No obj_ent_idx found for value: {value}")
                    same_triplet_list.append([triplet])
                    triplet_list.append(triplet)

            else:
                triplet = (ent, relation, row[relation], row['sent_idx'], int(row['ent_idx']), None)
                same_triplet_list.append([triplet])
                triplet_list.append(triplet)
                
    return triplet_list, same_triplet_list, sent_idx

def create_report_level_data(df, args):
    """Create report level data entry"""

    section_list = ['hist', 'find', 'impr']
    
    hist = df[df['section'] == 'hist']
    find = df[df['section'] == 'find']
    impr = df[df['section'] == 'impr']

    report = ""
    if not hist.empty:
        report += f"{hist['report'].iloc[0]}\n"
    if not find.empty:
        report += f"{find['report'].iloc[0]}\n"
    if not impr.empty:
        report += f"{impr['report'].iloc[0]}"

    report_triplet_list = []
    report_same_triplet_list = []

    for section in section_list:    
        df_section = df[df['section'] == section]
        for row_idx, row in df_section.iterrows():
            triplet_list, same_triplet_list = extract_triplets(row)
            report_triplet_list.extend(triplet_list)
            report_same_triplet_list.extend(same_triplet_list)

    return {
        'subject_id': df['subject_id'].iloc[0],
        'study_id': df['study_id'].iloc[0],
        'passage': report,
        'relations': RELATIONS,
        'triplet_list': report_triplet_list,
        'same_triplet_list': report_same_triplet_list,
        'data_from': args.mode
    }

def prepend_idx(df_section, args, col=None):
    """Add index numbers to sentences based on their position in the report"""
    
    numbered_report = ""
    numbered_report_list = []
    
    if args.mode in ['maira', 'maira_cascade', 'medversa', 'rgrg', 'cvt2distilgpt2', 'medgemma', 'lingshu', 'silver_eval', 'libra', 'chexagent']:
        for idx, row in df_section.iterrows():
            text = row[f'{args.mode}_report'] if args.mode != 'silver_eval' else row['report']
            
            if pd.isna(text) or not isinstance(text, str) or len(text) < 2:
                # Handle missing or non-string data
                continue
            
            # Split text into sentences using regex
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            for sent_idx, sent in enumerate(sentences):
                if sent.strip():  # Skip empty sentences
                    numbered_report += f"({sent_idx+1}) {sent.strip()}"
                    numbered_report_list.append(f"({sent_idx+1}) {sent.strip()}")
            return numbered_report.strip(), numbered_report_list
        
    elif args.mode == 'rexerr':
        if df_section['section'].iloc[0] == 'find':
            # Check if error_findings values are nan for all rows
            all_nan = df_section['error_findings'].isna().all()
            if all_nan:
                print("Warning: All error_findings values are NaN")
                return "", []
                
            for idx, row in df_section.iterrows():
                text = row['error_findings']

                if pd.isna(text) or not isinstance(text, str):
                    continue  # Skip this row and move to next
                
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
                for sent_idx, sent in enumerate(sentences):
                    if sent.strip():  # Skip empty sentences
                        numbered_report += f"({sent_idx+1}) {sent.strip()}"
                        numbered_report_list.append(f"({sent_idx+1}) {sent.strip()}")
            
            # Return results after processing all rows
            return numbered_report.strip(), numbered_report_list
            
        elif df_section['section'].iloc[0] == 'impr':
            # Check if error_impression values are nan for all rows
            all_nan = df_section['error_impression'].isna().all()
            if all_nan:
                print("Warning: All error_impression values are NaN")
                return "", []
                
            for idx, row in df_section.iterrows():
                text = row['error_impression']
                
                if pd.isna(text) or not isinstance(text, str):
                    continue  # Skip this row and move to next
                
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
                for sent_idx, sent in enumerate(sentences):
                    if sent.strip():  # Skip empty sentences
                        numbered_report += f"({sent_idx+1}) {sent.strip()}"
                        numbered_report_list.append(f"({sent_idx+1}) {sent.strip()}")
            
            # Return results after processing all rows
            return numbered_report.strip(), numbered_report_list
        
        # Default return value
        return "", []
    else:
        if 'section_report' not in df_section.columns:
            if 'sent_idx' in df_section.columns:
                max_num = int(df_section['sent_idx'].max())
                for idx in range(max_num):
                    sent = df_section[df_section['sent_idx'] == idx+1]['sent'].values[0]
                    numbered_report += f"({idx+1}) {sent}"
                    numbered_report_list.append(f"({idx+1}) {sent}")
            else:
                for idx, row in df_section.iterrows():
                    text = row[col]
                    # Split sentences while preserving numbers and periods (e.g. "1.", "2.")
                    # Modified to not split patterns starting with number + period
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\d\.)(?<=\.|\?|\!)\s', text)
                    for sent_idx, sent in enumerate(sentences):
                        if sent.strip():  # Skip empty sentences
                            numbered_report += f"({sent_idx+1}) {sent.strip()}"
                            numbered_report_list.append(f"({sent_idx+1}) {sent.strip()}")
            return numbered_report.strip(), numbered_report_list
        else:
            section_report = df_section['section_report'].iloc[0]
            numbered_report_list = []
            pattern = r'\(\d+\)\s*([^(]*?)(?=\s*\(\d+\)|$)'
            matches = re.finditer(pattern, section_report)
            
            for match in matches:
                # Find the complete text fragment including the original number pattern
                start_pos = match.start()
                # Find the number pattern
                number_match = re.search(r'\(\d+\)', section_report[start_pos:match.end()])
                if number_match:
                    number_part = number_match.group(0)
                    text_part = match.group(1).strip()
                    numbered_report_list.append(f"{number_part} {text_part}")
            
            # If the list is empty, add the original text to the list
            if not numbered_report_list:
                numbered_report_list = [section_report]
                
            return section_report, numbered_report_list

def create_section_level_data(df, args):
    """Create section level data entry"""

    vocab = pd.read_csv(args.vocab_path)
    vocab['category'] = vocab['category'].replace(['lf', 'If'], 'pf')

    if 'section' in df.columns:
        if args.mode in ['maira', 'maira_cascade', 'rexerr', 'medversa', 'rgrg', 'cvt2distilgpt2', 'lingshu', 'medgemma', 'libra', 'chexagent']:
            section_list = ['find', 'impr']
        else:
            section_list = ['hist', 'find', 'impr']
    
        section_data = []
        for section in section_list:
            section_triplet_list = []
            section_same_triplet_list = []

            df_section = df[df['section'] == section]
            # Skip empty sections
            if df_section.empty:
                continue

            gt_sro_review = defaultdict(list)
            gt_sro = defaultdict(list)
            gt_so = defaultdict(list)
            gt_s = defaultdict(list)
            gt_ent_rcg = defaultdict(list)
            gt_sro_rmd = defaultdict(list)
            gt_so_rmd = defaultdict(list)
            gt_s_rmd = defaultdict(list)
            gt_ent_rcg_rmd = defaultdict(list)
            
            if not df_section.empty:
                report_section, _ = prepend_idx(df_section, args)

                if len(df_section) == 0:
                    continue
                
                if args.mode == 'silver_eval':
                    section_data.append({
                        'subject_id': df['subject_id'].iloc[0],
                        'study_id': df['study_id'].iloc[0],
                        'section': section,
                        'passage': report_section,
                        'relations': RELATIONS,
                        'triplet_list': None,
                        'same_triplet_list': None,
                        'data_from': args.mode,
                        'json_input': None
                    })
                    continue

                df_section_idx_removed = df_section.copy()
                for row_idx, row in df_section_idx_removed.iterrows():
                    triplet_list, same_triplet_list, sent_idx = extract_triplets(row)
                    section_triplet_list.extend(triplet_list)
                    section_same_triplet_list.extend(same_triplet_list)

                    for triplet in triplet_list:
                        gt_sro[sent_idx].extend([(triplet[0], triplet[1], triplet[2])])
                        
                        ent_cat_list = vocab[vocab['target_term'] == triplet[0]]['category'].tolist()
                        gt_ent_rcg[sent_idx].extend([(triplet[0], ent_cat_list)])
                        gt_so[sent_idx].extend([(triplet[0])])
                        if triplet[1] not in ['cat', 'dx_status', 'dx_certainty', 'associate', 'evidence']:
                            gt_so[sent_idx].extend([(triplet[2])])
                            gt_ent_rcg[sent_idx].extend([(triplet[2], [triplet[1]])])

                        gt_s[sent_idx].extend([(triplet[0])])
                        
                        
                # Remove duplicates from gt_s_rmd
                for sent_idx, entities in gt_s.items():
                    gt_s_rmd[sent_idx] = list(dict.fromkeys(entities))
                    
                for sent_idx, entities in gt_so.items():
                    gt_so_rmd[sent_idx] = list(dict.fromkeys(entities))

                for sent_idx, entities in gt_ent_rcg.items():
                    # Convert list of tuples to dictionary to remove duplicates
                    # This won't work with dict.fromkeys since tuples with same entity name but different categories will be treated as duplicates
                    unique_entities = {}
                    for entity, categories in entities:
                        if entity in unique_entities:
                            # If entity already exists, merge the categories
                            unique_entities[entity] = list(set(unique_entities[entity] + categories))
                        else:
                            unique_entities[entity] = categories
                    
                    # Convert back to list of tuples
                    gt_ent_rcg_rmd[sent_idx] = [(entity, categories) for entity, categories in unique_entities.items()]
                
                # Remove duplicates from gt_sro
                for sent_idx, triplets in gt_sro.items():
                    # Use a set to track unique triplets
                    unique_triplets = []
                    seen = set()
                    for triplet in triplets:
                        # Create a hashable representation of the triplet
                        triplet_key = (triplet[0], triplet[1], triplet[2])
                        if triplet_key not in seen:
                            seen.add(triplet_key)
                            unique_triplets.append(triplet)
                    gt_sro_rmd[sent_idx] = unique_triplets

                if args.candidate_type == 'gt_sro_review':
                    final_candidates = gt_sro_rmd
                elif args.candidate_type == 'gt_sro':
                    final_candidates = gt_sro_rmd
                elif args.candidate_type == 'gt_so':
                    final_candidates = gt_so_rmd
                elif args.candidate_type == 'gt_s':
                    final_candidates = gt_s_rmd
                elif args.candidate_type == 'gt_ent_rcg':
                    final_candidates = gt_ent_rcg_rmd
                else:
                    final_candidates = gt_so_rmd

                if args.mode not in ['maira', 'maira_cascade', 'rexerr', 'medversa', 'rgrg', 'cvt2distilgpt2', 'medgemma', 'lingshu', 'silver_eval', 'libra', 'chexagent']:
                    json_data = {
                        'report_section': []
                    }
                    for sent_idx in range(1, df_section['sent_idx'].max() + 1):
                        if args.candidate_type == 'no_candidates':
                            json_data['report_section'].append({
                                'sent_idx': sent_idx,
                                'sentence': df_section[df_section['sent_idx'] == sent_idx]['sent'].values[0],
                            })
                        else:
                            json_data['report_section'].append({
                                'sent_idx': sent_idx,
                                'sentence': df_section[df_section['sent_idx'] == sent_idx]['sent'].values[0],
                                'candidates': final_candidates[sent_idx]
                            })

                    section_data.append({
                        'subject_id': df['subject_id'].iloc[0],
                        'study_id': df['study_id'].iloc[0],
                        'section': section,
                        'passage': report_section,
                        'relations': RELATIONS,
                        'triplet_list': section_triplet_list,
                        'same_triplet_list': section_same_triplet_list,
                        'data_from': args.mode,
                        'json_input': json_data # JSON input used when GT candidate or no-candidate mode is selected
                    })
                    
                else:
                    section_data.append({
                        'subject_id': df['subject_id'].iloc[0],
                        'study_id': df['study_id'].iloc[0],
                        'section': section,
                        'passage': report_section,
                        'relations': RELATIONS,
                        'triplet_list': section_triplet_list,
                        'same_triplet_list': section_same_triplet_list,
                        'data_from': args.mode,
                        'json_input': None
                    })
                    
    else:
        section_data = []
        for col in args.report_col_name:
            section_triplet_list = []
            section_same_triplet_list = []
            
            # Create a new dataframe with only the gt_report column
            report_section, _ = prepend_idx(df, args, col)
            # Iterate through each row in the dataframe
            section_data.append({
                'subject_id': int(df['subject_id'].iloc[0]) if 'subject_id' in df.columns else None,
                'study_id': int(df['study_id'].iloc[0]) if 'study_id' in df.columns else None,
                'section': col,
                'passage': report_section,
                'relations': RELATIONS,
                'triplet_list': None,
                'same_triplet_list': None,
                'data_from': args.mode
            })
            
            
    return section_data


def create_sent_level_data(df, args):
    """Create section level data entry"""

    vocab = pd.read_csv(args.vocab_path)
    
    if 'section' in df.columns:
        if args.mode in ['maira', 'maira_cascade', 'rexerr']:
            section_list = ['find', 'impr']
        else:
            section_list = ['hist', 'find', 'impr']
    
        section_data = []
        for section in section_list:
            section_triplet_list = []
            section_same_triplet_list = []

            df_section = df[df['section'] == section]
            
            if df_section.empty:
                continue

            gt_sro_review = defaultdict(list)
            gt_sro = defaultdict(list)
            gt_so = defaultdict(list)
            gt_s = defaultdict(list)
            gt_ent_rcg = defaultdict(list)
            gt_sro_rmd = defaultdict(list)
            gt_so_rmd = defaultdict(list)
            gt_s_rmd = defaultdict(list)
            gt_ent_rcg_rmd = defaultdict(list)
            
            if not df_section.empty:
                _, report_section_list = prepend_idx(df_section, args)

                if len(df_section) == 0:
                    continue
                
                df_section_idx_removed = df_section.copy()
                
                sent_idx_triplets = defaultdict(list)
                sent_idx_same_triplets = defaultdict(list)
                for row_idx, row in df_section_idx_removed.iterrows():
                    triplet_list, same_triplet_list, sent_idx = extract_triplets(row)
                    sent_idx_triplets[sent_idx].extend(triplet_list)
                    sent_idx_same_triplets[sent_idx].extend(same_triplet_list)
                    
                for sent_idx in sorted(sent_idx_triplets.keys()):
                    section_triplet_list.append(sent_idx_triplets[sent_idx])
                    section_same_triplet_list.append(sent_idx_same_triplets[sent_idx])

                    for triplet in triplet_list:
                        gt_sro[sent_idx].append((triplet[0], triplet[1], triplet[2]))
                        
                        ent_cat_list = vocab[vocab['target_term'] == triplet[0]]['category'].tolist()
                        gt_ent_rcg[sent_idx].append((triplet[0], ent_cat_list))
                        gt_so[sent_idx].append((triplet[0]))
                        if triplet[1] not in ['cat', 'dx_status', 'dx_certainty', 'associate', 'evidence']:
                            gt_so[sent_idx].append((triplet[2]))
                            gt_ent_rcg[sent_idx].append((triplet[2], [triplet[1]]))

                        gt_s[sent_idx].append((triplet[0]))
                        
                        
                # Remove duplicates from gt_s_rmd
                for sent_idx, entities in gt_s.items():
                    gt_s_rmd[sent_idx] = list(dict.fromkeys(entities))
                    
                for sent_idx, entities in gt_so.items():
                    gt_so_rmd[sent_idx] = list(dict.fromkeys(entities))

                for sent_idx, entities in gt_ent_rcg.items():
                    # Convert list of tuples to dictionary to remove duplicates
                    # This won't work with dict.fromkeys since tuples with same entity name but different categories will be treated as duplicates
                    unique_entities = {}
                    for entity, categories in entities:
                        if entity in unique_entities:
                            # If entity already exists, merge the categories
                            unique_entities[entity] = list(set(unique_entities[entity] + categories))
                        else:
                            unique_entities[entity] = categories
                    
                    # Convert back to list of tuples
                    gt_ent_rcg_rmd[sent_idx] = [(entity, categories) for entity, categories in unique_entities.items()]
                
                # Remove duplicates from gt_sro
                for sent_idx, triplets in gt_sro.items():
                    # Use a set to track unique triplets
                    unique_triplets = []
                    seen = set()
                    for triplet in triplets:
                        # Create a hashable representation of the triplet
                        triplet_key = (triplet[0], triplet[1], triplet[2])
                        if triplet_key not in seen:
                            seen.add(triplet_key)
                            unique_triplets.append(triplet)
                    gt_sro_rmd[sent_idx] = unique_triplets

                if args.candidate_type == 'gt_sro_review':
                    final_candidates = gt_sro_rmd
                elif args.candidate_type == 'gt_sro':
                    final_candidates = gt_sro_rmd
                elif args.candidate_type == 'gt_so':
                    final_candidates = gt_so_rmd
                elif args.candidate_type == 'gt_s':
                    final_candidates = gt_s_rmd
                elif args.candidate_type == 'gt_ent_rcg':
                    final_candidates = gt_ent_rcg_rmd
                else:
                    final_candidates = gt_so_rmd

                if args.mode not in ['maira', 'maira_cascade', 'rexerr']:
                    if len(section_triplet_list) != df_section['sent_idx'].max() :
                        print(f"len(section_triplet_list) != df_section['sent_idx'].max() : {len(section_triplet_list)} != {df_section['sent_idx'].max()}")
                    for sent_idx in range(1, df_section['sent_idx'].max() + 1):
                        if args.candidate_type == 'no_candidates':
                            section_data.append({
                            'subject_id': df['subject_id'].iloc[0],
                            'study_id': df['study_id'].iloc[0],
                            'section': section,
                            'passage': report_section_list[sent_idx-1],
                            'relations': RELATIONS,
                            'triplet_list': section_triplet_list[sent_idx-1],
                            'same_triplet_list': section_same_triplet_list[sent_idx-1],
                            'data_from': args.mode,
                            'json_input': {
                                'sent_idx': sent_idx,
                                'sentence': df_section[df_section['sent_idx'] == sent_idx]['sent'].values[0],
                            } 
                            })
                        else:
                            section_data.append({
                            'subject_id': df['subject_id'].iloc[0],
                            'study_id': df['study_id'].iloc[0],
                            'section': section,
                            'passage': report_section_list[sent_idx-1],
                            'relations': RELATIONS,
                            'triplet_list': section_triplet_list[sent_idx-1],
                            'same_triplet_list': section_same_triplet_list[sent_idx-1],
                            'data_from': args.mode,
                            'json_input': {
                                'sent_idx': sent_idx,
                                'sentence': df_section[df_section['sent_idx'] == sent_idx]['sent'].values[0],
                                'candidates': final_candidates[sent_idx]
                            }
                            })

                else:
                    for sent_idx in range(1, df_section['sent_idx'].max() + 1):
                        section_data.append({
                            'subject_id': df['subject_id'].iloc[0],
                            'study_id': df['study_id'].iloc[0],
                            'section': section,
                            'passage': report_section_list[sent_idx-1],
                            'relations': RELATIONS,
                            'triplet_list': section_triplet_list[sent_idx-1],
                            'same_triplet_list': section_same_triplet_list[sent_idx-1],
                            'data_from': args.mode,
                            'json_input': None
                        })
                    
    else:
        section_data = []
        for col in args.report_col_name:
            section_triplet_list = []
            section_same_triplet_list = []
            
            # Create a new dataframe with only the gt_report column
            _, report_section_list = prepend_idx(df, args, col)
            
            # Iterate through each row in the dataframe
            for sent_itr in report_section_list:
                section_data.append({
                    'subject_id': int(df['subject_id'].iloc[0]) if 'subject_id' in df.columns else None,
                    'study_id': int(df['study_id'].iloc[0]) if 'study_id' in df.columns else None,
                    'section': col,
                    'passage': sent_itr,
                    'relations': RELATIONS,
                    'triplet_list': None,
                    'same_triplet_list': None,
                    'data_from': args.mode
                })
            
            
    return section_data


def process_study_data(args_tuple):
    """Process a single study ID to create study data"""
    id, test_df, args = args_tuple
    
    df = test_df[test_df['study_id'] == id]
    results = []
    
    if args.unit == 'report':
        results.append((str(0), create_report_level_data(df, args)))
    elif args.unit == 'section':
        section_data = create_section_level_data(df, args)
        for i, data in enumerate(section_data):
            results.append((str(i), data))
    elif args.unit == 'sent':
        sent_data = create_sent_level_data(df, args)
        for i, data in enumerate(sent_data):
            results.append((str(i), data))
    
    return results

def create_input(args):
    if args.mode in ['gold_eval', 'maira', 'maira_cascade', 'rexerr', 'medversa', 'rgrg', 'cvt2distilgpt2', 'lingshu', 'medgemma', 'libra', 'chexagent']:
        gold = pd.read_csv(args.gold_path)
        gold = gold.copy()
        # gold.loc[:, 'cat'] = gold['cat'].replace(['lf', 'If'], 'pf')
        gold.loc[:, 'evidence'] = gold['evidence'].str.replace(r'idx(\d+)', r'obj_ent_idx\1', regex=True)
        gold.loc[:, 'associate'] = gold['associate'].str.replace(r'idx(\d+)', r'obj_ent_idx\1', regex=True)
        gold.loc[:, 'location'] = gold['location'].str.replace(r'loc:\s*', '', regex=True).str.replace(r'det:\s*', '', regex=True)
        
        # test_subject = ['p10274145', 'p10523725', 'p10886362', 'p10959054', 'p12433421', 
        #     'p15321868', 'p15446959', 'p15881535', 'p17720924', 'p18079481']

        test_subject = ['p10046166', 'p10532326', 'p10885696', 'p11540283', 'p11607628',
            'p11879886', 'p12966004', 'p15094735', 'p15109122', 'p15207316',
            'p15272972', 'p16059470', 'p17270742', 'p17288844', 'p17396677',
            'p17962324', 'p18417750', 'p18517718', 'p18570152', 'p19150427',
            'p10274145', 'p10523725', 'p10886362', 'p10959054', 'p12433421',
            'p15321868', 'p15446959', 'p15881535', 'p17720924', 'p18079481']

        
        if args.mode in ['maira_cascade', 'maira']:
            devset = gold
            test_df = gold
            if args.mode == 'maira_cascade':
                maira_df = pd.read_json(args.maira2_cascade_report_path, lines=True)
            
            elif args.mode == 'maira':
                maira_df = pd.read_json(args.maira2_report_path, lines=True)
            
            maira_df = maira_df.rename(columns={'report': f'{args.mode}_report'})
            
            test_df = test_df.merge(maira_df, on=['subject_id', 'sequence'], how='left')
            test_df = test_df[((test_df['section'] == 'impr')&
                        (test_df[f'{args.mode}_report'].str.len() > 2))|
                        ((test_df['section'] == 'find')&
                        (test_df[f'{args.mode}_report'].str.len() > 2))].drop_duplicates(subset=['study_id'])
            args.test_std_ids = test_df['study_id'].unique().tolist()
            print(f"Run Subject {len(test_df.subject_id.unique())}, Study {len(test_df.study_id.unique())}")

        elif args.mode in ['medversa', 'rgrg', 'cvt2distilgpt2', 'lingshu', 'medgemma', 'libra', 'chexagent']:
            devset = gold
            test_df = gold[gold['subject_id'].isin(test_subject)]

            if args.mode == 'medversa':
                df = pd.read_csv(args.medversa_report_path)
            
            elif args.mode == 'rgrg':
                df = pd.read_csv(args.rgrg_report_path)
            
            elif args.mode == 'cvt2distilgpt2':
                df = pd.read_csv(args.cvt2distilgpt2_report_path)
            
            elif args.mode == 'lingshu':
                df = pd.read_json(args.lingshu_path, lines=True)
                df['subject_id'] = df['image_path'].str.extract(r'/home/data_storage/mimic-cxr-jpg/2.0.0/files/p\d+/(p\d+)/s\d+/')[0]
                df['study_id'] = df['image_path'].str.extract(r'/home/data_storage/mimic-cxr-jpg/2.0.0/files/p\d+/p\d+/(s\d+)/')[0]

                df['report'] = df['raw_output'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
                df['report'] = df['report'].str.strip()
            
            elif args.mode == 'medgemma':
                df = pd.read_json(args.medgemma_path, lines=True)
                df['subject_id'] = df['image_path'].str.extract(r'/home/data_storage/mimic-cxr-jpg/2.0.0/files/p\d+/(p\d+)/s\d+/')[0]
                df['study_id'] = df['image_path'].str.extract(r'/home/data_storage/mimic-cxr-jpg/2.0.0/files/p\d+/p\d+/(s\d+)/')[0]
                df['report'] = df['raw_output'].fillna('')
                df['report'] = df['report'].str.strip()

            elif args.mode == 'libra':
                df = pd.read_csv(args.libra_path)
                df['report'] = df['generated_report'].str.strip()
            
            elif args.mode == 'chexagent':
                df = pd.read_csv(args.chexagent_path)
                df['report'] = df['generated_report'].str.strip()
                
            df = df.rename(columns={'report': f'{args.mode}_report'})
            
            if args.mode == 'medversa':#
                
                df_study_ids = df['study_id'].unique().tolist()
                
                for id in df_study_ids:
                    section_list = test_df[(test_df['study_id'] == id)&(test_df['section'] != 'hist')]['section'].unique().tolist()
                    
                    if 'impr' not in section_list:
                        new_row = test_df[(test_df['study_id'] == id)&(test_df['section'] == 'find')].copy()
                        new_row['section'] = 'impr'
                        test_df = pd.concat([test_df, new_row])
                    elif 'find' not in section_list:
                        new_row = test_df[(test_df['study_id'] == id)&(test_df['section'] == 'impr')].copy()
                        new_row['section'] = 'find'
                        test_df = pd.concat([test_df, new_row])
                    
                test_df = test_df.merge(df, on=['subject_id', 'study_id', 'section'], how='left')
                test_df = test_df[((test_df['section'] == 'impr')&
                            (test_df[f'{args.mode}_report'].str.len() > 2))|
                            ((test_df['section'] == 'find')&
                            (test_df[f'{args.mode}_report'].str.len() > 2))].drop_duplicates(subset=['study_id', 'section'])
            else:
                test_df = test_df.merge(df, on=['subject_id', 'study_id'], how='left')
                test_df = test_df[((test_df['section'] == 'impr')&
                            (test_df[f'{args.mode}_report'].str.len() > 2))|
                            ((test_df['section'] == 'find')&
                            (test_df[f'{args.mode}_report'].str.len() > 2))].drop_duplicates(subset=['study_id'])
            

            args.test_std_ids = test_df['study_id'].unique().tolist()
            print(f"Run Subject {len(test_df.subject_id.unique())}, Study {len(test_df.study_id.unique())}")

        elif args.mode == 'rexerr':
            rexerr_df = pd.read_csv(args.rexerr_report_path)
            rexerr_df['study_id'] = 's' + rexerr_df['study_id'].astype(str)
            rexerr_df = rexerr_df[['study_id', 'error_findings', 'error_impression']]
            devset = gold
            test_df = gold[gold['subject_id'].isin(test_subject)]
                
            test_df = test_df.merge(rexerr_df, on=['study_id'], how='left')
            args.test_std_ids = test_df['study_id'].unique().tolist()
            
            print(f"Run Subject {len(test_df.subject_id.unique())}, Study {len(test_df.study_id.unique())}")
            
        else:
            devset = gold
            test_df = gold
                
            args.test_std_ids = test_df['study_id'].unique().tolist()
                            
        print(f'\n {len(args.test_std_ids)} studies in test set')
        # Use multiprocessing to process study data in parallel
        print(f"Starting multiprocessing with {min(cpu_count(), 64)} workers")
        process_args = [(id, test_df, args) for id in args.test_std_ids]
        

        with Pool(processes=min(cpu_count(), 64)) as pool:
            results = list(tqdm(
                pool.imap(process_study_data, process_args),
                total=len(process_args),
                desc="Processing study data"
            ))
        
        # Flatten results and create study_data dictionary
        study_data = {}
        idx = 0
        for result_list in results:
            for offset, data in result_list:
                study_data[str(idx)] = data
                idx += 1

    elif args.mode == 'rexval' or args.mode == 'silver_eval':
        gold = pd.read_csv(args.gold_path)
        gold = gold.copy()
        gold.loc[:, 'cat'] = gold['cat'].replace(['lf', 'If'], 'pf')
        gold.loc[:, 'evidence'] = gold['evidence'].str.replace(r'idx(\d+)', r'obj_ent_idx\1', regex=True)
        gold.loc[:, 'associate'] = gold['associate'].str.replace(r'idx(\d+)', r'obj_ent_idx\1', regex=True)
        gold.loc[:, 'location'] = gold['location'].str.replace(r'loc:\s*', '', regex=True).str.replace(r'det:\s*', '', regex=True)
        # Delete the original location column and rename location2 to location
        if 'dx_uncertainty' in gold.columns:
            gold = gold.rename(columns={'dx_uncertainty': 'dx_certainty'})

        if args.mode == 'rexval':
            test_df = pd.read_csv(args.rexval_report_path)

        elif args.mode == 'silver_eval':
            test_df = pd.read_csv(args.silver_eval_report_path)
            # test_df = test_df[test_df['split'] == 'test'].sample(n=2, random_state=42)
        
                
        devset = gold
            
        args.test_std_ids = test_df['study_id'].unique().tolist()
        
        print(f'\n {len(args.test_std_ids)} studies in test set')
    
        print(f"Test std ids: {len(args.test_std_ids)}")
        
        input_file_path = os.path.join(
            args.output_dir,
            f'{args.mode}_{args.candidate_type}_{args.unit}_input.json'
        )
        
        if not os.path.exists(input_file_path):
            print(f"input_file does not exist: {input_file_path}")

            # Use multiprocessing to process study data in parallel
            print(f"Starting multiprocessing with {min(cpu_count(), 64)} workers")
            process_args = [(id, test_df, args) for id in args.test_std_ids]
            
            with Pool(processes=min(cpu_count(), 64)) as pool:
                results = list(tqdm(
                    pool.imap(process_study_data, process_args),
                    total=len(process_args),
                    desc="Processing study data"
                ))
            
            # Flatten results and create study_data dictionary
            study_data = {}
            idx = 0
            for result_list in results:
                for offset, data in result_list:
                    study_data[str(idx)] = data
                    idx += 1            
        else:
            print(f"input_file exists: {input_file_path}")


    if args.mode == 'silver_eval' and os.path.exists(input_file_path):
        study_data = json.load(open(input_file_path, 'r'))
        print(f"input_file loaded: {input_file_path}")
    
    else:
        # Save sentence-level data  
        output_file = os.path.join(
            args.output_dir,
            f'{args.mode}_{args.candidate_type}_{args.unit}_input.json'
        )

        # Remove directory if a path with the same name already exists
        if os.path.isdir(output_file):
            import shutil
            shutil.rmtree(output_file)

        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(study_data, f, ensure_ascii=False, indent=4)

        print(f'Saved input data to {output_file}')

    return study_data, devset


def generate_task(custom_id, conversation, args, system_message=None):

    if args.deployment_name.isin['gpt-4o-mini-batchapi', 'gpt-4o-batch']:
        task = {
            "custom_id": f"{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": args.deployment_name,
                    "temperature": 0.1,
                    "messages": conversation
            }
        }
    elif args.deployment_name == 'o3-mini-batch':
        task = {
            "custom_id": f"{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                # This is what you would have in your Chat Completions API call
                "model": args.deployment_name,
                "messages": conversation
            }
        }
    else:
        task = {
            "custom_id": f"{custom_id}",
            "system_message": system_message,
            "messages": conversation
        }

    return task

class RelationEnum(str, Enum):
    Cat = "Cat"
    Dx_Status = "Dx_Status"
    Dx_Certainty = "Dx_Certainty"
    Location = "Location"
    Associate = "Associate"
    Evidence = "Evidence"
    Morphology = "Morphology"
    Distribution = "Distribution"
    Measurement = "Measurement"
    Severity = "Severity"
    Comparison = "Comparison"
    Onset = "Onset"
    NoChange = "No Change"
    Improved = "Improved"
    Worsened = "Worsened"
    Placement = "Placement"
    PastHx = "Past Hx"
    OtherSource = "Other Source"
    AssessmentLimitations = "Assessment Limitations"

class EntityRelation(BaseModel):
    """Represents a single relation of an entity with its value"""
    relation: RelationEnum = Field(..., description="The relation name among the predefined types")
    value: str = Field(..., description="The value corresponding to the relation")
    obj_ent_idx: Optional[int] = Field(
        None,
        description="For Associate/Evidence relations, the ent_idx of the object entity"
    )

    @root_validator(skip_on_failure=True)
    def require_obj_ent_idx_for_certain_relations(cls, values):
        rel = values.get('relation')
        idx = values.get('obj_ent_idx')
        if rel in {RelationEnum.Associate, RelationEnum.Evidence} and idx is None:
            raise ValueError("'obj_ent_idx' must be provided for Associate and Evidence relations")
        return values

class Entity(BaseModel):
    """Represents a single entity with all its extracted relations"""
    name: str = Field(..., description="The name of the entity, exactly as provided")
    sent_idx: int = Field(..., description="Index of the sentence from which this entity was extracted")
    ent_idx: int = Field(..., description="Unique identifier for this entity within the report section")
    relations: List[EntityRelation] = Field(..., description="List of all relations for this entity")

    @root_validator(skip_on_failure=True)
    def validate_relations(cls, values):
        relations = values.get('relations', [])
        types = [rel.relation for rel in relations]
        # Must include exactly one Cat and one Status
        if types.count(RelationEnum.Cat) != 1 or types.count(RelationEnum.Dx_Status) != 1 or types.count(RelationEnum.Dx_Certainty) != 1:
            raise ValueError("Each entity must include exactly one 'Cat' relation and exactly one 'Status' relation and exactly one 'Dx_Certainty' relation")
        return values

class StructuredOutput(BaseModel):
    """Schema for structured extraction matching the provided output format"""
    entities: List[Entity] = Field(..., description="All extracted entities with their relations")

    @root_validator(skip_on_failure=True)
    def validate_entities(cls, values):
        entities = values.get('entities', [])
        if not entities:
            raise ValueError("Output must include at least one entity")
        # Ensure unique ent_idx and consistent
        idxs = [e.ent_idx for e in entities]
        if len(idxs) != len(set(idxs)):
            raise ValueError("Each entity must have a unique 'ent_idx'")
        return values

def convert_to_sr_structure(structured_output):
    """
    Convert a StructuredOutput object to a dictionary of triplets grouped by relation type.
    Format matches the output from the JSON parsing branch in parse_eval_format.
    
    Args:
        structured_output (StructuredOutput, dict, or str): Parsed output matching the Entity schema,
                                                           already formatted dictionary, or raw text containing JSON

    Returns:
        dict: Dictionary with relation types as keys and lists of triplets as values
    """
    import json
    import re
    
    # Initialize defaultdict to store triplets grouped by relation
    pred_triplet = defaultdict(list)
    
    # If structured_output is a string, extract JSON from it
    if isinstance(structured_output, str):
        try:
            # Try to extract JSON from text using various patterns
            json_text = None
            
            # Pattern 1: Look for ```json ... ``` blocks
            json_patterns = [
                r'```json\s*\n(.*?)\n```',
                r'```json\s*(.*?)```',
                r'```\s*json\s*\n(.*?)\n```',
                r'```\s*json\s*(.*?)```'
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, structured_output, re.DOTALL | re.IGNORECASE)
                if json_match:
                    json_text = json_match.group(1).strip()
                    break
            
            # Pattern 2: Look for { ... } blocks (standalone JSON)
            if not json_text:
                # Find the first complete JSON object
                brace_count = 0
                start_idx = structured_output.find('{')
                if start_idx != -1:
                    json_start = start_idx
                    for i, char in enumerate(structured_output[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_text = structured_output[json_start:i+1]
                                break
            
            # Pattern 3: Try to extract JSON after common keywords
            if not json_text:
                keywords = ['json:', 'output:', 'result:', 'answer:']
                for keyword in keywords:
                    idx = structured_output.lower().find(keyword)
                    if idx != -1:
                        remaining_text = structured_output[idx + len(keyword):].strip()
                        brace_idx = remaining_text.find('{')
                        if brace_idx != -1:
                            brace_count = 0
                            json_start = brace_idx
                            for i, char in enumerate(remaining_text[brace_idx:], brace_idx):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_text = remaining_text[json_start:i+1]
                                        break
                            if json_text:
                                break
            
            # Parse the extracted JSON
            if json_text:
                try:
                    structured_output = json.loads(json_text)
                    print(f"Successfully extracted and parsed JSON from text")
                except json.JSONDecodeError:
                    # Try to clean up common JSON formatting issues
                    json_text_cleaned = json_text.replace('\n', ' ').replace('\t', ' ')
                    # Remove extra commas before closing brackets
                    json_text_cleaned = re.sub(r',\s*}', '}', json_text_cleaned)
                    json_text_cleaned = re.sub(r',\s*]', ']', json_text_cleaned)
                    structured_output = json.loads(json_text_cleaned)
                    print(f"Successfully parsed JSON after cleanup")
            else:
                print(f"No JSON found in text. First 300 chars: {structured_output[:300]}...")
                return pred_triplet
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse extracted JSON: {e}")
            print(f"Extracted JSON text (first 500 chars): {json_text[:500] if json_text else 'None'}")
            return pred_triplet
        except Exception as e:
            print(f"Error extracting JSON from text: {e}")
            print(f"Input text (first 300 chars): {structured_output[:300]}...")
            return pred_triplet
    
    # Check if structured_output is already a dictionary with relation keys
    if isinstance(structured_output, dict) and any(key in RELATIONS for key in structured_output.keys()):
        # Already in the correct format, return as is
        return structured_output
    
    # If it's a dictionary but with 'entities' key (parsed JSON)
    if isinstance(structured_output, dict) and 'entities' in structured_output:
        entities = structured_output['entities']
    # If it's a StructuredOutput object
    elif hasattr(structured_output, 'entities'):
        entities = structured_output.entities
    else:
        print("structured_output", structured_output)
        raise ValueError("Invalid structured_output format: must be a StructuredOutput object or a dictionary with 'entities' key")
    
    # Process entities
    for entity in entities:
        # Handle both object and dict formats
        if isinstance(entity, dict):
            entity_name = entity['name'].lower().strip()
            sent_idx = entity.get('sent_idx')
            ent_idx = entity.get('ent_idx')
            relations = entity.get('relations', [])
        else:
            entity_name = entity.name.lower().strip()
            sent_idx = entity.sent_idx
            ent_idx = entity.ent_idx
            relations = entity.relations
        
        # Process relations
        for relation in relations:
            if isinstance(relation, dict):
                relation_name = relation['relation'].lower()
                relation_value = relation['value'].lower().strip()
                obj_ent_idx = relation.get('obj_ent_idx')
            else:
                relation_name = relation.relation.value.lower() if hasattr(relation.relation, 'value') else str(relation.relation).lower()
                relation_value = relation.value.lower().strip()
                obj_ent_idx = relation.obj_ent_idx
            
            # Add triplet to the appropriate relation category
            pred_triplet[relation_name].append((
                entity_name, 
                relation_name, 
                relation_value,
                sent_idx,
                ent_idx,
                obj_ent_idx
            ))
    
    return pred_triplet

def convert_to_assistant_string(structured_output):
    """
    Convert a StructuredOutput object to a dictionary of triplets grouped by relation type.
    Format matches the output from the JSON parsing branch in parse_eval_format.
    
    Args:
        structured_output (StructuredOutput or dict): Parsed output matching the Entity schema
                                                     or already formatted dictionary

    Returns:
        dict: Dictionary with relation types as keys and lists of triplets as values
    """
    # Initialize defaultdict to store triplets grouped by relation
    extracted_entities = []
    
    
    # Check if structured_output is already a dictionary with relation keys
    if isinstance(structured_output, dict) and any(key in RELATIONS for key in structured_output.keys()):
        # Already in the correct format, return as is
        return structured_output
    
    # If it's a dictionary but with 'entities' key (parsed JSON)
    if isinstance(structured_output, dict) and 'entities' in structured_output:
        entities = structured_output['entities']
    # If it's a StructuredOutput object
    elif hasattr(structured_output, 'entities'):
        entities = structured_output.entities
    else:
        print("structured_output", structured_output)
        raise ValueError("Invalid structured_output format: must be a StructuredOutput object or a dictionary with 'entities' key")
    
    # Process entities
    for entity in entities:
        
        extracted_relations = []
        # Handle both object and dict formats
        if isinstance(entity, dict):
            entity_name = entity['name'].lower().strip()
            sent_idx = entity.get('sent_idx')
            ent_idx = entity.get('ent_idx')
            relations = entity.get('relations', [])
        else:
            entity_name = entity.name.lower().strip()
            sent_idx = entity.sent_idx
            ent_idx = entity.ent_idx
            relations = entity.relations
        
        for relation in relations:
            if isinstance(relation, dict):
                relation_name = relation['relation']
                relation_value = relation['value'].strip()
                obj_ent_idx = relation.get('obj_ent_idx')
            else:
                relation_name = relation.relation.value if hasattr(relation.relation, 'value') else str(relation.relation)
                relation_value = relation.value.strip()
                obj_ent_idx = relation.obj_ent_idx
            
            if relation_name in ['Associate', 'Evidence']:
                extracted_relations.append({
                    "relation": relation_name,
                    "value": relation_value,
                    "obj_ent_idx": obj_ent_idx
                })
            else:
                extracted_relations.append({
                    "relation": relation_name,
                    "value": relation_value,
                })
            
        extracted_entities.append({
            "name": entity_name, 
            "sent_idx": sent_idx,
            "ent_idx": ent_idx,
            "relations": extracted_relations
        })
    
    assistant_string = "OUTPUT: " + json.dumps({"entities": extracted_entities}, indent=2)
    return assistant_string

def evaluate_funct(gold_file_path, exp_file_path, args):
    exp_file = json.load(open(exp_file_path))
    gpt_data = {}
    for ann in exp_file['annotations']:
        try:
            result = parse_eval_format(ann['custom_id'], ann['model_output'])
            if result is not None:
                custom_id, pred_triplet = result
                gpt_data[custom_id] = pred_triplet
            else:
                print(f"Warning: Failed to parse output for custom_id: {ann['custom_id']}")
                gpt_data[ann['custom_id']] = {}
        except Exception as e:
            print(f"Error processing custom_id {ann['custom_id']}: {str(e)}")
            print(f"Problematic model_output: {ann['model_output'][:100]}...")
            gpt_data[ann['custom_id']] = {}

    save_path = f'./singleSR/eval/{args.mode}/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
    
    if not os.path.exists(save_path):
        os.makedirs(f'./{save_path}', exist_ok=True)

    triplet_path = f'{save_path}/pred_triplet_path.json'

    with open(triplet_path, 'w', encoding="utf-8") as f:
        json.dump(gpt_data, f, ensure_ascii=False, indent=4)

def run_sr_eval(data_path, triplet_path, save_path):
    SR_EVAL(data_path=data_path, gpt_pred_path=triplet_path, save_path=save_path)
    
def run_sro_eval(data_path, triplet_path, save_path, jaccard):
    SRO_EVAL(data_path=data_path, gpt_pred_path=triplet_path, save_path=save_path, jaccard=jaccard)
    
def run_gen_report_sr_eval(data_path, triplet_path, save_path):
    Gen_report_SR_EVAL(data_path=data_path, gpt_pred_path=triplet_path, save_path=save_path)
    
def run_gen_report_sro_eval(data_path, triplet_path, save_path, jaccard):
    Gen_report_SRO_EVAL(data_path=data_path, gpt_pred_path=triplet_path, save_path=save_path, jaccard=jaccard)

def evaluate_funct(gold_file_path, exp_file_path, args):
    exp_file = json.load(open(exp_file_path))
    
    gpt_data = {}
    for ann in exp_file['annotations']:
        
        try:
            result = parse_eval_format(ann['custom_id'], ann['model_output'])
            if result is not None:
                custom_id, pred_triplet = result
                gpt_data[custom_id] = pred_triplet
            else:
                print(f"Warning: Failed to parse output for custom_id: {ann['custom_id']}")
                gpt_data[ann['custom_id']] = {}
        
        except Exception as e:
            print(f"Error processing custom_id {ann['custom_id']}: {str(e)}")
            print(f"Problematic model_output: {ann['model_output'][:100]}...")
            gpt_data[ann['custom_id']] = {}


    if not args.dynamic_retrieval:
        if args.multi:
            save_path = f'./singleSR/eval/{args.mode}/M{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
        else:
            save_path = f'./singleSR/eval/{args.mode}/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
    else:
        save_path = f'./singleSR/eval/{args.mode}/dynamic_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
    
    if not os.path.exists(save_path):
        os.makedirs(f'./{save_path}', exist_ok=True)

    triplet_path = f'{save_path}/pred_triplet_path.json'

    with open(triplet_path, 'w', encoding="utf-8") as f:
        json.dump(gpt_data, f, ensure_ascii=False, indent=4)
    # Run evaluation in parallel
    if args.mode in ['gold_eval', 'maira', 'maira_cascade', 'rexerr', 'medversa', 'rgrg', 'cvt2distilgpt2', 'lingshu', 'medgemma', 'libra', 'chexagent']:
        # Launch SR and SRO evaluation processes
        p1 = Process(target=run_sr_eval, args=(gold_file_path, triplet_path, save_path))
        p2 = Process(target=run_sro_eval, args=(gold_file_path, triplet_path, save_path, args.jaccard))

        p1.start()
        p2.start()

        # Wait for both processes to finish
        p1.join()
        p2.join()

        # Compute evaluation results sequentially
        cal_result_SR_EVAL(save_path)
        cal_result_SRO_EVAL(save_path)

    elif args.mode in ['rexval']:
        # Launch SR and SRO evaluation processes (generation report variant)
        p1 = Process(target=run_gen_report_sr_eval, args=(gold_file_path, triplet_path, save_path))
        p2 = Process(target=run_gen_report_sro_eval, args=(gold_file_path, triplet_path, save_path, args.jaccard))

        p1.start()
        p2.start()

        # Wait for both processes to finish
        p1.join()
        p2.join()

        # Compute evaluation results sequentially
        cal_result_gen_report_SR_EVAL(save_path)
        cal_result_gen_report_SRO_EVAL(save_path)
    
    return save_path          


def create_relation_dataframe(data_path=None, gpt_pred_path=None, save_path=None, relation_to_evaluate=None):
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    eval_results = []
    with open(f"{save_path}/subject_predict.json", 'r') as file:
        for line in file:
            if line.strip():
                eval_results.append(json.loads(line))
    
    entity_rows = []
    
    for result in eval_results:
        custom_id = result['custom_id']
        report_type = result['report_type'] if 'report_type' in result else None
        sentences = result['sentences']
        study_id = data[custom_id]['study_id']
        subject_id = data[custom_id]['subject_id'] if 'subject_id' in data[custom_id] else None
                
        all_entities = {}
        
        for relation in RELATIONS:
            triplets = result['pred_triplet'].get(relation, [])
            
            right_entities = set(result['right_entities'].get(relation, []))
            wrong_entities = set(result['wrong_entities'].get(relation, []))
            
            for triplet in triplets:
                entity = triplet[0]
                rel = triplet[1]
                value = triplet[2]
                if len(triplet) > 3:
                    sent_idx = triplet[3]
                    ent_idx = triplet[4]
                else:
                    sent_idx = None
                    ent_idx = None
                
                # Initialize entity info on first occurrence
                if entity not in all_entities:
                    # Set base fields
                    entity_data = {
                        'custom_id': custom_id,
                        'report_type': report_type,
                        'data_from': result['data_from'],
                        'study_id': study_id,
                        'subject_id': subject_id,
                        'sentences': sentences,
                        'sent_idx': sent_idx,
                        'ent_idx': ent_idx,
                        'entity': entity,
                        'is_correct': entity in right_entities
                    }

                    # Initialize all RELATIONS to None
                    for r in RELATIONS:
                        entity_data[r] = None

                    all_entities[entity] = entity_data

                # Set relation as a column
                all_entities[entity][rel] = value

        # Append collected entity info to rows
        for entity, entity_data in all_entities.items():
            entity_rows.append(entity_data)

    df = pd.DataFrame(entity_rows)

    # Save dataframe
    df.to_csv(f"{save_path}/pred_SR_df.csv", index=False)
    return df

def create_relation_dataframe2(data_path=None, gpt_pred_path=None, save_path=None, relation_to_evaluate=None):
    with open(data_path, 'r') as file:
        data = json.load(file)
        
    eval_results = []
    with open(f"{save_path}/triplets_predict.json", 'r') as file: ####
        for line in file:
            if line.strip():
                eval_results.append(json.loads(line))
    
    entity_rows = []
    
    for result in eval_results:
        custom_id = result['custom_id']
        report_type = result['report_type'] if 'report_type' in result else None
        sentences = result['sentence']
        study_id = data[custom_id]['study_id']
        subject_id = data[custom_id]['subject_id'] if 'subject_id' in data[custom_id] else None
                
        all_entities = {}
        
        right_triplets = result['right_triplet_list']
        wrong_triplets = result['wrong_triplet_list']

        all_triplets = defaultdict(list)

        for triplet in right_triplets + wrong_triplets:
            all_triplets[triplet[1]].append(triplet)

        all_right_entities = defaultdict(list)
        all_wrong_entities = defaultdict(list)

        for triplet in right_triplets:
            all_right_entities[triplet[1]].append(triplet[0])
        for triplet in wrong_triplets:
            all_wrong_entities[triplet[1]].append(triplet[0])
        
        for relation in RELATIONS:
            
            right_entities = set(all_right_entities.get(relation, []))
            wrong_entities = set(all_wrong_entities.get(relation, []))
            
            for triplet in all_triplets[relation]:
                entity = triplet[0]
                rel = triplet[1]
                value = triplet[2]
                if len(triplet) > 3:
                    sent_idx = triplet[3]
                    ent_idx = triplet[4]
                    obj_ent_idx = triplet[5]
                else:
                    sent_idx = None
                    ent_idx = None
                    obj_ent_idx = None
                # Initialize entity info on first occurrence
                if entity not in all_entities:
                    # Set base fields
                    entity_data = {
                        f'({sent_idx}, {ent_idx})': {
                            'custom_id': custom_id,
                            'report_type': report_type,
                            'data_from': result['data_from'],
                            'study_id': study_id,
                            'subject_id': subject_id,
                            'sentences': sentences,
                            'sent_idx': sent_idx,
                            'ent_idx': ent_idx,
                            'entity': entity,
                            'is_correct': entity in right_entities
                        }
                    }
                    
                    # Initialize all RELATIONS to None
                    for r in RELATIONS:
                        entity_data[f'({sent_idx}, {ent_idx})'][r] = None

                    all_entities[entity] = entity_data

                elif f'({sent_idx}, {ent_idx})' not in all_entities[entity]:
                    # Set base fields for new (sent_idx, ent_idx) position
                    entity_data = {
                        f'({sent_idx}, {ent_idx})': {
                            'custom_id': custom_id,
                            'report_type': report_type,
                            'data_from': result['data_from'],
                            'study_id': study_id,
                            'subject_id': subject_id,
                            'sentences': sentences,
                            'sent_idx': sent_idx,
                            'ent_idx': ent_idx,
                            'entity': entity,
                            'is_correct': entity in right_entities
                        }
                    }
                    
                    # Initialize all RELATIONS to None
                    for r in RELATIONS:
                        entity_data[f'({sent_idx}, {ent_idx})'][r] = None

                    # Merge new position info into existing entity data
                    all_entities[entity].update(entity_data)
                
                # Add relation as a column
                # If the relation value is None, set it to the new value
                # If it already has a value, append the new value with a comma
                current_value = all_entities[entity][f'({sent_idx}, {ent_idx})'][rel]
                if current_value is None:
                    if rel in ['associate', 'evidence']:
                        all_entities[entity][f'({sent_idx}, {ent_idx})'][rel] = f"{value}, idx{obj_ent_idx}"
                    else:
                        all_entities[entity][f'({sent_idx}, {ent_idx})'][rel] = value
                else:
                    if rel in ['associate', 'evidence']:
                        all_entities[entity][f'({sent_idx}, {ent_idx})'][rel] = f"{current_value}, {value}, idx{obj_ent_idx}"
                    else:
                        all_entities[entity][f'({sent_idx}, {ent_idx})'][rel] = f"{current_value}, {value}"
        
        # Append collected entity info to rows
        for entity, entity_datas in all_entities.items():
            for entity_data in entity_datas.values():
                entity_rows.append(entity_data)

    df = pd.DataFrame(entity_rows)

    # Save dataframe
    df.to_csv(f"{save_path}/pred_SR_df.csv", index=False)
    
    return df


def create_relation_dataframe3(data_path=None, gpt_pred_path=None, save_path=None, relation_to_evaluate=None):
    with open(data_path, 'r') as file:
        data = json.load(file)

    # Load prediction JSON from gpt_pred_path (using annotations > model_output)
    with open(gpt_pred_path, 'r') as file:
        pred_data = json.load(file)

    annotations = pred_data.get('annotations', [])

    # Buffer to collect rows for the final DataFrame
    entity_rows = []

    for ann in annotations:
        custom_id = ann.get('custom_id')
        if custom_id is None or custom_id not in data:
            continue

        # Gold metadata
        study_id = data[custom_id].get('study_id')
        subject_id = data[custom_id].get('subject_id')
        report_type = data[custom_id].get('section')
        sentences = data[custom_id].get('passage')

        model_output = ann.get('model_output', {})

        # Aggregate by (entity, sent_idx, ent_idx)
        rows_by_key = {}

        # Iterate over all relation channels
        for rel_key, triples in model_output.items():
            rel_norm = rel_key.lower()
            if rel_norm not in RELATIONS:
                continue
            for t in triples:
                if not isinstance(t, list) or len(t) < 3:
                    continue
                entity = t[0]
                value = t[2]
                sent_idx = t[3] if len(t) > 3 else None
                ent_idx = t[4] if len(t) > 4 else None
                obj_ent_idx = t[5] if len(t) > 5 else None

                key = (entity, sent_idx, ent_idx)
                if key not in rows_by_key:
                    row = {
                        'custom_id': custom_id,
                        'report_type': report_type,
                        'data_from': 'silver_eval',
                        'study_id': study_id,
                        'subject_id': subject_id,
                        'sentences': sentences,
                        'sent_idx': sent_idx,
                        'ent_idx': ent_idx,
                        'entity': entity,
                    }
                    for r in RELATIONS:
                        row[r] = None
                    rows_by_key[key] = row

                # Set value (accumulate if multiple values exist)
                current_val = rows_by_key[key][rel_norm]
                if rel_norm in ['associate', 'evidence'] and obj_ent_idx is not None:
                    val_to_set = f"{value}, idx{obj_ent_idx}"
                else:
                    val_to_set = value

                if current_val is None:
                    rows_by_key[key][rel_norm] = val_to_set
                else:
                    rows_by_key[key][rel_norm] = f"{current_val}, {val_to_set}"

        # Append collected entity instances
        entity_rows.extend(rows_by_key.values())

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    df = pd.DataFrame(entity_rows)
    df.to_csv(f"{save_path}/pred_SR_df.csv", index=False)
    return df


def visualize_table(data_path, output_path=None, metrics=None, highlight_best=True, 
                   figsize=(14, 0.5), style='whitegrid'):
    # Load data
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Set default metrics if not provided
    if metrics is None:
        metrics = ['SR P', 'SR R', 'SR F1', 'SRO P', 'SRO R', 'SRO F1']
    
    # Function to shorten model names
    def shorten_model_name(name):
        parts = name.split('-')
        return '-'.join(parts[:min(3, len(parts))])
    
    # Shorten model names
    df['Short Model'] = df['Model'].apply(shorten_model_name)
    
    # Select only the columns we want to display
    display_cols = ['Short Model', 'Cand. Rate'] + metrics
    df_display = df[['Short Model', 'Cand. Rate'] + metrics].copy()
    
    # Round numeric values for display
    for col in df_display.columns:
        if col not in ['Short Model', 'Cand. Rate'] and df_display[col].dtype in [np.float64, np.int64]:
            df_display[col] = df_display[col].round(1)
    
    # Sort by the last metric (usually the most important one) in descending order
    df_display = df_display.sort_values(metrics[-1], ascending=False)
    
    # Set the style - use a cleaner style for NeurIPS
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Calculate figure size based on number of rows
    num_rows = len(df_display)
    fig_height = max(figsize[1] * num_rows, 4)  # Ensure minimum height
    fig = plt.figure(figsize=(figsize[0], fig_height))
    
    # Create axis without frame
    ax = plt.subplot(111, frame_on=False)
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    
    # Format the data for display - add % to numeric values
    formatted_data = df_display.copy()
    for col in metrics:
        formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.1f}")
    
    # Rename the column header for better display
    formatted_data = formatted_data.rename(columns={'Short Model': 'Model'})
    display_cols[0] = 'Model'  # Update display_cols to match
    
    # Create the table with NeurIPS-style formatting
    table = ax.table(
        cellText=formatted_data.values,
        colLabels=formatted_data.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(formatted_data.columns)
    )
    
    # Set table properties for NeurIPS style
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Slightly smaller font for academic style
    table.scale(1.1, 1.6)   # Adjust cell size for NeurIPS style
    
    # Add grid lines using the older API
    # Set edges for all cells to create grid effect
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor('black')
    
    # Highlight the best value in each metric column if requested
    if highlight_best:
        # Create a custom colormap for highlighting (subtle blue gradient)
        cmap = LinearSegmentedColormap.from_list('neurips_highlight', ['#ffffff', '#e1effe'])
        
        # Find and highlight best values
        for col in metrics:
            col_idx = display_cols.index(col)
            best_idx = df_display[col].idxmax()
            row_idx = df_display.index.get_indexer([best_idx])[0]
            
            # Get the cell and set its color (adjust for header row)
            cell = table[(row_idx + 1, col_idx)]  # +1 for header
            cell.set_facecolor(cmap(0.8))  # Use a subtle highlight
            
            # Make the best value bold
            cell.get_text().set_fontweight('bold')
    
    # Add a title in NeurIPS style (more understated)
    plt.title('Performance Comparison', fontsize=16, fontweight='bold', pad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot with higher DPI for publication quality
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Table visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_overall_metrics(results):
    """
    Calculate overall metrics (F1, Precision, Recall, TP, FP, FN) from results.
    
    Args:
        results (dict): Results dictionary from SR_result.json or SRO_result.json
        
    Returns:
        dict: Dictionary containing overall metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Skip the 'all' key if it exists
    for relation, metrics in results.items():
        if relation == 'all':
            continue
        
        total_tp += metrics.get('tp', 0)
        total_fp += metrics.get('fp', 0)
        
        # Handle different formats between SR and SRO results
        if 'miss' in metrics:  # SRO format
            total_fn += metrics.get('miss', 0)
        elif 'all' in metrics:  # SR format
            total_fn += metrics.get('all', 0) - metrics.get('tp', 0)
    
    # Calculate precision, recall, and F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def generate_table_from_rexval(results, output_path=None):
    """
    Generate a comparison table from rexval mode results.
    
    Args:
        results (dict): Dictionary containing evaluation results by model
        output_path (str, optional): Path to save the output table. If None, returns the DataFrame
        
    Returns:
        pd.DataFrame or None: Comparison table if output_path is None, otherwise None
    """
    # Initialize results storage
    table_data = []
    
    print(f"Results keys: {list(results.keys())}")
    if 'triplets' in results:
        print(f"Triplets models: {list(results['triplets'].keys())}")
    if 'subj' in results:
        print(f"Subject models: {list(results['subj'].keys())}")
    

    # Process triplets results (SRO)
    if 'triplets' in results:
        for model_name, model_results in results['triplets'].items():
            # Calculate overall metrics
            total_tp = sum(model_results[r].get('tp', 0) for r in model_results if r in RELATIONS)
            total_fp = sum(model_results[r].get('fp', 0) for r in model_results if r in RELATIONS)
            total_miss = sum(model_results[r].get('miss', 0) for r in model_results if r in RELATIONS)
            
            # Calculate precision, recall, F1
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_miss) if (total_tp + total_miss) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add to table data
            table_data.append({
                'Model': model_name,
                'SRO P': round(precision * 100, 1),
                'SRO R': round(recall * 100, 1),
                'SRO F1': round(f1 * 100, 1)
            })
    print(f"Generated {len(table_data)} table entries")

    # Process subject results (SR)
    if 'subj' in results:
        for model_name, model_results in results['subj'].items():
            # Find existing entry or create new one
            entry = next((item for item in table_data if item['Model'] == model_name), None)
            if entry is None:
                entry = {'Model': model_name}
                table_data.append(entry)
            
            # Calculate overall metrics
            total_tp = sum(model_results[r].get('tp', 0) for r in model_results if r in RELATIONS)
            total_fp = sum(model_results[r].get('fp', 0) for r in model_results if r in RELATIONS)
            total_all = sum(model_results[r].get('all', 0) for r in model_results if r in RELATIONS)
            total_miss = total_all - total_tp
            
            # Calculate precision, recall, F1
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / total_all if total_all > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add to entry
            entry['SR P'] = round(precision * 100, 1)
            entry['SR R'] = round(recall * 100, 1)
            entry['SR F1'] = round(f1 * 100, 1)
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by SRO F1 score (descending)
    if not df.empty and 'SRO F1' in df.columns:
        df = df.sort_values('SRO F1', ascending=False)
    
    if df.empty:
        print("Warning: No data was generated for the table")
    else:
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
    
    # Save to file if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.1f')
        print(f"Results saved to {output_path}")
        return None
    
    return df

def generate_table(base_path='./singleSR/eval/test_2studies', args=None, output_path=None, mode=None):
    """
    Generate a comparison table of evaluation results across different models and candidate rates.
    
    Args:
        base_path (str): Base directory containing evaluation results
        output_path (str, optional): Path to save the output table. If None, returns the DataFrame
        mode (str, optional): Evaluation mode ('rexval' or None)
        
    Returns:
        pd.DataFrame or None: Comparison table if output_path is None, otherwise None
    """
    # Handle rexval mode
    if mode == 'rexval':
        results = {}
        
        # Debug: Print the base path
        print(f"Looking for rexval results in: {base_path}")
        
        # For rexval mode, we need to find all model directories
        model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        print(f"Found model directories: {model_dirs}")
        
        # Initialize results for all models
        results = {'triplets': {}, 'subj': {}}
        
        # Process each model directory
        for model in model_dirs:
            model_path = os.path.join(base_path, model, 'SROSRO', 'section', '1')
            
            # Check if the model has results
            if not os.path.exists(model_path):
                print(f"No results found for model: {model}")
                continue
                
            # Load SRO results
            sro_path = f'{model_path}/SRO_result_by_model.json'
            if os.path.exists(sro_path):
                print(f"Loading SRO results from: {sro_path}")
                with open(sro_path, 'r') as f:
                    model_sro_results = json.load(f)
                    # Add to the overall results
                    for metric_model, metric_results in model_sro_results.items():
                        results['triplets'][f"{model}_{metric_model}"] = metric_results
            
            # Load SR results
            sr_path = f'{model_path}/SR_result_by_model.json'
            if os.path.exists(sr_path):
                print(f"Loading SR results from: {sr_path}")
                with open(sr_path, 'r') as f:
                    model_sr_results = json.load(f)
                    # Add to the overall results
                    for metric_model, metric_results in model_sr_results.items():
                        results['subj'][f"{model}_{metric_model}"] = metric_results
        
        # Debug: Print the loaded results
        print(f"Loaded triplets models: {list(results['triplets'].keys())}")
        print(f"Loaded subject models: {list(results['subj'].keys())}")
        
        return generate_table_from_rexval(results, output_path)
    
    # Standard mode (original implementation)
    # Find all model directories
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Initialize results storage
    results = []
    
    # Process each model
    for model in model_dirs:
        # Find all candidate rates for this model
        cand_rates = []
        for rate_dir in glob.glob(f"{base_path}/{model}/SROSRO/{args.unit}/*"):
            if os.path.isdir(rate_dir):
                cand_rate = os.path.basename(rate_dir)
                if cand_rate.replace('.', '', 1).isdigit():  # Check if it's a number
                    cand_rates.append(cand_rate)
        
        # Process each candidate rate
        for cand_rate in cand_rates:
            sr_path = f"{base_path}/{model}/SROSRO/{args.unit}/{cand_rate}/SR_result.json"
            sro_path = f"{base_path}/{model}/SROSRO/{args.unit}/{cand_rate}/SRO_result.json"
            
            # Skip if either file doesn't exist
            if not (os.path.exists(sr_path) and os.path.exists(sro_path)):
                continue
                
            # Load results
            try:
                with open(sr_path, 'r') as f:
                    sr_results = json.load(f)
                with open(sro_path, 'r') as f:
                    sro_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading results for {model} with candidate rate {cand_rate}")
                continue
            
            # Calculate overall metrics for SR (subject recognition)
            sr_metrics = calculate_overall_metrics(sr_results)
            
            # Calculate overall metrics for SRO (subject-relation-object)
            sro_metrics = calculate_overall_metrics(sro_results)
            
            # Format metrics for publication (rounded to 3 decimal places)
            result_entry = {
                'Model': model,
                'Cand. Rate': float(cand_rate),
                'SR P': round(sr_metrics['precision'] * 100, 1),
                'SR R': round(sr_metrics['recall'] * 100, 1),
                'SR F1': round(sr_metrics['f1'] * 100, 1),
                'SRO P': round(sro_metrics['precision'] * 100, 1),
                'SRO R': round(sro_metrics['recall'] * 100, 1),
                'SRO F1': round(sro_metrics['f1'] * 100, 1),
            }
            
            results.append(result_entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by model and candidate rate
    if not df.empty:
        df = df.sort_values(['Model', 'Cand. Rate'])
    
    # Save to file if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.1f')
        print(f"Results saved to {output_path}")
        return None
    
    return df

def concat_batch_results(base_path, model=''):
    """
    Merge JSON and CSV files of the same name from multiple batch sub-folders into
    a single combined file at the model root.

    Args:
        base_path (str): Base path containing the model directory.
        model (str): Model name (sub-directory under base_path).

    Returns:
        dict: Mapping of file name → path of the merged output file.
    """

    # Resolve model directory
    model_path = os.path.join(base_path, model)

    if not os.path.exists(model_path):
        print(f"Path not found: {model_path}")
        return {}

    # Find batch sub-folders (e.g. 0_547, 1_547, ...)
    batch_folders = []
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and '_' in item:
            batch_folders.append(item_path)

    if not batch_folders:
        print(f"No batch folders found: {model_path}")
        return {}

    print(f"Found {len(batch_folders)} batch folder(s): {batch_folders}")

    # Dictionaries to accumulate data per file type
    json_data = defaultdict(list)        # regular JSON files
    csv_data = defaultdict(list)         # CSV files
    jsonl_data = defaultdict(list)       # line-delimited JSON files
    array_json_data = defaultdict(list)  # array-format JSON files

    # Collect JSON and CSV files from each batch folder
    for batch_folder in batch_folders:
        print(f"Processing: {batch_folder}")

        # Process JSON files
        for json_file in glob.glob(os.path.join(batch_folder, "*.json")):
            file_name = os.path.basename(json_file)

            # These files use JSONL (line-delimited) format
            if file_name in ['triplets_predict.json', 'subject_predict.json']:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:  # skip blank lines
                                try:
                                    data = json.loads(line)
                                    jsonl_data[file_name].append(data)
                                except json.JSONDecodeError as e:
                                    print(f"JSONL parse error: {json_file}, line: {line[:50]}..., error: {e}")
                except Exception as e:
                    print(f"Error processing JSONL file: {json_file}, error: {e}")

            # Array-format JSON files
            elif file_name in ['all_subject.json', 'all.json']:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content and content[0] == '[' and content[-1] == ']':
                            try:
                                data = json.loads(content)
                                if isinstance(data, list):
                                    # Add each item in the array
                                    array_json_data[file_name].extend(data)
                                else:
                                    print(f"Not an array JSON file: {json_file}")
                            except json.JSONDecodeError as e:
                                print(f"Array JSON parse error: {json_file}, error: {e}")
                except Exception as e:
                    print(f"Error processing array JSON file: {json_file}, error: {e}")

            # Regular JSON files
            elif file_name in ['SR_result.json', 'SRO_result.json', 'SR_result_no_sent_idx.json', 'SRO_result_no_sent_idx.json', 'pred_triplet_path.json']:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            json_data[file_name].append(data)
                        except json.JSONDecodeError:
                            print(f"JSON format error: {json_file} (empty or malformed)")
                except Exception as e:
                    print(f"Error processing JSON file: {json_file}, error: {e}")

            # Skip unsupported file types
            else:
                print(f"Unsupported JSON file format: {file_name}, skipping.")

        # Process CSV files
        for csv_file in glob.glob(os.path.join(batch_folder, "*.csv")):
            file_name = os.path.basename(csv_file)

            try:
                # Skip empty files
                if os.path.getsize(csv_file) == 0:
                    print(f"Empty CSV file: {csv_file}, skipping.")
                    continue

                df = pd.read_csv(csv_file)
                if not df.empty:
                    csv_data[file_name].append(df)
                else:
                    print(f"Empty DataFrame: {csv_file}, skipping.")
            except Exception as e:
                print(f"Error processing CSV file: {csv_file}, error: {e}")

    # Dict to store result file paths
    result_files = {}

    # Merge and save regular JSON files
    for file_name, data_list in json_data.items():
        if not data_list:
            print(f"No data to merge: {file_name}, skipping.")
            continue

        output_file = os.path.join(model_path, file_name)

        if file_name in ['SR_result.json', 'SRO_result.json', 'SR_result_no_sent_idx.json', 'SRO_result_no_sent_idx.json']:
            # Result files: merge as dict (sum numeric values for duplicate keys)
            combined_data = {}
            for data in data_list:
                for key, value in data.items():
                    if key not in combined_data:
                        combined_data[key] = value
                    else:
                        # Merge nested dicts; add numeric values
                        if isinstance(value, dict) and isinstance(combined_data[key], dict):
                            for subkey, subvalue in value.items():
                                if subkey in combined_data[key]:
                                    if isinstance(subvalue, (int, float)) and isinstance(combined_data[key][subkey], (int, float)):
                                        combined_data[key][subkey] += subvalue
                                else:
                                    combined_data[key][subkey] = subvalue

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
        else:
            # Other JSON files: merge as list
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
        result_files[file_name] = output_file
        print(f"Saved: {output_file}")

    # Merge and save array JSON files
    for file_name, data_list in array_json_data.items():
        if not data_list:
            print(f"No data to merge: {file_name}, skipping.")
            continue

        output_file = os.path.join(model_path, file_name)

        # Combine all items into a single array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        result_files[file_name] = output_file
        print(f"Saved: {output_file}")

    # Merge and save JSONL files
    for file_name, data_list in jsonl_data.items():
        if not data_list:
            print(f"No data to merge: {file_name}, skipping.")
            continue

        output_file = os.path.join(model_path, file_name)

        # Write each object on a new line
        with open(output_file, 'w', encoding='utf-8') as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        result_files[file_name] = output_file
        print(f"Saved: {output_file}")

    # Merge and save CSV files
    for file_name, df_list in csv_data.items():
        if not df_list:
            print(f"No data to merge: {file_name}, skipping.")
            continue

        output_file = os.path.join(model_path, file_name)

        try:
            combined_df = pd.concat(df_list, ignore_index=True)
            if not combined_df.empty:
                combined_df.to_csv(output_file, index=False)
                result_files[file_name] = output_file
                print(f"Saved: {output_file}")
            else:
                print(f"Merged DataFrame is empty: {file_name}, not saving.")
        except Exception as e:
            print(f"Error merging CSV file: {file_name}, error: {e}")



def visualize_rexval_metrics(data_path, output_path=None, metrics=['SR F1', 'SRO F1'], 
                            figsize=(16, 12), style='whitegrid'):
    """
    Visualize metrics for different models in rexval mode using line plots.
    
    Args:
        data_path (str): Path to the CSV file with comparison data
        output_path (str, optional): Path to save the visualization
        metrics (list): Metrics to visualize (e.g., ['SR F1', 'SRO F1'])
        figsize (tuple): Figure size (width, height)
        style (str): Seaborn style for the plot
    """
    # Load data
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Filter out gt_report
    # df = df[~df['Model'].str.contains('gt_report')]
    
    if df.empty:
        print("No data left after filtering out gt_report")
        return
    
    # Extract model name from the full model name (e.g., 'gpt-4.1_radgraph' -> 'radgraph')
    df['Short Model'] = df['Model'].apply(lambda x: x.split('_')[-1] if '_' in x else x)
    
    # Set the style
    sns.set_style(style)
    
    # Create a figure with higher DPI for publication quality
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    
    # Define markers and line styles for the metrics
    markers = ['o', 's', '^', 'D', 'v']  # Different markers for each metric
    line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for each metric
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    # Create x positions for the models
    models = df['Short Model'].unique()
    x = np.arange(len(models))
    
    # Store text objects for later adjustment
    texts = []
    
    # Create legend handles
    legend_handles = []
    
    # Plot each metric
    for j, metric in enumerate(metrics):
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the data.")
            continue
            
        # Get values for this metric across all models
        y_values = []
        for model in models:
            model_data = df[df['Short Model'] == model]
            if not model_data.empty:
                model_value = model_data[metric].values[0]
                y_values.append(model_value)
            else:
                y_values.append(0)  # Default value if model data is missing
        
        # Plot the line with explicit color
        line, = ax.plot(x, y_values, 
                      marker=markers[j % len(markers)], 
                      linestyle=line_styles[j % len(line_styles)],
                      linewidth=3, 
                      markersize=16, 
                      alpha=0.8,
                      color=colors[j],
                      label=metric)
        
        # Add to legend handles
        legend_handles.append(line)
        
        # Add labels at each point
        for i, (xi, yi) in enumerate(zip(x, y_values)):
            t = ax.text(xi, yi, f'{yi:.1f}', 
                       ha='center', 
                       va='bottom', 
                       fontsize=24, 
                       fontweight='bold',
                       color=colors[j])
            texts.append(t)
    
    # Use adjust_text to prevent overlapping
    try:
        from adjustText import adjust_text
        adjust_text(texts, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
                    expand_points=(1.7, 1.7),
                    force_points=(0.6, 0.6))
    except ImportError:
        print("Warning: adjustText package not found. Text labels may overlap.")
    
    # Enhance the plot for publication quality
    ax.set_title('Model Performance Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Score (%)', fontsize=16, fontweight='bold', labelpad=15)
    
    # Set x-axis ticks to model names
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14, fontweight='bold', rotation=45, ha='right')
    
    # Format y-axis as percentage with larger ticks
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax.tick_params(axis='y', which='major', labelsize=14)
    
    # Set y-axis to start from 0
    max_value = 100
    if not df[metrics].empty:
        max_value = max(df[metrics].max().max() * 1.1, 100)
    ax.set_ylim(0, max_value)
    
    # Make tick marks thicker
    ax.tick_params(width=2, length=8)
    
    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=1.5)
    
    # Add legend with larger font - only if we have handles
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=14, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()        
        
def visualize_metrics(data_path, output_path=None, metrics=['SR F1', 'SRO F1'], 
                            figsize=(16, 12), style='whitegrid', y_min=40, y_max=100):
    """
    Visualize metrics for different models using separate plots for each metric.
    
    Args:
        data_path (str): Path to the CSV file with comparison data
        output_path (str, optional): Path to save the visualization
        metrics (list): Metrics to visualize (e.g., ['SR F1', 'SRO F1'])
        figsize (tuple): Figure size (width, height)
        style (str): Seaborn style for the plot
        y_min (int): Minimum value for y-axis
        y_max (int): Maximum value for y-axis
    """
    # Load data
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Set the style
    sns.set_style(style)
    
    # Create a separate plot for each metric
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the data.")
            continue
        
        # Create figure
        plt.figure(figsize=figsize, dpi=100)
        
        # Get unique model-candidate combinations
        model_candidates = df[['Model']].copy()
        model_candidates['Short Model'] = model_candidates['Model'].apply(lambda x: x.split('_')[-1])
        model_candidates['Shot Type'] = model_candidates['Model'].apply(lambda x: x.split('_')[0])
        model_candidates['Candidate'] = model_candidates['Model'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'no')
        
        # Get unique combinations
        unique_combinations = model_candidates[['Short Model', 'Candidate']].drop_duplicates()
        
        # Plot each combination
        for idx, (model, candidate) in unique_combinations.iterrows():
            # Get data for this combination
            mask = (model_candidates['Short Model'] == model) & (model_candidates['Candidate'] == candidate)
            combination_data = df[mask].copy()
            
            if len(combination_data) > 0:
                # Sort by shot type for proper line connection
                combination_data['Shot Type'] = combination_data['Model'].apply(lambda x: x.split('_')[0])
                combination_data = combination_data.sort_values('Shot Type')
                
                # Plot line
                plt.plot(combination_data['Shot Type'], combination_data[metric], 
                        marker='o', linestyle='-', linewidth=2, markersize=8,
                        label=f"{model}-{candidate}")
                
                # Add value labels
                for _, row in combination_data.iterrows():
                    plt.text(row['Shot Type'], row[metric], f'{row[metric]:.1f}',
                            ha='center', va='bottom', fontsize=12)
        
        # Customize plot
        plt.title(f'{metric} Performance by Shot Type', fontsize=20, pad=20)
        plt.xlabel('Shot Type', fontsize=16, labelpad=10)
        plt.ylabel('Score (%)', fontsize=16, labelpad=10)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        
        # Set y-axis limits
        plt.ylim(y_min, y_max)
        
        # Customize grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the plot
        if output_path:
            metric_output_path = os.path.join(output_path, f'{metric.replace(" ", "_")}.png')
            os.makedirs(os.path.dirname(metric_output_path), exist_ok=True)
            plt.savefig(metric_output_path, dpi=300, bbox_inches='tight')
            print(f"{metric} visualization saved to {metric_output_path}")
        else:
            plt.show()
        
        plt.close()