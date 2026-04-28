import os
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .utils import (generate_graph, create_input, create_batch_file, run_llm_batch, read_batch_results, monitor_batch_job, process_chunk, merge_results, 
                  initialize_llm_client, evaluate_funct, create_relation_dataframe, create_relation_dataframe2, create_relation_dataframe3,
                  generate_table, visualize_table, visualize_metrics, visualize_rexval_metrics, create_fp, create_fn)

def main(args, client, tokenizer):
    if not args.dynamic_retrieval:
        if args.multi:
            results_path = f'./singleSR/exp/M{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'
        else:
            results_path = f'./singleSR/exp/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'
    else:
        results_path = f'./singleSR/exp/dynamic_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'

    if not args.multi and (args.deployment_name.startswith('gpt') or args.deployment_name.startswith('claude')) and args.api_key != 'local_LLM':
        
        if not args.dynamic_retrieval:
            batch_path = f'./singleSR/data/batch_files/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'
        else:
            batch_path = f'./singleSR/data/batch_files/dynamic_{args.candidate_type}_{args.deployment_name}/{args.mode}/{args.unit}/{args.candidate_usage}'
        # 1. Generate batch file
        if not os.path.exists(batch_path):
            create_batch_file(args)
        elif not any(f.endswith('.jsonl') for f in os.listdir(batch_path)):
            create_batch_file(args)

        if not os.path.exists(f'{results_path}/batch_results.jsonl'):
            print("\n LLM result file does not exist. Running LLM Batch Processing!!")

            # 2. Get batch file paths (supports single file or multi-chunk files)
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"Batch path does not exist: {batch_path}")
            batch_file_list = sorted(f for f in os.listdir(batch_path) if f.endswith('.jsonl'))
            if not batch_file_list:
                raise FileNotFoundError(f"No .jsonl files found in: {batch_path}")

            # 3. Submit and collect all chunks sequentially
            all_result_lines = []
            for chunk_idx, batch_fname in enumerate(batch_file_list):
                batch_file_path = os.path.join(batch_path, batch_fname)
                print(f"\n[Chunk {chunk_idx + 1}/{len(batch_file_list)}] Submitting: {batch_fname}")

                batch_job = run_llm_batch(batch_file_path, client, args)
                print(f"batch_job id: {batch_job.id}")
                batch_job_id = batch_job.id

                batch_status = monitor_batch_job(batch_job_id, client, args, check_interval=60)

                # Final status check
                if args.deployment_name.startswith('claude'):
                    _raw = getattr(client, 'client', client)
                    batch = _raw.messages.batches.retrieve(batch_job_id)
                    completed_count = batch.request_counts.succeeded
                    total_count = (batch.request_counts.succeeded + batch.request_counts.errored +
                                   batch.request_counts.processing + batch.request_counts.canceled)
                else:
                    batch = client.batches.retrieve(batch_job_id)
                    completed_count = batch.request_counts.completed
                    total_count = batch.request_counts.total

                print(f"\nChunk {chunk_idx + 1} final status: {batch_status}  ({completed_count}/{total_count})")

                if batch_status in ["completed", "succeeded", "ended"]:
                    if args.deployment_name.startswith('gpt'):
                        chunk_content = client.files.content(batch.output_file_id).text
                        all_result_lines.extend(chunk_content.strip().splitlines())
                    else:
                        chunk_list = []
                        _raw = getattr(client, 'client', client)
                        for result in _raw.messages.batches.results(batch_job_id):
                            if result.result.type == "succeeded":
                                content = result.result.message.content[0]
                                # Serialize tool_use input dict to JSON string for uniform downstream handling
                                content_str = (json.dumps(content.input)
                                               if content.type == 'tool_use'
                                               else content.text)
                                chunk_list.append(json.dumps({
                                    "custom_id": result.custom_id,
                                    "content": content_str,
                                }))
                        all_result_lines.extend(chunk_list)
                        print(f"  Collected {len(chunk_list)} results from chunk {chunk_idx + 1}")
                else:
                    print(f"  WARNING: chunk {chunk_idx + 1} ended with status '{batch_status}' — skipping")

            os.makedirs(results_path, exist_ok=True)
            with open(f'{results_path}/batch_results.jsonl', 'w') as f:
                f.write("\n".join(all_result_lines))
            print(f"\nAll chunks done. Total results saved: {len(all_result_lines)}")
            print("-----")
        else:
            print("LLM result file exists. Reading LLM result...")

        _, LLM_SR_file = read_batch_results(batch_path, results_path, args) 
                     
    else:
        if args.run_model:
            final_output_file = os.path.join(results_path, f'{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}_{args.output_format}_{args.mode}_{args.unit}.json')

            if not os.path.exists(final_output_file):                
                input_data, devset = create_input(args)

                # Split data into chunks
                if args.mode in ['medversa', 'rgrg', 'cvt2distilgpt2']:
                    chunk_size = 5
                else:
                    chunk_size = 225  # Adjust this based on your needs

                data_items = list(input_data.items())
                chunks = [dict(data_items[i:i + chunk_size]) for i in range(0, len(data_items), chunk_size)]

                # Process chunks in parallel
                with Pool(processes=min(cpu_count(), 8)) as pool:
                    chunk_args = [(chunk, devset, results_path, args, i, None, None) for i, chunk in enumerate(chunks)]
                    chunk_files = list(tqdm(
                        pool.starmap(process_chunk, chunk_args),
                        total=len(chunks),
                        desc="Processing chunks"
                    ))
    
                merge_results(os.path.join(results_path, 'intermediate'), final_output_file)

            LLM_SR_file = final_output_file
        else:
            LLM_SR_file = os.path.join(results_path, f'{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}_{args.output_format}_{args.mode}_{args.unit}.json')
                
    gold_file_path = f'{args.output_dir}/{args.mode}_{args.candidate_type}_{args.unit}_input.json'
    
    if args.mode == 'silver_eval':
        create_relation_dataframe3(gold_file_path, LLM_SR_file, f'./singleSR/eval/{args.mode}/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}')

    elif args.mode in ['gold_eval', 'rexval', 'maira', 'maira_cascade', 'rexerr', 'medversa', 'rgrg', 'cvt2distilgpt2', 'libra', 'chexagent']:
        eval_path = evaluate_funct(gold_file_path, LLM_SR_file, args)
        
        create_fp(eval_path, gold_file_path)
        create_fn(eval_path, gold_file_path)
        
        if args.mode in ['rexval']:
            create_relation_dataframe(gold_file_path, LLM_SR_file, eval_path)
        else:
            create_relation_dataframe2(gold_file_path, LLM_SR_file, eval_path)
        
        generate_graph(eval_path, args)   
        
        generate_table(base_path=f'./singleSR/eval/{args.mode}', args=args,
                    output_path=f'./singleSR/model_comparison/{args.mode}/all_models_comparison.csv',
                    mode=args.mode)
        
        if args.mode != 'rexval':
            visualize_table(f'./singleSR/model_comparison/{args.mode}/all_models_comparison.csv', 
                            output_path=f'./singleSR/model_comparison/figures/{args.mode}/table.png')

        if args.mode == 'rexval':
            visualize_rexval_metrics(
            f'./singleSR/model_comparison/{args.mode}/all_models_comparison.csv',
            f'./singleSR/model_comparison/figures/{args.mode}/model_comparison.png',
            metrics=['SR F1', 'SRO F1'])
        
        else:
            visualize_metrics(f'./singleSR/model_comparison/{args.mode}/all_models_comparison.csv', 
                            output_path=f'./singleSR/model_comparison/figures/{args.mode}',
                            metrics=['SR F1', 'SRO F1'])

            visualize_metrics(f'./singleSR/model_comparison/{args.mode}/all_models_comparison.csv', 
                            output_path=f'./singleSR/model_comparison/figures/{args.mode}',
                            metrics=['SRO P', 'SRO R'])
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create input data for GPT-SR')
    parser.add_argument('--gold_path', type=str, default='./dataset/Lunguage.csv',
                        help='Path to gold dataset')
    
    ########### SILVER EVAL ###########
    parser.add_argument('--silver_eval_report_path', type=str, default='./dataset/processed_reports/mimic_cxr_reports.csv',
                        help='Raw Report Path')
    #######################################
    
    ########### REXVAL  ###########
    parser.add_argument('--rexval_report_path', type=str, default='./benchmark/rexval/50_samples_gt_and_candidates.csv',
                        help='Raw Report Path')
    parser.add_argument('--report_col_name', type=list, default=['gt_report', 'radgraph', 'bertscore', 's_emb', 'bleu'],
                        help='Report column name')
    #######################################
    
    ########### MAIRA 2 ###########
    parser.add_argument('--maira2_cascade_report_path', type=str, default='./benchmark/maira_cascade.ndjson',
                        help='Path to mira2 cascade report')
    parser.add_argument('--maira2_report_path', type=str, default='./benchmark/maira.ndjson',
                        help='Path to mira2 report')
    #######################################


    ########### MEDVERSA ###########
    parser.add_argument('--medversa_report_path', type=str, default='./benchmark/medversa.csv',
                        help='Path to medversa report')
    #######################################
    
    ########### RGRG ###########
    parser.add_argument('--rgrg_report_path', type=str, default='./benchmark/rgrg.csv',
                        help='Path to rgrg report')
    #######################################
    
    ########### Cvt2distilgpt2 ###########
    parser.add_argument('--cvt2distilgpt2_report_path', type=str, default='./benchmark/cvt2distilgpt2.csv',
                        help='Path to cvt2distilgpt2 report')
    #######################################

    ########### REXERR ###########
    parser.add_argument('--rexerr_report_path', type=str, default='./benchmark/rexerr_gold.csv',
                        help='Path to rexerr report')
    #######################################

    parser.add_argument('--lingshu_path', type=str, default='./benchmark/lingshu.csv',
                        help='Path to lingshu report')
    parser.add_argument('--medgemma_path', type=str, default='./benchmark/medgemma.csv',
                        help='Path to rexerr report')
    parser.add_argument('--libra_path', type=str, default='./benchmark/libra_results.csv',
                        help='Path to libra report')
    parser.add_argument('--chexagent_path', type=str, default='./benchmark/chexagent_results.csv',
                        help='Path to chexagent report')

    parser.add_argument('--run_model', action='store_true', default=True,
                        help='Run model')
    parser.add_argument('--toy_set', action='store_true', default=True,
                        help='Use toy set')
    parser.add_argument('--output_dir', type=str, default='./singleSR/data',
                        help='Directory to save output files')
    parser.add_argument('--deployment_name', type=str, default='gpt-oss-20b',
                        choices=['gpt-4.1',
                            'gpt-5',
                            'gpt-oss-20b',
                            'gpt-oss-120b',
                            'baichuan',
                            'medgemma-27b-text-it',
                            'qwen3-235b-a22b',
                            'deepseek-v3-0324',
                            'llama4-maverick-instruct-basic',
                            'claude-haiku-4-5-20251001',
                            'claude-sonnet-4-6',
                            'claude-opus-4-6'])
    
    parser.add_argument('--mode', type=str, default='gold_eval',
                        choices=['rexval', 'rexerr', 'gold_eval', 'maira_cascade', 'maira', 'medversa', 'rgrg', 'cvt2distilgpt2', 'silver_eval', 'libra', 'chexagent'],
                        help='Mode for data processing')
        
    parser.add_argument('--candidate_type', type=str, default='vocab_ent_rcg',
                        choices=['gt_sro_review', 'gt_sro', 'gt_so', 'gt_s', 'gt_ent_rcg', 'vocab_so', 'vocab_s', 'vocab_ent_rcg', 'no_candidates'],
                        help='Use gold standard')
    parser.add_argument('--unit', type=str, default='section',
                        choices=['report', 'section', 'sent'],
                        help='Unit for SR extraction')
    parser.add_argument('--n_retrieval', type=int, default=5,
                        help='Number of retrievals')
    parser.add_argument('--multi', action='store_true', default=False,
                        help='Use multi-turn')
    parser.add_argument('--context_width', type=int, default=4,
                        help='Context width')
    
    
    ########### Batch split ###########
    parser.add_argument('--batch_itr', type=int, default=None,
                        help='Batch iteration')
    
    parser.add_argument('--batch_file_size', type=int, default=2,
                        help='Batch split size')
    #######################################
    
    
    parser.add_argument('--entity_types', nargs='+', default=['COF', 'NCD', 'PATIENT INFO.', 'PF', 'CF', 'OTH'],
                        help='Entity types')
    
    parser.add_argument('--relation_types', nargs='+', default=['Location', 'Associate', 'Evidence'],
                        help='Relation types')
    
    parser.add_argument('--attribute_types', nargs='+', 
                        default=['Morphology', 'Distribution', 'Measurement', 'Severity',
                                'Comparison', 'Onset', 'No Change', 'Improved', 'Worsened',
                                'Placement', 'Past Hx', 'Other Source', 'Assessment Limitations'],
                        help='Attribute types')
    
    
    parser.add_argument('--output_format', type=str, default='SROSRO',
                        help='Output format')
    
    parser.add_argument('--diverse_retrieval', action='store_true', default=True,
                        help='Use diverse retrieval')
    parser.add_argument('--fast_retrieval', action='store_true', default=True,
                        help='Use fast retrieval')
    parser.add_argument('--dynamic_retrieval', action='store_true', default=False,
                        help='Use dynamic retrieval')
    parser.add_argument('--candidate_usage', type=float, default=1,
                        help='Candidate usage ratio')
    parser.add_argument('--candidate_discontinuous', action='store_true', default=False,
                        help='Use discontinuous candidate')
    parser.add_argument('--jaccard', action='store_true', default=True,
                        help='Use Jaccard similarity')
    parser.add_argument('--vocab_path', type=str, default='./dataset/Lunguage_vocab.csv',
                        help='Path to vocab file')

    parser.add_argument('--api_key', type=str, default='local_LLM',
                        help='API key for the selected model for third party usage like OpenAI, Fireworks, Anthropic, etc.')
    
    args = parser.parse_args()
    
    args.dev_list = []
    
    client, tokenizer = initialize_llm_client(args.deployment_name, args.api_key)
    
    main(args, client, tokenizer)