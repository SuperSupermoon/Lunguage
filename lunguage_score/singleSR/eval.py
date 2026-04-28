import os
import json
from fuzzywuzzy import fuzz
from collections import defaultdict
import re
import pandas as pd
fuzzy_threshold = 60

RELATIONS = ['cat', 'dx_status', 'dx_certainty', 'location', 'placement', 'associate', 'evidence',
'morphology', 'distribution', 'measurement', 'severity', 'comparison',
'onset', 'no change', 'improved', 'worsened', 'past hx', 'other source', 'assessment limitations']

def fuzzy_match(triplet1, triplet2, relation, jaccard=True, th=fuzzy_threshold):

    if relation != 'location' or not jaccard:
        if triplet1[1] == triplet2[1]:  # If the relation is the same
            if relation in ['cat', 'dx_status', 'dx_certainty']:  # If relation is 'cat' or 'status'
                if fuzz.ratio(triplet1[0], triplet2[0]) > th and triplet1[2] == triplet2[2]:
                    return True
            else:  # For all other relations
                if fuzz.ratio(triplet1[0], triplet2[0]) > th and fuzz.ratio(triplet1[2], triplet2[2]) > th:
                    return True
    else:
        if triplet1[1] == 'location' and triplet2[1] == 'location':
            if fuzz.ratio(triplet1[0], triplet2[0]) > th:
                # Remove location and detail identifiers (loc:, det:) and create sets
                triplet1_location_set = set(re.sub(r'(loc|det):\w+', '', triplet1[2]).strip().split(','))
                triplet2_location_set = set(re.sub(r'(loc|det):\w+', '', triplet2[2]).strip().split(','))
                
                # Calculate fuzzy Jaccard similarity between the two location sets
                match_count = 0
                for loc1 in triplet1_location_set:
                    for loc2 in triplet2_location_set:
                        if fuzz.ratio(loc1.strip(), loc2.strip()) > th:
                            match_count += 1
                            break
                
                # If all locations in both sets match (Jaccard similarity = 1)
                if match_count == len(triplet1_location_set) and match_count == len(triplet2_location_set):
                    return True
    return False

def enhanced_fuzzy_match(triplet1, triplet2, relation, jaccard=True, th=fuzzy_threshold):
    # 1. Apply original fuzzy_match logic
    original_match = fuzzy_match(triplet1, triplet2, relation, jaccard, th)
    if original_match:
        return True
 
    if triplet1[1] != triplet2[1]:
        return False
    
    # Handle special relations
    if relation in ['cat', 'dx_status', 'dx_certainty']:
        # These relations require an exact object match
        if triplet1[2] != triplet2[2]:
            return False

        # Match subject flexibly
        # Check original subject against reversed position
        if fuzz.ratio(triplet1[0], triplet2[2]) > th:
            return True
        return False
    
    elif relation == 'location':
        # Special handling for location relations
        # Check original subject against reversed position
        if fuzz.ratio(triplet1[0], triplet2[2]) > th:
            # Check original object against reversed position (location special case)
            triplet1_location_set = set(re.sub(r'(loc|det):\w+', '', triplet1[2]).strip().split(','))
            triplet2_location_set = set(re.sub(r'(loc|det):\w+', '', triplet2[0]).strip().split(','))
            
            match_count = 0
            for loc1 in triplet1_location_set:
                for loc2 in triplet2_location_set:
                    if fuzz.ratio(loc1.strip(), loc2.strip()) > th:
                        match_count += 1
                        break
            
            if match_count > 0 and match_count >= min(len(triplet1_location_set), len(triplet2_location_set)) * 0.5:
                return True
        return False
    
    else:
        # Flexible matching for general relations
        # 1. Normal position matching (subject-subject, object-object)
        normal_match = fuzz.ratio(triplet1[0], triplet2[0]) > th and fuzz.ratio(triplet1[2], triplet2[2]) > th

        # 2. Cross-position matching (subject-object, object-subject)
        cross_match = fuzz.ratio(triplet1[0], triplet2[2]) > th and fuzz.ratio(triplet1[2], triplet2[0]) > th

        # 3. Partial matching (subject/object substring contained in the other)
        partial_match = False

        # subject1 is a substring of object2, or vice versa
        if (triplet1[0] in triplet2[2] or triplet2[2] in triplet1[0]) and fuzz.ratio(triplet1[2], triplet2[0]) > th:
            partial_match = True

        # object1 is a substring of subject2, or vice versa
        if (triplet1[2] in triplet2[0] or triplet2[0] in triplet1[2]) and fuzz.ratio(triplet1[0], triplet2[2]) > th:
            partial_match = True

        set_match = False

        # Convert subject and object of triplet1 to a set
        triplet1_elements = set([triplet1[0].strip(), triplet1[2].strip()])
        # Convert subject and object of triplet2 to a set
        triplet2_elements = set([triplet2[0].strip(), triplet2[2].strip()])

        # Perform fuzzy matching between the two sets
        match_count = 0
        for elem1 in triplet1_elements:
            for elem2 in triplet2_elements:
                if fuzz.ratio(elem1, elem2) > th:
                    match_count += 1
                    break

        # If all elements in both sets are matched, set_match = True
        if match_count == len(triplet1_elements) and match_count == len(triplet2_elements):
            set_match = True
        
        return normal_match or set_match #or cross_match #

def flexible_fuzzy_match(triplet1, triplet2, relation, jaccard=True, th=fuzzy_threshold):
    """
    Combines original fuzzy_match with the new flexible (enhanced) matching.
    """
    # Try original fuzzy_match first
    if fuzzy_match(triplet1, triplet2, relation, jaccard, th):
        return True

    # If original match fails, try enhanced matching
    return enhanced_fuzzy_match(triplet1, triplet2, relation, jaccard, th)

def SR_EVAL(data_path, gpt_pred_path, save_path, relation_to_evaluate=None):
    print("\n=== Starting SR_EVAL with debugging ===")
    
    if os.path.exists(f"{save_path}/subject_predict.json"):
        with open(f"{save_path}/subject_predict.json", "w") as file:
            pass

    with open(data_path, 'r') as file:
        data = json.load(file)

    with open(gpt_pred_path, 'r') as file:
        gpt_pred_data = json.load(file)

    debug_log_path = f"{save_path}/sr_eval_debug.log"
    with open(debug_log_path, "w") as debug_file:
        debug_file.write("=== SR_EVAL Debugging Log ===\n\n")

    for custom_id in data.keys():
        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"\n\n=== Processing custom_id: {custom_id} ===\n")
            debug_file.write(f"Passage: {data[custom_id]['passage']}\n\n")
        
        sentences = data[custom_id]['passage']
        
        # Handle case where custom_id might not exist in gpt_pred_data
        if custom_id not in gpt_pred_data:
            with open(debug_log_path, "a") as log_file:
                log_file.write(f"GPT prediction not found for {custom_id}\n")
            save = {
                'custom_id': custom_id,
                "data_from": data[custom_id]['data_from'],
                "report_type": data[custom_id]['section'],
                "sentences": sentences,
                "relations": data[custom_id]['relations'],
                "right_entities": {},
                "wrong_entities": {},
                "right_entities_no_sent_idx": {},
                "wrong_entities_no_sent_idx": {},
                "pred_triplet": {},
                "same_triplet_list": data[custom_id]['same_triplet_list']
            }
            with open(f"{save_path}/subject_predict.json", "a") as file:
                json.dump(save, file)
                file.write('\n')
            continue
        
        gpt_pred = gpt_pred_data[custom_id]
        
        right = {}
        wrong = {}
        right_no_sent_idx = {}
        wrong_no_sent_idx = {}
        save_pred_triplet_list = {}

        relations_to_check = relation_to_evaluate if relation_to_evaluate else data[custom_id]['relations']

        for relation in relations_to_check:
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"\n--- Processing relation: {relation} ---\n")
            
            right[relation] = []
            wrong[relation] = []
            right_no_sent_idx[relation] = []
            wrong_no_sent_idx[relation] = []
            
            pred_triplet_list = gpt_pred.get(relation, [])
            save_pred_triplet_list[relation] = pred_triplet_list
            
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"Predicted triplets for {relation}: {pred_triplet_list}\n\n")
                debug_file.write(f"GT triplets for {relation}:\n")
                gt_triplets = [triplets for triplets in data[custom_id]['same_triplet_list'] if triplets[0][1] == relation]
                for i, triplet in enumerate(gt_triplets):
                    debug_file.write(f"  {i}: {triplet}\n")
                debug_file.write("\n")
            
            triplet_index = []
            for i, pred_triplet in enumerate(pred_triplet_list):
                entity = pred_triplet[0]
                pred_sent_idx = pred_triplet[3] if len(pred_triplet) > 3 else None
                
                with open(debug_log_path, "a") as debug_file:
                    debug_file.write(f"MODE 1 - Processing pred triplet {i}: {pred_triplet}\n")
                    debug_file.write(f"  Entity: {entity}, Pred sent idx: {pred_sent_idx}\n")
                
                flag = 0
                subject_triplets = [triplets for triplets in data[custom_id]['same_triplet_list'] if triplets[0][1] == relation]
                
                for index, true_triplet in enumerate(subject_triplets):
                    true_entity = true_triplet[0][0]
                    true_sent_idx = true_triplet[0][3] if len(true_triplet[0]) > 3 else None
                    
                    similarity = fuzz.ratio(entity, true_entity)
                    
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"  Comparing with GT triplet {index}: {true_triplet[0]}\n")
                        debug_file.write(f"    GT Entity: {true_entity}, GT sent idx: {true_sent_idx}\n")
                        debug_file.write(f"    Similarity: {similarity}\n")
                    
                    if similarity > 70:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"    Similarity > 70: YES\n")
                        
                        if pred_sent_idx is not None and true_sent_idx is not None:
                            with open(debug_log_path, "a") as debug_file:
                                debug_file.write(f"    Both sent idx exist: YES\n")
                                debug_file.write(f"    Sent idx match: {pred_sent_idx == true_sent_idx}\n")
                            
                            if pred_sent_idx == true_sent_idx:
                                flag = 1
                                if index not in triplet_index:
                                    with open(debug_log_path, "a") as debug_file:
                                        debug_file.write(f"    MATCH FOUND! Adding to right[{relation}]\n")
                                    
                                    right[relation].append(entity)
                                    triplet_index.append(index)
                                    break
                        else:
                            with open(debug_log_path, "a") as debug_file:
                                debug_file.write(f"    Both sent idx exist: NO (pred_sent_idx={pred_sent_idx}, true_sent_idx={true_sent_idx})\n")
                    else:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"    Similarity > 70: NO\n")
                
                if not flag:
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"  No match found. Adding to wrong[{relation}]\n")
                    
                    wrong[relation].append(entity)
            
            triplet_index_no_sent = []
            for i, pred_triplet in enumerate(pred_triplet_list):
                entity = pred_triplet[0]
                
                with open(debug_log_path, "a") as debug_file:
                    debug_file.write(f"MODE 2 - Processing pred triplet {i}: {pred_triplet}\n")
                    debug_file.write(f"  Entity: {entity}\n")
                
                flag = 0
                subject_triplets = [triplets for triplets in data[custom_id]['same_triplet_list'] if triplets[0][1] == relation]
                
                for index, true_triplet in enumerate(subject_triplets):
                    true_entity = true_triplet[0][0]
                    
                    similarity = fuzz.ratio(entity, true_entity)
                    
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"  Comparing with GT triplet {index}: {true_triplet[0]}\n")
                        debug_file.write(f"    GT Entity: {true_entity}\n")
                        debug_file.write(f"    Similarity: {similarity}\n")
                    
                    if similarity > 70:
                        flag = 1
                        if index not in triplet_index_no_sent:
                            with open(debug_log_path, "a") as debug_file:
                                debug_file.write(f"    MATCH FOUND! Adding to right_no_sent_idx[{relation}]\n")
                            
                            right_no_sent_idx[relation].append(entity)
                            triplet_index_no_sent.append(index)
                            break
                
                if not flag:
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"  No match found. Adding to wrong_no_sent_idx[{relation}]\n")
                    
                    wrong_no_sent_idx[relation].append(entity)
            
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"\n--- Results comparison for relation {relation} ---\n")
                debug_file.write(f"Mode 1 right: {right[relation]}\n")
                debug_file.write(f"Mode 2 right: {right_no_sent_idx[relation]}\n")
                debug_file.write(f"Mode 1 wrong: {wrong[relation]}\n")
                debug_file.write(f"Mode 2 wrong: {wrong_no_sent_idx[relation]}\n")
                debug_file.write(f"Difference in right: {set(right[relation]) != set(right_no_sent_idx[relation])}\n")
                debug_file.write(f"Difference in wrong: {set(wrong[relation]) != set(wrong_no_sent_idx[relation])}\n\n")
        
        save = {
            'custom_id': custom_id,
            "data_from": data[custom_id]['data_from'],
            "report_type": data[custom_id]['section'],
            "sentences": sentences,
            "relations": data[custom_id]['relations'],
            "right_entities": right,
            "wrong_entities": wrong,
            "right_entities_no_sent_idx": right_no_sent_idx,
            "wrong_entities_no_sent_idx": wrong_no_sent_idx,
            "pred_triplet": save_pred_triplet_list,
            "same_triplet_list": data[custom_id]['same_triplet_list']
        }
        
        with open(f"{save_path}/subject_predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')
    
    print(f"=== SR_EVAL completed. Debug log saved to {debug_log_path} ===")

def SRO_EVAL(data_path, gpt_pred_path, save_path, relation_to_evaluate=None, cat_to_evaluate=['pf', 'cf', 'oth', 'patient info.', 'ncd', 'cof', 'ncd/cof'], jaccard=True, check_ent_idx=False, check_evidence_associate=False):
    print("\n=== Starting SRO_EVAL with debugging ===")
    
    debug_log_path = f"{save_path}/sro_eval_debug.log"
    with open(debug_log_path, "w") as log_file:
        log_file.write("=== SRO_EVAL Debugging Log ===\n\n")
    
    if os.path.exists(f"{save_path}/triplets_predict.json"):
        with open(f"{save_path}/triplets_predict.json", "w") as file:
            pass
    
    with open(data_path, 'r') as file:
        data = json.load(file)

    with open(gpt_pred_path, 'r') as file:
        gpt_pred_data = json.load(file)

    for custom_id in data.keys():
        with open(debug_log_path, "a") as log_file:
            log_file.write(f"\n\n=== Processing custom_id: {custom_id} ===\n")
            log_file.write(f"Passage: {data[custom_id]['passage']}\n\n")
        
        wrong = []
        right = []
        triplet_index = []
        wrong_no_sent_idx = []
        right_no_sent_idx = []
        triplet_index_no_sent = []
        
        sentence = data[custom_id]['passage']
        
        # Handle case where custom_id might not exist in gpt_pred_data
        if custom_id not in gpt_pred_data:
            with open(debug_log_path, "a") as log_file:
                log_file.write(f"GPT prediction not found for {custom_id}\n")
            save = {
                'custom_id': custom_id,
                "data_from": data[custom_id]['data_from'],
                "report_type": data[custom_id]['section'],
                "sentence": sentence,
                "true_triplet_list": data[custom_id]['same_triplet_list'],
                "right_triplet_list": [],
                "wrong_triplet_list": [],
                "miss_triplet_list": data[custom_id]['same_triplet_list'],
                "right_triplet_list_no_sent_idx": [],
                "wrong_triplet_list_no_sent_idx": [],
                "miss_triplet_list_no_sent_idx": data[custom_id]['same_triplet_list']
            }
            with open(f"{save_path}/triplets_predict.json", "a") as file:
                json.dump(save, file)
                file.write('\n')
            continue
            
        gpt_pred = gpt_pred_data[custom_id]
        
        save = {
            'custom_id': custom_id,
            "data_from": data[custom_id]['data_from'],
            "report_type": data[custom_id]['section'],
            "sentence": sentence,
        }
        
        relations_to_check = relation_to_evaluate if relation_to_evaluate else data[custom_id]['relations']
        
        # Filter ground truth triplets based on relations_to_check
        gt_triplet_list = []
        for triplets in data[custom_id]['same_triplet_list']:
            filtered_triplets = []
            for triplet in triplets:
                if triplet[1] in relations_to_check:
                    if triplet[1] == 'cat':
                        if triplet[2] not in cat_to_evaluate:
                            continue
                    filtered_triplets.append(triplet)
            if filtered_triplets:
                gt_triplet_list.append(filtered_triplets)
        
        with open(debug_log_path, "a") as log_file:
            log_file.write(f"GT triplet list (filtered):\n")
            for i, triplet_group in enumerate(gt_triplet_list):
                log_file.write(f"  Group {i}: {triplet_group}\n")
            log_file.write("\n")

        for relation in relations_to_check:
            with open(debug_log_path, "a") as log_file:
                log_file.write(f"--- Processing relation: {relation} ---\n")
            
            save[relation] = {}
            pred_triplet_list = gpt_pred.get(relation, [])
            
            with open(debug_log_path, "a") as log_file:
                log_file.write(f"Predicted triplets for {relation}: {pred_triplet_list}\n\n")
                log_file.write(f"GT triplets for {relation} (filtered):\n")
                for i, triplet_group in enumerate(gt_triplet_list):
                    if triplet_group[0][1] == relation:
                        log_file.write(f"  {i}: {triplet_group}\n")
                log_file.write("\n")
            
            
            # for pred_idx, pred_triplet in enumerate(pred_triplet_list):
            #     flag = 0
                
            #     pred_sent_idx = pred_triplet[3] if len(pred_triplet) > 3 else None
            #     pred_ent_idx = pred_triplet[4] if len(pred_triplet) > 4 else None
                
            #     with open(debug_log_path, "a") as log_file:
            #         log_file.write(f"MODE 1 - Processing pred triplet {pred_idx}: {pred_triplet}\n")
            #         log_file.write(f"  Entity: {pred_triplet[0]}, Relation: {pred_triplet[1]}, Object: {pred_triplet[2]}, Pred sent idx: {pred_sent_idx}\n")
                    
            #         if check_ent_idx:
            #             log_file.write(f", Pred ent idx: {pred_ent_idx}")
            #         log_file.write("\n")

            #     for index, true_triplet_group in enumerate(gt_triplet_list):
            #         true_triplet = true_triplet_group[0]                    
            #         true_sent_idx = true_triplet[3] if len(true_triplet) > 3 else None
            #         true_ent_idx = true_triplet[4] if len(true_triplet) > 4 else None
                    
            #         with open(debug_log_path, "a") as log_file:
            #             log_file.write(f"  Comparing with GT triplet group {index}: {true_triplet_group[0]}\n")
            #             log_file.write(f"    GT Entity: {true_triplet[0]}, GT Relation: {true_triplet[1]}, GT Object: {true_triplet[2]}, GT sent idx: {true_sent_idx}\n")
            #             if check_ent_idx:
            #                 log_file.write(f", GT ent idx: {true_ent_idx}")
            #             log_file.write("\n")
                        
            #             log_file.write(f"    Both sent idx exist: {pred_sent_idx is not None and true_sent_idx is not None}\n")
            #             if check_ent_idx:
            #                 log_file.write(f"    Both ent idx exist: {pred_ent_idx is not None and true_ent_idx is not None}\n")
                        
            #             if pred_sent_idx is not None and true_sent_idx is not None:
            #                 log_file.write(f"    Sent idx match: {pred_sent_idx == true_sent_idx}\n")
            #             if check_ent_idx and pred_ent_idx is not None and true_ent_idx is not None:
            #                 log_file.write(f"    Ent idx match: {pred_ent_idx == true_ent_idx}\n")
                    
            #         #########################################################
            #         # Check if indices match based on configuration
            #         indices_match = True
            #         if pred_sent_idx is not None and true_sent_idx is not None:
            #             if pred_sent_idx != true_sent_idx:
            #                 indices_match = False
                    
            #         if check_ent_idx and pred_ent_idx is not None and true_ent_idx is not None:
            #             if pred_ent_idx != true_ent_idx:
            #                 indices_match = False
                    
            #         # Special check for evidence and associate relations
            #         evidence_associate_match = True
            #         if check_evidence_associate and relation in ['evidence', 'associate']:
            #             pred_last_value = pred_triplet[-1] if len(pred_triplet) > 3 else None
            #             true_last_value = true_triplet[-1] if len(true_triplet) > 3 else None
                        
            #             with open(debug_log_path, "a") as log_file:
            #                 log_file.write(f"    Special check for {relation}: pred_last={pred_last_value}, true_last={true_last_value}\n")
                        
            #             if pred_last_value != true_last_value:
            #                 evidence_associate_match = False

            #         if indices_match and evidence_associate_match:
            #             match_result = fuzzy_match(pred_triplet, true_triplet, relation, jaccard=jaccard)
                        
            #             with open(debug_log_path, "a") as log_file:
            #                 log_file.write(f"    Fuzzy match result: {match_result}\n")
                        
            #             if match_result:
            #                 flag = 1
            #                 if index not in triplet_index:
            #                     with open(debug_log_path, "a") as log_file:
            #                         log_file.write(f"    MATCH FOUND! Adding to right triplets (Mode 1)\n")
            #                     right.append(pred_triplet)
            #                     triplet_index.append(index)
            #                     break
                
            #     if not flag:
            #         with open(debug_log_path, "a") as log_file:
            #             log_file.write(f"    No match found. Adding to wrong triplets (Mode 1)\n")
            #         wrong.append(pred_triplet)
            
            # for pred_idx, pred_triplet in enumerate(pred_triplet_list):
            #     flag = 0
                
            #     with open(debug_log_path, "a") as log_file:
            #         log_file.write(f"MODE 2 - Processing pred triplet {pred_idx}: {pred_triplet}\n")
            #         log_file.write(f"  Entity: {pred_triplet[0]}, Relation: {pred_triplet[1]}, Object: {pred_triplet[2]}\n")
                
            #     for index, true_triplet_group in enumerate(gt_triplet_list):
            #         true_triplet = true_triplet_group[0]
                    
            #         with open(debug_log_path, "a") as log_file:
            #             log_file.write(f"  Comparing with GT triplet group {index}: {true_triplet_group[0]}\n")
            #             log_file.write(f"    GT Entity: {true_triplet[0]}, GT Relation: {true_triplet[1]}, GT Object: {true_triplet[2]}\n")
                    
            #         # Special check for evidence and associate relations in Mode 2
            #         evidence_associate_match = True
            #         if check_evidence_associate and relation in ['evidence', 'associate']:
            #             pred_last_value = pred_triplet[-1] if len(pred_triplet) > 3 else None
            #             true_last_value = true_triplet[-1] if len(true_triplet) > 3 else None
                        
            #             with open(debug_log_path, "a") as log_file:
            #                 log_file.write(f"    Special check for {relation}: pred_last={pred_last_value}, true_last={true_last_value}\n")
                        
            #             if pred_last_value != true_last_value:
            #                 evidence_associate_match = False

            #         if evidence_associate_match:
            #             match_result = fuzzy_match(pred_triplet, true_triplet, relation, jaccard=jaccard)
                        
            #             with open(debug_log_path, "a") as log_file:
            #                 log_file.write(f"    Fuzzy match result: {match_result}\n")
                        
            #             if match_result:
            #                 flag = 1
            #                 if index not in triplet_index_no_sent:
            #                     with open(debug_log_path, "a") as log_file:
            #                         log_file.write(f"    MATCH FOUND! Adding to right_no_sent_idx triplets (Mode 2)\n")
            #                     right_no_sent_idx.append(pred_triplet)
            #                     triplet_index_no_sent.append(index)
            #                     break
                
            #     if not flag:
            #         with open(debug_log_path, "a") as log_file:
            #             log_file.write(f"    No match found. Adding to wrong_no_sent_idx triplets (Mode 2)\n")
            #         wrong_no_sent_idx.append(pred_triplet)

            # MODE 1: Flexible matching without considering sentence index            
            for pred_idx, pred_triplet in enumerate(pred_triplet_list):
                flag = 0
                
                pred_sent_idx = pred_triplet[3] if len(pred_triplet) > 3 else None
                pred_ent_idx = pred_triplet[4] if len(pred_triplet) > 4 else None
                
                with open(debug_log_path, "a") as log_file:
                    log_file.write(f"MODE 1 - Processing pred triplet {pred_idx}: {pred_triplet}\n")
                    log_file.write(f"  Entity: {pred_triplet[0]}, Relation: {pred_triplet[1]}, Object: {pred_triplet[2]}, Pred sent idx: {pred_sent_idx}\n")
                    
                    if check_ent_idx:
                        log_file.write(f", Pred ent idx: {pred_ent_idx}")
                    log_file.write("\n")

                for index, true_triplet_group in enumerate(gt_triplet_list):
                    true_triplet = true_triplet_group[0]                    
                    true_sent_idx = true_triplet[3] if len(true_triplet) > 3 else None
                    true_ent_idx = true_triplet[4] if len(true_triplet) > 4 else None
                    
                    with open(debug_log_path, "a") as log_file:
                        log_file.write(f"  Comparing with GT triplet group {index}: {true_triplet_group[0]}\n")
                        log_file.write(f"    GT Entity: {true_triplet[0]}, GT Relation: {true_triplet[1]}, GT Object: {true_triplet[2]}, GT sent idx: {true_sent_idx}\n")
                        if check_ent_idx:
                            log_file.write(f", GT ent idx: {true_ent_idx}")
                        log_file.write("\n")
                    
                    # Special check for evidence and associate relations
                    evidence_associate_match = True
                    if check_evidence_associate and relation in ['evidence', 'associate']:
                        pred_last_value = pred_triplet[-1] if len(pred_triplet) > 3 else None
                        true_last_value = true_triplet[-1] if len(true_triplet) > 3 else None
                        
                        with open(debug_log_path, "a") as log_file:
                            log_file.write(f"    Special check for {relation}: pred_last={pred_last_value}, true_last={true_last_value}\n")
                        
                        if pred_last_value != true_last_value:
                            evidence_associate_match = False

                    # Flexible matching without considering sentence index
                    if evidence_associate_match:
                        # Use flexible_fuzzy_match, which combines fuzzy_match and enhanced_fuzzy_match
                        match_result = flexible_fuzzy_match(pred_triplet, true_triplet, relation, jaccard=jaccard)
                        
                        with open(debug_log_path, "a") as log_file:
                            log_file.write(f"    Flexible fuzzy matching result: {match_result}\n")
                        
                        if match_result:
                            flag = 1
                            if index not in triplet_index_no_sent:
                                with open(debug_log_path, "a") as log_file:
                                    log_file.write(f"    MATCH FOUND! Adding to right_no_sent_idx triplets (Mode 1)\n")
                                right_no_sent_idx.append(pred_triplet)
                                triplet_index_no_sent.append(index)
                                break
                
                if not flag:
                    with open(debug_log_path, "a") as log_file:
                        log_file.write(f"    No match found. Adding to wrong_no_sent_idx triplets (Mode 1)\n")
                    wrong_no_sent_idx.append(pred_triplet)            
                    
            # MODE 2: Flexible matching only when sentence indices match
            for pred_idx, pred_triplet in enumerate(pred_triplet_list):
                flag = 0
                
                pred_sent_idx = pred_triplet[3] if len(pred_triplet) > 3 else None
                
                with open(debug_log_path, "a") as log_file:
                    log_file.write(f"MODE 2 - Processing pred triplet {pred_idx}: {pred_triplet}\n")
                    log_file.write(f"  Entity: {pred_triplet[0]}, Relation: {pred_triplet[1]}, Object: {pred_triplet[2]}, Pred sent idx: {pred_sent_idx}\n")
                
                for index, true_triplet_group in enumerate(gt_triplet_list):
                    true_triplet = true_triplet_group[0]
                    true_sent_idx = true_triplet[3] if len(true_triplet) > 3 else None
                    
                    with open(debug_log_path, "a") as log_file:
                        log_file.write(f"  Comparing with GT triplet group {index}: {true_triplet_group[0]}\n")
                        log_file.write(f"    GT Entity: {true_triplet[0]}, GT Relation: {true_triplet[1]}, GT Object: {true_triplet[2]}, GT sent idx: {true_sent_idx}\n")
                    
                    # Check if sentence indices match
                    sent_idx_match = True
                    if pred_sent_idx is not None and true_sent_idx is not None:
                        if pred_sent_idx != true_sent_idx:
                            sent_idx_match = False
                            with open(debug_log_path, "a") as log_file:
                                log_file.write(f"    Sentence indices don't match. Skipping.\n")
                            continue
                    
                    # Special check for evidence and associate relations in Mode 2
                    evidence_associate_match = True
                    if check_evidence_associate and relation in ['evidence', 'associate']:
                        pred_last_value = pred_triplet[-1] if len(pred_triplet) > 3 else None
                        true_last_value = true_triplet[-1] if len(true_triplet) > 3 else None
                        
                        with open(debug_log_path, "a") as log_file:
                            log_file.write(f"    Special check for {relation}: pred_last={pred_last_value}, true_last={true_last_value}\n")
                        
                        if pred_last_value != true_last_value:
                            evidence_associate_match = False
                            with open(debug_log_path, "a") as log_file:
                                log_file.write(f"    Evidence/associate values don't match. Skipping.\n")
                            continue

                    # Only proceed with flexible matching if sentence indices match
                    if sent_idx_match and evidence_associate_match:
                        # Flexible matching for Mode 2
                        match_result = flexible_fuzzy_match(pred_triplet, true_triplet, relation, jaccard=jaccard)
                        
                        with open(debug_log_path, "a") as log_file:
                            log_file.write(f"    Flexible fuzzy matching result: {match_result}\n")
                        
                        if match_result:
                            flag = 1
                            if index not in triplet_index:
                                with open(debug_log_path, "a") as log_file:
                                    log_file.write(f"    MATCH FOUND! Adding to right triplets (Mode 2)\n")
                                right.append(pred_triplet)
                                triplet_index.append(index)
                                break
                
                if not flag:
                    with open(debug_log_path, "a") as log_file:
                        log_file.write(f"    No match found. Adding to wrong triplets (Mode 2)\n")
                    wrong.append(pred_triplet)

            with open(debug_log_path, "a") as log_file:
                log_file.write("\n")
        
        miss = []
        for i, s_f_l in enumerate(gt_triplet_list):
            if i not in triplet_index:
                with open(debug_log_path, "a") as log_file:
                    log_file.write(f"Mode 1 - Missing GT triplet group {i}: {s_f_l}\n")
                miss.append(s_f_l)
        
        miss_no_sent_idx = []
        for i, s_f_l in enumerate(gt_triplet_list):
            if i not in triplet_index_no_sent:
                with open(debug_log_path, "a") as log_file:
                    log_file.write(f"Mode 2 - Missing GT triplet group {i}: {s_f_l}\n")
                miss_no_sent_idx.append(s_f_l)

        with open(debug_log_path, "a") as log_file:
            log_file.write("\n--- Results comparison ---\n")
            log_file.write(f"Mode 1 right (TP): {len(right)}\n")
            log_file.write(f"Mode 2 right (TP): {len(right_no_sent_idx)}\n")
            log_file.write(f"Mode 1 wrong (FP): {len(wrong)}\n")
            log_file.write(f"Mode 2 wrong (FP): {len(wrong_no_sent_idx)}\n")
            log_file.write(f"Mode 1 miss (FN): {len(miss)}\n")
            log_file.write(f"Mode 2 miss (FN): {len(miss_no_sent_idx)}\n")
            log_file.write(f"Difference in right: {len(right) != len(right_no_sent_idx)}\n")
            log_file.write(f"Difference in wrong: {len(wrong) != len(wrong_no_sent_idx)}\n")
            log_file.write(f"Difference in miss: {len(miss) != len(miss_no_sent_idx)}\n\n")
            
            if len(right) != len(right_no_sent_idx):
                log_file.write("Detailed right differences:\n")
                mode1_only = [t for t in right if t not in right_no_sent_idx]
                mode2_only = [t for t in right_no_sent_idx if t not in right]
                log_file.write(f"  Mode 1 only: {mode1_only}\n")
                log_file.write(f"  Mode 2 only: {mode2_only}\n\n")
            
            if len(wrong) != len(wrong_no_sent_idx):
                log_file.write("Detailed wrong differences:\n")
                mode1_only = [t for t in wrong if t not in wrong_no_sent_idx]
                mode2_only = [t for t in wrong_no_sent_idx if t not in wrong]
                log_file.write(f"  Mode 1 only: {mode1_only}\n")
                log_file.write(f"  Mode 2 only: {mode2_only}\n\n")

        save["true_triplet_list"] = gt_triplet_list
        save["right_triplet_list"] = right
        save["wrong_triplet_list"] = wrong
        save["miss_triplet_list"] = miss
        save["right_triplet_list_no_sent_idx"] = right_no_sent_idx
        save["wrong_triplet_list_no_sent_idx"] = wrong_no_sent_idx
        save["miss_triplet_list_no_sent_idx"] = miss_no_sent_idx

        with open(f"{save_path}/triplets_predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')
    
    print(f"=== SRO_EVAL completed. Debug log saved to {debug_log_path} ===")
                                                                  
def cal_result_SR_EVAL(file_path, relations=None):
    """
        Calculate the results of each subject
    :param file_path:
    :return:
    """
    datas = []
    for file in os.listdir(file_path):
        if "subject_predict.json" in file:
            datas += [json.loads(line) for line in open(os.path.join(file_path, file)).readlines()]

    json.dump(datas, open(os.path.join(file_path, "all_subject.json"), "w"), indent=4)
    
    entity_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
    entity_result_no_sent_idx = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
    
    for data in datas:        
        for relation in data["right_entities"]:
            entity_result[relation]['tp'] += len(data["right_entities"][relation])
            
        for relation in data["wrong_entities"]:
            entity_result[relation]['fp'] += len(data["wrong_entities"][relation])
            
        if "right_entities_no_sent_idx" in data and "wrong_entities_no_sent_idx" in data:
            for relation in data["right_entities_no_sent_idx"]:
                entity_result_no_sent_idx[relation]['tp'] += len(data["right_entities_no_sent_idx"][relation])
                
            for relation in data["wrong_entities_no_sent_idx"]:
                entity_result_no_sent_idx[relation]['fp'] += len(data["wrong_entities_no_sent_idx"][relation])
        else:
            print(f"Warning: No 'right_entities_no_sent_idx' or 'wrong_entities_no_sent_idx' in data")
            
        for relation in data['relations']:
            relation_count = len([triplets for triplets in data['same_triplet_list'] if triplets[0][1] == relation])
            entity_result[relation]['all'] += relation_count
            entity_result_no_sent_idx[relation]['all'] += relation_count
    
    for relation in RELATIONS:
        if relation not in entity_result:
            continue
        right = entity_result[relation]['tp']
        wrong = entity_result[relation]['fp']
        all_count = entity_result[relation]['all']
        recall = right / all_count if all_count != 0 else 0
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0

        entity_result[relation]['recall'] = recall
        entity_result[relation]['precision'] = precision
        entity_result[relation]['f1'] = f1

    valid_relations = [r for r in RELATIONS if r in entity_result]
    if valid_relations:
        entity_result['macro_avg'] = {
            'recall': sum(entity_result[r]['recall'] for r in valid_relations) / len(valid_relations),
            'precision': sum(entity_result[r]['precision'] for r in valid_relations) / len(valid_relations),
            'f1': sum(entity_result[r]['f1'] for r in valid_relations) / len(valid_relations),
            'num_relations': len(valid_relations)
        }

    for relation in RELATIONS:
        if relation not in entity_result_no_sent_idx:
            continue
        right = entity_result_no_sent_idx[relation]['tp']
        wrong = entity_result_no_sent_idx[relation]['fp']
        all_count = entity_result_no_sent_idx[relation]['all']
        recall = right / all_count if all_count != 0 else 0
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0

        entity_result_no_sent_idx[relation]['recall'] = recall
        entity_result_no_sent_idx[relation]['precision'] = precision
        entity_result_no_sent_idx[relation]['f1'] = f1

    valid_relations_no_sent_idx = [r for r in RELATIONS if r in entity_result_no_sent_idx]
    if valid_relations_no_sent_idx:
        entity_result_no_sent_idx['macro_avg'] = {
            'recall': sum(entity_result_no_sent_idx[r]['recall'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'precision': sum(entity_result_no_sent_idx[r]['precision'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'f1': sum(entity_result_no_sent_idx[r]['f1'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'num_relations': len(valid_relations_no_sent_idx)
        }

    json.dump(entity_result, open(os.path.join(file_path, "SR_result.json"), "w"), indent=4)
    json.dump(entity_result_no_sent_idx, open(os.path.join(file_path, "SR_result_no_sent_idx.json"), "w"), indent=4)

def cal_result_SRO_EVAL(file_path, relations=None):
    """
        Calculate the results of triplets
    :param file_path:
    :return:
    """
    true_relation_count = {}
    true_relation_count_no_sent_idx = {}
    
    datas = [json.loads(line) for line in open(os.path.join(file_path, "triplets_predict.json"), encoding='utf-8').readlines()]
    json.dump(datas, open(os.path.join(file_path, "all.json"), "w"), indent=4)
    
    relations_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0, "precision": 0, "f1": 0})
    relations_result_no_sent_idx = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0, "precision": 0, "f1": 0})
    
    for data in datas:
        if "right_triplet_list" in data:
            for triplet in data["right_triplet_list"]:
                relations_result[triplet[1]]['tp'] += 1
                
        if "wrong_triplet_list" in data:
            for triplet in data["wrong_triplet_list"]:
                relations_result[triplet[1]]['fp'] += 1
                
        if "miss_triplet_list" in data:
            for triplet in data["miss_triplet_list"]:
                relations_result[triplet[0][1]]['miss'] += 1
        
        if "right_triplet_list_no_sent_idx" in data:
            for triplet in data["right_triplet_list_no_sent_idx"]:
                relations_result_no_sent_idx[triplet[1]]['tp'] += 1
                
        if "wrong_triplet_list_no_sent_idx" in data:
            for triplet in data["wrong_triplet_list_no_sent_idx"]:
                relations_result_no_sent_idx[triplet[1]]['fp'] += 1
                
        if "miss_triplet_list_no_sent_idx" in data:
            for triplet in data["miss_triplet_list_no_sent_idx"]:
                relations_result_no_sent_idx[triplet[0][1]]['miss'] += 1

    for relation in RELATIONS:
        if relation not in relations_result:
            continue
        true_relation_count[relation] = relations_result[relation]['tp'] + relations_result[relation]['miss']
        relations_result[relation]['all'] = true_relation_count[relation]

        right = relations_result[relation]['tp']
        wrong = relations_result[relation]['fp']
        miss = relations_result[relation]['miss']
        all_count = true_relation_count[relation]

        recall = right / all_count if all_count != 0 else 0
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0

        relations_result[relation]['recall'] = recall
        relations_result[relation]['precision'] = precision
        relations_result[relation]['f1'] = f1

    valid_relations = [r for r in RELATIONS if r in relations_result]
    if valid_relations:
        relations_result['macro_avg'] = {
            'recall': sum(relations_result[r]['recall'] for r in valid_relations) / len(valid_relations),
            'precision': sum(relations_result[r]['precision'] for r in valid_relations) / len(valid_relations),
            'f1': sum(relations_result[r]['f1'] for r in valid_relations) / len(valid_relations),
            'num_relations': len(valid_relations)
        }

    for relation in RELATIONS:
        if relation not in relations_result_no_sent_idx:
            continue
        true_relation_count_no_sent_idx[relation] = relations_result_no_sent_idx[relation]['tp'] + relations_result_no_sent_idx[relation]['miss']
        relations_result_no_sent_idx[relation]['all'] = true_relation_count_no_sent_idx[relation]

        right = relations_result_no_sent_idx[relation]['tp']
        wrong = relations_result_no_sent_idx[relation]['fp']
        miss = relations_result_no_sent_idx[relation]['miss']
        all_count = true_relation_count_no_sent_idx[relation]

        recall = right / all_count if all_count != 0 else 0
        precision = right / (right + wrong) if (right + wrong) != 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0

        relations_result_no_sent_idx[relation]['recall'] = recall
        relations_result_no_sent_idx[relation]['precision'] = precision
        relations_result_no_sent_idx[relation]['f1'] = f1

    valid_relations_no_sent_idx = [r for r in RELATIONS if r in relations_result_no_sent_idx]
    if valid_relations_no_sent_idx:
        relations_result_no_sent_idx['macro_avg'] = {
            'recall': sum(relations_result_no_sent_idx[r]['recall'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'precision': sum(relations_result_no_sent_idx[r]['precision'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'f1': sum(relations_result_no_sent_idx[r]['f1'] for r in valid_relations_no_sent_idx) / len(valid_relations_no_sent_idx),
            'num_relations': len(valid_relations_no_sent_idx)
        }

    json.dump(relations_result, open(os.path.join(file_path, "SRO_result.json"), "w"), ensure_ascii=False, indent=4)
    json.dump(relations_result_no_sent_idx, open(os.path.join(file_path, "SRO_result_no_sent_idx.json"), "w"), ensure_ascii=False, indent=4)

def get_triplet_from_gt(gpt_pred_data, data):
    gt_triplet_list = {}
    last_gt_triplet = None
    
    for custom_id in gpt_pred_data.keys():
        # Check if the section is 'gt_report' and collect those custom_ids
        if data[custom_id].get('section') == 'gt_report':
            # Create a list of lists where each inner list contains triplets
            triplet_groups = []
            for relation, triplets in gpt_pred_data[custom_id].items():
                for triplet in triplets:
                    triplet_groups.append([triplet])
            
            gt_triplet_list[custom_id] = triplet_groups
            last_gt_triplet = triplet_groups
        else:
            # Use the last seen gt_report triplet if available, otherwise use current triplet
            gt_triplet_list[custom_id] = last_gt_triplet if last_gt_triplet is not None else []
            if last_gt_triplet is None:
                # If no gt_report was seen yet, create triplet groups from current data
                triplet_groups = []
                for relation, triplets in gpt_pred_data[custom_id].items():
                    for triplet in triplets:
                        triplet_groups.append([triplet])
                gt_triplet_list[custom_id] = triplet_groups

    return gt_triplet_list

def Gen_report_SR_EVAL(data_path=None, gpt_pred_path=None, save_path=None, relation_to_evaluate=None):
    if os.path.exists(f"{save_path}/subject_predict.json"):
        with open(f"{save_path}/subject_predict.json", "w") as file:
            pass
    
    # if data_path is not None:
    with open(data_path, 'r') as file:
        data = json.load(file)

    with open(gpt_pred_path, 'r') as file:
        gpt_pred_data = json.load(file)

    if data['0']['same_triplet_list'] is None:
        all_gt_triplet_list = get_triplet_from_gt(gpt_pred_data, data)
    else:
        all_gt_triplet_list = None
    
    for custom_id in gpt_pred_data.keys():
        # if data[custom_id]['section'] == 'gt_report':
        #     continue
        
        report_type = data[custom_id]['section']
        sentences = data[custom_id]['passage']
        gpt_pred = gpt_pred_data[custom_id]
        right = {}
        wrong = {}
        pred_triplet = {}
        relations_to_check = relation_to_evaluate if relation_to_evaluate else data[custom_id]['relations']

        for relation in relations_to_check:
            right[relation] = []
            wrong[relation] = []
            pred_triplet_list = gpt_pred.get(relation, [])
            pred_entity_list = [triplet[0] for triplet in pred_triplet_list]
            pred_val = [triplet[2] for triplet in pred_triplet_list]
            gt_triplet_list = data[custom_id]['same_triplet_list'] if all_gt_triplet_list is None else all_gt_triplet_list[custom_id]

            pred_triplet[relation] = pred_triplet_list
            
            triplet_index = []
            for entity in pred_entity_list:
                flag = 0
                subject_triplets = [triplets for triplets in gt_triplet_list if triplets[0][1] == relation]
                
                for index, true_triplet in enumerate(subject_triplets):
                    true_entities = true_triplet[0][0]
                    if fuzz.ratio(entity, true_entities) > 70:
                        flag = 1
                        if index not in triplet_index:
                            right[relation].append(entity)
                            triplet_index.append(index)
                if not flag:
                    wrong[relation].append(entity)
        save = {
            'custom_id': custom_id,
            "report_type": report_type,
            "pred_triplet": pred_triplet,
            "data_from": data[custom_id]['data_from'],
            "sentences": sentences,
            "relations": data[custom_id]['relations'],
            "right_entities": right,
            "wrong_entities": wrong,
            "same_triplet_list": gt_triplet_list
        }
        # Duplicates may be generated here.
        with open(f"{save_path}/subject_predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')

def Gen_report_SRO_EVAL(data_path=None, gpt_pred_path=None, save_path=None, relation_to_evaluate=None, cat_to_evaluate=['pf', 'cf', 'oth', 'patient info.', 'ncd', 'cof', 'ncd/cof'], jaccard=True):
    if os.path.exists(f"{save_path}/triplets_predict.json"):
        with open(f"{save_path}/triplets_predict.json", "w") as file:
            pass
        
    with open(data_path, 'r') as file:
        data = json.load(file)

    with open(gpt_pred_path, 'r') as file:
        gpt_pred_data = json.load(file)

    if data['0']['same_triplet_list'] is None:
        all_gt_triplet_list = get_triplet_from_gt(gpt_pred_data, data)
    else:
        all_gt_triplet_list = None
    
    for custom_id in gpt_pred_data.keys():
        # if data[custom_id]['section'] == 'gt_report':
        #     continue

        report_type = data[custom_id]['section']
        wrong = []
        right = []
        triplet_index = []
        sentence = data[custom_id]['passage']
        gpt_pred = gpt_pred_data[custom_id]
        #print("sentence: ", sentence)
        save = {
            'custom_id': custom_id,
            "report_type": report_type,
            "data_from": data[custom_id]['data_from'],
            "sentence": sentence,
        }

        relations_to_check = relation_to_evaluate if relation_to_evaluate else data[custom_id]['relations']
        
        gt_triplet_list = data[custom_id]['same_triplet_list'] if all_gt_triplet_list is None else all_gt_triplet_list[custom_id]
        
        # Filter ground truth triplets based on relations_to_check
        filtered_gt_triplet_list = []
        for triplets in gt_triplet_list:
            filtered_triplets = []
            for triplet in triplets:
                if triplet[1] in relations_to_check:
                    if triplet[1] == 'cat':
                        if triplet[2] not in cat_to_evaluate:
                            continue
                    filtered_triplets.append(triplet)
            if filtered_triplets:
                filtered_gt_triplet_list.append(filtered_triplets)

        for relation in relations_to_check:
            save[relation] = {}
   
            pred_triplet_list = gpt_pred.get(relation, [])
            
            for triplet in pred_triplet_list:
                flag = 0
                ######### Fuzzy matching with sentence index
                # for index, true_triplet in enumerate(filtered_gt_triplet_list):
                #     if fuzzy_match(triplet, true_triplet[0], relation, jaccard=jaccard):
                #         flag = 1
                #         if index not in triplet_index:
                #             right.append(triplet)
                #             triplet_index.append(index)
                # if not flag:
                #     wrong.append(triplet)
        
                # Flexible matching that doesn't care about subject/object order
                flag = 0
                for index, true_triplet in enumerate(filtered_gt_triplet_list):
                    if flexible_fuzzy_match(triplet, true_triplet[0], relation, jaccard=jaccard):
                        flag = 1
                        if index not in triplet_index:
                            right.append(triplet)
                            triplet_index.append(index)
                            break
                if not flag:
                    wrong.append(triplet)
        
        miss = [s_f_l for i, s_f_l in enumerate(filtered_gt_triplet_list) if i not in triplet_index]

        save["true_triplet_list"] = filtered_gt_triplet_list  # Full GT
        save["right_triplet_list"] = right  # TP
        save["wrong_triplet_list"] = wrong # GPT
        save["miss_triplet_list"] = miss # GT FN

        with open(f"{save_path}/triplets_predict.json", "a") as file:
            json.dump(save, file)
            file.write('\n')

def cal_result_gen_report_SR_EVAL(file_path, relations=None):
    """
        Calculate the results of each subject
    :param file_path:
    :return:
    """
    
    datas = []
    for file in os.listdir(file_path):
        if "subject_predict.json" in file:
            datas += [json.loads(line) for line in open(os.path.join(file_path, file)).readlines()]

    json.dump(datas, open(os.path.join(file_path, "all_subject.json"), "w"), indent=4)

    # Group data by report_type
    model_data = {}
    for data in datas:
        report_type = data.get("report_type", "unknown")
        if report_type not in model_data:
            model_data[report_type] = []
        model_data[report_type].append(data)

    model_results = {}
    for model_name, model_datas in model_data.items():
        model_entity_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0.0, "precision": 0.0, "f1": 0.0})
        
        for data in model_datas:
            for relation in data["right_entities"]:
                model_entity_result[relation]['tp'] += len(data["right_entities"][relation])
            for relation in data["wrong_entities"]:
                model_entity_result[relation]['fp'] += len(data["wrong_entities"][relation])
            for relation in data['relations']:
                model_entity_result[relation]['all'] += len([triplets for triplets in data['same_triplet_list'] if triplets[0][1] == relation])
        
        # Calculate metrics for this model
        for relation in RELATIONS:
            if relation not in model_entity_result:
                continue
            right = model_entity_result[relation]['tp']
            wrong = model_entity_result[relation]['fp'] 
            recall = right / model_entity_result[relation]['all'] if model_entity_result[relation]['all'] != 0 else 0
            precision = right / (right + wrong) if (right + wrong) != 0 else 0
            model_entity_result[relation]['recall'] = recall
            model_entity_result[relation]['precision'] = precision
            model_entity_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
        
        model_results[model_name] = model_entity_result
    
    # Save per-model results
    json.dump(model_results, open(os.path.join(file_path, "SR_result_by_model.json"), "w"), indent=4)

def cal_result_gen_report_SRO_EVAL(file_path, relations=None):
    """
        Calculate the results of triplets
    :param file_path:
    :return:
    """
    true_relation_count = {}
    datas = [json.loads(line) for line in open(os.path.join(file_path, "triplets_predict.json"), encoding='utf-8').readlines()]
    json.dump(datas, open(os.path.join(file_path, "all.json"), "w"), indent=4)

    # Group data by report_type
    model_data = {}
    for data in datas:
        report_type = data.get("report_type", "unknown")
        if report_type not in model_data:
            model_data[report_type] = []
        model_data[report_type].append(data)

    # Calculate results by model
    model_results = {}
    for model_name, model_datas in model_data.items():
        model_true_relation_count = {}
        model_relations_result = defaultdict(lambda: {'all': 0, 'tp': 0, 'fp': 0, "miss": 0, "recall": 0, "precision": 0, "f1": 0})
        
        for data in model_datas:
            if "right_triplet_list" in data:
                for triplet in data["right_triplet_list"]:
                    model_relations_result[triplet[1]]['tp'] += 1
            if "wrong_triplet_list" in data:
                for triplet in data["wrong_triplet_list"]:
                    model_relations_result[triplet[1]]['fp'] += 1
            if "miss_triplet_list" in data:
                for triplet in data["miss_triplet_list"]:
                    model_relations_result[triplet[0][1]]['miss'] += 1
        
        for relation in RELATIONS:
            if relation not in model_relations_result:
                continue
            model_true_relation_count[relation] = model_relations_result[relation]['tp'] + model_relations_result[relation]['miss']
            model_relations_result[relation]['all'] = model_true_relation_count[relation]

            right = model_relations_result[relation]['tp']
            wrong = model_relations_result[relation]['fp']
            recall = right / model_true_relation_count[relation] if model_true_relation_count[relation] != 0 else 0
            precision = right / (right + wrong) if (right + wrong) != 0 else 0
            model_relations_result[relation]['recall'] = recall
            model_relations_result[relation]['precision'] = precision
            model_relations_result[relation]['f1'] = 2 * recall * precision / (recall + precision) if recall != 0 and precision != 0 else 0
        
        model_results[model_name] = model_relations_result
    
    # Save per-model results
    json.dump(model_results, open(os.path.join(file_path, "SRO_result_by_model.json"), "w"), ensure_ascii=False, indent=4)


def include(triplet1, triplet2):  # Does not require exact match on the other side; uses fuzzy matching
    if triplet1[1] in ['cat', 'dx_status', 'dx_certainty']:
        if triplet1[0] in triplet2[0] and triplet1[2] == triplet2[2]:
            return True
    elif triplet1[0] in triplet2[0] and fuzz.ratio(triplet1[2], triplet2[2]) > fuzzy_threshold:
        return True
    elif triplet1[2] in triplet2[2] and fuzz.ratio(triplet1[0], triplet2[0]) > fuzzy_threshold:
        return True
    return False

def remove_same_triplet(right_triplet_list, miss_triplet_list):
    # Remove duplicates
    for right in right_triplet_list:
        for miss in miss_triplet_list:
            if right[1] == miss[0][1]:
                if include(miss[0], right) or include(right, miss[0]):
                    miss_triplet_list.remove(miss)
                
    return right_triplet_list, miss_triplet_list
