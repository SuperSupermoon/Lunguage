import os
import sys

import numpy as np
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch.distributed as dist


def compute_largest_cluster(sentences):
    if len(sentences) == 0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]
    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster


def compute_kmeans(sentences):
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences)
    embeddings = preprocessing.normalize(embeddings)
    kmeans = binary_search_optimal_kmeans(
        embeddings, min_k=0, max_k=(len(sentences) - 1)
    )
    return embeddings, kmeans


def binary_search_optimal_kmeans(data, min_k, max_k):
    best_k = min_k
    best_score = -1
    best_kmeans = KMeans(n_clusters=1, random_state=42).fit(data)

    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:
            break

        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)

        if score > best_score:
            best_score = score
            best_k = mid_k
            best_kmeans = kmeans
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    return best_kmeans


def flatten_values_lists_of_list_dicts_to_dict(item):
    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)

    return result


def gather_processes(local_candidates, local_references=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("RANK", "0"))
    global_candidates_list = None
    global_references_list = None

    if local_rank == 0:
        global_candidates_list = [None for _ in range(world_size)]
        global_references_list = [None for _ in range(world_size)]
    try:
        dist.gather_object(local_candidates, global_candidates_list, dst=0)

        if local_references is not None:
            dist.gather_object(local_references, global_references_list, dst=0)

    except Exception as e:
        print(f"Error during result gathering: {e}")

    if local_rank != 0:
        dist.destroy_process_group()
        sys.exit()

    candidates_list = []
    for i in global_candidates_list:
        candidates_list.extend(i)

    if global_references_list[0] is not None:
        references_list = []
        for i in global_references_list:
            references_list.extend(i)
        print(f"References list: {len(references_list)}")
        return candidates_list, references_list

    return candidates_list


def clean_responses(response):
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if ("[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>") in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len=300):
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"""Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

Process Overview: You will be presented with:

1. The criteria for making a judgment.
2. The reference radiology report.
3. The candidate radiology report.
4. The desired format for your assessment.

1. Criteria for Judgment:

For each candidate report, determine:

The count of clinically significant errors.
The count of clinically insignificant errors.

Errors can fall into one of these categories:

a) False report of a finding in the candidate.
b) Missing a finding present in the reference.
c) Misidentification of a finding's anatomic location/position.
d) Misassessment of the severity of a finding.
e) Mentioning a comparison that isn't in the reference.
f) Omitting a comparison detailing a change from a prior study.
Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.

2. Reference Report:
{text1}

3. Candidate Report:
{text2}

4. Reporting Your Assessment:

Follow this specific format for your output, even if no errors are found:
```
[Explanation]:
<Explanation>

[Clinically Significant Errors]:
(a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
....
(f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

[Clinically Insignificant Errors]:
(a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
....
(f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

[Matched Findings]:
<The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
```
"""
    return prompt


__all__ = [
    "compute_largest_cluster",
    "compute_kmeans",
    "binary_search_optimal_kmeans",
    "flatten_values_lists_of_list_dicts_to_dict",
    "gather_processes",
    "clean_responses",
    "make_prompt",
]





