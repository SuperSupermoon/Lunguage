from .template import templates

def create_sys_msg(args):
    if args.candidate_type == 'gt_sro_review':
        return create_sys_msg_sro_review(args)
    elif args.candidate_type == 'gt_sro':
        return create_sys_msg_sro(args)
    elif (args.candidate_type == 'gt_ent_rcg') or (args.candidate_type == 'vocab_ent_rcg'):
        if args.multi:
            return create_sys_msg_ent_rcg2(args)
        else:
            return create_sys_msg_ent_rcg(args)
    elif args.candidate_type == 'no_candidates':
        return create_sys_msg_no_candidates(args)
    else:
        return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
            "candidates": [
                "pneumonia"
            ]
        },
        {
            "sent_idx": 2,
            "sentence": "A new small pleural effusion is seen on the left side.",
            "candidates": [
                "pleural effusion"
            ]
        }
    ]
}

Allowed relations (exact names):
Cat, Status, Location, Placement, Associate, Evidence,
Morphology, Distribution, Measurement, Severity, Comparison,
Onset, No Change, Improved, Worsened, Past Hx, Other Source, Assessment Limitations

Allowed Cat values:
RF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• One of: RF, OTH, COF, NCD, PATIENT INFO.

Dx_Status
• Exactly one per entity.
• Positive if stated with certainty (“definite,” clear presence/absence).
• Negative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Dx_Certainty
• Exactly one per entity.
• Definitive if stated with certainty (“definite,” clear presence/absence).
• Tentative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Location
• Only for anatomic sites of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”).

Associate / Evidence
• Only when an explicit link is stated within or across sentences.
• Include obj_ent_idx pointing to the related entity.

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat and one Status.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. When labeling entity names, refer back to the input candidates to guide spelling and casing—ensure consistency with those terms.
6. Leverage the candidate entity names as much as possible.
7. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
8. Location captures only the anatomic site of that entity.
9. Placement applies only to devices.
10. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
11. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
12. Extract only relations supported by explicit or inferable text evidence.
13. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def create_sys_msg_sro_review(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
            "candidates": [
                [
                    "pneumonia",
                    "cat",
                    "rf"
                ],
                [
                    "pneumonia",
                    "dx_status",
                    "positive"
                ],
                [
                    "pneumonia",
                    "dx_certainty",
                    "tentative"
                ],
                [
                    "pneumonia",
                    "location",
                    "right lower lobe"
                ],
                [
                    "pneumonia",
                    "evidence",
                    "findings"
                ],
                [
                    "findings",
                    "cat",
                    "rf"
                ],
                [
                    "findings",
                    "dx_status",
                    "positive"
                ],
                [
                    "findings",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "findings",
                    "associate",
                    "pneumonia"
                ],
                [
                    "opacity",
                    "cat",
                    "rf"
                ],
                [
                    "opacity",
                    "dx_status",
                    "positive"
                ],
                [
                    "opacity",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "opacity",
                    "location",
                    "right lower lobe"
                ]
            ]
        },
        {
            "sent_idx": 2,
            "sentence": "A new small pleural effusion is seen on the left side.",
            "candidates": [
                [
                    "pleural effusion",
                    "cat",
                    "rf"
                ],
                [
                    "pleural effusion",
                    "dx_status",
                    "positive"
                ],
                [
                    "pleural effusion",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "pleural effusion",
                    "location",
                    "left side"
                ],
                [
                    "pleural effusion",
                    "onset",
                    "new"
                ],
                [
                    "pleural effusion",
                    "measurement",
                    "small"
                ] 
            ]
        }
    ]
}

Allowed relations (exact names):
Cat, Status, Location, Placement, Associate, Evidence,
Morphology, Distribution, Measurement, Severity, Comparison,
Onset, No Change, Improved, Worsened, Past Hx, Other Source, Assessment Limitations

Allowed Cat values:
RF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• One of: RF, OTH, COF, NCD, PATIENT INFO.

Dx_Status
• Exactly one per entity.
• Positive if stated with certainty (“definite,” clear presence/absence).
• Negative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Dx_Certainty
• Exactly one per entity.
• Definitive if stated with certainty (“definite,” clear presence/absence).
• Tentative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Location
• Only for anatomic sites of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”).

Associate / Evidence
• Only when an explicit link is stated within or across sentences.
• Include obj_ent_idx pointing to the related entity.

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat and one Status.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. When labeling entity names, refer back to the input candidates to guide spelling and casing—ensure consistency with those terms.
6. Review the candidate triplets(subject, relation, object). Keep correct triplets as-is. Fix incorrect ones. Add any valid triplets not included in the candidates.
7. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
8. Location captures only the anatomic site of that entity.
9. Placement applies only to devices.
10. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
11. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
12. Extract only relations supported by explicit or inferable text evidence.
13. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def create_sys_msg_sro(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
            "candidates": [
                [
                    "pneumonia",
                    "cat",
                    "rf"
                ],
                [
                    "pneumonia",
                    "dx_status",
                    "positive"
                ],
                [
                    "pneumonia",
                    "dx_certainty",
                    "tentative"
                ],
                [
                    "pneumonia",
                    "location",
                    "right lower lobe"
                ],
                [
                    "pneumonia",
                    "evidence",
                    "findings"
                ],
                [
                    "findings",
                    "cat",
                    "rf"
                ],
                [
                    "findings",
                    "dx_status",
                    "positive"
                ],
                [
                    "findings",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "findings",
                    "associate",
                    "pneumonia"
                ],
                [
                    "opacity",
                    "cat",
                    "rf"
                ],
                [
                    "opacity",
                    "dx_status",
                    "positive"
                ],
                [
                    "opacity",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "opacity",
                    "location",
                    "right lower lobe"
                ]
            ]
        },
        {
            "sent_idx": 2,
            "sentence": "A new small pleural effusion is seen on the left side.",
            "candidates": [
                [
                    "pleural effusion",
                    "cat",
                    "rf"
                ],
                [
                    "pleural effusion",
                    "dx_status",
                    "positive"
                ],
                [
                    "pleural effusion",
                    "dx_certainty",
                    "definitive"
                ],
                [
                    "pleural effusion",
                    "location",
                    "left side"
                ],
                [
                    "pleural effusion",
                    "onset",
                    "new"
                ],
                [
                    "pleural effusion",
                    "measurement",
                    "small"
                ] 
            ]
        }
    ]
}

Allowed relations (exact names):
Cat, Status, Location, Placement, Associate, Evidence,
Morphology, Distribution, Measurement, Severity, Comparison,
Onset, No Change, Improved, Worsened, Past Hx, Other Source, Assessment Limitations

Allowed Cat values:
RF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• One of: RF, OTH, COF, NCD, PATIENT INFO.

Dx_Status
• Exactly one per entity.
• Positive if stated with certainty (“definite,” clear presence/absence).
• Negative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Dx_Certainty
• Exactly one per entity.
• Definitive if stated with certainty (“definite,” clear presence/absence).
• Tentative if hedged or uncertain (“possible,” “suggests,” “cannot exclude”).

Location
• Only for anatomic sites of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”).

Associate / Evidence
• Only when an explicit link is stated within or across sentences.
• Include obj_ent_idx pointing to the related entity.

Morphology / Distribution / Measurement / Severity / Comparison / Onset / No Change / Improved / Worsened / Past Hx / Other Source / Assessment Limitations
• Use only the exact relation names and values as defined.

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat and one Status.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. When labeling entity names, refer back to the input candidates to guide spelling and casing—ensure consistency with those terms.
6. Leverage the candidate triplets(subject, relation, object) as much as possible. 
7. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
8. Location captures only the anatomic site of that entity.
9. Placement applies only to devices.
10. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
11. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
12. Extract only relations supported by explicit or inferable text evidence.
13. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def create_sys_msg_ent_rcg(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
            "candidates": [
                [
                    "pneumonia",
                    [
                        "entity",
                    ]
                ],
                [
                    "findings",
                    [
                        "entity",
                    ]
                ],
                [
                    "right lower lobe",
                    [
                        "location"
                    ]
                ],
                [
                    "opacity",
                    [
                        "entity"
                    ]
                ]
            ]
        },
        {
            "sent_idx": 2,
            "sentence": "A new small pleural effusion is seen on the left side.",
            "candidates": [
                [
                    "pleural effusion",
                    [
                        "entity"
                    ]
                ],
                [
                    "left side",
                    [
                        "location"
                    ]
                ],
                [
                    "new",
                    [
                        "onset"
                    ]
                ],
                [
                    "small",
                    [
                        "measurement"
                    ]
                ]
            ]
        }
    ]
}

Allowed relations (exact names):
Cat, Status, Location, Placement, Associate, Evidence,
Morphology, Distribution, Measurement, Severity, Comparison,
Onset, No Change, Improved, Worsened, Past Hx, Other Source, Assessment Limitations

Allowed Cat values:
PF, CF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• PF (Perceptual Findings): Findings that can be observed in chest X-ray images.
• CF (Contextual Findings): Contextual diagnoses that require physician interpretation based on external information such as the patient's history, lab findings, or prior studies.
• OTH: Other Objects: Any medical devices observed in chest X-ray images or procedures and surgeries, as well as treatment and medications.
• COF (Clinical Observations): Evidence-based objective medical observation without clinician's subjective opinion which are often obtained through lab tests, physical exam, and other diagnostic procedures that are not based on chest x-ray imaging.
• NCD (Non-CXR Diagnosis): Contextual diagnosis given COF, which are often obtained from other sources like CT or lab tests, but are difficult to determine from chest X-ray images.
• PATIENT INFO. (Patient Information): Subjective indication of a disease or a change in condition as perceived by the patient. This is based on personal experiences and feelings, and they are not directly measurable.

Dx_Status
• Indicates whether a clinical entity is present or absent in the radiograph.
• Positive:
    • The entity is explicitly stated to be present, or there is a described change (e.g., improved, worsened, unchanged), implying that the entity currently exists or is progressing.
    • Examples: "Pleural effusion has improved", "Consolidation is stable", "No change in pneumothorax"
• Negative:
    • The entity is explicitly stated to be absent or removed.
    • Examples: "No pleural effusion", "ET tube has been removed", "No lines seen"

Dx_Certainty
• Indicates the level of certainty expressed regarding the presence or absence of a clinical entity.
• Definitive:
    • The statement conveys a clear and confident assertion about the presence or absence of an entity.
    • Examples: "No pneumothorax", "Definite consolidation", "Heart size is stable"
• Tentative:
    • The statement conveys uncertainty, possibility, or a lack of definitive conclusion.
    • Examples: "Possible pneumonia", "Suggests effusion", "Cannot exclude pneumothorax"

Location
• Only for position of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”, “stable position”).

Associate
• This relation identifies explicit, non-causal links between medical findings, conditions, or objects within or across sentences.
• Normally bidirectional when used alone (i.e., (A -Associate→ B) and (B -Associate→ A)). If Evidence is also present between the entities, then Associate is unidirectional, from finding → diagnosis.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → (dual-lead pacemaker/ICD device -Associate-> leads), (leads -Associate-> dual-lead pacemaker/ICD device)

Evidence
• This relation is used when one entity serves as evidence for another. The subject must be a diagnosis or suspected condition, while the object must be a radiological finding or observable support.
• Evidence is always unidirectional: from diagnosis → finding. When Evidence is annotated, a single corresponding Associate relation must also be included.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. lung opacity might be pulmonary edema → (pulmonary edema -Evidence-> lung opacity), (lung opacity -Associate-> pulmonary edema)

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent", "intact")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior", "recent")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat, one Dx_Status, and one Dx_Certainty.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. Each item in the input candidates represents a candidate span from the input text, along with its category.
6. When labeling entity names, refer back to the input candidates to guide spelling and casing—ensure consistency with those terms.
7. Leverage the candidate tuples(text span, category list) as much as possible. 
8. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
9. Placement applies only to devices.
10. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
11. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
12. Extract only relations supported by explicit or inferable text evidence.
13. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
14. Use the provided candidates directly to construct subject, relation, and object triplets.
15. Extract any additional entities or relations that are present in the text but not included in the candidates.
"""

def create_sys_msg_ent_rcg2(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
            "candidates": [
                [
                    "pneumonia",
                    [
                        "cf",
                    ]
                ],
                [
                    "findings",
                    [
                        "pf",
                    ]
                ],
                [
                    "right lower lobe",
                    [
                        "location"
                    ]
                ],
                [
                    "opacity",
                    [
                        "pf"
                    ]
                ]
            ]
        }
    ]
}

Allowed relations (exact names):
Cat, Status, Location, Placement, Associate, Evidence,
Morphology, Distribution, Measurement, Severity, Comparison,
Onset, No Change, Improved, Worsened, Past Hx, Other Source, Assessment Limitations

Allowed Cat values:
PF, CF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• PF (Perceptual Findings): Findings that can be observed in chest X-ray images.
• CF (Contextual Findings): Contextual diagnoses that require physician interpretation based on external information such as the patient's history, lab findings, or prior studies.
• OTH: Other Objects: Any medical devices observed in chest X-ray images or procedures and surgeries, as well as treatment and medications.
• COF (Clinical Observations): Evidence-based objective medical observation without clinician's subjective opinion which are often obtained through lab tests, physical exam, and other diagnostic procedures that are not based on chest x-ray imaging.
• NCD (Non-CXR Diagnosis): Contextual diagnosis given COF, which are often obtained from other sources like CT or lab tests, but are difficult to determine from chest X-ray images.
• PATIENT INFO. (Patient Information): Subjective indication of a disease or a change in condition as perceived by the patient. This is based on personal experiences and feelings, and they are not directly measurable.

Dx_Status
• Indicates whether a clinical entity is present or absent in the radiograph.
• Positive:
    • The entity is explicitly stated to be present, or there is a described change (e.g., improved, worsened, unchanged), implying that the entity currently exists or is progressing.
    • Examples: "Pleural effusion has improved", "Consolidation is stable", "No change in pneumothorax"
• Negative:
    • The entity is explicitly stated to be absent or removed.
    • Examples: "No pleural effusion", "ET tube has been removed", "No lines seen"

Dx_Certainty
• Indicates the level of certainty expressed regarding the presence or absence of a clinical entity.
• Definitive:
    • The statement conveys a clear and confident assertion about the presence or absence of an entity.
    • Examples: "No pneumothorax", "Definite consolidation", "Heart size is stable"
• Tentative:
    • The statement conveys uncertainty, possibility, or a lack of definitive conclusion.
    • Examples: "Possible pneumonia", "Suggests effusion", "Cannot exclude pneumothorax"

Location
• Only for position of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”, “stable position”).

Associate
• This relation identifies explicit, non-causal links between medical findings, conditions, or objects within or across sentences.
• Normally bidirectional when used alone (i.e., (A -Associate→ B) and (B -Associate→ A)). If Evidence is also present between the entities, then Associate is unidirectional, from finding → diagnosis.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → (dual-lead pacemaker/ICD device -Associate-> leads), (leads -Associate-> dual-lead pacemaker/ICD device)

Evidence
• This relation is used when one entity serves as evidence for another. The subject must be a diagnosis or suspected condition, while the object must be a radiological finding or observable support.
• Evidence is always unidirectional: from diagnosis → finding. When Evidence is annotated, a single corresponding Associate relation must also be included.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. lung opacity might be pulmonary edema → (pulmonary edema -Evidence-> lung opacity), (lung opacity -Associate-> pulmonary edema)

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent", "intact")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior", "recent")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat, one Dx_Status, and one Dx_Certainty.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. Each item in the input candidates represents a candidate span from the input text, along with its category.
6. When labeling entity names, refer back to the input candidates to guide spelling and casing—ensure consistency with those terms.
7. Leverage the candidate tuples(text span, category list) as much as possible. 
8. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
9. Placement applies only to devices.
10. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
11. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
12. Extract only relations supported by explicit or inferable text evidence.
13. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def create_sys_msg_no_candidates(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "report_section": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity."
        },
        {
            "sent_idx": 2,
            "sentence": "A new small pleural effusion is seen on the left side."
        }
    ]
}

Allowed Cat values:
PF, CF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• PF (Perceptual Findings): Findings that can be observed in chest X-ray images.
• CF (Contextual Findings): Contextual diagnoses that require physician interpretation based on external information such as the patient's history, lab findings, or prior studies.
• OTH: Other Objects: Any medical devices observed in chest X-ray images or procedures and surgeries, as well as treatment and medications.
• COF (Clinical Observations): Evidence-based objective medical observation without clinician's subjective opinion which are often obtained through lab tests, physical exam, and other diagnostic procedures that are not based on chest x-ray imaging.
• NCD (Non-CXR Diagnosis): Contextual diagnosis given COF, which are often obtained from other sources like CT or lab tests, but are difficult to determine from chest X-ray images.
• PATIENT INFO. (Patient Information): Subjective indication of a disease or a change in condition as perceived by the patient. This is based on personal experiences and feelings, and they are not directly measurable.

Dx_Status
• Indicates whether a clinical entity is present or absent in the radiograph.
• Positive:
    • The entity is explicitly stated to be present, or there is a described change (e.g., improved, worsened, unchanged), implying that the entity currently exists or is progressing.
    • Examples: "Pleural effusion has improved", "Consolidation is stable", "No change in pneumothorax"
• Negative:
    • The entity is explicitly stated to be absent or removed.
    • Examples: "No pleural effusion", "ET tube has been removed", "No lines seen"

Dx_Certainty
• Indicates the level of certainty expressed regarding the presence or absence of a clinical entity.
• Definitive:
    • The statement conveys a clear and confident assertion about the presence or absence of an entity.
    • Examples: "No pneumothorax", "Definite consolidation", "Heart size is stable"
• Tentative:
    • The statement conveys uncertainty, possibility, or a lack of definitive conclusion.
    • Examples: "Possible pneumonia", "Suggests effusion", "Cannot exclude pneumothorax"

Location
• Only for position of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”, “stable position”).

Associate
• This relation identifies explicit, non-causal links between medical findings, conditions, or objects within or across sentences.
• Normally bidirectional when used alone (i.e., (A -Associate→ B) and (B -Associate→ A)). If Evidence is also present between the entities, then Associate is unidirectional, from finding → diagnosis.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → (dual-lead pacemaker/ICD device -Associate-> leads), (leads -Associate-> dual-lead pacemaker/ICD device)

Evidence
• This relation is used when one entity serves as evidence for another. The subject must be a diagnosis or suspected condition, while the object must be a radiological finding or observable support.
• Evidence is always unidirectional: from diagnosis → finding. When Evidence is annotated, a single corresponding Associate relation must also be included.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. lung opacity might be pulmonary edema → (pulmonary edema -Evidence-> lung opacity), (lung opacity -Associate-> pulmonary edema)

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent", "intact")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior", "recent")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat and one Status.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
6. Placement applies only to devices.
7. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
8. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
9. Extract only relations supported by explicit or inferable text evidence.
10. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def create_sys_msg_no_candidates2(args):
    return """You are a high‑precision relation‑extraction engine for chest X‑ray report sections.

**Input (JSON):**
{
    "context": [
        {
            "sent_idx": 1,
            "sentence": "Findings suggest possible pneumonia in the right lower lobe with opacity.",
        }
    ],
    "query": {
        "sent_idx": 2,
        "sentence": "A new small pleural effusion is seen on the left side.",
    }
}

Allowed Cat values:
PF, CF, OTH, COF, NCD, PATIENT INFO.

Relation definitions & labeling rules
Cat (Category)
• Exactly one per entity.
• PF (Perceptual Findings): Findings that can be observed in chest X-ray images.
• CF (Contextual Findings): Contextual diagnoses that require physician interpretation based on external information such as the patient's history, lab findings, or prior studies.
• OTH: Other Objects: Any medical devices observed in chest X-ray images or procedures and surgeries, as well as treatment and medications.
• COF (Clinical Observations): Evidence-based objective medical observation without clinician's subjective opinion which are often obtained through lab tests, physical exam, and other diagnostic procedures that are not based on chest x-ray imaging.
• NCD (Non-CXR Diagnosis): Contextual diagnosis given COF, which are often obtained from other sources like CT or lab tests, but are difficult to determine from chest X-ray images.
• PATIENT INFO. (Patient Information): Subjective indication of a disease or a change in condition as perceived by the patient. This is based on personal experiences and feelings, and they are not directly measurable.

Dx_Status
• Indicates whether a clinical entity is present or absent in the radiograph.
• Positive:
    • The entity is explicitly stated to be present, or there is a described change (e.g., improved, worsened, unchanged), implying that the entity currently exists or is progressing.
    • Examples: "Pleural effusion has improved", "Consolidation is stable", "No change in pneumothorax"
• Negative:
    • The entity is explicitly stated to be absent or removed.
    • Examples: "No pleural effusion", "ET tube has been removed", "No lines seen"

Dx_Certainty
• Indicates the level of certainty expressed regarding the presence or absence of a clinical entity.
• Definitive:
    • The statement conveys a clear and confident assertion about the presence or absence of an entity.
    • Examples: "No pneumothorax", "Definite consolidation", "Heart size is stable"
• Tentative:
    • The statement conveys uncertainty, possibility, or a lack of definitive conclusion.
    • Examples: "Possible pneumonia", "Suggests effusion", "Cannot exclude pneumothorax"

Location
• Only for position of findings or devices (e.g. “right lower lobe”).

Placement
• Only for devices (Cat=OTH). Indicates position or movement (e.g. “malpositioned,” “in expected position”, “stable position”).

Associate
• This relation identifies explicit, non-causal links between medical findings, conditions, or objects within or across sentences.
• Normally bidirectional when used alone (i.e., (A -Associate→ B) and (B -Associate→ A)). If Evidence is also present between the entities, then Associate is unidirectional, from finding → diagnosis.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → (dual-lead pacemaker/ICD device -Associate-> leads), (leads -Associate-> dual-lead pacemaker/ICD device)

Evidence
• This relation is used when one entity serves as evidence for another. The subject must be a diagnosis or suspected condition, while the object must be a radiological finding or observable support.
• Evidence is always unidirectional: from diagnosis → finding. When Evidence is annotated, a single corresponding Associate relation must also be included.
• Include obj_ent_idx pointing to the related entity.
• e.g. left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)
• e.g. lung opacity might be pulmonary edema → (pulmonary edema -Evidence-> lung opacity), (lung opacity -Associate-> pulmonary edema)

Morphology
• Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text. (e.g. "irregular", "rounded", "dense", "patchy", "linear", "plate-like", "nodular")

Distribution
• Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text. (e.g. "focal", "multifocal/multi-focal", "scattered", "widespread")

Measurement
• Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text. (e.g. "small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace")

Severity
• Attributes referring to the severity of an entity. Both subject and object are found in the text. (e.g. "mild", "moderate", "severe", "low-grade", "benign")

Comparison
• Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text. (e.g. "left brighter than right lung")

Onset
• Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text. (e.g. "new", "old", "acute", "subacute", "chronic", "remote", "recurrent")

No Change
• Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text. (e.g. "no change", "unchanged", "similar", "persistent", "intact")

Improved
• Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text. (e.g. "improved", "decreased", "resolved")

Worsened
• Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text. (e.g. "worsened", "increased")

Past Hx
• Reference to the patient's past medical or surgical history. Both subject and object are found in the text. (e.g. "history", "status post", "known", "prior", "recent")

Other Source
• Indicates information from non-CXR imaging sources. Both subject and object are found in the text. (e.g. "CT", "MRI")

Assessment Limitations
• Notes any technical issues affecting image interpretation. Both subject and object are found in the text. (e.g. "out of view", "differences in technique")

Extraction rules
1. Only findings, diagnoses or devices may be subjects.
2. Subject entity must have exactly one Cat and one Status.    
3. Assign each subject entity a unique ent_idx.
4. Use the provided sent_idx for each entity.
5. You may discover extra entities—assign them new ent_idx values and append in order of first occurrence.
6. Placement applies only to devices.
7. Associate and Evidence connect entities only when explicitly stated; include obj_ent_idx.
8. If an entity appears in multiple sites or with different attributes, output multiple entries (same name) with distinct sent_idx/ent_idx.
9. Extract only relations supported by explicit or inferable text evidence.
10. Output exactly one JSON object conforming to the Pydantic StructuredOutput schema. No plain‑text triples or commentary.
"""

def outdated_create_sys_msg(args):
    # Filter templates to only include specified entity types
    filtered_ent_templates = {k: v for k, v in templates['ent_type'].items() if k in args.entity_types}
    filtered_rel_templates = {k: v for k, v in templates['relation_type'].items() if k in args.relation_types}
    filtered_attr_templates = {k: v for k, v in templates['attribute_type'].items() if k in args.attribute_types}
    
    numbering = 1

    sys_msg_content = f"""You are an advanced medical relation extraction system specializing in chest X-ray reports.
    You will be given a text and a list of entity candidates. 
    Your task is to extract all valid relations between them using only the predefined relation types. 

## TASK
Your task is extracting triplets of **Subject, Relation, and Object** from chest X-ray reports. 
The triplets should be formatted as: **Subject -Relation-> Object**.
User will provide you with a sentence from a chest X-ray report along with candidate subjects and objects. 
You must extract at least the relations between the provided candidates. However, be aware that:

### **Key Requirements:**
1. You must **evaluate all predefined relation types** for every provided candidate **but only extract those that are valid given the candidate's role**.
2. **Only findings, diagnoses, or devices** can serve as subjects. Note that **location descriptors, change indicators, and descriptive attributes** cannot serve as subjects in relations.
3. If an entity can serve as a subject, you **must always extract both a -Cat-> relation and a -Status-> relation for that entity**.
4. You **must use the provided CANDIDATES exactly as they are, without any modifications**. **Do not change, rephrase, or standardize their text in any way**.
5. If additional relevant entities are found **(even if not in the candidate list)**, extract their relations as well. These entities can serve as subjects or objects in valid relations, as long as they meet the criteria for relation roles.
6. The subject and object of a relation **must not be the same entity**. Each relation **must connect two distinct entities**.
7. When the same entity appears in different relations, it **must be represented consistently** with exactly the same text. Do not use different phrasings or variations to refer to the same entity across relations.
8. If an entity **has different attributes or different locations**, you **must create separate relation sets for each location**. For example, if "pulmonary edema" is improved on the left but unchanged on the right, extract:

    (pulmonary edema -Cat-> RF)
    (pulmonary edema -Status-> DP)
    (pulmonary edema -Location-> loc: left)
    (pulmonary edema -Improved-> improved)

    AND

    (pulmonary edema -Cat-> RF)
    (pulmonary edema -Status-> DP)
    (pulmonary edema -Location-> loc: right)
    (pulmonary edema -No Change-> unchanged)

9. Use only the **predefined relation types** (explained below).
10. Follow the **strict output format** specified at the end.
11. You must evaluate **all 18 relation types** for each entity, but only extract those that are supported by textual or inferred evidence in the report.

---

## RELATION TYPES
Relations in chest X-ray reports typically fall into several categories. If any entity is determined to be a subject of a relation, you must also extract its -Cat-> and -Status-> relations. These two relations are mandatory for every subject entity:
"""

    if len(filtered_ent_templates) > 0:
        # Add each entity type description
        sys_msg_content += f"""
### **{numbering}. Cat (Category)**: This relation identifies the **category** of an entity. Assign this to every **finding/diagnosis/device** entity. The subject should be an entity mentioned in the text, but the object must be one of the following predefined categories:
"""
        numbering += 1
        for ent_type, details in filtered_ent_templates.items():
            sys_msg_content += f"""
    - **{ent_type} ({details['name']})**: {details['description']}
    subject examples: {details['examples']}
    """
    
    if len(filtered_rel_templates) > 0:
        sys_msg_content += f"""
### **{numbering}. Status**: This relation indicates the **status or condition** of an entity. The subject is mentioned in the text, but the object must be one of these four status codes based on two factors:

    a) Diagnosis certainty: Definitive or Tentative
    b) Mention context: Positively mentioned (present/abnormal) or Negatively mentioned (absent/normal)

    Status codes:
   - **DP**: Definitive Positive (definitely present or abnormal)
   - **DN**: Definitive Negative (definitely absent or normal)
   - **TP**: Tentative Positive (possibly present or abnormal)
   - **TN**: Tentative Negative (possibly absent or normal)
"""
    numbering += 1
    if len(filtered_rel_templates) > 0:

        for rel_type, details in filtered_rel_templates.items():
            sys_msg_content += f"""
### **{numbering}. {rel_type}**: {details['description']}
examples: {details['examples']}
"""
            numbering += 1

    if len(filtered_attr_templates) > 0:
        for rel_type, details in filtered_attr_templates.items():
            sys_msg_content += f"""
### **{numbering}. {rel_type}**: {details['description']}
object examples: {details['examples']}
""" 
            numbering += 1

    sys_msg_content += f"""
---

## **Input format**
INPUT : (Text including the entity 1 and entity 2)
CANDIDATES : [entity1, entity2]

## **Final output format**
After your analysis, present your findings in the following format for each extracted entity:

[Entity] -Cat-> [Category]
[Entity] -Status-> [Status]
[Entity] -Location-> loc: [Location], det: [Details]
[Entity] -[Relation Type]-> [Related Entity or Attribute]

Example:
pneumonia -Cat-> RF
pneumonia -Status-> TP
pneumonia -Location-> loc: right lower lobe
pneumonia -Evidence-> opacity
pneumonia -Onset-> new

Repeat this structure for each entity and its relations. If a specific relation doesn't apply to an entity, omit that line.
"""
    
    return sys_msg_content

if __name__ == "__main__":
    # Example usage with selected entity types
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate system message for GPT-SR')
    parser.add_argument('--entity_types', nargs='+', default=['COF', 'NCD', 'PATIENT INFO.', 'RF', 'OTH'],
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
    
    args = parser.parse_args()
    
    sys_msg_content = create_sys_msg(args)
    print(sys_msg_content)
    #print(factualness_sys_msg)