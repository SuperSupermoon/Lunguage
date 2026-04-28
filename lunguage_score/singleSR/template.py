templates = {
    "ent_type": {
        "COF": {
            "name": "Clinical Objective Findings",
            "description": "Refers to evidence-based objective medical observation without clinician's subjective opinion which are often obtained through lab tests, physical exam, and other diagnostic procedures that are not based on chest x-ray imaging.",
            "examples": [
                "hemoglobin levels", "white blood cell count", "liver function tests",
                "heart rate", "Systemic inflammatory response syndrome (SIRS)", "temperature"
            ]
        },
        "NCD": {
            "name": "Non-CXR Diagnosis",
            "description": "Refers to contextual diagnosis given COF, which are often obtained from other sources like CT or lab tests, but are difficult to determine from chest X-ray images.",
            "examples": ["AIDS"]
        },
        "PATIENT INFO.": {
            "name": "Patient Information",
            "description": "A subjective indication of a disease or a change in condition as perceived by the patient. This is based on personal experiences and feelings, and they are not directly measurable.",
            "examples": ["fatigue", "cough", "shortness of breath", "vomiting"]
        },
        "OTH": {
            "name": "Other Objects",
            "description": "This category pertains to any medical devices observed in chest X-ray images or procedures and surgeries, as well as treatment and medications.",
            "examples": ['chest tubes', 'picc line', 'pacemakers', 'catheters', 'PEG', 'sternotomy', 'stem cell transplant', 'prednisone', 'kenalog injection']
        },
        "RF": {
            "name": "Radiological Findings",
            "description": "Refers to findings that can be observed in chest X-ray images. It includes both directly visible anatomical and pathological findings, as well as contextual diagnoses that require physician interpretation based on external information such as the patient's history, lab findings, or prior studies.",
            "examples": [
                # LF (Low-level Features) - directly observable anatomical findings
                "Lung", "Cardiomediastinal silhouette", "Opacity", "Collapse",
                # PF (Perceptual Findings) - directly observable pathological findings
                "Consolidation", "Infiltration", "Atelectasis", "Pleural Effusion", "Bronchiectasis",
                "Calcification", "Pneumomediastinum", "Pneumoperitoneum", "Pneumothorax",
                "Hydropneumothorax", "Lesion", "Mass", "Nodule",
                # CF (Contextual Findings) - diagnoses requiring external information (history, labs, prior studies)
                "pneumonia", "fluid overload/heart failure", "copd", "emphysema", "granulomatous disease",
                "interstitial lung disease", "goiter", "lung cancer", "aspiration",
                "alveolar hemorrhage", "pericardial effusion", "pericarditis",
                "pulmonary hypertension", "tumor", "Pulmonary Edema"
            ]
        }
    },
    'relation_type':{
        "Location": {
            "description": """This relation describes the specific position of an entity. Both subject and object are found within the text. The location may have two components:

   - **loc**: Primary location (anatomical structure or device placement)
   - **det**: Additional descriptive details for the location

   If the same entity appears in multiple locations, you **must extract each instance separately.** Once an object has been used as a location, you **must not use it as a subject in another relation.**""",
            "description_for_factualness": "A specific position of an entity",
            "examples": [
                "Right pleural effusion, evaluation for interval change. → (pleural effusion -Location-> loc: right)",
                "A dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → **(leads -CAT-> OTH), (leads -Status-> DP), (leads -Location-> loc: right atrium), (leads -CAT-> OTH), (leads -Status-> DP), (leads -Location-> loc: right ventricle)**",
                "Endotracheal tube in carina 5cm below. → (endotracheal tube -Location-> loc: carina, det: 5cm below)",
                "Pneumonia in lung above pacemaker. → (pneumonia -Location-> loc: lung, det: above pacemaker)",
                "Pacer lead is unchanged in position, tip projecting over the floor of the right ventricle close to the anticipated location of the apex. → (tip -Location-> loc: right ventricle, det: floor close to the apex)",
                "Peribronchial opacities in the left lower lung and right lung base are concerning for aspiration. → **(peribronchial opacities -CAT-> RF), (peribronchial opacities -Status-> DP), (peribronchial opacities -Location-> loc: left lower lung), (peribronchial opacities -CAT-> RF), (peribronchial opacities -Status-> DP), (peribronchial opacities -Location-> loc: right lung base)**",
            ]
        },
        "Associate": {
            "description": "This relation identifies links between medical findings, conditions, or objects (excluding causal relationships). All entities involved are mentioned in the text.",
            "description_for_factualness": "A relationship between two medical findings or conditions",
            "examples": [
                "Pneumonia is associated with lung opacity and fatigue. → (pneumonia -Associate-> lung opacity), (pneumonia -Associate-> fatigue)",
                "A dual-lead pacemaker/ICD device appears unchanged with leads again terminating in the right atrium and ventricle, respectively. → (dual-lead pacemaker/ICD device -Associate-> leads), (leads -Associate-> dual-lead pacemaker/ICD device)"
            ]
        },
        "Evidence": {
            "description": """This relation is used when one entity serves as evidence for another. The subject must be a diagnosis or suspected condition, while the object must be a radiological finding or observable support. Both subject and object are found in the text.

    - If A is evidenced by B, extract (A -Evidence-> B)
    - However, (B -Evidence-> A) is not valid; instead, extract (B -Associate-> A)""",
            "description_for_factualness": "A causal implication between two medical findings or conditions. Object should be an evidence for the relationship.",
            "examples": [
                "Left lung opacity suggests pneumonia → (pneumonia -Evidence-> left lung opacity), (left lung opacity -Associate-> pneumonia)",
                "Lung opacity might be pulmonary edema → (pulmonary edema -Evidence-> lung opacity), (lung opacity -Associate-> pulmonary edema)",
            ]
            }
    },
    'attribute_type':{
        "Morphology": {
            "description": "Physical form, structure, shape, pattern or texture of an object or substance. Both subject and object are found in the text.",
            "examples": ["irregular", "rounded", "dense", "ground-glass", "patchy", "linear", "plate-like", "nodular"]
        },
        "Distribution": {
            "description": "Arrangement, spread of objects or elements within a particular area or space. Both subject and object are found in the text.",
            "examples": ["focal", "multifocal/multi-focal", "scattered", "hazy", "widespread"]
        },
        "Measurement": {
            "description": "Physical dimensions or overall magnitude of an entity and attributes about counting or quantifying individual occurrences or components. Both subject and object are found in the text.",
            "examples": ["small", "large", "massive", "subtle", "minor", "xx cm", "single", "multiple", "few", "trace"]
        },
        "Severity": {
            "description": "Attributes referring to the severity of an entity. Both subject and object are found in the text.",
            "examples": ["mild", "moderate", "severe", "low-grade", "benign", "malignant"]
        },
        "Comparison": {
            "description": "Refers to any comparison in the single study, such as left, right comparison except for the temporal comparison. Both subject and object are found in the text.",
            "examples": ["left brighter than right lung"]
        },
        "Onset": {
            "description": "Refers to the chronological progression or appearance of a medical finding or device. Both subject and object are found in the text.",
            "examples": ["new", "old", "acute", "subacute", "chronic", "remote", "recurrent"]
        },
        "No Change": {
            "description": "Refers to the consistent state or condition that remains unaltered from a prior study. Both subject and object are found in the text.",
            "examples": ["no change", "unchanged", "similar", "persistent"]
        },
        "Improved": {
            "description": "Refers to a positive change or stabilization in a patient's clinical state when compared to a prior assessment. Both subject and object are found in the text.",
            "examples": ["improved", "decreased", "stable", "resolved"]
        },
        "Worsened": {
            "description": "Refers to the negative change in a patient's clinical state in comparison to a prior assessment. Both subject and object are found in the text.",
            "examples": ["worsened", "increased"]
        },
        "Placement": {
                "description": "Refers to the positional status of a medical device. Both subject and object are found in the text.",
                "examples": ["displaced", "repositioned", "withdrawn", "expected position", "standard placement", "malpositioned"]
        },
        "Past Hx": {
            "description": "Reference to the patient's past medical or surgical history. Both subject and object are found in the text.",
            "examples": ["history", "status post", "known", "prior"]
        },
        "Other Source": {
            "description": "Indicates information from non-CXR imaging sources. Both subject and object are found in the text.",
            "examples": ["CT", "MRI"]
        },
        "Assessment Limitations": {
            "description": "Notes any technical issues affecting image interpretation. Both subject and object are found in the text.",
            "examples": ["out of view", "differences in technique"]
        }
    },
    'relation_cot':{
        'Cat': 'categorized as',
        'Status': 'has diagnostic status',
        'Location': 'located at',
        'Associate': 'associated with',
        'Evidence': 'suggests',
        'Morphology': 'has structure, shape, texture of',
        'Distribution': 'has arrangement, spread pattern of',
        'Measurement': 'measured as',
        'Severity': 'indicates severity',
        'Comparison': 'compared to',
        'Onset': 'has onset',
        'No Change': 'has no change',
        'Improved': 'has improved',
        'Worsened': 'has worsened',
        'Placement': 'positioned at',
        'Past Hx': 'has medical history',
        'Other Source': 'reported from other source',
        'Assessment Limitations': 'affected by technical or other assessment issues',
    },
    'factualness':{
        'cat': [
            ("Blood tests showed an elevated white blood cell count, which is indicative of an infection.",
             ["white blood cell count", "cat", "cof"], 'true'),

            ("A chest X-ray revealed a large pleural effusion on the left side, causing significant lung compression.",
             ["pleural effusion", "cat", "rf"], 'true'),

            ("The patient complained of severe shortness of breath and fatigue over the past few days.",
             ["shortness of breath", "cat", "rf"], 'false'),  # Should be Patient Info.

            ("The patient reported experiencing persistent vomiting for three days.",
             ["vomiting", "cat", "rf"], 'false')  # Should be Patient Info.
        ],
        'status': [
            # Definitive Positive (DP)
            ("The patient has had left lower lobectomy and posterior chest wall reconstruction.",
             ["left lower lobectomy", "status", "dp"], 'true'),

            ("There is a possibility that the patient has lung cancer, but further tests are needed.",
             ["lung cancer", "status", "tp"], 'true'),  # Should be TP

            # False case: Wrong categorization (DN → TN)
            ("The doctor is uncertain but thinks tuberculosis is unlikely.",
             ["tuberculosis", "status", "dn"], 'false'),  # Should be TN

            # False case: Incorrect relation
            ("The patient has been diagnosed with pneumonia based on the chest X-ray findings.",
             ["pneumonia", "status", "tp"], 'false'),  # Should be DP
        ],
        'location': [
            # True Case: Single location
            ("There is a right pleural effusion, requiring evaluation for interval change.",
             ["pleural effusion", "location", "right"], 'true'),

            # True Case: Multiple locations
            ("The cardiac and mediastinal contours are normal.",
             ["contours", "location", "cardiac, mediastinal"], 'true'),

            # False Case: Wrong location mapping
            ("The cardiac and mediastinal contours are normal.",
             ["contours", "location", "lungs"], 'false'),

            # False Case: Incorrect or missing details
            ("Endotracheal tube is in the carina, 5 cm below the vocal cords.",
             ["Endotracheal tube", "location", "carina, 5cm above the vocal cords"], 'false'),
        ],
        'associate': [
            # True Case 1: Multiple valid associations
            ("Pneumonia is associated with lung opacity and fatigue.",
             ["pneumonia", "associate", "lung opacity"], 'true'),

            # True Case 2: Symptoms correctly associated with a condition
            ("Opacity in the lower lung suggests the presence of pneumonia.",
             ["opacity", "associate", "pneumonia"], 'true'),

            ("Opacity in the lower lung suggests the presence of pneumonia.",
             ["pneumonia", "associate", "opacity"], 'false'),

            ("Pneumonia is associated with lung opacity and fatigue.",
             ["pneumonia", "associate", "bronchiectasis"], 'false'),  # bronchiectasis not in the text
        ],
        'evidence': [
            # True Case 1: Opacity as evidence for pneumonia
            ("Opacity in the lower lung suggests the presence of pneumonia.",
             ["pneumonia", "evidence", "opacity"], 'true'),

            # True Case 2: Lung opacity as possible evidence for pulmonary edema
            ("Lung opacity might be due to pulmonary edema.",
             ["pulmonary edema", "evidence", "lung opacity"], 'true'),

            # False Case 1: Incorrect causal relationship
            ("The patient has pneumonia and a history of smoking.",
             ["pneumonia", "evidence", "smoking"], 'false'),  # pneumonia does not directly evidence smoking

            ("Opacity in the lower lung suggests the presence of pneumonia.",
             ["opacity", "evidence", "pneumonia"], 'false'),
        ],
        'morphology': [
            # True Case 1: Correct morphology classification
            ("The lung lesion appears nodular in shape.",
             ["lung lesion", "morphology", "nodular"], 'true'),

            # True Case 2: Correctly identified ground-glass opacity
            ("A ground-glass opacity is present in the right lower lobe.",
             ["opacity", "morphology", "ground-glass"], 'true'),

            # False Case 1: Incorrect relation
            ("The opacity is focal in the right upper lung.",
             ["opacity", "morphology", "focal"], 'false'),

            # False Case 2: Morphology type not present in text
            ("There are widespread consolidations throughout both lungs.",
             ["consolidations", "morphology", "widespread"], 'false'),
        ],
        'distribution': [
            # True Case 1: Correctly identified focal distribution
            ("The opacity is focal in the right upper lung.",
             ["opacity", "distribution", "focal"], 'true'),

            # True Case 2: Correctly identified widespread distribution
            ("There are widespread consolidations throughout both lungs.",
             ["consolidations", "distribution", "widespread"], 'true'),

            ("The lung lesion appears nodular in shape.",
             ["lung lesion", "distribution", "nodular"], 'false'),

            ("A ground-glass opacity is present in the right lower lobe.",
             ["opacity", "distribution", "ground-glass"], 'false'),
        ],
        'measurement': [
            # True Case 1: Correct measurement descriptor
            ("A small nodule was detected in the left lung.",
             ["nodule", "measurement", "small"], 'true'),

            # True Case 2: Correct quantification
            ("Multiple opacities are present throughout both lungs.",
             ["opacities", "measurement", "multiple"], 'true'),

            # False Case 1: Incorrect measurement descriptor
            ("A small nodule was detected in the left lung.",
             ["nodule", "measurement", "large"], 'false'),  # actual text says "small", not "large"

            # False Case 2: Measurement type not present in text
            ("A large mass is visible in the right lung.",
             ["mass", "measurement", "trace"], 'false')  # "trace" does not appear in the text
        ],
        'severity': [
            # True Case 1: Correct severity classification
            ("The patient has mild pneumonia.",
             ["pneumonia", "severity", "mild"], 'true'),

            # True Case 2: Correct identification of malignancy
            ("A malignant tumor was detected in the right lung.",
             ["tumor", "severity", "malignant"], 'true'),

            # False Case 1: Incorrect severity classification
            ("The patient has mild pneumonia.",
             ["pneumonia", "severity", "severe"], 'false'),  # actual text says "mild", not "severe"

            # False Case 2: Severity type not present in text
            ("A large mass is visible in the left lung.",
             ["mass", "severity", "low-grade"], 'false')  # "low-grade" does not appear in the text
        ],
        'comparison': [
            # True Case 1: Correct left vs. right comparison
            ("Again visualized are small bilateral pleural effusions, greater on the right than the left with bibasilar atelectasis.",
             ["pleural effusions", "comparison", "greater on the right than the left"], 'true'),

            # True Case 2: Correct size comparison
            ("The right heart border is more prominent than the left.",
             ["right heart border", "comparison", "more prominent than the left"], 'true'),

            # False Case 1: Incorrect comparison direction
            ("The left lung appears brighter than the right lung.",
             ["right lung", "comparison", "brighter than the left lung"], 'false'),  # comparison direction is reversed

            # False Case 2: Comparison type not present in text
            ("There is an opacity in the right lung.",
             ["right lung", "comparison", "brighter than the left lung"], 'false')  # no comparison expression in the text
        ],
        'onset': [
            # True Case 1: Correct onset classification (new)
            ("A new opacity is seen in the left lung.",
             ["opacity", "onset", "new"], 'true'),

            # True Case 2: Correct onset classification (chronic)
            ("The patient has a history of chronic bronchitis.",
             ["bronchitis", "onset", "chronic"], 'true'),

            # False Case 1: Incorrect onset classification
            ("A new opacity is seen in the left lung.",
             ["opacity", "onset", "old"], 'false'),  # actual text says "new", not "old"

            # False Case 2: Onset type not present in text
            ("There is an opacity in the right lung.",
             ["opacity", "onset", "acute"], 'false')  # "acute" does not appear in the text
        ],
        'no change': [
            # True Case 1: Correct no change classification (unchanged)
            ("The size of the nodule remains unchanged since the last exam.",
             ["nodule", "no change", "unchanged"], 'true'),

            # True Case 2: Correct no change classification (persistent)
            ("The pleural effusion is persistent compared to the prior study.",
             ["pleural effusion", "no change", "persistent"], 'true'),

            # False Case 1: Incorrect classification (wrong term)
            ("The size of the nodule remains unchanged since the last exam.",
             ["nodule", "no change", "new"], 'false'),  # actual text says "unchanged", not "new"

            # False Case 2: No indication of stability in the text
            ("A new consolidation is seen in the right lung.",
             ["consolidation", "no change", "similar"], 'false')  # "similar" does not appear in the text
        ],
        'improved': [
            # True Case 1: Correct improvement classification (improved)
            ("The patient's pneumonia has improved compared to the previous examination.",
             ["pneumonia", "improved", "improved"], 'true'),

            # True Case 2: Correct resolution of a condition
            ("The previously noted pleural effusion has resolved.",
             ["pleural effusion", "improved", "resolved"], 'true'),

            # False Case 1: Incorrect improvement classification (worsened condition)
            ("The patient's pneumonia has worsened compared to the previous examination.",
             ["pneumonia", "improved", "worsened"], 'false'),  # actual text says "worsened", not "improved"

            # False Case 2: No improvement mentioned in the text
            ("Left lung volume has decreased.",
             ["opacity", "improved", "decreased"], 'false')
        ],
        'worsened': [
            # True Case 1: Correct worsening classification (worsened)
            ("The patient's pneumonia has worsened since the last examination.",
             ["pneumonia", "worsened", "worsened"], 'true'),

            # True Case 2: Correct increase in condition severity
            ("The size of the pleural effusion has increased compared to the prior study.",
             ["pleural effusion", "worsened", "increased"], 'true'),

            # False Case 1: Incorrect worsening classification (improved condition)
            ("The size of the pleural effusion has decreased compared to the prior study.",
             ["pneumonia", "worsened", "decreased"], 'false'),  # actual text says "improved", not "worsened"

            # False Case 2: No worsening mentioned in the text
            ("A stable opacity is noted in the right lung.",
             ["opacity", "worsened", "increased"], 'false')  # "increased" does not appear in the text
        ],
        'placement': [
            # True Case 1: Correct placement classification (expected position)
            ("The endotracheal tube is in its expected position.",
             ["endotracheal tube", "placement", "expected position"], 'true'),

            # True Case 2: Correctly identifying a displaced device
            ("The nasogastric tube appears displaced into the right main bronchus.",
             ["nasogastric tube", "placement", "displaced"], 'true'),

            # False Case 1: Incorrect classification (wrong descriptor)
            ("The endotracheal tube is in its expected position of the carina.",
             ["endotracheal tube", "placement", "carina"], 'false'),

            # False Case 2: Placement description not present in text
            ("A central line is present in the right internal jugular vein.",
             ["central line", "placement", "right internal jugular vein"], 'false')  # "malpositioned" not in text
        ],
        'past hx': [
            # True Case 1: Correct reference to past medical history (history)
            ("The patient has a history of hypertension.",
             ["hypertension", "past hx", "history"], 'true'),

            # True Case 2: Correct use of "status post" indicating past surgery
            ("The patient is status post appendectomy.",
             ["appendectomy", "past hx", "status post"], 'true'),

            # False Case 1: Incorrect classification (wrong descriptor)
            ("The patient has a history of hypertension.",
             ["hypertension", "past hx", "new"], 'false'),  # actual text says "history", not "new"

            # False Case 2: No past medical history mentioned
            ("The patient currently has pneumonia.",
             ["pneumonia", "past hx", "prior"], 'false')  # "prior" does not appear in the text
        ],
        'other source': [
            # True Case 1: Correctly identified non-CXR imaging source (CT)
            ("A CT scan confirmed the presence of a pulmonary embolism.",
             ["pulmonary embolism", "other source", "CT"], 'true'),

            # True Case 2: Correctly identified non-CXR imaging source (MRI)
            ("An MRI revealed a brain tumor in the left hemisphere.",
             ["brain tumor", "other source", "MRI"], 'true'),

            # False Case 1: Incorrect classification (wrong source)
            ("A CT scan confirmed the presence of a pulmonary embolism.",
             ["pulmonary embolism", "other source", "MRI"], 'false'),  # actual text says "CT", not "MRI"

            # False Case 2: No non-CXR imaging source mentioned
            ("A chest X-ray showed bilateral infiltrates.",
             ["bilateral infiltrates", "other source", "CT"], 'false')  # CT not mentioned in text
        ],
        'assessment limitations': [
            # True Case 1: Correctly identified technical issue (out of view)
            ("The lower lung fields are partially out of view due to poor positioning.",
             ["lower lung fields", "assessment limitations", "out of view"], 'true'),

            # True Case 2: Correctly identified technical issue (differences in technique)
            ("Differences in imaging technique limit the comparison with prior studies.",
             ["comparison with prior studies", "assessment limitations", "differences in technique"], 'true'),

            # False Case 1: Incorrect classification (wrong issue)
            ("The lower lung fields are partially out of view due to poor positioning.",
             ["lower lung fields", "assessment limitations", "differences in technique"], 'false'),  # actual text says "out of view"

            # False Case 2: No technical issue mentioned
            ("A well-centered chest X-ray was obtained with good image quality.",
             ["chest X-ray", "assessment limitations", "out of view"], 'false')  # no technical issue in text
        ]
    }
}


