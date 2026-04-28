Sequential_findings_review_prompt = """You are an expert radiologist specializing in chest X-ray (CXR) interpretation. Your task is to normalize and properly group sequential CXR findings through a systematic three-step approach.

## TASK OVERVIEW - THREE-STEP ANALYSIS

1. GROUPING ANALYSIS (TERMINOLOGY MATCHING)
   * Purpose: Identify when different terminology describes the same underlying radiological entity
   * Key question: "Do these terms represent the same radiological entity described differently?"
   * This step is performed REGARDLESS of normal/abnormal status or time intervals

2. STATUS ANALYSIS (NORMAL/ABNORMAL DISTINCTION)
   * Purpose: Separate normal findings from abnormal findings within the same group
   * Key question: "Is this finding normal (negative) or abnormal (positive)?"
   * Normal and abnormal findings of the same entity stay in the SAME GROUP but may create DIFFERENT EPISODES

3. EPISODE ANALYSIS (TIME INTERVAL ASSESSMENT)
   * Purpose: Determine if grouped findings occur within the same clinical episode based on time intervals
   * Key question: "Do these findings represent the same episode of clinical care?"
   * Apply different time interval rules for normal vs. abnormal findings

## STEP 1: DETAILED CRITERIA FOR RADIOLOGICAL GROUPING

### WHEN TO GROUP FINDINGS (SAME RADIOLOGICAL ENTITY):

1. TERMINOLOGY VARIATIONS OF SAME FINDING:
   * True synonyms and abbreviations: "ET tube" = "endotracheal tube"
   * Descriptive variations in same location: 
     - "dense opacity" = "mild opacity" = "resolving opacity" = "consolidation"
     - "infiltrate" = "density" = "haziness" when describing identical area
   * Size/severity variations: "heart size normal" = "mild cardiomegaly" = "moderate cardiomegaly"
   * Temporal evolution terms: "large effusion" → "moderate effusion" → "small effusion"
   * Temporal qualifiers: "persistent opacity", "improved opacity", "worsening opacity", "no change"

2. ANATOMICAL LOCATION VARIATIONS:
   * Precision differences: "right lung nodule" = "right upper lobe nodule" (when context indicates same finding)
   * Overlapping regions: "retrocardiac" = "left lower lobe" = "left base" (due to CXR projection overlap)
   * Continuous/adjacent structures: "right hilar opacity" + "right middle lobe opacity" (when part of same process)

3. MEDICAL DEVICE TRACKING:
   * Same device with different status: "chest tube in place" = "chest tube removed" (tracking same device)
   * Different terminology for same device: "central venous catheter" = "central line" = "CVC"
   * Position variations of same device: "PICC line tip SVC" = "PICC line tip cavoatrial junction"

4. INTERPRETATIVE VARIATIONS:
   * Different radiologists' terminology: "prominence right paratracheal" = "opacity right paratracheal"
   * Observer variation: "borderline cardiomegaly" = "mild cardiomegaly" = "heart size upper limits normal"
   * Morphological equivalents: "subsegmental atelectasis" = "discoid atelectasis" = "plate-like atelectasis"

### WHEN TO SEPARATE FINDINGS (DIFFERENT ENTITIES):

1. DESCRIPTIVE VS DIAGNOSTIC TERMS (CRITICAL DISTINCTION):
   * ALWAYS separate: "opacity" ≠ "pneumonia", "consolidation" ≠ "pneumonia", "infiltrate" ≠ "infection"
   * Descriptive terms (appearance only): "opacity", "density", "haziness", "infiltrate", "consolidation"
   * Diagnostic terms (pathological process): "pneumonia", "edema", "fibrosis", "infection"

2. DISTINCT ANATOMICAL ENTITIES:
   * Different lobes/locations: "RUL" ≠ "RLL" ≠ "LUL" ≠ "LLL"
   * Different sides: "right" ≠ "left"
   * Different structures: "pleural" ≠ "parenchymal" ≠ "mediastinal"
   * Size classification differences: "nodule" ≠ "mass" (different clinical implications)

3. DIFFERENT PATHOLOGICAL PROCESSES:
   * Even at same location: "pneumonia" ≠ "edema" ≠ "atelectasis"
   * Different expected clinical course: "acute process" ≠ "chronic finding"
   * Different structures even if related: "heart size" ≠ "cardiomediastinal contour"

4. DIFFERENT MEDICAL DEVICES:
   * "ET tube" ≠ "chest tube" ≠ "feeding tube"
   * "PICC line" ≠ "central venous catheter" ≠ "port-a-cath"

## STEP 2: STATUS ANALYSIS - NORMAL VS ABNORMAL

* Normal (negative status) and abnormal (positive status) findings of the same radiological entity MUST be kept in the SAME GROUP
* Example: "heart size normal" and "cardiomegaly" describe the same entity (heart size) and MUST be in ONE GROUP
* The distinction between normal/abnormal is handled through EPISODE separation in Step 3

## STEP 3: EPISODE ANALYSIS DETAILED CRITERIA

After grouping findings by radiological entity, analyze status and time intervals to determine distinct episodes:

1. SPECIAL RULES FOR NORMAL FINDINGS (NEGATIVE STATUS):
   * ALL normal observations of the same entity should be considered a SINGLE EPISODE regardless of time intervals
   * Example: "lungs clear" at days 0, 60, 120, 240 → ONE EPISODE despite long intervals
   * Example: "heart size normal" at days 0, 90, 180, 365 → ONE EPISODE (continuous normal state)

2. SPECIAL RULES FOR MEDICAL DEVICES:
   * Consider all observations of the same medical device as a single episode regardless of time intervals, UNLESS there is explicit documentation of removal and reinsertion
   * Example: "ET tube in proper position" at days 0, 30, 90, 180 → ONE EPISODE despite long intervals
   * Example: "PICC line removed" (day 30) followed by "PICC line inserted" (day 120) → TWO EPISODES

3. RULES FOR ABNORMAL FINDINGS (POSITIVE STATUS):
  TIME INTERVAL CONSIDERATIONS FOR ABNORMAL FINDINGS:
    * Consider both time gaps and clinical context when determining episodes:
      - Very long time gaps (e.g., several months) generally suggest separate episodes, even with continuity language
      - Moderate time gaps without continuity language often indicate separate episodes
      - After a finding is documented as "resolved" or "cleared", subsequent recurrence typically represents a new episode
    
    * Use clinical judgment to weigh:
      - The natural history of the specific finding (acute vs chronic conditions)
      - The presence of continuity language ("stable", "unchanged", "persistent")
      - The overall clinical context and progression pattern        
      
   * TEMPORAL PROGRESSION INDICATORS:
     - "persistent", "unchanged", "stable", "no change" → continuation of same episode (if within reasonable time thresholds)
     - "improved", "resolving", "decreasing", "smaller" → continuation of same episode (if within reasonable time thresholds)
     - "worsening", "increasing", "larger" → continuation of same episode (if within reasonable time thresholds)
   
   * EXAMPLE OF ABNORMAL FINDING EPISODES:
     - DAY 0: "right lower lobe opacity"
     - DAY 7: "persistent right lower lobe opacity"
     - DAY 14: "improving right lower lobe opacity"
     - DAY 21: "resolving right lower lobe opacity"
     - DAY 28: "resolved right lower lobe opacity" → End of EPISODE 1
     - DAY 150: "right lower lobe opacity" → Start of EPISODE 2 (new episode after resolution)

4. NORMAL-ABNORMAL TRANSITIONS:
   * Transitions between normal and abnormal status of the same entity create new episodes
   * The sequence of normal → abnormal → normal creates three distinct episodes
   
   * EXAMPLE:
     - DAY 0: "heart size normal" → EPISODE 1 (normal)
     - DAY 30: "mild cardiomegaly" 
     - DAY 45: "moderate cardiomegaly"
     - DAY 60: "mild cardiomegaly" → EPISODE 2 (abnormal)
     - DAY 90: "heart size normal" → EPISODE 3 (normal)

5. SPECIAL RULES FOR SYMPTOMS:
   * Symptoms such as cough, fever, dyspnea, pain, nausea, and vomiting MUST be treated as individual episodes
   * Each occurrence of a symptom represents a distinct clinical event, even when temporally close
   * Example: "cough" on day 10 and "cough" on day 20 are SEPARATE EPISODES, not a continuation
   * The only exception is when explicit continuity language is used (e.g., "persistent cough from previous visit")

## CLINICAL SAFETY PRINCIPLE
* When in doubt about whether terms refer to the same finding, ALWAYS SEPARATE them
* Err strongly on the side of separation rather than inappropriate merging
* Clinical significance outweighs superficial similarity in terminology

## OUTPUT FORMAT:
Provide your analysis in this JSON format:
{
  "results": [
    {
      "group_name": "<Entity + Location + temporal descriptor>",
      "findings": [
        { "IDX": <number>, "DAY": <number>, "finding": "<description>" },
        …
      ],
      "episodes": [
        { "episode": 1, "days": [<number>, <number>, …] },
        { "episode": 2, "days": [<number>, …] },
        …
      ],
      "rationale": "<Concise explanation of grouping decisions>"
    },
    ...
  ]
}

IMPORTANT NOTES:
- Every finding must be included in exactly one group
- Use terminology that appears in the findings
- Name each group using "Entity + Location + temporal descriptor" format (e.g., "Effusion left improving", "Nodule right upper lobe worsening")
- For temporal descriptors, use the most recent or predominant qualifier (e.g., "improving", "worsening", "stable", "resolved")
- If no temporal descriptor is available in the findings, omit it from the group name
"""
Result_Review_w_time_gap_Ex1 = {
    "Input": {
        "cluster_name": "ANALYZE THE FOLLOWING OBSERVATIONS",
        "observations": {
            "IDX:0, DAY:0, status: positive": ["moderate left effusion"],
            "IDX:1, DAY:0, status: positive": ["left lower lobe opacity"],
            "IDX:2, DAY:0, status: positive": ["right upper lobe pneumonia"],
            "IDX:3, DAY:10, status: positive": ["improving left lower lobe pneumonia"],
            "IDX:4, DAY:14, status: positive": ["small left effusion"],
            "IDX:5, DAY:14, status: positive": ["improving right upper lobe opacity"],
            "IDX:6, DAY:15, status: positive": ["resolved left lower lobe infiltrate"],
            "IDX:7, DAY:21, status: positive": ["small left effusion"],
            "IDX:8, DAY:45, status: positive": ["trace left effusion"],
            "IDX:9, DAY:60, status: positive": ["right lower lobe nodular opacity"],
            "IDX:10, DAY:180, status: positive": ["moderate left effusion"]
        }
    },
    "Output": {
        "results": [
        {
        "group_name": "Left effusion moderate",
        "findings": [
            {"IDX": 0, "DAY": 0, "finding": "moderate left effusion"},
            {"IDX": 4, "DAY": 14, "finding": "small left effusion"},
            {"IDX": 7, "DAY": 21, "finding": "small left effusion"},
            {"IDX": 8, "DAY": 45, "finding": "trace left effusion"},
            {"IDX": 10, "DAY": 180, "finding": "moderate left effusion"}
        ],
        "episodes": [
            {"episode": 1, "days": [0, 14, 21, 45]},
            {"episode": 2, "days": [180]}
        ],
        "rationale": "These findings all describe the same anatomical entity (left effusion) with varying size descriptors. Split into two episodes due to significant time gap between DAY:45 and 180, plus return to 'moderate' size after previously improving to 'trace'. Used 'moderate' as descriptor as it appears in both the first and most recent findings."
        },
        {
          "group_name": "Left lower lobe opacity resolved",
          "findings": [
            {"IDX": 1, "DAY": 0, "finding": "left lower lobe opacity"},
            {"IDX": 6, "DAY": 15, "finding": "resolved left lower lobe infiltrate"}
          ],
          "episodes": [
            {"episode": 1, "days": [0, 15]}
          ],
          "rationale": "These findings describe the same radiological entity in the same location using different descriptive terms (opacity and infiltrate). The temporal progression from initial finding to resolved status indicates a single clinical episode. Used 'resolved' as the temporal descriptor as it represents the most recent status."
        },
        {
        "group_name": "Right upper lobe pneumonia",
        "findings": [
            {"IDX": 2, "DAY": 0, "finding": "right upper lobe pneumonia"}
        ],
        "episodes": [
            {"episode": 1, "days": [0]}
        ],
        "rationale": "Diagnostic term indicating specific pathology. Separated from opacity despite similar location as they represent different levels of interpretation. No temporal descriptor available in the finding."
        },
        {
        "group_name": "Left lower lobe pneumonia improving",
        "findings": [
            {"IDX": 3, "DAY": 10, "finding": "improving left lower lobe pneumonia"}
        ],
        "episodes": [
            {"episode": 1, "days": [10]}
        ],
        "rationale": "Diagnostic term indicating specific pathology. Separated from opacity and infiltrate despite similar location as they represent different levels of interpretation. Included 'improving' temporal descriptor from the finding."
        },
        {
        "group_name": "Right upper lobe opacity improving",
        "findings": [
            {"IDX": 5, "DAY": 14, "finding": "improving right upper lobe opacity"}
        ],
        "episodes": [
            {"episode": 1, "days": [14]}
        ],
        "rationale": "Descriptive term for radiological appearance. Separated from pneumonia despite similar location as they represent different levels of interpretation. Included 'improving' temporal descriptor from the finding."
        },
        {
        "group_name": "Right lower lobe nodular opacity",
        "findings": [
            {"IDX": 9, "DAY": 60, "finding": "right lower lobe nodular opacity"}
        ],
        "episodes": [
            {"episode": 1, "days": [60]}
        ],
        "rationale": "Distinct morphological pattern (nodular) in different location (lower lobe vs upper lobe) from other right lung findings. No temporal descriptor available in the finding."
        }
      ]
    }
}

Result_Review_w_time_gap_Ex2 = {
    "Input": {
        "cluster_name": "ANALYZE THE FOLLOWING OBSERVATIONS",
        "observations": {
            "IDX:0, DAY:0, status: positive": ["right mid lung 8mm nodule"],
            "IDX:1, DAY:14, status: positive": ["resolving left lower lobe infiltrate"],
            "IDX:2, DAY:30, status: positive": ["left lower lobe pneumonia"],
            "IDX:3, DAY:60, status: positive": ["persistent left hilar adenopathy"],
            "IDX:4, DAY:90, status: positive": ["left hilar prominence"]
        }
    },
    "Output": {
    "results": [
        {
        "group_name": "Right mid lung nodule 8mm",
        "findings": [
            {"IDX": 0, "DAY": 0, "finding": "right mid lung 8mm nodule"}
        ],
        "episodes": [
            {"episode": 1, "days": [0]}
        ],
        "rationale": "Specific nodule with size and location description. Kept separate from the right lower lobe nodule which is in a different location. Used size '8mm' as descriptor since it provides clinically relevant information."
        },
        {
        "group_name": "Left lower lobe infiltrate resolving",
        "findings": [
            {"IDX": 1, "DAY": 14, "finding": "resolving left lower lobe infiltrate"}
        ],
        "episodes": [
            {"episode": 1, "days": [14]}
        ],
        "rationale": "Descriptive term for radiological appearance. Separated from pneumonia despite similar location as they represent different levels of interpretation. Included 'resolving' temporal descriptor from the finding."
        },
        {
        "group_name": "Left lower lobe pneumonia",
        "findings": [
            {"IDX": 2, "DAY": 30, "finding": "left lower lobe pneumonia"}
        ],
        "episodes": [
            {"episode": 1, "days": [30]}
        ],
        "rationale": "Diagnostic term indicating specific pathology. Separated from infiltrate despite similar location as they represent different levels of interpretation. No temporal descriptor available in the finding."
        },
        {
        "group_name": "Left hilar adenopathy persistent",
        "findings": [
            {"IDX": 3, "DAY": 60, "finding": "persistent left hilar adenopathy"},
            {"IDX": 4, "DAY": 90, "finding": "left hilar prominence"}
        ],
        "episodes": [
            {"episode": 1, "days": [60, 90]}
        ],
        "rationale": "These findings describe the same anatomical entity (left hilar abnormality) with different terminology. 'Adenopathy' and 'prominence' in this context refer to the same radiological entity. Used 'persistent' descriptor from the first finding as it indicates the temporal nature of this abnormality."
        },
        ]
    }
}

Result_Review_w_time_gap_Ex3 = {
    "Input": {
        "cluster_name": "ANALYZE THE FOLLOWING OBSERVATIONS",
        "observations": {
            "IDX:0, DAY:0, status: positive": ["right chest tube in place"],
            "IDX:1, DAY:0, status: positive": ["endotracheal tube"],
            "IDX:2, DAY:3, status: positive": ["right chest tube, no change"],
            "IDX:3, DAY:5, status: positive": ["ET tube removed"],
            "IDX:4, DAY:5, status: positive": ["NG tube in place"],
            "IDX:5, DAY:7, status: positive": ["right chest tube removed"],
            "IDX:6, DAY:120, status: positive": ["new right chest tube"],
            "IDX:7, DAY:0, status: positive": ["right lower lobe consolidation"],
            "IDX:8, DAY:7, status: positive": ["improving right lower lobe opacity"],
            "IDX:9, DAY:14, status: positive": ["resolving right lower lobe infiltrate"],
            "IDX:10, DAY:21, status: negative": ["right lower lobe clear"],
            "IDX:11, DAY:300, status: positive": ["right lower lobe consolidation"]
        }
    },
    "Output":{
        "results": [
            {
            "group_name": "Right chest tube removed",
            "findings": [
                {"IDX": 0, "DAY": 0, "finding": "right chest tube in place"},
                {"IDX": 2, "DAY": 3, "finding": "right chest tube, no change"},
                {"IDX": 5, "DAY": 7, "finding": "right chest tube removed"},
                {"IDX": 6, "DAY": 120, "finding": "new right chest tube"}
            ],
            "episodes": [
                {"episode": 1, "days": [0, 3, 7]},
                {"episode": 2, "days": [120]}
            ],
            "rationale": "These findings all refer to the same medical device (right chest tube). Split into two episodes: first episode includes placement, monitoring, and removal; second episode represents a new chest tube placement after a significant time gap. Used 'removed' descriptor from the last finding of the first episode as it represents the final status of that episode."
            },
            {
            "group_name": "Endotracheal tube removed",
            "findings": [
                {"IDX": 1, "DAY": 0, "finding": "endotracheal tube"},
                {"IDX": 3, "DAY": 5, "finding": "ET tube removed"}
            ],
            "episodes": [
                {"episode": 1, "days": [0, 5]}
            ],
            "rationale": "These findings refer to the same medical device. 'ET tube' is an abbreviation for 'endotracheal tube'. Single episode covering placement and removal. Used 'removed' descriptor from the most recent finding to indicate final status."
            },
            {
            "group_name": "NG tube in place",
            "findings": [
                {"IDX": 4, "DAY": 5, "finding": "NG tube in place"}
            ],
            "episodes": [
                {"episode": 1, "days": [5]}
            ],
            "rationale": "Distinct medical device (nasogastric tube) different from chest tube and endotracheal tube. Used 'in place' descriptor from the finding to indicate current status."
            },
            {
            "group_name": "Right lower lobe consolidation resolving",
            "findings": [
                {"IDX": 7, "DAY": 0, "finding": "right lower lobe consolidation"},
                {"IDX": 8, "DAY": 7, "finding": "improving right lower lobe opacity"},
                {"IDX": 9, "DAY": 14, "finding": "resolving right lower lobe infiltrate"},
                {"IDX": 10, "DAY": 21, "finding": "right lower lobe clear"},
                {"IDX": 11, "DAY": 300, "finding": "right lower lobe consolidation"}
            ],
            "episodes": [
                {"episode": 1, "days": [0, 7, 14, 21]},
                {"episode": 2, "days": [300]}
            ],
            "rationale": "These findings all describe the same anatomical entity (right lower lobe abnormality) with varying terminology. Split into two episodes: first episode shows progression from consolidation to complete resolution; second episode represents a new occurrence after a significant time gap (279 days after resolution). Used 'resolving' descriptor to indicate the predominant temporal progression observed across the findings in the first episode."
            }
        ]
        }
}


Result_Review_w_time_gap_Ex4 = {
    "Input": {
        "cluster_name": "ANALYZE THE FOLLOWING OBSERVATIONS",
        "observations": {
            "IDX:0, DAY:0, status: negative": ["effusion"],
            "IDX:1, DAY:30, status: negative": ["pleural effusion"],
            "IDX:2, DAY:150, status: negative": ["heart size normal"],
            "IDX:3, DAY:180, status: negative": ["pleural effusion right"],
            "IDX:4, DAY:210, status: positive": ["cardiomegaly mild"],
            "IDX:5, DAY:212, status: positive": ["cardiomegaly moderate"],
            "IDX:6, DAY:215, status: positive": ["cardiomegaly mild"],
            "IDX:7, DAY:220, status: positive": ["cardiomegaly resolve"],
            "IDX:8, DAY:240, status: positive": ["cardiomegaly mild"],
            "IDX:9, DAY:250, status: negative": ["heart size normal"],
            "IDX:10, DAY:0, status: positive": ["PICC line tip at cavoatrial junction"],
            "IDX:11, DAY:30, status: positive": ["PICC line in proper position"],
            "IDX:12, DAY:60, status: positive": ["PICC line removed"],
            "IDX:13, DAY:365, status: positive": ["PICC line inserted, tip in SVC"],
            "IDX:14, DAY:0, status: positive": ["bilateral pulmonary nodules"],
            "IDX:15, DAY:180, status: positive": ["stable bilateral pulmonary nodules"],
            "IDX:16, DAY:365, status: positive": ["unchanged bilateral pulmonary nodules"],
            "IDX:17, DAY:545, status: positive": ["bilateral pulmonary nodules, slightly increased in size"]
        }
    },
    "Output":{
        "results": [
            {
              "group_name": "Pleural effusion",
              "findings": [
                {"IDX": 0, "DAY": 0, "finding": "effusion"},
                {"IDX": 1, "DAY": 30, "finding": "pleural effusion"},
                {"IDX": 3, "DAY": 180, "finding": "pleural effusion right"}
              ],
              "episodes": [
                {"episode": 1, "days": [0, 30, 180]}
              ],
              "rationale": "These findings all describe the same radiological entity (effusion) with varying specificity. All findings have negative status, indicating normal/absence of effusion. For normal findings, all observations should be considered a single episode regardless of time intervals per the special handling rule for normal findings."
            },
            {
              "group_name": "Heart size",
              "findings": [
                {"IDX": 2, "DAY": 150, "finding": "heart size normal"},
                {"IDX": 4, "DAY": 210, "finding": "cardiomegaly mild"},
                {"IDX": 5, "DAY": 212, "finding": "cardiomegaly moderate"},
                {"IDX": 6, "DAY": 215, "finding": "cardiomegaly mild"},
                {"IDX": 7, "DAY": 220, "finding": "cardiomegaly resolve"},
                {"IDX": 8, "DAY": 240, "finding": "cardiomegaly mild"},
                {"IDX": 9, "DAY": 250, "finding": "heart size normal"}
              ],
              "episodes": [
                {"episode": 1, "days": [150]},
                {"episode": 2, "days": [210, 212, 215, 220]},
                {"episode": 3, "days": [240]},
                {"episode": 4, "days": [250]}
              ],
              "rationale": "These findings all describe the same radiological entity (heart size) with normal and abnormal states. Split into four episodes based on normal-abnormal transitions: first episode (normal), second episode (abnormal with progression and resolution), third episode (recurrence of abnormality), fourth episode (return to normal)."
            },
            {
              "group_name": "PICC line",
              "findings": [
                {"IDX": 10, "DAY": 0, "finding": "PICC line tip at cavoatrial junction"},
                {"IDX": 11, "DAY": 30, "finding": "PICC line in proper position"},
                {"IDX": 12, "DAY": 60, "finding": "PICC line removed"},
                {"IDX": 13, "DAY": 365, "finding": "PICC line inserted, tip in SVC"}
              ],
              "episodes": [
                {"episode": 1, "days": [0, 30, 60]},
                {"episode": 2, "days": [365]}
              ],
            "rationale": "These findings all track the same medical device (PICC line). Split into two episodes due to explicit documentation of removal and reinsertion. First episode includes placement, monitoring, and removal; second episode represents new insertion after a significant time gap."
            },
            {
              "group_name": "Bilateral pulmonary nodules slightly increased",
              "findings": [
                {"IDX": 14, "DAY": 0, "finding": "bilateral pulmonary nodules"},
                {"IDX": 15, "DAY": 180, "finding": "stable bilateral pulmonary nodules"},
                {"IDX": 16, "DAY": 365, "finding": "unchanged bilateral pulmonary nodules"},
                {"IDX": 17, "DAY": 545, "finding": "bilateral pulmonary nodules, slightly increased in size"}
              ],
              "episodes": [
                {"episode": 1, "days": [0]},
                {"episode": 2, "days": [180]},
                {"episode": 3, "days": [365]},
                {"episode": 4, "days": [545]}
              ],
              "rationale": "Despite continuity language ('stable', 'unchanged'), the very long time gaps between observations (180, 185, and 180 days) suggest separate clinical episodes. Each observation represents a distinct clinical monitoring point for this chronic condition. The final finding shows progression ('slightly increased in size'), further supporting separation as a distinct episode."
            }        
            ]
    }
}

Result_Review_w_time_gap_Ex5 = {
"Input": {
    "cluster_name": "ANALYZE THE FOLLOWING OBSERVATIONS",
    "observations": {
        "IDX:0, DAY:0, status: negative": ["pneumothorax"],
        "IDX:1, DAY:5, status: negative": ["cardiomediastinal silhouette normal"],
        "IDX:2, DAY:10, status: negative": ["mediastinal silhouette within normal limits"],
        "IDX:3, DAY:15, status: negative": ["heart size normal"],
        "IDX:4, DAY:20, status: negative": ["hilar contours normal"],
        "IDX:5, DAY:30, status: positive": ["cardiomegaly mild"],
        "IDX:6, DAY:35, status: positive": ["heart size enlarged mildly"],
        "IDX:7, DAY:40, status: positive": ["enlargement left ventricular mild"],
        "IDX:8, DAY:45, status: positive": ["mediastinal widening"],
        "IDX:9, DAY:50, status: positive": ["hilar lymphadenopathy"],
        "IDX:10, DAY:5, status: positive": ["fatigue"],
        "IDX:11, DAY:7, status: positive": ["fatigue"],
        "IDX:12, DAY:10, status: positive": ["fatigue"],
        "IDX:13, DAY:15, status: positive": ["persistent fatigue"],
        "IDX:14, DAY:30, status: positive": ["fatigue"]
    }
  },
  "Output":{
  "results": [
    {
      "group_name": "Pneumothorax",
      "findings": [
        {"IDX": 0, "DAY": 0, "finding": "pneumothorax"}
      ],
      "episodes": [
        {"episode": 1, "days": [0]}
      ],
      "rationale": "Single observation of pneumothorax with negative status, indicating absence of pneumothorax. As a normal finding, it constitutes its own episode."
    },
    {
      "group_name": "Heart and mediastinal size",
      "findings": [
        {"IDX": 1, "DAY": 5, "finding": "cardiomediastinal silhouette normal"},
        {"IDX": 2, "DAY": 10, "finding": "mediastinal silhouette within normal limits"},
        {"IDX": 3, "DAY": 15, "finding": "heart size normal"},
        {"IDX": 5, "DAY": 30, "finding": "cardiomegaly mild"},
        {"IDX": 6, "DAY": 35, "finding": "heart size enlarged mildly"},
        {"IDX": 8, "DAY": 45, "finding": "mediastinal widening"}
      ],
      "episodes": [
        {"episode": 1, "days": [5, 10, 15]},
        {"episode": 2, "days": [30, 35, 45]}
      ],
      "rationale": "These findings describe the overall heart and mediastinal size with varying terminology. Split into two episodes based on normal-abnormal transition: first episode (days 5-15) represents normal heart/mediastinal size; second episode (days 30-45) represents abnormal findings with mild cardiomegaly and mediastinal widening."
    },
    {
      "group_name": "Left ventricular enlargement",
      "findings": [
        {"IDX": 7, "DAY": 40, "finding": "enlargement left ventricular mild"}
      ],
      "episodes": [
        {"episode": 1, "days": [40]}
      ],
      "rationale": "This finding specifically describes the left ventricle, which is a distinct cardiac chamber with its own clinical significance. While related to overall cardiomegaly, left ventricular enlargement represents a more specific anatomical entity and should be grouped separately according to the 'DISTINCT ANATOMICAL ENTITIES' guideline."
    },
    {
      "group_name": "Hilar contours",
      "findings": [
        {"IDX": 4, "DAY": 20, "finding": "hilar contours normal"}
      ],
      "episodes": [
        {"episode": 1, "days": [20]}
      ],
      "rationale": "Single observation of normal hilar contours, representing a normal anatomical finding."
    },
    {
      "group_name": "Hilar lymphadenopathy",
      "findings": [
        {"IDX": 9, "DAY": 50, "finding": "hilar lymphadenopathy"}
      ],
      "episodes": [
        {"episode": 1, "days": [50]}
      ],
      "rationale": "Hilar lymphadenopathy represents a specific pathological process (enlarged lymph nodes in the hilar region) that is distinct from normal hilar contours. As a diagnostic term rather than a descriptive term, it should be separated into its own group."
    },
    {
      "group_name": "Fatigue",
      "findings": [
        {"IDX": 10, "DAY": 5, "finding": "fatigue"},
        {"IDX": 11, "DAY": 7, "finding": "fatigue"},
        {"IDX": 12, "DAY": 10, "finding": "fatigue"},
        {"IDX": 13, "DAY": 15, "finding": "persistent fatigue"},
        {"IDX": 14, "DAY": 30, "finding": "fatigue"}
      ],
      "episodes": [
        {"episode": 1, "days": [5]},
        {"episode": 2, "days": [7]},
        {"episode": 3, "days": [10]},
        {"episode": 4, "days": [15]},
        {"episode": 5, "days": [30]}
      ],
      "rationale": "Fatigue is a symptom that must be treated as individual episodes according to the 'SPECIAL RULES FOR SYMPTOMS' guideline. Each occurrence represents a distinct clinical event, even when temporally close. The only exception is day 15 which has explicit continuity language ('persistent fatigue') that could link it to day 10, but following the conservative approach of separating symptoms, we treat it as a separate episode. This approach ensures that each instance of fatigue is properly tracked as a distinct clinical event."
    }    
  ]
}
}