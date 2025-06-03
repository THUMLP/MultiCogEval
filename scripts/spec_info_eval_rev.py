# import openai
'''
    如果前面大的检查项识别了，后面括号里的就不管了
'''
import time
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import json
from tqdm import tqdm
from functools import lru_cache
from multiprocessing import Pool
import argparse
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    set_seed,
)
from vllm import LLM, SamplingParams
import pandas as pd
import argparse
import random
import pickle
import re
from fuzzywuzzy import fuzz
from utils import *


ACTION_LIST = ['PE', 'LAB', 'IMAGE', 'MICRO', 'OUTPUT']
SYSTEM_PROMPT = """You are an experienced medical AI assistant. Your ultimate goal is to help the doctor diagnose the patient's condition and recommend a treatment plan (procedure and prescription). You will be provided with the patient's history and the results of any tests that the doctor has already performed. You can also order additional tests for more information, including physical examinations, laboratory tests, microbiology tests, and imaging. 

The action you can choose are:

1. PE: Perform physical examination of patient and receive the observations.
2. LAB: Run laboratory tests and receive their values. You will get all the lab tests results at once.
3. MICRO: Run microbiology tests and receive their values. You will get all the microbiology tests results at once.
4. IMAGE: Do specific imaging scans and receive the radiologist report. You will get all the imaging results at once.
5. OUTPUT: Output the final diagnosis and treatment plan.

Note: To improve diagnostic efficiency, please perform physical examinations (PE), laboratory tests (LAB), microbiological tests (MICRO), and imaging scans (IMAGE) only when necessary for diagnosis. When you are confident, choose the "OUTPUT" action and you will be asked to output the corresponding diagnosis and treatment plan.

Your output format should be:

Rationale: (your reasoning process for choosing the next action)
Action: (one of the actions in [PE, LAB, IMAGE, MICRO, OUTPUT])
"""
START_PROMPT = """Now a patient comes to see the doctor. 
Patient History: {}. 
Please choose your next action from [PE, LAB, IMAGE, MICRO, OUTPUT].
"""

PE_PROMPT = """Physical Examination of this patient: {}
Please choose your next action from [{}].
"""

LAB_SPECIFIC_PROMPT = """You choose LAB as the next action. Please provide the specific list of laboratory tests you want to run:

please output your choice in the following format:
Rationale: (your reasoning process for choosing the next action)
Lab Tests: (the specific laboratory tests you want to run, separated by commas)
"""

LAB_PROMPT = """Here are the results of laboratory tests this patient has taken: 
{}
Please choose your next action from [{}].
"""

WARNING_EXAM_PROMPT = """NO VALID TESTS FOUND. Please choose your next action from [{0}].

Your output format should be:

Rationale: (your reasoning process for choosing the next action)
Action: (one of the actions in [{0}])
"""

IMAGE_PROMPT = """Here are the radiologist reports of this patient: 
{}
Please choose your next action from [{}].
"""

IMAGE_SPECIFIC_PROMPT = """You choose IMAGE as the next action. Please provide the specific list of imaging scans you want to run:

please output your choice in the following format:
Rationale: (your reasoning process for choosing the next action)
Imaging scans: (the specific imaging scans you want to run, separated by commas)
"""


MICRO_PROMPT = """Here are the microbiology test results of this patient: 
{}
Please choose your next action from [{}].
"""

MICRO_SPECIFIC_PROMPT = """You choose MICRO as the next action. Please provide the specific list of microbiology tests you want to run:

please output your choice in the following format:
Rationale: (your reasoning process for choosing the next action)
Microbiology tests: (the specific microbiology tests you want to run, separated by commas)
"""


SUMMARY_PROMPT = """Summarize our chat history, condense the content as much as possible while preserving all essential information related to the diagnosis. Eliminate redundant or irrelevant information, and ensure the summary maintains coherence and clarity.

Your output format should be:

Summary: (your summary of the dialogue)
"""

DIAGNOSIS_ENFORCE_PROMPT = """You have accessed all the available information of this patient. Now, directly output the diagnosis and treatment plan for this patient based on the information you have. 

Your output format should be:

Diagnosis: (the final diagnosis, specific disease name)
Treatment: (the treatment plan, including procedure and prescription)
"""

DIAGNOSIS_PROMPT = """You choose OUTPUT as the next action. Please directly output the diagnosis and treatment plan for this patient based on the information you have. 

Your output format should be:

Diagnosis: (the final diagnosis, specific disease name)
Treatment: (the treatment plan, including procedure and prescription)
"""

WARNING_ACTION_PROMPT = """You have chosen an action that is not available or used the wrong format. 

Your output format should be:

Rationale: (your reasoning process for choosing the next action)
Action: (one of the actions in [{0}])

Please choose your next action from [{0}].
"""


WARNING_DIAGNOSIS_PROMPT = """Diagnosis or treatment plan NOT FOUND. Please recheck the format and output the diagnosis and treatment plan for this patient based on the information you have, using the correct format below:

Diagnosis: (the final diagnosis, specific disease name)
Treatment: (the treatment plan, including procedure and prescription)

"""

supplementary_lab_tests = {
    'Rapid Strep Test': [
        "Strep A, Rapid Antigen"
    ],
    
    'Blood Chemistry Panel':[
        'Electrolyte Panel',
        'Renal Function Panel (RFP)',
        'Glucose',
        'Calcium'
    ],
    'Blood Chemistry':[
        'Electrolyte Panel',
        'Renal Function Panel (RFP)',
        'Glucose',
        'Calcium'
    ],
    'Blood Chemistry Test':[
        'Electrolyte Panel',
        'Renal Function Panel (RFP)',
        'Glucose',
        'Calcium'
    ],
    'Arterial Blood Gases': [
        'pH',
        'pCO2',
        'pO2',
        'Bicarbonate',
        'Oxygen Saturation',
        "Base Excess",
        "Lactate"
    ],
    'Arterial Blood Gas': [
        'pH',
        'pCO2',
        'pO2',
        'Bicarbonate',
        'Oxygen Saturation',
        "Base Excess",
        "Lactate"
    ],
    'BGA': [
        'pH',
        'pCO2',
        'pO2',
        'Bicarbonate',
        'Oxygen Saturation',
        "Base Excess",
        "Lactate"
    ],
    "Blood gas analysis":[
        'pH',
        'pCO2',
        'pO2',
        'Bicarbonate',
        'Oxygen Saturation',
        "Base Excess",
        "Lactate"
    ],
    "ABG": [
        'pH',
        'pCO2',
        'pO2',
        'Bicarbonate',
        'Oxygen Saturation',
        "Base Excess",
        "Lactate"
    ],
    "HIV antibodies":[
        "HIV 1 Ab Confirmation",
        "HIV 2 Ab Confirmation",
    ],
    "Neutrophil-Lymphocyte Ratio":[
        "Absolute Neutrophil Count",
        "Absolute Lymphocyte Count"
    ],
    "NLR":[
        "Absolute Neutrophil Count",
        "Absolute Lymphocyte Count"
    ],
    'HBV': [
        'Hepatitis B Core Antibody, IgM',
        'Hepatitis B Surface Antibody',
        'Hepatitis B Surface Antigen',
        'Hepatitis B Virus Core Antibody',
        'Hepatitis B Viral Load'
    ],
    'HCV':[
        "Hepatitis C Viral Load",
        "Reflex Confirmatory Hepatitis C Viral Load",
        "Hepatitis C Virus Antibody"
    ],
    'Cardiac Enzyme Panel':[
        'Troponin I',
        'Troponin T',
        'Creatine Kinase',
        'Creatine Kinase, MB Isoenzyme',
        "Myoglobin, Urine",
        'Lactate Dehydrogenase',
        'AST'
    ],
    "Cardiac Enzymes": [
        'Troponin I',
        'Troponin T',
        'Creatine Kinase',
        'Creatine Kinase, MB Isoenzyme',
        "Myoglobin, Urine",
        'Lactate Dehydrogenase',
        'AST'
    ],
    "ANA":[
        'Double Stranded DNA',
    ],
    'reticulocyte count':[
        "Reticulocyte Count, Absolute",
        "Reticulocyte Count, Automated", 
        "Reticulocyte Count, Manual"
    ],
    "Hepatitis panel":['Hepatitis A Virus Antibody',
                       'Hepatitis A Virus IgM Antibody',
                       'Hepatitis B Core Antibody, IgM',
                       'Hepatitis B Surface Antibody',
                       'Hepatitis B Surface Antigen',
                       'Hepatitis B Virus Core Antibody',
                        'Hepatitis C Virus Antibody',
                        'Hepatitis B Viral Load',
                         'Hepatitis C Viral Load',
                         'Reflex Confirmatory Hepatitis C Viral Load'],
    "Cholesterol": [
        'Total Cholesterol',
        'Cholesterol, HDL',
        'Cholesterol, LDL, Calculated',
        "Cholesterol, LDL, Measured",
        "Cholesterol Ratio"
    ],
    'urine drug screen':[
        "Amphetamine Screen, Urine",
        'Barbiturate Screen, Urine',
        'Benzodiazepine Screen, Urine',
        'Opiate Screen, Urine',
    ],
    "PCR for influenza":[
       "Influenza A by PCR",
        "Influenza B by PCR",
    ],
    'CSF':['Bilirubin, Total, CSF',
           'Chloride, CSF',
           'Glucose, CSF',
            'Lactate Dehydrogenase, CSF',
            'Miscellaneous, CSF',
            'PEP, CSF',
            'Total Protein, CSF',
            'Hematocrit, CSF',
            'RBC, CSF',
            'Total Nucleated Cells, CSF',
            'Total Bilirubin, CSF'],
    'Swab Culture':[
        'Swab - R/O Yeast - IC',
        'SWAB- R/O YEAST',
        
        ],
    'VZV':[
        "VZV IgG Ab",
        "VZV IgG Ab Value"
        ],
    'Rapid Influenza Diagnostic Test':[
        "Influenza A, Rapid Antigen",
        "Influenza B, Rapid Antigen"
    ],
    "Urine PCR":[
        "URINE CULTURE",
        "Urine Analysis (UA)"
    ],
    "Urine Antigen Test":[
        "Legionella Urinary Antigen"
    ]
}

standardize_lab_tests = {
    "Neutrophil Count": "Absolute Neutrophil Count",
    'HBsAg': 'Hepatitis B Surface Antigen',
    'HBsAb': 'Hepatitis B Surface Antibody',
    'HCV genotyping': 'HCV GENOTYPE',
    'HCV RNA': "Hepatitis C Viral Load",
    "TBili": "Total Bilirubin",
    "DBili": "Direct Bilirubin",
    "TB": "Total Bilirubin",
    "DB": "Direct Bilirubin",
    "estimated glomerular filtration rate": 'Estimated GFR',
    "eGFR": 'Estimated GFR',
    "LDH": 'Lactate Dehydrogenase (LD)',
    "BC": 'Blood Culture',
    "FBG":'Glucose',
    "URINE": "Urine Analysis (UA)",
    "Urine Microscopy": "Urine Analysis (UA)",
    "RPR": "RAPID PLASMA REAGIN TEST",
    "Legionella antigen": "Legionella Urinary Antigen",
    "Bone marrow biopsy": "Tissue Culture-Bone Marrow",
    "Tuberculosis":"GEN-PROBE AMPLIFIED M. TUBERCULOSIS DIRECT TEST (MTD)",
    "Hepatitis C antibodies":"Hepatitis C Virus Antibody",
    "Hepatitis C antibody":"Hepatitis C Virus Antibody",
    "Lipid Panel": "Lipid Profile",
    "Renal profile": "Renal Function Panel (RFP)",
    'ALB':'Albumin',
    'Urinal Culture': 'URINE CULTURE',
    'Urinary culture': 'URINE CULTURE',
    'BC':'Blood Culture',
    'UC':'URINE CULTURE',
    'free T3': 'T3',
    "CK-MB": "Creatine Kinase, MB Isoenzyme",
    "Blood Sugar": "Glucose",
    "international normalized ratio": "INR",
    "HCV antibody": "Hepatitis C Virus Antibody",
    "HCV antibodies": "Hepatitis C Virus Antibody",
    "HAV antibody": "Hepatitis A Virus Antibody",
    "HAV antibodies": "Hepatitis A Virus Antibody",
    "Rapid Plasma Reagin": "RAPID PLASMA REAGIN TEST",
    'respiratory virus panel':"Respiratory Viral Culture",
    "Coagulation panel":"Coagulation Profile",
    "kidney function":"Kidney Function Tests (KFT)",
    "Respiratory Panel":"Respiratory Viral Culture",
    "respiratory viruses":"Respiratory Viral Culture",
    "liver function":"Liver Function Test (LFT)",
    "HbA1c":"% Hemoglobin A1c",
    "serum iron": "Iron",
    "carcinoembryonic antigen": "Carcinoembyronic Antigen (CEA)",
    "respiratory viral panel": "Respiratory Viral Culture",
    "Alanine Transaminase": "ALT",
    "Aspartate Transaminase": "AST",
    "CBCT": 'CBC',
    'blood cell count': 'CBC',
    'UCS': "URINE CULTURE",
    "Direct Smear Examination of Blood":"SMEAR FOR BACTERIAL VAGINOSIS",
    "Blood Agar Culture": "Blood Culture",
    "Blood Mycology": "FUNGAL CULTURE",
    'herpes simplex virus':'VIRAL CULTURE: R/O HERPES SIMPLEX VIRUS'
}

# supplementary_lab_tests = {key.lower():value for key, value in supplementary_lab_tests.items()}
# standardize_lab_tests = {key.lower():value for key, value in standardize_lab_tests.items()}

base_new = "/hdd/zhouyx/mimic-clinical-decision-making"
meta_lab_tests = json.load(open(os.path.join(base_new, 'meta_lab_tests_UMLS_ext.json'),'r'))
for key in supplementary_lab_tests:
    if key not in meta_lab_tests:
        tmp_list = []
        for name in supplementary_lab_tests[key]:
            if name in meta_lab_tests:
                tmp_list.extend(meta_lab_tests[name])
        meta_lab_tests[key] = tmp_list
for key in standardize_lab_tests:
    if key not in meta_lab_tests:
        meta_lab_tests[key] = meta_lab_tests[standardize_lab_tests[key]]
# stop_tests = ['CO2','PCT','Na','Ca','QFT']
# if not os.path.exists("lab_test_buffer.json"):
#     print("No buffer found")
#     buffer = {}
# else:
#     print("Loading buffer")
#     buffer = json.load(open("lab_test_buffer.json",'r'))
nofound = set()
def load_data(path):
    return json.load(open(path,'r'))

# buffer_count = {}

@lru_cache(maxsize=10000)
def fuzzy_lab_match(item):
    flag = False
    best_match = ''
    best_score = 0
    for lab_name in meta_lab_tests:
        score = fuzz.ratio(item.lower(), lab_name.lower())
        if item[-1] == 's' and item[:-1].lower() == lab_name.lower():
            score = 100
        if score == 100:
            flag = True
            best_score = score
            best_match = lab_name
            break
        elif score > 90 or (len(lab_name) > 1 and re.search(rf'\b{re.escape(lab_name.lower())}\b', item.lower())):
            flag = True
            if score > best_score:
                best_score = score
                best_match = lab_name
    if best_match == '':
        flag = False
    return flag, best_match

def apply_chat_template(message, tokenizer, template):
    if len(template) == 0:   # default
        return tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
    elif template == 'qwen':
        return apply_chat_template_qwen(message)
    elif template == 'med42':
        return apply_chat_template_med42(message)
    elif template == 'cc':
        return apply_chat_template_clinicalcamel(message)
    else:
        raise ValueError("Template not recognized")

def apply_chat_template_qwen(message):
    #  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    text = ''
    for i, m in enumerate(message):
        if m['role'] == 'system':
            text += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
            # user = m['content']
        
        elif m['role'] == 'assistant':
            text += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
        elif m['role'] == 'user':
            text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        else:
            raise ValueError("Role not recognized")
    text += f"<|im_start|>assistant\n"
    return text

def apply_chat_template_med42(message):
    #  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    text = ''
    for i, m in enumerate(message):
        if m['role'] == 'system':
            text += f"<|system|>: {m['content']}\n"
            # user = m['content']
        
        elif m['role'] == 'assistant':
            text += f"<|assistant|>: {m['content']}\n"
        elif m['role'] == 'user':
            text += f"<|prompter|>: {m['content']}\n"
        else:
            raise ValueError("Role not recognized")
    text += f"<|assistant|>:"
    return text

def apply_chat_template_clinicalcamel(message):
    #  '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n'
    text = ''
    for i, m in enumerate(message):
        # if m['role'] == 'system':
        #     text += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
            # user = m['content']
        
        if m['role'] == 'assistant':
            text += f"\n\n### Assistant: {m['content']}"
        elif m['role'] == 'user':
            text += f"\n\n### User: {m['content']}"
        else:
            raise ValueError("Role not recognized")
    text += f"\n\n### Assistant:"
    return text

def specification_parser_test(action, response, exam_name_mapping):
    # outf = open(f"unfound_lab_test_{model_name}.jsonl",'a')
    # outf_2 = open(f"unfound_rab_{model_name}.jsonl",'a')
    if action == 'LAB_s':
        prefix = r'Lab Tests:'
    elif action == 'IMAGE_s':
        prefix = r'Imaging scans:'
    elif action == 'MICRO_s':
        prefix = r'Microbiology tests:'
    pattern = prefix+r"(.*?)\."
    action_spc = re.search(pattern, response, re.IGNORECASE)
    if not action_spc:
        action_spc = response.split(prefix)[-1]
    else:
        action_spc = action_spc.group(1)
    
    while '*' in action_spc:
        action_spc = action_spc.replace('*','')
    
    if action in ['LAB_s', 'MICRO_s']:
        action_spc = re.sub(r'\band\b', ',', action_spc)
        action_spc = re.sub(r'\bor\b', ',', action_spc)
        action_spc_text = action_spc
        for word in ["order", "run", "level[s]?", "repeat", "check",'including','the','e.g.','etc.','for','to','with']:
            action_spc = re.sub(
                rf"\b{word}\b", "", action_spc, flags=re.IGNORECASE
            )
        new_action_spc = []
        # action_spc = re.split(r',', action_spc)
        buf = ""
        abbr_buf = ""
        long_to_abbr = {}
        paren_flag = False  # if there is a parenthese
        for w in action_spc:    
            if w == '(':
                paren_flag = True
                new_action_spc.append(buf) 
            elif w == ')':
                paren_flag = False
                abbrs = abbr_buf.split(',')
                new_abbrs = []
                for i in range(len(abbrs)):
                    abbrs[i] = abbrs[i].strip()
                    if len(abbrs[i]) > 0:
                        new_abbrs.append(abbrs[i])
                abbrs = new_abbrs
                long_to_abbr[buf.strip()] = abbrs    
                abbr_buf = ""
                buf = ""
            elif w == ',':
                if not paren_flag and len(buf.strip()) > 0:
                    new_action_spc.append(buf)
                    buf = ""
                else:
                    abbr_buf += w
            else:
                if paren_flag:
                    abbr_buf += w
                else:
                    buf += w
        if len(buf.strip()) > 0:
            new_action_spc.append(buf)
        action_spc = new_action_spc
        action_spc = [item.strip() for item in action_spc]
        new_action_spc = []
        for item in action_spc:
            if len(item) > 0:
                new_action_spc.append(item)
        action_spc = new_action_spc
        res_list = []
    
        # for name in standardize_lab_tests:
        #     if re.search(rf'\b{name}\b', action_spc_text, re.IGNORECASE):
        #         res_list.update(meta_lab_tests[standardize_lab_tests[name]])
        
        
        
        matched_pairs = []
        abbr_action_spc = []
        for item in action_spc:
            if len(item) <=1:
                continue
            flag = False
            # if item.lower() in supplementary_lab_tests:
            #     flag = True
            #     for name in supplementary_lab_tests[item.lower()]:
            #         res_list.update(meta_lab_tests[name])
            # elif 'Panel' in item:
            if len(item.split()) > 10:
                continue    # long sentence
            else:
                flag, best_match = fuzzy_lab_match(item)
                if flag:
                    res_list.append(best_match)

            if not flag:    
                if item in long_to_abbr:
                    abbr_action_spc.extend(long_to_abbr[item])
                # if item.lower() not in nofound:
                # outf.write(json.dumps([item, action_spc, response])+'\n')
                nofound.add(item.lower())
        for item in abbr_action_spc:
            if len(item) <=1:
                continue
            flag = False
            # if item.lower() in supplementary_lab_tests:
            #     flag = True
            #     for name in supplementary_lab_tests[item.lower()]:
            #         res_list.update(meta_lab_tests[name])
            if len(item.split()) > 10:
                continue    # long sentence
            else:
                flag, best_match = fuzzy_lab_match(item)
                
                if flag:
                    res_list.append(best_match)

            if not flag:
                # if item.lower() not in nofound:
                # outf.write(json.dumps([item, action_spc, response])+'\n')
                nofound.add(item.lower())
    else:
        # region = 
        if '\n' in action_spc:
            DELIM = '\n'
        elif ';' in action_spc:
            DELIM = ';'
        else:
            DELIM = ','
            
        # Split the action_spc
        buf = ""
        abbr_buf = ""
        long_to_abbr = {}
        new_action_spc = []
        paren_flag = False  # if there is a parenthese
        for w in action_spc:    
            if w == '(':
                paren_flag = True
            elif w == ')':
                paren_flag = False
            elif w == DELIM:
                if not paren_flag and len(buf.strip()) > 0:
                    new_action_spc.append(buf)
                    buf = ""
            else:
                buf += w
        if len(buf.strip()) > 0:
            new_action_spc.append(buf)
        action_spc = new_action_spc
            
        while ', including' in action_spc:
            action_spc = action_spc.replace(', including', 'including')
        action_spc = [item.strip() for item in action_spc]
        action_spc = [item for item in action_spc if len(item) > 0]
        
        res_list = []
        for item in action_spc:
            if len(item) <=1:
                continue
            flag = False
            modality, region = parse_radiology_request(item)
            if item.upper() in exam_name_mapping:
                flag = True
                res_list.append({'Exam Name': item.upper(), 'Region': exam_name_mapping[item.upper()]['regions'], 'Modality': exam_name_mapping[item.upper()]['modalities']})
            else:
                
                if len(modality) == 0 or len(region) == 0:
                    if len(res_list) > 0 and len(region) > 0:   # Maybe the region belongs to the previous exam
                        res_list[-1]['Region'].extend(region)
                        res_list[-1]['Region'] = list(set(res_list[-1]['Region']))
                    elif len(region) == 0 and len(modality) > 0:
                        global_modal, global_region = parse_radiology_request(response) # Regions may also exist in the rationale
                        if len(global_region) > 0:
                            res_list.append({'Exam Name': item.upper(), 'Region': global_region, 'Modality': modality})
                    # else:
                        # outf_2.write(json.dumps([item, modality, region, action_spc, response])+'\n')
                else:
                    res_list.append({'Exam Name': item.upper(), 'Region': region, 'Modality': modality})
    res_list = list(res_list)
    # outf.close()
    # outf_2.close()
    # for item in valid_list:
    #     if item.lower() in action_spc.lower():
    #         res_list.append(item)
    return res_list

def specification_parser(action, response, exam_name_mapping, model_name):
    outf = open(f"unfound_lab_test_{model_name}.jsonl",'a')
    outf_2 = open(f"unfound_rab_{model_name}.jsonl",'a')
    if action == 'LAB_s':
        prefix = r'Lab Tests:'
    elif action == 'IMAGE_s':
        prefix = r'Imaging scans:'
    elif action == 'MICRO_s':
        prefix = r'Microbiology tests:'
    pattern = prefix+r"(.*?)\."
    action_spc = re.search(pattern, response, re.IGNORECASE)
    if not action_spc:
        action_spc = response.split(prefix)[-1]
    else:
        action_spc = action_spc.group(1)
        
    if len(action_spc.strip()) == 0:
        action_spc = response.split(prefix)[-1]
    
    while '*' in action_spc:
        action_spc = action_spc.replace('*','')
    
    if action in ['LAB_s', 'MICRO_s']:
        action_spc = re.sub(r'\band\b', ',', action_spc)
        action_spc = re.sub(r'\bor\b', ',', action_spc)
        action_spc_text = action_spc
        for word in ["order", "run", "level[s]?", "repeat", "check",'including','the','e.g.','etc.','for','to','with']:
            action_spc = re.sub(
                rf"\b{word}\b", "", action_spc, flags=re.IGNORECASE
            )
        new_action_spc = []
        # action_spc = re.split(r',', action_spc)
        buf = ""
        abbr_buf = ""
        long_to_abbr = {}
        paren_flag = False  # if there is a parenthese
        for w in action_spc:    
            if w == '(':
                paren_flag = True
                new_action_spc.append(buf) 
            elif w == ')':
                paren_flag = False
                abbrs = abbr_buf.split(',')
                new_abbrs = []
                for i in range(len(abbrs)):
                    abbrs[i] = abbrs[i].strip()
                    if len(abbrs[i]) > 0:
                        new_abbrs.append(abbrs[i])
                abbrs = new_abbrs
                long_to_abbr[buf.strip()] = abbrs    
                abbr_buf = ""
                buf = ""
            elif w == ',':
                if not paren_flag and len(buf.strip()) > 0:
                    new_action_spc.append(buf)
                    buf = ""
                else:
                    abbr_buf += w
            else:
                if paren_flag:
                    abbr_buf += w
                else:
                    buf += w
        if len(buf.strip()) > 0:
            new_action_spc.append(buf)
        action_spc = new_action_spc
        action_spc = [item.strip() for item in action_spc]
        new_action_spc = []
        for item in action_spc:
            if len(item) > 0:
                new_action_spc.append(item)
        action_spc = new_action_spc
        res_list = set()
    
        # for name in standardize_lab_tests:
        #     if re.search(rf'\b{name}\b', action_spc_text, re.IGNORECASE):
        #         res_list.update(meta_lab_tests[standardize_lab_tests[name]])
        
        
        
        matched_pairs = []
        abbr_action_spc = []
        for item in action_spc:
            if len(item) <=1:
                continue
            flag = False
            # if item.lower() in supplementary_lab_tests:
            #     flag = True
            #     for name in supplementary_lab_tests[item.lower()]:
            #         res_list.update(meta_lab_tests[name])
            # elif 'Panel' in item:
            if len(item.split()) > 10:
                continue    # long sentence
            else:
                flag, best_match = fuzzy_lab_match(item)
                if flag:
                    res_list.update(meta_lab_tests[best_match])

            if not flag:    
                if item in long_to_abbr:
                    abbr_action_spc.extend(long_to_abbr[item])
                # if item.lower() not in nofound:
                outf.write(json.dumps([item, action_spc, response])+'\n')
                nofound.add(item.lower())
        for item in abbr_action_spc:
            if len(item) <=1:
                continue
            flag = False
            # if item.lower() in supplementary_lab_tests:
            #     flag = True
            #     for name in supplementary_lab_tests[item.lower()]:
            #         res_list.update(meta_lab_tests[name])
            if len(item.split()) > 10:
                continue    # long sentence
            else:
                flag, best_match = fuzzy_lab_match(item)
                
                if flag:
                    res_list.update(meta_lab_tests[best_match])

            if not flag:
                # if item.lower() not in nofound:
                outf.write(json.dumps([item, action_spc, response])+'\n')
                nofound.add(item.lower())
    else:
        # region = 
        if '\n' in action_spc:
            DELIM = '\n'
        elif ';' in action_spc:
            DELIM = ';'
        else:
            DELIM = ','
            
        # Split the action_spc
        buf = ""
        abbr_buf = ""
        long_to_abbr = {}
        new_action_spc = []
        paren_flag = False  # if there is a parenthese
        for w in action_spc:    
            if w == '(':
                paren_flag = True
            elif w == ')':
                paren_flag = False
            elif w == DELIM:
                if not paren_flag and len(buf.strip()) > 0:
                    new_action_spc.append(buf)
                    buf = ""
            else:
                buf += w
        if len(buf.strip()) > 0:
            new_action_spc.append(buf)
        action_spc = new_action_spc
            
        while ', including' in action_spc:
            action_spc = action_spc.replace(', including', 'including')
        action_spc = [item.strip() for item in action_spc]
        action_spc = [item for item in action_spc if len(item) > 0]
        
        res_list = []
        for item in action_spc:
            if len(item) <=1:
                continue
            flag = False
            modality, region = parse_radiology_request(item)
            if item.upper() in exam_name_mapping:
                flag = True
                res_list.append({'Exam Name': item.upper(), 'Region': exam_name_mapping[item.upper()]['regions'], 'Modality': exam_name_mapping[item.upper()]['modalities']})
            else:
                
                if len(modality) == 0 or len(region) == 0:
                    if len(res_list) > 0 and len(region) > 0:   # Maybe the region belongs to the previous exam
                        res_list[-1]['Region'].extend(region)
                        res_list[-1]['Region'] = list(set(res_list[-1]['Region']))
                    elif len(region) == 0 and len(modality) > 0:
                        global_modal, global_region = parse_radiology_request(response) # Regions may also exist in the rationale
                        if len(global_region) > 0:
                            res_list.append({'Exam Name': item.upper(), 'Region': global_region, 'Modality': modality})
                    else:
                        outf_2.write(json.dumps([item, modality, region, action_spc, response])+'\n')
                else:
                    res_list.append({'Exam Name': item.upper(), 'Region': region, 'Modality': modality})
    res_list = list(res_list)
    outf.close()
    outf_2.close()
    # for item in valid_list:
    #     if item.lower() in action_spc.lower():
    #         res_list.append(item)
    return res_list


def action_parser(response):
    if 'Rationale:' in response:
        # find the last rationale
        response = response.split('Rationale:')[-1]
        response = 'Rationale:' + response
    action_list = ['IMAGE_s','LAB_s','MICRO_s']
    for i,prefix in enumerate([r'Imaging scans:', r'Lab Tests:', r'Microbiology tests:']):
        if prefix in response:
            action = action_list[i]
            
            return action, False, response
        
    if 'Diagnosis:' in response and 'Treatment:' in response:
        diagnosis_patterns = [r"Diagnosis:(.*)\n", r"Diagnosis:(.*)\Z"]
        treatment_patterns = [r"Treatment:(.*)\Z"]
        diagnosis, treatment = "None", "None"
        for p in diagnosis_patterns:
            res = re.search(p, response, re.IGNORECASE)
            if res:
                diagnosis = res.group(1).strip()
                break
        for p in treatment_patterns:
            res = re.search(p, response, re.IGNORECASE|re.DOTALL)
            if res:
                treatment = res.group(1).strip()
                break
        # if diagnosis == None or treatment == None:
        #     raise ValueError("Action parsing failed")
        
        return [diagnosis, treatment], True, response
    elif 'Action:' in response:
        action = None
        patterns = [r"Action:(.*)\Z", r"Action:(.*)\n"]
        for p in patterns:
            res = re.search(p, response, re.IGNORECASE)
            if res:
                action = res.group(1).strip()
                break
        if action == None:
            return [None, None], False, response
        # discard the content after action
        response = response.split('Action: ')[0] + 'Action: ' + action
        
        return action, False, response
    
    
    else:
        # Unable to parse
        return response, False, response


def post_processing(ids, response, summarized_failed_ids, summarize_ids, ids_chat_history, ids_action_history, ids_resources, ids_test_set, exam_name_mapping, exam_spec_mapping,lab_test_id_to_name, itemid_to_name):
    ids_lab_event_request, ids_image_event_request, ids_micro_event_request = [], [], []
    finish = False
    if ids in summarized_failed_ids:
        ids_chat_history.append({'role': 'assistant', 'content': "Failed to summarize"})
        result = ["failed", "failed"]
        finish = True
        return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
    result, finish, response = action_parser(response)
    if finish:
        ids_chat_history.append({'role': 'assistant', 'content': response})    # ends with assistant

        return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        # valid_ids.remove(ids)
    else:
        # new_ids.append(ids)
        if len(ids_action_history) > 0 and ids_action_history[-1] == 'WRONG': # last action is invalid:
            # assert len(ids_chat_history) > 2
            if ids not in summarize_ids and len(ids_chat_history) > 2:
                ids_chat_history = ids_chat_history[:-2]  # remove the last two messages (wrong response and user rectification)
            ids_action_history = ids_action_history[:-1]
        assert ids_chat_history[-1]['role'] == 'user'
        ids_chat_history.append({'role': 'assistant', 'content': response})
        tmp_actions = ids_action_history
        available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
        
        if 'OUTPUT' in ids_action_history:
            # already in the output stage
            assert ids_chat_history[-1]['role'] == 'assistant'
            ids_chat_history.append({'role': 'user', 'content': WARNING_DIAGNOSIS_PROMPT})
            ids_action_history.append("WRONG")
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        if len(available_actions) == 1 and 'OUTPUT' in available_actions and (('LAB_s' not in result) and ('IMAGE_s' not in result) and ('MICRO_s' not in result)):
            # only output left, enforce output
            assert ids_chat_history[-1]['role'] == 'assistant'
            ids_chat_history.append({'role': 'user', 'content': DIAGNOSIS_ENFORCE_PROMPT})
            ids_action_history.append('OUTPUT')
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        if 'PE' in result:
            if 'PE' in ids_action_history:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': 'You have already performed Physical Examination. Please choose another action from [{}].'.format(available_actions_text)})
                
            else:
                ids_action_history.append('PE')
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                if len(ids_test_set[3]) == 0:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    ids_chat_history.append({'role': 'user', 'content': 'No Physical Examination. Please choose another action.'}) 
                else:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    ids_chat_history.append({'role': 'user', 'content': PE_PROMPT.format(ids_test_set[3], available_actions_text)})
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        elif 'LAB_s' in result:
            available_tests = list(ids_test_set[4].keys())
            request_tests = specification_parser('LAB_s', response, exam_name_mapping,args.model_name)
            ids_lab_event_request = [request_tests, response]
            valid_tests = [test for test in request_tests if test in ids_test_set[4]]
            if len(valid_tests) == 0:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_EXAM_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            elif 'LAB' not in ids_resources:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_ACTION_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            else:
                assert ids_chat_history[-1]['role'] == 'assistant'
                lab_result = ''
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                for test in valid_tests:
                    if test in ids_test_set[4]:
                        lab_result += f"{lab_test_id_to_name[test]}: {ids_test_set[4][test]}\n"
                ids_chat_history.append({'role': 'user', 'content': LAB_PROMPT.format(lab_result, available_actions_text)})
                ids_resources.remove('LAB')  
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request  
            
        elif 'LAB' in result:
            if 'LAB' in ids_action_history:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': 'You have already performed Laboratory Test. Please choose another action from [{}].'.format(available_actions_text)})
            else:
                ids_action_history.append('LAB')
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                if len(ids_test_set[4]) == 0:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    ids_chat_history.append({'role': 'user', 'content': 'No Laboratory Test. Please choose another action.'}) 
                else:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    # available_tests = ', '.join(list(ids_test_set[4].keys()))
                    ids_chat_history.append({'role': 'user', 'content': LAB_SPECIFIC_PROMPT})
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        elif 'IMAGE_s' in result:
            available_tests = ids_test_set[6]
            
            exam_name_lists = [exam for exam in available_tests]
            request_tests = specification_parser('IMAGE_s', response, exam_name_mapping,args.model_name)
            ids_image_event_request = [request_tests, response]
            valid_tests = []
            for i,test in enumerate(request_tests):
                append_modality = []
                for modal in test['Modality']:
                    if modal in UNIQUE_TO_BROAD_MODALITY:
                        append_modality += [UNIQUE_TO_BROAD_MODALITY[modal]]
                test['Modality'] += append_modality
                test['Modality'] = list(set(test['Modality']))
                request_tests[i] = test
            
            for test in request_tests:
                
                if test['Exam Name'].upper() in exam_name_lists:
                    valid_tests.append(test['Exam Name'].upper())
                else:
                    if len(test['Region']) == 0 or len(test['Modality']) == 0:
                        continue
                    for region in test['Region']:
                        for modal in test['Modality']:
                            # if modal in UNIQUE_TO_BROAD_MODALITY
                            if region in exam_spec_mapping and modal in exam_spec_mapping[region]:
                                candidate_exams = exam_spec_mapping[region][modal]
                                matched_exams = list(set(candidate_exams) & set(exam_name_lists))
                                if len(matched_exams) > 0:
                                    valid_tests += matched_exams
            valid_tests = list(set(valid_tests))    
            if len(valid_tests) == 0:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_EXAM_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            elif 'IMAGE' not in ids_resources:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_ACTION_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            else:
                assert ids_chat_history[-1]['role'] == 'assistant'
                image_result = ''
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                for test in valid_tests:
                    if test in ids_test_set[6]:
                        image_result += f"Exam Name: {test}\nContent: {ids_test_set[6][test]}\n\n"
                ids_chat_history.append({'role': 'user', 'content': IMAGE_PROMPT.format(image_result, available_actions_text)})
                ids_resources.remove('IMAGE')
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
            
        elif 'IMAGE' in result:
            if 'IMAGE' in ids_action_history:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': 'You have already performed Imaging Test. Please choose another action from [{}].'.format(available_actions_text)})
            else:
                ids_action_history.append('IMAGE')
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                if len(ids_test_set[6]) == 0:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    ids_chat_history.append({'role': 'user', 'content': 'No Imaging Test. Please choose another action.'}) 
                else:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    # available_tests = ', '.join(list(ids_test_set[6].keys()))
                    ids_chat_history.append({'role': 'user', 'content': IMAGE_SPECIFIC_PROMPT})
                    
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
                    # ids_chat_history.append({'role': 'user', 'content': IMAGE_PROMPT.format(ids_test_set[6],available_actions_text)})
        elif 'MICRO_s' in result:
            available_tests = list(ids_test_set[5].keys())
            request_tests = specification_parser('MICRO_s', response, exam_name_mapping,args.model_name)
            ids_micro_event_request = [request_tests, response]
            valid_tests = [test for test in request_tests if test in ids_test_set[5]]
            if len(valid_tests) == 0:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_EXAM_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            elif 'MICRO' not in ids_resources:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': WARNING_ACTION_PROMPT.format(available_actions_text)})
                ids_action_history.append("WRONG")
            else:
                assert ids_chat_history[-1]['role'] == 'assistant'
                micro_result = ''
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                for test in valid_tests:
                    if test in ids_test_set[5]:
                        micro_result += f"{itemid_to_name[test]}: {ids_test_set[5][test]}\n"
                ids_chat_history.append({'role': 'user', 'content': MICRO_PROMPT.format(micro_result, available_actions_text)})
                ids_resources.remove('MICRO')
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        elif 'MICRO' in result:
            if 'MICRO' in ids_action_history:
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                assert ids_chat_history[-1]['role'] == 'assistant'
                ids_chat_history.append({'role': 'user', 'content': 'You have already performed Microbiology Test. Please choose another action from [{}].'.format(available_actions_text)})
            else:
                ids_action_history.append('MICRO')
                tmp_actions = ids_action_history
                available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
                available_actions_text = ', '.join(available_actions)
                if len(ids_test_set[5]) == 0:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    ids_chat_history.append({'role': 'user', 'content': 'No Microbiology Test. Please choose another action.'}) 
                else:
                    assert ids_chat_history[-1]['role'] == 'assistant'
                    # available_tests = ', '.join(list(ids_test_set[5].keys()))
                    ids_chat_history.append({'role': 'user', 'content': MICRO_SPECIFIC_PROMPT})
                    # ids_chat_history.append({'role': 'user', 'content': MICRO_PROMPT.format(ids_test_set[5],available_actions_text)})
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        elif 'OUTPUT' in result:
            assert ids_chat_history[-1]['role'] == 'assistant'
            ids_chat_history.append({'role': 'user', 'content': DIAGNOSIS_PROMPT})
            ids_action_history.append('OUTPUT')
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request
        else:
            tmp_actions = ids_action_history
            available_actions = [action for action in ACTION_LIST if action not in tmp_actions]
            available_actions_text = ', '.join(available_actions)
            assert ids_chat_history[-1]['role'] == 'assistant'
            ids_chat_history.append({'role': 'user', 'content': WARNING_ACTION_PROMPT.format(available_actions_text)})
            ids_action_history.append("WRONG")
            return finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request

def main(args):
    # 1. Load Model
    random.seed(args.seed)
    model = LLM(model=args.model,tensor_parallel_size=args.num_cuda, gpu_memory_utilization=args.util,max_model_len=args.max_length,trust_remote_code=True, swap_space=32, max_num_seqs=args.max_num_seqs)
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    # 2. Load Data
    base_new = "/hdd/zhouyx/mimic-clinical-decision-making"
    all_data = pickle.load(
            open(os.path.join(base_new, f"final_filtered_rev_all_hadm_info_first_diag.pkl"), "rb"),
        )
    
    microbiologyevents = pd.read_csv(os.path.join('/hdd/zhouyx/mimic-iv-2.2/hosp', 'microbiologyevents.csv'))
    itemid_to_name = microbiologyevents.set_index('test_itemid')['test_name'].to_dict()
    lab_test_mapping = pickle.load(open('/hdd/zhouyx/mimic-iv-2.2/hosp/lab_test_mapping.pkl','rb'))
    lab_test_id_to_name = lab_test_mapping.set_index('itemid')['label'].to_dict()
    exam_name_mapping = json.load(open('radiology_name_dict.json','r'))
    exam_spec_mapping = json.load(open('radiology_spec_dict.json','r'))
    for region in exam_spec_mapping:
        for modality in exam_spec_mapping[region]:
            if modality in UNIQUE_TO_BROAD_MODALITY:
                exam_spec_mapping[region][UNIQUE_TO_BROAD_MODALITY[modality]] += exam_spec_mapping[region][modality]
                exam_spec_mapping[region][UNIQUE_TO_BROAD_MODALITY[modality]] = list(set(exam_spec_mapping[region][UNIQUE_TO_BROAD_MODALITY[modality]]))
                
    test_set = []
    for disease in tqdm(all_data):
        for _id in all_data[disease]:
            present_illness = all_data[disease][_id]['Patient History']
            physical_examination = all_data[disease][_id]['Physical Examination']
            lab_events = all_data[disease][_id]['Laboratory Tests']
            # lab_event_text = ''
                # lab_event_text += f"{lab_test_id_to_name[lab_event]}: {lab_events[lab_event]}\n"
            microbiology_events = all_data[disease][_id]['Microbiology']
            radiology_report = {one['Exam Name']:one for one in all_data[disease][_id]['Radiology']}
            # radiology_report_dict = {}
            # for i, radiology in enumerate(radiology_report):
            #     radiology_report_dict[radiology['Exam Name']] = radiology['Report']
            test_set.append([disease, _id, present_illness, physical_examination, lab_events, microbiology_events, radiology_report])
    print("Data Loaded")
    pool = Pool(args.num_workers)
    # 3. Chatting
    stack = []
    chat_history = {}
    lab_event_request = {}
    micro_event_request = {}
    image_event_request = {}
    
    action_history = {}
    valid_ids = []
    virtual_id_to_real_id = {}
    # initial prefill
    for i, item in enumerate(test_set):
        if args.no_system:
            msg = [
                {'role': 'user', 'content': SYSTEM_PROMPT+START_PROMPT.format(item[2])}
            ]
        else:
            msg = [
                {'role': 'system', 'content': "You are a helpful medical AI assistant."},
                {'role': 'user', 'content': SYSTEM_PROMPT+START_PROMPT.format(item[2])}
            ]
        chat_history[i] = msg
        valid_ids.append(i)
        # virtual_id_to_real_id[i*args.repeat+j] = i
            
    results = {}
    turn = 0
    # sampled_valid_ids = random.sample(valid_ids, 100)
    # valid_ids = sampled_valid_ids
    resources = {}
    for ids in valid_ids:
        action_history[ids] = []
        resources[ids] = ['PE', 'LAB', 'IMAGE', 'MICRO']
    # valid_ids = [i for i in range(len(test_set)) if test_set[i][0] == 'appendicitis']
    # valid_ids = random.sample(valid_ids, 100)
    last_len_ids = len(valid_ids)
    patience = 0
    while len(valid_ids) > 0:
        if turn > 10:
            if len(valid_ids) == last_len_ids:
                patience += 1
                if patience > 3 or turn > 30:
                    print("Early Stopping")
                    for ids in valid_ids:
                        chat_history[ids].append({'role': 'assistant', 'content': "Early Stopping"})
                        results[ids] = ["failed", "failed"]
                        # valid_ids.remove(ids)
                    break
            else:
                patience = 0
                if turn > 30:
                    print("Early Stopping")
                    for ids in valid_ids:
                        chat_history[ids].append({'role': 'assistant', 'content': "Early Stopping"})
                        results[ids] = ["failed", "failed"]
                        # valid_ids.remove(ids)
                    break
        last_len_ids = len(valid_ids)
        print("Turn {} Remaining: {}".format(turn, len(valid_ids)))
        input_texts = []
        summarize_ids = []
        summarized_failed_ids = []
        for ids in valid_ids:
            msg_text = apply_chat_template(chat_history[ids], tokenizer, args.custom_template)
            
            tokens = tokenizer.tokenize(msg_text, return_tensors="pt")
            if len(tokens) > args.max_length - 100:
                # print("Too long")
                tmp_num = 2 if not args.no_system else 1
                if len(chat_history[ids]) == tmp_num:
                    recent_input = chat_history[ids][-1]['content']
                    while len(tokens) > args.max_length-100:
                        try:
                            assert 'Please choose your next action' in recent_input
                            instruct_index = recent_input.index("Please choose your next action")
                            assert instruct_index - 50 > 0
                        except Exception:
                            summarized_failed_ids.append(ids)
                        recent_input = recent_input[:instruct_index-50]+recent_input[instruct_index:]
                        if args.no_system:
                            chat_history[ids] = [
                                {'role': 'user', 'content': recent_input},
                            ]
                        else:
                            chat_history[ids] = [
                                {'role': 'system', 'content': "You are a helpful medical AI assistant."},
                                {'role': 'user', 'content': recent_input},
                                ]
                        msg_text = apply_chat_template(chat_history[ids], tokenizer, args.custom_template)
                        tokens = tokenizer.tokenize(msg_text, return_tensors="pt")
                    
                        
                else:
                    summarize_ids.append(ids)
                # continue
            input_texts.append(msg_text)
        summarize_prompt = []
        
        if len(summarize_ids) > 0:
            for ids in summarize_ids:
                old_history = chat_history[ids][:-1]    # ends with assistant
                start_text = old_history[1]['content']
                if "Now a patient comes to see the doctor." in start_text:
                    start_text = start_text[start_text.index("Now a patient comes to see the doctor."):]
                    old_history[1]['content'] = start_text
                old_history.append({'role': 'user', 'content': SUMMARY_PROMPT}) # ends with user
                tmp_input = apply_chat_template(old_history, tokenizer, args.custom_template)
                tokens = tokenizer.tokenize(tmp_input, return_tensors="pt")
                if len(tokens) > args.max_length-100:
                    
                    while len(tokens) > args.max_length-100:
                        # Decide which dialogue to truncate
                        max_len = 0
                        max_index = -1
                        for i in range(len(old_history)-1):
                            if old_history[i]['role'] == 'user' and len(old_history[i]['content']) > max_len and 'Please choose your next action' in old_history[i]['content']:
                                max_len = len(old_history[i]['content'])
                                max_index = i
                            elif old_history[i]['role'] == 'assistant' and len(old_history[i]['content']) > max_len:
                                max_len = len(old_history[i]['content'])
                                max_index = i
                        if max_index == -1:
                            summarized_failed_ids.append(ids)
                            tmp_input = 'failed'
                            break
                        recent_input = old_history[max_index]['content']  # last user input
                        if old_history[max_index]['role'] == 'assistant':
                            recent_input = recent_input[:-50]
                            old_history[max_index]['content'] = recent_input
                            tmp_input = apply_chat_template(old_history, tokenizer, args.custom_template)
                            tokens = tokenizer.tokenize(tmp_input, return_tensors="pt")
                        else:
                            instruct_index = recent_input.index("Please choose your next action")
                            # start truncate
                            if instruct_index - 50 <= 0:
                                summarized_failed_ids.append(ids)
                                break
                            recent_input = recent_input[:instruct_index-50]+recent_input[instruct_index:]
                            old_history[max_index]['content'] = recent_input
                            tmp_input = apply_chat_template(old_history, tokenizer, args.custom_template)
                            tokens = tokenizer.tokenize(tmp_input, return_tensors="pt")
                    # recent_input = old_history[-2]['content']
                    
                summarize_prompt.append(tmp_input)
        if len(summarize_prompt) > 0:
            print("Summarize")
            
            summarize = model.generate(summarize_prompt, SamplingParams(temperature=0.8, max_tokens=args.max_length))
            summarize = [response.outputs[0].text for response in summarize]
            for ids, response in zip(summarize_ids, summarize):
                recent_input = chat_history[ids][-1]['content']
                if args.no_system:
                    chat_history[ids] = [
                        {'role': 'user', 'content': SYSTEM_PROMPT+"Following are the crucial information of a patient: "+response+"\n"+recent_input},
                    ]
                else:
                    chat_history[ids] = [
                    {'role': 'system', 'content': "You are a helpful medical AI assistant."},
                    {'role': 'user', 'content': SYSTEM_PROMPT+"Following are the crucial information of a patient: "+response+"\n"+recent_input},
                    ]   # ends with user
                msg_text = apply_chat_template(chat_history[ids], tokenizer, args.custom_template)
                tokens = tokenizer.tokenize(msg_text, return_tensors="pt")
                truncate_summary_flag = False
                while len(tokens) > args.max_length-100:
                    if "Please choose your next action" not in recent_input:
                        truncate_summary_flag = True
                        
                    instruct_index = recent_input.index("Please choose your next action")
                    if instruct_index - 50 <= 0:
                        truncate_summary_flag = True
                    if not truncate_summary_flag:
                        recent_input = recent_input[:instruct_index-50]+recent_input[instruct_index:]
                    else:
                        if len(response) <= 50:
                            summarized_failed_ids.append(ids)
                            break
                        response = response[:-50]
                    if args.no_system:
                        chat_history[ids] = [
                            {'role': 'user', 'content': SYSTEM_PROMPT+"Following are the crucial information of a patient: "+response+"\n"+recent_input},
                        ]
                    else:
                        chat_history[ids] = [
                            {'role': 'system', 'content': "You are a helpful medical AI assistant."},
                            {'role': 'user', 'content': SYSTEM_PROMPT+"Following are the crucial information of a patient: "+response+"\n"+recent_input},
                            ]   # ends with user
                        
                    msg_text = apply_chat_template(chat_history[ids], tokenizer, args.custom_template)
                    tokens = tokenizer.tokenize(msg_text, return_tensors="pt")
                input_texts[valid_ids.index(ids)] = msg_text
        responses = model.generate(input_texts, SamplingParams(temperature=0.8, max_tokens=args.max_length))
        responses = [response.outputs[0].text for response in responses]
        new_ids = []
        print("START POSTPROCESSING...")
        start_time = time.time()
        batch_results = pool.starmap(post_processing, zip(valid_ids, responses, [summarized_failed_ids]*len(valid_ids), [summarize_ids]*len(valid_ids), [chat_history[ids] for ids in valid_ids], [action_history[ids] for ids in valid_ids], [resources[ids] for ids in valid_ids], [test_set[ids] for ids in valid_ids], [exam_name_mapping]*len(valid_ids), [exam_spec_mapping]*len(valid_ids), [lab_test_id_to_name]*len(valid_ids), [itemid_to_name]*len(valid_ids)))
        print("POSTPROCESSING DONE")
        print("Time: ", time.time()-start_time)
        for i, (finish, ids_chat_history, ids_action_history, ids_resources, result, ids_lab_event_request, ids_image_event_request, ids_micro_event_request) in enumerate(batch_results):
            ids = valid_ids[i]
            chat_history[ids] = ids_chat_history
            action_history[ids] = ids_action_history
            resources[ids] = ids_resources
            if len(ids_lab_event_request) > 0:
                lab_event_request[ids] = ids_lab_event_request
            if len(ids_image_event_request) > 0:
                image_event_request[ids] = ids_image_event_request
            if len(ids_micro_event_request) > 0:
                micro_event_request[ids] = ids_micro_event_request
            if finish:
                results[ids] = result
            else:
                new_ids.append(ids)
                
        turn += 1
        valid_ids = new_ids
    for ids in results:
        results[ids] = [test_set[ids][1]] + results[ids]
    if args.seed > 0:
        if not os.path.exists(f'results_part_{args.seed}'):
            os.makedirs(f'results_part_{args.seed}')
        pickle.dump(results, open(f"results_part_{args.seed}/results_{args.model_name}.pkl",'wb'))
        pickle.dump(chat_history, open(f"results_part_{args.seed}/chat_history_{args.model_name}.pkl",'wb'))
        pickle.dump({'lab': lab_event_request, 'micro': micro_event_request, 'image': image_event_request}, open(f"results_part_{args.seed}/event_request_{args.model_name}.pkl",'wb'))
    else:
        if not os.path.exists(f'results_part'):
            os.makedirs(f'results_part')
        pickle.dump(results, open(f"results_part/results_{args.model_name}.pkl",'wb'))
        pickle.dump(chat_history, open(f"results_part/chat_history_{args.model_name}.pkl",'wb'))
        pickle.dump({'lab': lab_event_request, 'micro': micro_event_request, 'image': image_event_request}, open(f"results_part/event_request_{args.model_name}.pkl",'wb'))
    # json.dump(buffer, open("lab_test_buffer.json",'w'))
                
            
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--nchain", "-c", type=int, default=5)
    parser.add_argument("--nbatch", "-b", type=int, default=-1)
    parser.add_argument("--typs", action="extend",nargs="+", type=str)
    parser.add_argument("--dataset",type=str,default='medqa')
    # parser.add_argument("--tokenizer_path", type=str, default="/home/zhouyx/llama2/vicuna_13B")
    parser.add_argument("--model", type=str, default='/hdd/zhouyx/llm/Llama-2-7b-chat-hf')
    parser.add_argument("--model_name", type=str, default='llama2-7B-it')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_cuda", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_system", action='store_true')
    parser.add_argument("--custom_template", type=str, default="")
    parser.add_argument("--util",type=float,default=0.98)
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--max_num_seqs",type=int,default=256)
    # parser.add_argument("--max_tokens",type=int, default=2048)
    # parser.add_argument("--subjects", type=list, default=['cat_qa','isa_qa1'])
    args = parser.parse_args()
    main(args)

