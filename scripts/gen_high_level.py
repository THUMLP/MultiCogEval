'''
    Script to generate high-level evaluation data from MIMIC-IV-2.2 dataset.
    We follow the same data preprocessing steps as in the MIMIC-IV-Ext Clinical Decision Making dataset, while we align the covered diseases with the low- and mid-level tasks.
'''

import os
import pandas as pd
from os.path import join
from dataset.dataset import load_data, extract_info_general
from tqdm import tqdm
import numpy as np
import json
import re
import time
import importlib
from tools.nlp import extract_primary_diagnosis
from tqdm import tqdm
from multiprocessing import Pool
import pickle

base_mimic = "data/mimic-iv/origin"
# base_mimic = "/hdd/zhouyx/mimic-iv-2.2"
MIMIC_hosp_base = join(base_mimic, "hosp")
(
    admissions_df,
    transfers_df,
    diag_icd,
    procedures_df,
    discharge_df,
    radiology_report_df,
    radiology_report_details_df,
    lab_events_df,
    microbiology_df,
) = load_data(base_mimic)


hadm_dict = json.load(open("data/mimic-iv/hadm_ids_dict.json", "r"))

start = time.time()
hadm_info_dict, hadm_info_clean_dict = {}, {}
def process_diseses(disease, hadm_ids):
    hadm_ids = [int(i) for i in hadm_ids]
    demo_hadm_info, demo_hadm_info_clean  = extract_info_general(
        hadm_ids,
        disease,
        [disease],
        discharge_df,
        admissions_df,
        transfers_df,
        lab_events_df,
        microbiology_df,
        radiology_report_df,
        radiology_report_details_df,
        diag_icd,
        procedures_df,
    )
    first_diag_ids = []
    for p in demo_hadm_info_clean:
        # if p in multi_diag_ids:
        #     continue
        dd = demo_hadm_info_clean[p]["Discharge Diagnosis"]
        dd = dd.lower()
        first_diag = extract_primary_diagnosis(dd)
        if first_diag and disease.lower() in first_diag.lower():
            first_diag_ids.append(p)
    hadm_info_firstdiag = {}
    for _id in first_diag_ids:
        hadm_info_firstdiag[_id] = demo_hadm_info_clean[_id]
    return hadm_info_firstdiag
pool = Pool(16)
# res = process_diseses("prostatitis", hadm_dict["prostatitis"])
# results = [res]
results = list(
            tqdm(pool.starmap(process_diseses, [(disease, hadm_dict[disease]) for disease in hadm_dict])))
hadm_info_clean_dict = {}
disease_list = list(hadm_dict.keys())
for i in range(len(disease_list)):
    hadm_info_clean_dict[disease_list[i]] = results[i]
base_new = "data/mimic-iv"
pickle.dump(
        hadm_info_clean_dict,
        open(join(base_new, f"all_hadm_info_first_diag.pkl"), "wb"),
    )