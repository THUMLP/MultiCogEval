import json
import pickle
import os
import random
import re
from utils import UNIQUE_TO_BROAD_MODALITY,UNIQUE_MODALITY_TO_ORGAN_MAPPING
import numpy as np
random.seed(0)

alternative_diagnosis = {
    "hepatitis B":['hep B','hep-B','HBV','Hep B','Hep-B','hepatitis B','Hepatitis B'],
    "prostatitis":['prostatitis','Prostatitis'],
    "type 1 diabetes mellitus":['T1DM','insulin-dependent diabetes','IDDM','Type 1 diabetes'],
    "Herpes zoster":['presumed  zoster','zoster','Zoster','shingles','Varicella-Zoster Virus','VZV'],
    'Pancreatitis':[],
    'infectious mononucleosis':['mononucleosis'],
    'lupus erythematosus':['SLE','lupus','Lupus'],
    "atrial fibrillation":['A-fib','Atrial fibrillation','afib','a-fib','A. fib','A. Fib','atrial  fibrillation','AF'],
    "Hepatitis C":['hep C','hep-C','HCV','Hep C','Hep-C','hepatitis C','Hepatitis C','hepatitic C'],
    "Subarachnoid hemorrhage":['SAH','Subarachnoid','sub-arachnoid hemorrhage',],
    "Hepatitis A":['hep A','hep-A','HAV','Hep A','Hep-A','hepatitis A','Hepatitis A','hepatitic A'],
    "Pelvic inflammatory disease":['PID','pelvic inflammatory'],
    "Hypothyroidism":[],
    "heart failure":['heart  failure','HFrEF','HFpEF','CHF','HF','dCHF'],
    "eosinophilia":[],
    "Adhesive capsulitis":['adhesive  capsulitis'],
    "Normal pressure hydrocephalus":['NPH'],
    "Esophageal varices":["esophageal  variceal",'esophageal/gastric varices','esophageal varix','Esophageal variceal','esophageal variceal'],
    "hypertension":["HTN",'cHTN','elevated blood pressure','high blood pressure'],
    "Pericarditis":["pericardial inflammation"],
    "hematuria":['blood in urine'],
    "Ventricular tachycardia":['VT'],
    "colitis":[['inflammation','colon']],
    "deficiency anemia":[['deficiency','anemia']],
    "bronchitis":[],
    "type 2 diabetes mellitus":['T2DM','non-insulin-dependent diabetes','NIDDM','Type 2 diabetes','NIDDM2'],
    "Sick sinus":[],
    "sleep apnea":["OSA"],
    "sinusitis":['sphenoiditis','Sinus infection'],
    "Sepsis":[],
    "cholecystitis":[["gallbladder","inflammation"]],
    "Myasthenia":[],
    "Irritable bowel syndrome":["IBS",'ibs'],
    "obstructive pulmonary disease":['COPD','copd'],
    "unstable angina":['UA'],
    "Orthostatic Hypotension":[],
    "tubular necrosis":[],
    "claudication":[],
    "carpal tunnel":[],
    "bacteremia":[['blood','infection']],
    "asthma":[],
    "Hemolytic anemia":[['hemolysis','anemia']],
    "multiple sclerosis":["secondary progressive MS",'RRMS','MS'],
    "osteoarthritis":['OA'],
    "Necrotizing fasciitis":[],
    "Lyme disease":['Lyme'],
    "rheumatoid arthritis":['RA'],
    "Hypocalcemia":["hypocalcemic"],
    "Cardiac tamponade":[['pericardial','tamponade'],['cardiac','tamponade']],
    "Polymyalgia rheumatica":["polymyalgia  rheumatica"],
    "Dilated cardiomyopathy":[],
    "appendicitis":[]
}


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
base_new = os.path.join(project_root,"data/mimic-iv")
mimic_path = 'data/mimic-iv/origin'  # Replace with your MIMIC-IV path


radiology_name_dict = json.load(open(os.path.join(project_root,'data/mimic-iv/radiology_name_dict.json'),'r'))
threshold = 0.1
def get_diagnosis_results(model_name):
    
    all_data = pickle.load(
            open(os.path.join(base_new, f"all_hadm_info_first_diag.pkl"), "rb"),
        )
    
    lab_test_mapping = pickle.load(open(os.path.join(mimic_path,'hosp','lab_test_mapping.pkl'),'rb'))
    lab_test_id_to_name = lab_test_mapping.set_index('itemid')['label'].to_dict()
    micro_acc_list, micro_acc_strict_list, macro_acc_list, macro_acc_strict_list = [],[],[],[]
    micro_recall_list, lab_recall_list, image_recall_list = [],[],[]
    all_disease_acc = {}
    all_disease_endpoint_acc = {}
    all_disease_ttl = {}
    for seed in os.listdir(os.path.join(project_root,'results','mimic-iv')):
        # if os.path.exists(os.path.join(project_root, "f"{dir_name}/chat_history_{model_name}.pkl"):
        try:
            chat_history = pickle.load(open(f"{project_root}/results/mimic-iv/{seed}/chat_history_{model_name}.pkl", "rb"))
            results = pickle.load(open(f"{project_root}/results/mimic-iv/{seed}/results_{model_name}.pkl", "rb"))
            event_request = pickle.load(open(f'{project_root}/results/mimic-iv/{seed}/event_request_{model_name}.pkl', 'rb'))
        except Exception:
            continue
        # else:
        #     continue
        data_dict = {}
        for disease in all_data:
            for _id in all_data[disease]:
                data_dict[_id] = all_data[disease][_id]
                data_dict[_id]['disease'] = disease
        synonyms = {}
        acc = 0
        ttl = 0
        failed_samples = {}
        acc_dict = {}
        ttl_dict = {}
        acc_strict_dict = {}
        acc_strict = 0
        # item = results[2602]
        multi_diagnosis = 0
        disease_count = {}
        results_dict = {}
        for ids in results:
            item = results[ids]
            _id = item[0]
            results_dict[_id] = ids
        
        for _id in data_dict:
            ids = results_dict[_id]
            item = results[ids]
            diagnosis = item[1]
            # if item[1] == 'failed':
            #     continue
            # else:
            #     print(chat_history[ids])
            label = data_dict[_id]['disease']
            final_output = chat_history[ids][-1]['content']
            discharge = data_dict[_id]['Discharge']
            discharge_diagnosis = data_dict[_id]['Discharge Diagnosis']
            lab_events, micro_events, image_events = data_dict[_id]['Laboratory Tests'], data_dict[_id]['Microbiology'], data_dict[_id]['Radiology']
            lab_events_dict = {lab_test_id_to_name[key]: lab_events[key] for key in lab_events}
            requested_lab_events = []
            if ids in event_request['lab']:
                requested_lab_events = event_request['lab'][ids][0]
            requested_micro_events = []
            if ids in event_request['micro']:
                requested_micro_events = event_request['micro'][ids][0]
            requested_image_events = []
            if ids in event_request['image']:
                requested_image_events = event_request['image'][ids][0]
            # print(requested_lab_events)
            
            lab_events_list = list(lab_events.keys())
            micro_events_list = list(micro_events.keys())
            # print(len(set(requested_lab_events) & set(lab_events_list)), len(set(requested_micro_events) & set(micro_events_list)))
            # if len(requested_lab_events) == 0 and len(requested_micro_events) == 0:
            
            #     lab_precision = len(set(requested_lab_events) & set(lab_events_list))/(len(requested_lab_events)+1e-10)
            #     lab_recall = len(set(requested_lab_events) & set(lab_events_list))/(len(lab_events_list)+1e-10)
            
            recall_list = []
            
            if len(lab_events_list) == 0:
                lab_recall = 1
            else:
                lab_recall = len(set(requested_lab_events) & set(lab_events_list))/(len(lab_events_list)+1e-10)
                recall_list.append(lab_recall)
                lab_recall_list.append(lab_recall)
                
            if len(micro_events_list) == 0:
                micro_recall = 1
            else:
                micro_recall = len(set(requested_micro_events) & set(micro_events_list))/(len(micro_events_list)+1e-10)
                recall_list.append(micro_recall)
                micro_recall_list.append(micro_recall)
                
            
            image_hit = 0
            image_events_list = []
            for exam in image_events:
                exam_name = exam['Exam Name']
                if exam_name in radiology_name_dict:
                    regions = radiology_name_dict[exam_name]['regions']
                    modalities = radiology_name_dict[exam_name]['modalities']
                image_events_list.append([regions, modalities])
                # print(region, modality)
            for item in image_events_list:
                regions, modalities = item
                for exam in requested_image_events:
                    req_region = exam['Region']
                    req_modalities = exam['Modality']
                    append_modals = []
                    append_regions = []
                    for modal in req_modalities:
                        if modal in UNIQUE_TO_BROAD_MODALITY:
                            append_modals.append(UNIQUE_TO_BROAD_MODALITY[modal])
                            if modal in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
                                req_region.append(UNIQUE_MODALITY_TO_ORGAN_MAPPING[modal])
                        else:
                            append_modals.append(modal)
                        
                    req_modalities = append_modals
                    if len(set(req_region) & set(regions)) > 0 and len(set(req_modalities) & set(modalities)) > 0:
                        image_hit += 1
                        break
            if len(image_events_list) == 0:
                image_recall = 1
            else:
                image_recall = image_hit/len(image_events_list)
                recall_list.append(image_recall)
                image_recall_list.append(image_recall)
            # if len(re)
            
            possible_labels = []
            # for disease in all_data:
                # if disease.lower() in discharge_diagnosis.lower():
                #     possible_labels.append(disease)
            # if len(possible_labels) > 1:
            #     print(possible_labels)
            #     multi_diagnosis += 1
            #     continue
            # else:
            #     if label not in disease_count:
            #         disease_count[label] = 0
            #     disease_count[label] += 1
            # if 'differential diagnosis' not in discharge.lower():
            #     print('y')
            
            # if len(data_dict[_id]['Microbiology']) == 0 or len(data_dict[_id]['Laboratory Tests']) == 0 or len(data_dict[_id]['Radiology']) == 0 or len(data_dict[_id]['Physical Examination']) == 0:
            #     continue
                
                # label = 'appendicitis'
            pattern = r"diagnosis:(.*?)(?:treatment:|treatment plan:|\Z)"
            # possible_diags = []
            # for disease in all_data:
            #     if disease.lower() in final_output.lower():
            #         possible_diags.append(disease)
            try:
                diagnosis = re.search(pattern, final_output, re.IGNORECASE|re.DOTALL).group(1)
                diagnosis = diagnosis.strip()
            except:
                diagnosis = ''
            # possible_diags = []
            # for disease in all_data:
            #     if disease.lower() in final_output.lower():
            #         possible_diags.append(disease)
            # if len(possible_diags) >1:
            #     print(possible_diags)
                # '# Pulmonary Hypertension\n# Hypertensive emergency\n# New cardiomyopathy with acute decompensated diastolic heart \nfailure'
            # if label == 'Hepatitis C':
            #     print(diagnosis)
            if label not in acc_dict:
                acc_dict[label] = 0
                ttl_dict[label] = 0
                acc_strict_dict[label] = 0
                
            # if len(recall_list) == 0:   # No lab, micro, image events
            #     complete_flag = True
            # else:
            #     if any([recall < threshold for recall in recall_list]): # All the recalls are below threshold
            #         complete_flag = False
            #     else:
            #         complete_flag = True
            
            if len(recall_list) == 0:
                complete_index = 1
            else:
                complete_index = sum(recall_list)/len(recall_list)  # AVG Recall
            
            # if lab_recall < threshold and micro_recall < threshold and image_recall < threshold:
            #     complete_flag = False
            # else:
            #     complete_flag = True
            
            if label.lower() not in diagnosis.lower():
                for item in alternative_diagnosis[label]:
                    if isinstance(item, list):
                        flag = True
                        for one in item:
                            pattern = re.compile(r"\b" + re.escape(one) + r"\b", re.IGNORECASE)
                            if not pattern.search(diagnosis):
                                flag = False
                                break
                        if flag:
                            acc += 1
                            acc_dict[label] += 1
                            # if complete_flag:
                            acc_strict += complete_index
                            acc_strict_dict[label] += complete_index
                            break
                    else:
                        pattern = re.compile(r"\b" + re.escape(item) + r"\b", re.IGNORECASE)
                        res = pattern.search(diagnosis)
                        if res:
                            acc += 1
                            acc_dict[label] += 1
                            # if complete_flag:
                            acc_strict += complete_index
                            acc_strict_dict[label] += complete_index
                            break
                # print(label, diagnosis)
                # if label == 'heart failure':
                #     history = chat_history[ids]
                #     print(label, diagnosis)
                #     print(data_dict[_id]['Discharge Diagnosis'])
                #     print('\n\n')
                if label not in failed_samples:
                    failed_samples[label] = []
                failed_samples[label].append(diagnosis)
            else:
                acc += 1
                acc_dict[label] += 1
                # if complete_flag:
                acc_strict += complete_index
                acc_strict_dict[label] += complete_index
            
            ttl += 1
            ttl_dict[label] += 1
        # for disease in failed_samples:
        #     print(disease)
        #     samples = random.sample(failed_samples[disease], 5)
        #     for i in range(5):
        #         print(samples[i])
        # print(acc/ttl)
        macro = 0
        macro_strict = 0
        for disease in all_data:
            macro += acc_dict[disease]/ttl_dict[disease]
            macro_strict += acc_strict_dict[disease]/ttl_dict[disease]
            all_disease_acc[disease] = all_disease_acc.get(disease, 0) + acc_strict_dict[disease]
            all_disease_ttl[disease] = all_disease_ttl.get(disease, 0) + ttl_dict[disease]
            all_disease_endpoint_acc[disease] = all_disease_endpoint_acc.get(disease, 0) + acc_dict[disease]
        macro /= len(all_data)
        macro_strict /= len(all_data)
        micro_acc_list.append(acc/ttl)
        micro_acc_strict_list.append(acc_strict/ttl)
        macro_acc_list.append(macro)
        macro_acc_strict_list.append(macro_strict)
        
    
    se_micro =  np.std(micro_acc_list, ddof=1) / np.sqrt(len(micro_acc_list))
    se_micro_strict =  np.std(micro_acc_strict_list, ddof=1) / np.sqrt(len(micro_acc_strict_list))
    se_macro =  np.std(macro_acc_list, ddof=1) / np.sqrt(len(macro_acc_list))
    se_macro_strict =  np.std(macro_acc_strict_list, ddof=1) / np.sqrt(len(macro_acc_strict_list))
    all_disease_acc = {key: all_disease_acc[key]/all_disease_ttl[key] for key in all_disease_acc}
    all_disease_endpoint_acc = {key: all_disease_endpoint_acc[key]/all_disease_ttl[key] for key in all_disease_endpoint_acc}
    # print(f"{model_name}: {np.mean(micro_acc_list):.3f} ({se_micro:.3f}), {np.mean(micro_acc_strict_list):.3f} ({se_micro_strict:.3f}), {np.mean(macro_acc_list):.3f} ({se_macro:.3f}), {np.mean(macro_acc_strict_list):.3f} ({se_macro_strict:.3f})")
    return [np.mean(micro_acc_list), np.mean(micro_acc_strict_list), np.mean(macro_acc_list), np.mean(macro_acc_strict_list), se_micro, se_micro_strict, se_macro, se_macro_strict, len(micro_acc_list), np.mean(lab_recall_list), np.mean(micro_recall_list), np.mean(image_recall_list),all_disease_acc,all_disease_endpoint_acc]
        # print(model_name,acc/ttl, macro)
        # print(acc_strict/ttl, macro_strict)

def get_diagnosis_results_selected(model_name, selected_ids):
    if model_name in ["gpt-4o","gpt-4o_T0.4",'deepseek-V3','deepseek-R1','o3-mini_medium','o1_medium','o3-mini_low','o3-mini_high']:
        all_data = pickle.load(
            open(os.path.join(base_new, f"sampled_final_filtered_rev_all_hadm_info_first_diag.pkl"), "rb"),
        )
    else:
        all_data = pickle.load(
                open(os.path.join(base_new, f"final_filtered_rev_all_hadm_info_first_diag.pkl"), "rb"),
            )
    selected_all_data = {}
    for disease in all_data:
        selected_all_data[disease] = {}
        for _id in all_data[disease]:
            if _id in selected_ids:
                selected_all_data[disease][_id] = all_data[disease][_id]
    new_dict = {}
    for disease in selected_all_data:
        if len(selected_all_data[disease]) > 0:
            new_dict[disease] = selected_all_data[disease]
    
    all_data = new_dict
    lab_test_mapping = pickle.load(open(f'{mimic_path}/hosp/lab_test_mapping.pkl','rb'))
    lab_test_id_to_name = lab_test_mapping.set_index('itemid')['label'].to_dict()
    micro_acc_list, micro_acc_strict_list, macro_acc_list, macro_acc_strict_list = [],[],[],[]
    micro_recall_list, lab_recall_list, image_recall_list = [],[],[]
    all_disease_acc = {}
    all_disease_endpoint_acc = {}
    all_disease_ttl = {}
    for seed in os.listdir(os.path.join(project_root,'results','mimic-iv')):
        try:
            chat_history = pickle.load(open(f"{project_root}/results/mimic-iv/{seed}/chat_history_{model_name}.pkl", "rb"))
            results = pickle.load(open(f"{project_root}/results/mimic-iv/{seed}/results_{model_name}.pkl", "rb"))
            event_request = pickle.load(open(f'{project_root}/results/mimic-iv/{seed}/event_request_{model_name}.pkl', 'rb'))
        except Exception:
            continue
        data_dict = {}
        for disease in all_data:
            for _id in all_data[disease]:
                data_dict[_id] = all_data[disease][_id]
                data_dict[_id]['disease'] = disease
        synonyms = {}
        acc = 0
        ttl = 0
        failed_samples = {}
        acc_dict = {}
        ttl_dict = {}
        acc_strict_dict = {}
        acc_strict = 0
        # item = results[2602]
        multi_diagnosis = 0
        disease_count = {}
        results_dict = {}
        for ids in results:
            item = results[ids]
            _id = item[0]
            results_dict[_id] = ids
        
        for _id in data_dict:
            ids = results_dict[_id]
            item = results[ids]
            diagnosis = item[1]
            # if item[1] == 'failed':
            #     continue
            # else:
            #     print(chat_history[ids])
            label = data_dict[_id]['disease']
            final_output = chat_history[ids][-1]['content']
            discharge = data_dict[_id]['Discharge']
            discharge_diagnosis = data_dict[_id]['Discharge Diagnosis']
            lab_events, micro_events, image_events = data_dict[_id]['Laboratory Tests'], data_dict[_id]['Microbiology'], data_dict[_id]['Radiology']
            lab_events_dict = {lab_test_id_to_name[key]: lab_events[key] for key in lab_events}
            requested_lab_events = []
            if ids in event_request['lab']:
                requested_lab_events = event_request['lab'][ids][0]
            requested_micro_events = []
            if ids in event_request['micro']:
                requested_micro_events = event_request['micro'][ids][0]
            requested_image_events = []
            if ids in event_request['image']:
                requested_image_events = event_request['image'][ids][0]
            # print(requested_lab_events)
            
            lab_events_list = list(lab_events.keys())
            micro_events_list = list(micro_events.keys())
            # print(len(set(requested_lab_events) & set(lab_events_list)), len(set(requested_micro_events) & set(micro_events_list)))
            # if len(requested_lab_events) == 0 and len(requested_micro_events) == 0:
            
            #     lab_precision = len(set(requested_lab_events) & set(lab_events_list))/(len(requested_lab_events)+1e-10)
            #     lab_recall = len(set(requested_lab_events) & set(lab_events_list))/(len(lab_events_list)+1e-10)
            
            recall_list = []
            
            if len(lab_events_list) == 0:
                lab_recall = 1
            else:
                lab_recall = len(set(requested_lab_events) & set(lab_events_list))/(len(lab_events_list)+1e-10)
                recall_list.append(lab_recall)
                lab_recall_list.append(lab_recall)
                
            if len(micro_events_list) == 0:
                micro_recall = 1
            else:
                micro_recall = len(set(requested_micro_events) & set(micro_events_list))/(len(micro_events_list)+1e-10)
                recall_list.append(micro_recall)
                micro_recall_list.append(micro_recall)
                
            
            image_hit = 0
            image_events_list = []
            for exam in image_events:
                exam_name = exam['Exam Name']
                if exam_name in radiology_name_dict:
                    regions = radiology_name_dict[exam_name]['regions']
                    modalities = radiology_name_dict[exam_name]['modalities']
                image_events_list.append([regions, modalities])
                # print(region, modality)
            for item in image_events_list:
                regions, modalities = item
                for exam in requested_image_events:
                    req_region = exam['Region']
                    req_modalities = exam['Modality']
                    append_modals = []
                    append_regions = []
                    for modal in req_modalities:
                        if modal in UNIQUE_TO_BROAD_MODALITY:
                            append_modals.append(UNIQUE_TO_BROAD_MODALITY[modal])
                            if modal in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
                                req_region.append(UNIQUE_MODALITY_TO_ORGAN_MAPPING[modal])
                        else:
                            append_modals.append(modal)
                        
                    req_modalities = append_modals
                    if len(set(req_region) & set(regions)) > 0 and len(set(req_modalities) & set(modalities)) > 0:
                        image_hit += 1
                        break
            if len(image_events_list) == 0:
                image_recall = 1
            else:
                image_recall = image_hit/len(image_events_list)
                recall_list.append(image_recall)
                image_recall_list.append(image_recall)
            # if len(re)
            
            possible_labels = []
            # for disease in all_data:
                # if disease.lower() in discharge_diagnosis.lower():
                #     possible_labels.append(disease)
            # if len(possible_labels) > 1:
            #     print(possible_labels)
            #     multi_diagnosis += 1
            #     continue
            # else:
            #     if label not in disease_count:
            #         disease_count[label] = 0
            #     disease_count[label] += 1
            # if 'differential diagnosis' not in discharge.lower():
            #     print('y')
            # diagnosis = item[1]
            # if len(data_dict[_id]['Microbiology']) == 0 or len(data_dict[_id]['Laboratory Tests']) == 0 or len(data_dict[_id]['Radiology']) == 0 or len(data_dict[_id]['Physical Examination']) == 0:
            #     continue
                
                # label = 'appendicitis'
            pattern = r"diagnosis:(.*?)(?:treatment:|treatment plan:|\Z)"
            # possible_diags = []
            # for disease in all_data:
            #     if disease.lower() in final_output.lower():
            #         possible_diags.append(disease)
            try:
                diagnosis = re.search(pattern, final_output, re.IGNORECASE|re.DOTALL).group(1)
                diagnosis = diagnosis.strip()
            except:
                diagnosis = ''
            # possible_diags = []
            # for disease in all_data:
            #     if disease.lower() in final_output.lower():
            #         possible_diags.append(disease)
            # if len(possible_diags) >1:
            #     print(possible_diags)
                # '# Pulmonary Hypertension\n# Hypertensive emergency\n# New cardiomyopathy with acute decompensated diastolic heart \nfailure'
            # if label == 'Hepatitis C':
            #     print(diagnosis)
            if label not in acc_dict:
                acc_dict[label] = 0
                ttl_dict[label] = 0
                acc_strict_dict[label] = 0
                
            # if len(recall_list) == 0:   # No lab, micro, image events
            #     complete_flag = True
            # else:
            #     if any([recall < threshold for recall in recall_list]): # All the recalls are below threshold
            #         complete_flag = False
            #     else:
            #         complete_flag = True
            
            if len(recall_list) == 0:
                complete_index = 1
            else:
                complete_index = sum(recall_list)/len(recall_list)  # AVG Recall
            
            # if lab_recall < threshold and micro_recall < threshold and image_recall < threshold:
            #     complete_flag = False
            # else:
            #     complete_flag = True
            
            if label.lower() not in diagnosis.lower():
                for item in alternative_diagnosis[label]:
                    if isinstance(item, list):
                        flag = True
                        for one in item:
                            pattern = re.compile(r"\b" + re.escape(one) + r"\b", re.IGNORECASE)
                            if not pattern.search(diagnosis):
                                flag = False
                                break
                        if flag:
                            acc += 1
                            acc_dict[label] += 1
                            # if complete_flag:
                            acc_strict += complete_index
                            acc_strict_dict[label] += complete_index
                            break
                    else:
                        pattern = re.compile(r"\b" + re.escape(item) + r"\b", re.IGNORECASE)
                        res = pattern.search(diagnosis)
                        if res:
                            acc += 1
                            acc_dict[label] += 1
                            # if complete_flag:
                            acc_strict += complete_index
                            acc_strict_dict[label] += complete_index
                            break
                # print(label, diagnosis)
                # if label == 'heart failure':
                #     history = chat_history[ids]
                #     print(label, diagnosis)
                #     print(data_dict[_id]['Discharge Diagnosis'])
                #     print('\n\n')
                if label not in failed_samples:
                    failed_samples[label] = []
                failed_samples[label].append(diagnosis)
            else:
                acc += 1
                acc_dict[label] += 1
                # if complete_flag:
                acc_strict += complete_index
                acc_strict_dict[label] += complete_index
            
            ttl += 1
            ttl_dict[label] += 1
        # for disease in failed_samples:
        #     print(disease)
        #     samples = random.sample(failed_samples[disease], 5)
        #     for i in range(5):
        #         print(samples[i])
        # print(acc/ttl)
        macro = 0
        macro_strict = 0
        for disease in all_data:
            macro += acc_dict[disease]/ttl_dict[disease]
            macro_strict += acc_strict_dict[disease]/ttl_dict[disease]
            all_disease_acc[disease] = all_disease_acc.get(disease, 0) + acc_strict_dict[disease]
            all_disease_ttl[disease] = all_disease_ttl.get(disease, 0) + ttl_dict[disease]
            all_disease_endpoint_acc[disease] = all_disease_endpoint_acc.get(disease, 0) + acc_dict[disease]
        macro /= len(all_data)
        macro_strict /= len(all_data)
        micro_acc_list.append(acc/ttl)
        micro_acc_strict_list.append(acc_strict/ttl)
        macro_acc_list.append(macro)
        macro_acc_strict_list.append(macro_strict)
        
    
    se_micro =  np.std(micro_acc_list, ddof=1) / np.sqrt(len(micro_acc_list))
    se_micro_strict =  np.std(micro_acc_strict_list, ddof=1) / np.sqrt(len(micro_acc_strict_list))
    se_macro =  np.std(macro_acc_list, ddof=1) / np.sqrt(len(macro_acc_list))
    se_macro_strict =  np.std(macro_acc_strict_list, ddof=1) / np.sqrt(len(macro_acc_strict_list))
    all_disease_acc = {key: all_disease_acc[key]/all_disease_ttl[key] for key in all_disease_acc}
    all_disease_endpoint_acc = {key: all_disease_endpoint_acc[key]/all_disease_ttl[key] for key in all_disease_endpoint_acc}
    return [np.mean(micro_acc_list), np.mean(micro_acc_strict_list), np.mean(macro_acc_list), np.mean(macro_acc_strict_list), se_micro, se_micro_strict, se_macro, se_macro_strict, len(micro_acc_list), np.mean(lab_recall_list), np.mean(micro_recall_list), np.mean(image_recall_list),all_disease_acc,all_disease_endpoint_acc]
    
if __name__ == "__main__":
    model_name = 'llama3-8B'
    results = get_diagnosis_results(model_name)
    for disease in results[-1]:
        print("{}/{}".format(disease,results[-1][disease]))
