'''
    Extract answer choice from the plain text with the re package
'''
import re
import os
import json
import random
import pandas as pd
import numpy as np
random.seed(48)
def extract_ans(response_str):
    if len(response_str) == 0:
        return None
    response_str_clean = response_str.strip().lower()
    if response_str_clean.startswith('true'):
        return 'T'
    if response_str_clean.startswith('false'):
        return 'F'
    # if response_str == 'false':
    #     return 'F'
    # if response_str == '是的':
    #     return 'T'
    # for one in ['wrong']:
    #     if response_str.startswith(one):
    #         return 'F'
    # for one in ['正确']:
    #     if response_str.startswith(one):
    #         return 'T'
    for one in ['incorrect','not correct','false','False','No','wrong']:
        if one in response_str:
            return 'F'
    for one in ['true','correct','consistent','True','Yes']:
        if one in response_str:
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return None



def get_SV_results(dataset, name, digit=5):
    # result_dir = 'results/{}_repeat/SV'.format(dataset)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    result_dir = os.path.join(project_root, 'results/{}_repeat/SV'.format(dataset))
    full_name = os.path.join(result_dir, name+'_SV_results.json')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    inconsistent = 0
    acc_list = np.zeros((digit))
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if dataset == 'medqa':
                pos_T_ques = entry[1]
                neg_T_ques = entry[2]
                pos_F_ques = entry[3]
                neg_F_ques = entry[4]
                pos_T_ans = entry[5]
                neg_T_ans = entry[6]
                pos_F_ans = entry[7]
                neg_F_ans = entry[8]
                pos_T_replys = entry[-digit*4:-digit*3]
                neg_T_replys = entry[-digit*3:-digit*2]
                pos_F_replys = entry[-digit*2:-digit]
                neg_F_replys = entry[-digit:]
            else:
                pos_T_ans = entry[3]
                pos_F_ans = entry[4]
                pos_T_replys = entry[5:digit+5]
                pos_F_replys = entry[digit+5:digit*2+5]
            pos_T_pred = [extract_ans(pos_T_reply) for pos_T_reply in pos_T_replys]
            pos_F_pred = [extract_ans(pos_F_reply) for pos_F_reply in pos_F_replys]
            acc_list += np.array([int(pred==pos_T_ans) for pred in pos_T_pred])
            # acc_list += np.array([int(pred==neg_T_ans) for pred in neg_T_pred])
            acc_list += np.array([int(pred==pos_F_ans) for pred in pos_F_pred])
            # acc_list += np.array([int(pred==neg_F_ans) for pred in neg_F_pred])
            ttl += 2
    acc_list /= ttl
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    return acc_list.mean(), se, acc_list

def get_SV_results_selected(dataset, name, digit=5, selected_list=None):
    result_dir = 'results/{}_repeat/SV'.format(dataset)
    full_name = os.path.join(result_dir, name+'_SV_results.json')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    inconsistent = 0
    acc_list = np.zeros((digit))
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            idx = entry[0]
            if idx not in selected_list:
                continue
            if dataset == 'medqa':
                pos_T_ques = entry[1]
                neg_T_ques = entry[2]
                pos_F_ques = entry[3]
                neg_F_ques = entry[4]
                pos_T_ans = entry[5]
                neg_T_ans = entry[6]
                pos_F_ans = entry[7]
                neg_F_ans = entry[8]
                pos_T_replys = entry[-digit*4:-digit*3]
                neg_T_replys = entry[-digit*3:-digit*2]
                pos_F_replys = entry[-digit*2:-digit]
                neg_F_replys = entry[-digit:]
            else:
                pos_T_ans = entry[3]
                pos_F_ans = entry[4]
                pos_T_replys = entry[5:digit+5]
                pos_F_replys = entry[digit+5:digit*2+5]
            pos_T_pred = [extract_ans(pos_T_reply) for pos_T_reply in pos_T_replys]
            pos_F_pred = [extract_ans(pos_F_reply) for pos_F_reply in pos_F_replys]
            acc_list += np.array([int(pred==pos_T_ans) for pred in pos_T_pred])
            # acc_list += np.array([int(pred==neg_T_ans) for pred in neg_T_pred])
            acc_list += np.array([int(pred==pos_F_ans) for pred in pos_F_pred])
            # acc_list += np.array([int(pred==neg_F_ans) for pred in neg_F_pred])
            ttl += 2
    acc_list /= ttl
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    return acc_list.mean(), se, acc_list
