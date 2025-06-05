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
def extract_option_ans(response_str, options):
    if len(response_str) == 0:
        return []
    for key in options:
        if key in response_str:
            response_str = response_str.replace(key, options[key])
    pattern=[
        r"^\s*Option\s*([A-E])",
        r"^\s*([A-E])\s*\Z",
        r"^\s*([A-E])\s+",
        r"^\s*([A-E])(?:,|:|\n|\)|\.)",
        r"^\s*\"([A-E])\"",
        r"Answers?:\s*([A-E])",
        r"(?:The|the) (?:correct )?answer (?:for .*? )?is:?\s*(?:option )?([A-E])",
        r"^Option ([A-E]) is the (?:correct )answer",
        r"Correct Answer:\s*([A-E])"
        
    ]
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return []
    # if response_str[0] in ['A','B','C','D','E']:
    #     ans_list.append(response_str[0])
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
        else:
            break
    # if len(ans_list) == 0:
    #     print(response_str)
    if len(ans_list)>0:
        ans_list  = ans_list[0]
    else:
        ans_list = None
    return ans_list

def extract_judge_ans(response_str):
    if len(response_str) == 0:
        return None
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
    pattern=[
        r"(incorrect|not correct|false|wrong)",
        r"(correct|consistent|true|yes)(?! answer)",
        r"(no)\.?\s*"
        r"^\s*([A-E])\s+",
        
    ]
    ans_list = []
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.I|re.DOTALL)
        else:
            break
    if len(ans_list) == 0:
        if 'correct' in response_str.lower():
            print('w')
        return None
    answer = ans_list[0]
    for one in ['incorrect','not correct','false','no','wrong']:
        if one in answer.lower():
            return 'F'
    for one in ['true','correct','consistent','yes']:
        if one in answer.lower():
            return 'T'
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    return None

def extract_ans(response_str,options, alice_ans):
    try:
        response_str = response_str.strip()
        patterns = [
            r'answer: (correct|incorrect, the correct answer is [A-E])',
        ]
        for p in patterns:
            res = re.findall(p, response_str,re.IGNORECASE)
            if len(res) > 0:
                response_str = res[0]
                break
        if response_str.startswith('correct'):
            return ['T',None]
        else:
            pattern = r'^incorrect, the correct answer is ([A-E])'
            res = re.search(pattern,response_str)
            return ['F',res[1]]
    except Exception as e:
        answer,miss = missing_value_padding([None, None], alice_ans)
        return answer



def missing_value_padding(answer, alice_ans):
    # 1. check the verify result:
    miss = 1 if (answer[0] == None or (answer[0] == 'F' and answer[1] == None)) else 0
    if answer[0] is not None and answer[1] is not None:
        return answer,miss
    
    if answer[0] == None:
        answer[0] = random.choice(['T','F'])

    if answer[1] == None:
        if answer[0] == 'T':
            answer[1] = alice_ans
        else:
            answer[1] = random.choice(list({'A','B','C','D','E'}-{alice_ans}))
    return answer,miss       
    
def get_MR_results(dataset,name,digit=5):
    # result_dir = 'results/{}_repeat/MR'.format(dataset)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    result_dir = os.path.join(project_root, 'results/{}_repeat/MR'.format(dataset))
    full_name = os.path.join(result_dir, name+'_MR_results.json')
    # out_full_name = os.path.join(result_dir, name+'_MR_results_processed.json')
    # outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    acc_list = np.asarray([0] * digit,dtype=np.float64)
    
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            T_ques = entry[1]
            F_ques = entry[2]
            direction = 'forward'
            if direction not in results:
                results[direction] = {'T':[0,0,0,0,0],'F':[0,0,0,0,0]}  # fail, ttl, acc, judge_acc, option_acc
            options = re.findall(r"([A-E]): (.*?)(?:\t|\Z|\.)",T_ques)
            options = {v[1]:v[0] for v in options}
            T_ans = entry[3]
            F_ans = entry[4]
            T_replys = entry[-digit*2:-digit]
            F_replys = entry[-digit:]
            
            T_alice, F_alice = re.search(r"Alice\'s answer: ([A-E])",T_ques)[1],re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]
            T_preds = [extract_ans(T_reply,options,T_alice) for T_reply in T_replys]
            F_preds = [extract_ans(F_reply,options,F_alice) for F_reply in F_replys]
            if dataset == 'medmcqa':
                acc_list += np.asarray([0.25 * float(T_pred[0] == 'T')+0.75 * float(F_pred[0] == 'F' and F_pred[1] == F_ans[1]) for T_pred, F_pred in zip(T_preds,F_preds)])

            else:
                acc_list += np.asarray([0.2 * float(T_pred[0] == 'T')+0.8 * float(F_pred[0] == 'F' and F_pred[1] == F_ans[1]) for T_pred, F_pred in zip(T_preds,F_preds)])
            
            entry += T_preds + F_preds
            # outf.write(json.dumps(entry,ensure_ascii=False)+'\n')
            ttl += 1
    # outf.close()
    acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)
    se = std_dev / np.sqrt(len(acc_list))
    # CI = 1.96 * np.std(acc_list) / np.sqrt(len(acc_list))
    return acc_list.mean(), se, acc_list

    
def get_MR_results_selected(dataset,name,digit=5,select_list=None):
    result_dir = 'results/{}_repeat/MR'.format(dataset)
    full_name = os.path.join(result_dir, name+'_MR_results.json')
    # out_full_name = os.path.join(result_dir, name+'_MR_results_processed.json')
    # outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    acc_list = np.asarray([0] * digit,dtype=np.float64)
    
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            ids = entry[0]
            if ids not in select_list:
                continue
            T_ques = entry[1]
            F_ques = entry[2]
            direction = 'forward'
            if direction not in results:
                results[direction] = {'T':[0,0,0,0,0],'F':[0,0,0,0,0]}  # fail, ttl, acc, judge_acc, option_acc
            options = re.findall(r"([A-E]): (.*?)(?:\t|\Z|\.)",T_ques)
            options = {v[1]:v[0] for v in options}
            T_ans = entry[3]
            F_ans = entry[4]
            T_replys = entry[-digit*2:-digit]
            F_replys = entry[-digit:]
            
            T_alice, F_alice = re.search(r"Alice\'s answer: ([A-E])",T_ques)[1],re.search(r"Alice\'s answer: ([A-E])",F_ques)[1]
            T_preds = [extract_ans(T_reply,options,T_alice) for T_reply in T_replys]
            F_preds = [extract_ans(F_reply,options,F_alice) for F_reply in F_replys]
            if dataset == 'medmcqa':
                acc_list += np.asarray([0.25 * float(T_pred[0] == 'T')+0.75 * float(F_pred[0] == 'F' and F_pred[1] == F_ans[1]) for T_pred, F_pred in zip(T_preds,F_preds)])

            else:
                acc_list += np.asarray([0.2 * float(T_pred[0] == 'T')+0.8 * float(F_pred[0] == 'F' and F_pred[1] == F_ans[1]) for T_pred, F_pred in zip(T_preds,F_preds)])
            
            entry += T_preds + F_preds
            # outf.write(json.dumps(entry,ensure_ascii=False)+'\n')
            ttl += 1
    # outf.close()
    acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1) 
    se = std_dev / np.sqrt(len(acc_list))
    # CI = 1.96 * np.std(acc_list) / np.sqrt(len(acc_list))
    return acc_list.mean(), se, acc_list

if __name__ == '__main__':
    dataset = 'medmcqa'
    model_names = ['llama-7B']
    digit = 5
    get_MR_results(dataset,model_names[0],digit)