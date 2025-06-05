'''
    Extract answer choice from the plain text with the re package
'''
import re
import os
import json
import random
import pandas as pd
random.seed(48)
import numpy as np

def extract_MCQ_ans(response_str, options):
    if 'The most likely diagnosis is contact dermatitis.' in response_str:
        print('y')
    for key in options:
        if key in response_str:
            response_str = response_str.replace(key, options[key])
        if key.lower() in response_str:
            response_str = response_str.replace(key.lower(), options[key])
        if key[0].lower()+key[1:] in response_str:
            response_str = response_str.replace(key[0].lower()+key[1:], options[key])
    pattern=[
        r"(?:Therefore, )?(?:The|the) (?:correct )?answer (?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option|Option)?\s*([A-E])",
        r"(?:Therefore, )?(?:The|the)[^\.]*(?:are|is|would be|should be)\s*(?: a)?:?\s*\**\"?(?:option|Option)?\s*([A-E])",
        r"(?:The|the)[^\.]*(?:most|best) [^\.]* (?:is|would be|are|should be)\s*:?(?:an|the)?\s*([A-E])",
        # r"answer is therefore: ([A-E])",
        r"options?:? ([A-E]) (?:are|is) the correct answers?",
        r"The answer is that options? ([A-E])",
        r"(?:Option )?([A-E])[^\.]*(?:most|best)[^\.]*\.",
        r"(?:are|is) consistent with ([A-E])",
        r"^([A-E])",
        r"^[^\.]*([A-E])\.",
        # r"[^\.]*(?:include|involve)?s?\s*:?(?:an|the)?([A-E])(?:,|\.)",
        r"(?:Option|option) ([A-E]) is correct",
        # r"Answers?:?\s*([A-E])",
    ]
    ans_list=[]
    response_str = response_str.strip()
    if len(response_str) == 0:
        return '', -1
    # if response_str[0] in ['A','B','C','D','E']:
    #     ans_list.append(response_str[0])
    pattern_idx = -1
    for i,p in enumerate(pattern):
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
            if len(ans_list) > 0:
                pattern_idx = i
        else:
            break
    
    if len(ans_list) == 0:
        # print(response_str)
        ans_list = ''
    else:
        # if pattern_idx == 9:
        ans_list = ans_list[0]
    # if pattern_idx == 2:
    #     print(response_str)
    return ans_list, pattern_idx

def extract_ans(response_str):
    response_str = response_str.strip().lower()
    if response_str.startswith('yes'):
        return 1
    elif response_str.startswith('no'):
        return 0
    else:
        patterns = [
            r"answer: (yes|no)"
        ]
        for p in patterns:
            ans = re.findall(p, response_str)
            if len(ans) > 0:
                if ans[0] == 'yes':
                    return 1
                else:
                    return 0
        return random.choice([0,1])
    
    # if len(ans) == 0:  # LLM answer the MCQ
    #     # return 1
    #     # print(response_str)
    #     pass
    #     # return 0
    # else:
    #     return 1
    
    # if not response_str.startswith('Question') and len(response_str.strip().split())<=10:
    #     print(response_str)
    # print(response_str)
    # return None

def answer_vote(answers):
    ans_cnt = [0,0]
    for one in answers:
        if one == 0 or one == 1:
            ans_cnt[one] += 1
    if ans_cnt[0] > ans_cnt[1]:
        ans = 0
    elif ans_cnt[0] < ans_cnt[1]:
        ans = 1
    else:
        ans = random.choice([0,1])
    if sum(ans_cnt) == 0:
        return None
    else:
        return ans


# names = ['med42-70B']
# outff = open('wrong_ans.txt','w')
def get_AE_results(dataset, name, digit=5):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    result_dir = os.path.join(project_root, 'results/{}_repeat/AE'.format(dataset))
    full_name = os.path.join(result_dir, name+'_AE_results.json') if dataset == 'medqa' else os.path.join(result_dir, name+'_AE_results.json')
    # outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    inconsistent = 0
    acc,ttl = 0,0
    acc_t, ttl_t, acc_f, ttl_f = 0,0,0,0
    hit = 0
    acc_list = np.zeros(digit)
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            pos_T_ques = entry[1]
            pos_options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\.|\Z)",pos_T_ques)
            pos_options = {v[1]:v[0] for v in pos_options}
            neg_T_ques = entry[2]
            neg_options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\.|\Z)",neg_T_ques)
            neg_options = {v[1]:v[0] for v in neg_options}
            direction = 'forward'
            if direction not in results:
                results[direction] = {'pos_T':[0,0,0,0],'neg_T':[0,0,0,0],'pos_F':[0,0,0,0],'neg_F':[0,0,0,0]}  # fail, ttl, corr, hit_corr
            pos_T_ans = entry[3]
            neg_T_ans = entry[4]
            if dataset == 'medmcqa':
                if pos_T_ans == 'T':
                    pos_T_ans = 1
                else:
                    pos_T_ans = 0
                if neg_T_ans == 'T':
                    neg_T_ans = 1
                else:
                    neg_T_ans = 0
            pos_T_replys = entry[-digit*2:-digit]
            neg_T_replys = entry[-digit:]
            pos_T_pred = [extract_ans(pos_T_reply) for pos_T_reply in pos_T_replys]
            neg_T_pred = [extract_ans(neg_T_reply) for neg_T_reply in neg_T_replys]
            acc_list += np.asarray([1 if pos_T_pred[i] == pos_T_ans else 0 for i in range(digit)])
            acc_list += np.asarray([1 if neg_T_pred[i] == neg_T_ans else 0 for i in range(digit)])
            ttl += 2
    acc_list = acc_list / ttl
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    return acc_list.mean(), se, acc_list
    # # outff.close()
# final_results = pd.DataFrame(final_results)
# final_results.to_excel('results/medqa/AE/results_AE_cot.xlsx')

def get_AE_results_selected(dataset, name, digit=5, selected_list=None):
    result_dir = 'results/medqa_repeat/AE' if dataset == 'medqa' else 'results/medmcqa_repeat/AE'
    full_name = os.path.join(result_dir, name+'_AE_results.json') if dataset == 'medqa' else os.path.join(result_dir, name+'_AE_results.json')
    # out_full_name = os.path.join(result_dir, name+'_AE_results_processed.json')
    # outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    inconsistent = 0
    acc,ttl = 0,0
    acc_t, ttl_t, acc_f, ttl_f = 0,0,0,0
    hit = 0
    acc_list = np.zeros(digit)
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            idx = entry[0]
            if idx not in selected_list:
                continue
            pos_T_ques = entry[1]
            pos_options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\.|\Z)",pos_T_ques)
            pos_options = {v[1]:v[0] for v in pos_options}
            neg_T_ques = entry[2]
            neg_options = re.findall(r"([A-E]): (.*?)(?:\t|\n|\.|\Z)",neg_T_ques)
            neg_options = {v[1]:v[0] for v in neg_options}
            direction = 'forward'
            if direction not in results:
                results[direction] = {'pos_T':[0,0,0,0],'neg_T':[0,0,0,0],'pos_F':[0,0,0,0],'neg_F':[0,0,0,0]}  # fail, ttl, corr, hit_corr
            pos_T_ans = entry[3]
            neg_T_ans = entry[4]
            if dataset == 'medmcqa':
                if pos_T_ans == 'T':
                    pos_T_ans = 1
                else:
                    pos_T_ans = 0
                if neg_T_ans == 'T':
                    neg_T_ans = 1
                else:
                    neg_T_ans = 0
            pos_T_replys = entry[-digit*2:-digit]
            neg_T_replys = entry[-digit:]
            pos_T_pred = [extract_ans(pos_T_reply) for pos_T_reply in pos_T_replys]
            neg_T_pred = [extract_ans(neg_T_reply) for neg_T_reply in neg_T_replys]
            acc_list += np.asarray([1 if pos_T_pred[i] == pos_T_ans else 0 for i in range(digit)])
            acc_list += np.asarray([1 if neg_T_pred[i] == neg_T_ans else 0 for i in range(digit)])
            ttl += 2
    acc_list = acc_list / ttl
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    return acc_list.mean(), se, acc_list
if __name__ == '__main__':
    model_name = 'deepseek-V3'
    digit = 3
    dataset = 'medqa'
    acc, se, acc_list = get_AE_results(dataset, model_name, digit)
    print(f'Acc: {acc:.4f} +/- {se:.4f}')