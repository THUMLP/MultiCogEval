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
def extract_ans(response_str, options):
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
        r"The correct answer (?:for .*? )?is:?\s*([A-E])",
        r"^Option ([A-E]) is the correct answer"
        
    ]
    ans_list=[]
    response_str = response_str.strip()
    if '\n\n' in response_str:
        response_str = response_str.split('\n\n')[0]
    if len(response_str) == 0:
        return random.choice(['A','B','C','D','E'])
    # if response_str[0] in ['A','B','C','D','E']:
    #     ans_list.append(response_str[0])
    for p in pattern:
        if len(ans_list)==0:
            ans_list=re.findall(p,response_str,re.DOTALL)
        else:
            break
    if len(ans_list) == 0:
        ans_list = random.choice(['A','B','C','D','E'])
    else:
        ans_list = ans_list[0]
    return ans_list



def get_MCQ_results(dataset, name, digit=5):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    result_dir = os.path.join(project_root,'results/{}_repeat/MCQ'.format(dataset))
    full_name = os.path.join(result_dir, name+'_MCQ_results.json')
    out_full_name = os.path.join(result_dir, name+'_MCQ_results_processed.json')
    outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    acc_list = [0] * digit
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            ques = entry[1]
            direction = 'forward'
            if direction not in results:
                results[direction] = {'pos':[0,0,0,0],'neg':[0,0,0,0]}  # fail, ttl, corr, hit_corr
            options = re.findall(r"([A-E]): (.*?)(?:\t|\Z)",ques)
            options = {v[1]:v[0] for v in options}
            pos_ans = entry[2]
            pos_reply = entry[-digit:]
            pos_pred_list = [extract_ans(reply,options) for reply in pos_reply]
            
            entry += pos_pred_list
            outf.write(json.dumps(entry)+'\n')
            ttl += 1
            acc_list = [acc_list[i]+(pos_pred_list[i]==pos_ans) for i in range(digit)]
    acc_list = np.asarray([acc/ttl for acc in acc_list])
    # print(acc_list.mean())
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    # print(CI)
    return acc_list.mean(), se, acc_list
    
def get_MCQ_results_selected(dataset, name, digit=5, selected_list=None):
    result_dir = 'results/{}_repeat/MCQ'.format(dataset)
    full_name = os.path.join(result_dir, name+'_MCQ_results.json')
    out_full_name = os.path.join(result_dir, name+'_MCQ_results_processed.json')
    outf = open(out_full_name,'w')
    model_name = name
    fail, ttl, corr, hit_corr = 0,0,0,0
    results = {}
    acc_list = [0] * digit
    with open(full_name, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            idx = entry[0]
            if idx not in selected_list:
                continue
            ques = entry[1]
            direction = 'forward'
            if direction not in results:
                results[direction] = {'pos':[0,0,0,0],'neg':[0,0,0,0]}  # fail, ttl, corr, hit_corr
            options = re.findall(r"([A-E]): (.*?)(?:\t|\Z)",ques)
            options = {v[1]:v[0] for v in options}
            pos_ans = entry[2]
            pos_reply = entry[-digit:]
            pos_pred_list = [extract_ans(reply,options) for reply in pos_reply]
            
            entry += pos_pred_list
            outf.write(json.dumps(entry)+'\n')
            ttl += 1
            acc_list = [acc_list[i]+(pos_pred_list[i]==pos_ans) for i in range(digit)]
    acc_list = np.asarray([acc/ttl for acc in acc_list])
    # print(acc_list.mean())
    # acc_list = acc_list / ttl
    std_dev = np.std(acc_list, ddof=1)  
    se = std_dev / np.sqrt(len(acc_list))  
    # print(CI)
    return acc_list.mean(), se, acc_list
    
if __name__ == '__main__':
    dataset = 'medmcqa'
    model_list = ['gpt-4o-mini','llama3-8B']
    final_results = []
    for model in model_list:
        acc, se, acc_list = get_MCQ_results(dataset, model, digit=3)
        print(model, acc, se)
        final_results.append([model, acc, se, acc_list.tolist()])
    # final_results = pd.DataFrame(final_results, columns=['model','acc','CI','acc_list'])
    # final_results.to_excel('results/medqa/MCQ/results_MCQ.xlsx')
# final_results = pd.DataFrame(final_results)
# final_results.to_excel('results/medqa/MCQ/results_MCQ.xlsx')


    # final_results = pd.DataFrame(final_results, columns=['model','acc','CI','acc_list'])
    # final_results.to_excel('results/medqa/MCQ/results_MCQ.xlsx')
# final_results = pd.DataFrame(final_results)
# final_results.to_excel('results/medqa/MCQ/results_MCQ.xlsx')