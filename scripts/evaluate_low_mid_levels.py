import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # del
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
import json
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import random
random.seed(0)

def chating(tmp_inputs, dev_data, num_examples, model, tokenizer,task, repeat=0):
    input_text_final = []
    for text in tmp_inputs:
        for i in range(args.repeat):
            new_len = -1
            num_examples = args.ntrain
            while new_len <= 0:
                demonstration = prepare_examples(dev_data, text[1], num_examples,task, text[2])
                input_text = demonstration + text[0]
                new_len = 2048-tokenizer.encode(input_text, return_tensors="pt").size(1)
                if new_len > 50:
                    new_len = 50
                num_examples -= 1
            input_text_final.append(input_text)


    sampling_params = SamplingParams(
    temperature=0,
    max_tokens=50,
    stop=['\n\nQuestion']
    )
    outputs = model.generate(
        input_text_final,
        sampling_params
        )
    replys = [output.outputs[0].text for output in outputs]
    return replys


def load_data(path):
    return json.load(open(path,'r'))

def prepare_examples(dev_data,neg,nums,task, idx):
    example_ids = set(range(len(dev_data))) - set([idx])
    example_ids = list(example_ids)
    sampled_example_ids = random.sample(example_ids, nums)
    examples = [dev_data[i] for i in sampled_example_ids]
    tmp = ''
    rand_list = random.choice([[0,0,0,1,1],[1,1,1,0,0]])
    random.shuffle(rand_list)
    if task == 'SV':
        for i,one in enumerate(examples):
            if rand_list[i] == 0:
                ques = one[1]
                ans = one[3]
            else:
                ques = one[2]
                ans = one[4]
            
            ans = 'true' if ans == 'T' else 'false'
            tmp += ques + '\nAnswer: '+ans+'\n\n'
    elif task == 'MCQ':
        for one in examples:
            ques = one[1]
            ans = one[2]
            # print(one)
            tmp += ques + '\nAnswer: '+ans+'\n\n'

    elif task == 'MR':
        for i,one in enumerate(examples):

            if rand_list[i] == 0:
                ques = one[1]
                ans = one[3]
            else:
                ques = one[2]
                ans = one[4]
            
            if ans[0] == 'T':
                tmp += ques + '\nAnswer: correct'+'\n\n'
            else:
                tmp += ques + '\nAnswer: incorrect, the correct answer is ' + ans[1]+'\n\n'
    elif task == 'AE':
        for i, one in enumerate(examples):

            if rand_list[i] == 0:
                ques = one[1]
                ans = one[3]
            else:
                ques = one[2]
                ans = one[4]
                
            if ans == 'T':
                tmp += ques + f'\nAnswer: yes\n\n'
            else:
                tmp += ques + f'\nAnswer: no\n\n'

                
    return tmp

def batch_prepare_inputs(inputs, dev_data, nums, task):
    pos_example = prepare_examples(dev_data,False,nums,task)
    neg_example = prepare_examples(dev_data, True, nums, task)
    examples = []
    for one in inputs:
        if one[1]:
            examples += ['Please complete the final sample with the same format as the given examples.'+ neg_example + one[0]]
        else:
            examples += ['Please complete the final sample with the same format as the given examples.'+ pos_example + one[0]]
    return examples    


def main(args):
    typs = args.typs
    print(typs)
    cnt = -1
    all_task_pool, all_item_pool, all_outf,all_dev = [],[],[],[]
    if not os.path.exists('results/{}_repeat'.format(args.dataset)):
        os.makedirs('results/{}_repeat'.format(args.dataset))
    for typ in typs:
        forward_path = os.path.join('data/{}/{}'.format(args.dataset,typ), 'test.json')

        forward_data = load_data(forward_path)
        
        if not os.path.exists('results/{}_repeat/{}'.format(args.dataset,typ)):
            os.makedirs('results/{}_repeat/{}'.format(args.dataset,typ))
        f = open('results/{0}_repeat/{2}/{1}_{2}_results.json'.format(args.dataset,args.model_name,typ),'w',encoding='utf8')
        all_outf.append(f)
        test_data = {'forward':forward_data}
        task_pool = []
        item_pool = []
        for direction in tqdm(test_data):
            temp_test_data = test_data[direction]
            for item in tqdm(temp_test_data):
                cnt += 1
                if cnt < args.start:
                    continue
                if typ == 'SV' or typ == 'MR' or typ == 'AE':
                    idx = item[0]
                    ques = item[1]
                    ques_2 = item[2]
                    input_text = ques + '\nAnswer:'
                    input_text_2 = ques_2 + '\nAnswer:'
                    task_pool += [[input_text, False, idx],[input_text_2,False, idx]]
                    item_pool += [item]
                elif typ == 'MCQ':
                    idx = item[0]
                    ques = item[1]
                    input_text = ques + '\nAnswer:'
                    task_pool += [[input_text, False, idx]]
                    item_pool += [item]
        all_task_pool.append(task_pool)
        all_item_pool.append(item_pool)
        all_dev.append(forward_data)
    model = LLM(model=args.model,tensor_parallel_size=args.num_cuda, gpu_memory_utilization=0.95,max_model_len=2048,trust_remote_code=True, swap_space=32)
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    
    for f, task_pool, item_pool, typ,dev_data in zip(all_outf, all_task_pool, all_item_pool, typs,all_dev):
        ncnt = 0
        if typ == 'SV' or typ == 'MR' or typ == 'AE':
            loop=2
        elif typ == 'MCQ':
            loop=1
        batch = task_pool
        all_results = chating(batch, dev_data, args.ntrain, model, tokenizer,typ)
        while len(all_results) > 0:
            item = item_pool[ncnt]
            ncnt += 1
            item += all_results[:loop*args.repeat]
            all_results = all_results[loop*args.repeat:]
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
            f.flush()
        f.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of demonstration examples to use for each input.")
    parser.add_argument("--typs", action="extend",nargs="+", type=str, help="Task types to evaluate on, e.g., SV, MR, AE, MCQ")
    parser.add_argument("--dataset",type=str,default='medmcqa', choices=['medmcqa','medqa'], help="Dataset to use for evaluation. Options are 'medmcqa' or 'medqa'.")
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3-8B',help="Model name or path to use for evaluation.")
    parser.add_argument("--model_name", type=str, default='llama3-8B',help="Model name for saving results.")
    parser.add_argument("--start", type=int, default=0, help="Start index for evaluation. Useful for resuming from a specific point.")
    parser.add_argument("--num_cuda", type=int, default=2, help="Number of CUDA devices to use for parallel processing.")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repeated evaluations for each input to ensure robustness.")
    args = parser.parse_args()
    main(args)

