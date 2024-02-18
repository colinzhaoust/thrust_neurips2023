import argparse
from datasets import load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates
import torch
import random
import copy
import os
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, GPT2Tokenizer, GPTJForCausalLM, GPTNeoForCausalLM, OPTForCausalLM, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
import json
# from data_utils import load_jsonl_file, save_jsonl_file

root_dir = "./output"


def save_to_json(list, path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(list, f)


def demo_loading(train_set, args):
    label_list = []
    demos = {}
    for record in train_set:
        try:
            label = record["option"+record["label"]] 
        except:
            # dummy
            label = record["option1"] 

        if label not in label_list:
            label_list.append(label)
            demos[label] = [record]

    for record in train_set:    
        if len(demos[label]) < args.demo_num:
            demos[label].append(record)
        
        prepared = 1
        for label in label_list:
            if len(demos[label]) < args.demo_num:
                prepared = 0

        if prepared:
            break

    return demos


def text_loading(record, task_name, args):

    if task_name in ["cikqa","strategyqa","boolq","esnli"]:
        question = record["sent"].strip() 
        if question[-1] not in [".",",","?"]:
            question += "."

        question += " Yes or No?"

        if args.use_know == "yes":
            question  = question + " " + size_limit(record["knowledge"], 120)

        if record["option"+record["label"]] in ["correct","similar"]:
            target = "yes"
        elif record["option"+record["label"]] in ["wrong","different"]:
            target = "no"

        answer_choices = ["yes","no"]

    # elif task_name in ["esnli"]:
    #     question = record["sent"].strip() 
    #     if question[-1] not in [".",",","?"]:
    #         question += "."

    #     question += " similar or different? "

    #     if args.use_know == "yes":
    #         question  = question + " " + size_limit(record["knowledge"], 120)

    #     if record["option"+record["label"]] in ["correct","similar"]:
    #         target = "similar"
    #     elif record["option"+record["label"]] in ["wrong","different"]:
    #         target = "different"

    #     answer_choices = ["similar","different"]

    elif task_name == "agnews":
        # change world to political
        answer_choices = ["political news","sports news","business news","technology news"]
        target = record["option"+record["label"]].replace("about ","")
        target += " news"

        if target == "world news":
            target = "political news"

        question = record["sent"].strip()

        question += " The news is about?"

        if args.use_know == "yes":
            question  = question + " " + size_limit(record["knowledge"], 120)

        for choice in answer_choices:
            question += choice
            if choice != "technology news":
                question += " or "
            else:
                # the last one
                question += "?"

    elif task_name in ["arc-easy","arc-hard"]:
        answer_choices = ["(A)","(B)","(C)","(D)"]

        question = record["sent"].strip().replace("?  ?","?") 
 
        names = [record['option1'],record['option2'],record['option3']]

        if "option4" not in record:
            # name4 = record['option3']
            answer_choices = ["(A)","(B)","(C)"]
        else:
            names.append(record['option4'])

        if not 'label' in record:
            # This is a dummy label for test prediction.
            # test.jsonl doesn't include the `answer`.
            label = "1"
        else:
            label = record['label']

        target = ""#answer_choices[int(label)-1]

        for n, name in enumerate(names):
            question += " "+answer_choices[n]+" "+ name

        if args.use_know == "yes":
            question  = question + " " + size_limit(record["knowledge"], 120)

        # add the real options
        for n, name in enumerate(names):
            answer_choices[n] = name #+= " "+name

        target += names[int(label)-1] #" " + 

    return question, target, answer_choices


def init_model(args, model_name):
    # bert-base-uncased, bert-large-uncased, roberta-base, roberta-large
    if "bert-" in model_name or "roberta-" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)

    elif "t5" in model_name:
        # model_name = "allenai/unifiedqa-t5-large" # you can specify the model size here
        if "11b" in model_name:
            num_gpus = torch.cuda.device_count()
            max_memory = {i: args.per_gpu_mem for i in range(num_gpus)} # adjust the max memory based on your gpu
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name, max_memory=max_memory, torch_dtype=torch.float16).to(args.device)
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name).to(args.device)

    elif model_name == "gpt-j-6B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(args.device)

    elif model_name == "gpt-neo-1.3B":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(args.device)

    elif model_name == "gpt-neo-2.7B":
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(args.device)

    elif model_name == "gpt-neox-20b":
        config = AutoConfig.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        num_gpus = torch.cuda.device_count()
        max_memory = {i: args.per_gpu_mem for i in range(num_gpus)} # adjust the max memory based on your gpu
        model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", torch_dtype=torch.float16, device_map="auto", offload_folder="offload_folder", max_memory=max_memory, low_cpu_mem_usage=True)

    elif "opt" in model_name:
        num_gpus = torch.cuda.device_count()
        max_memory = {i: args.per_gpu_mem for i in range(num_gpus)} # adjust the max memory based on your gpu
        tokenizer = AutoTokenizer.from_pretrained("facebook/{}".format(model_name), use_fast=False)
        model = OPTForCausalLM.from_pretrained("facebook/{}".format(model_name), torch_dtype=torch.float16, device_map="auto", offload_folder="offload_folder", max_memory=max_memory, low_cpu_mem_usage=True)
    else:
        print("Error: model not supported!!!\nSupported models: [gpt-neo-1.3B, gpt-neo-2.7B, gpt-j-6B, opt-30b]")
        raise NotImplementedError

    return tokenizer, model


def get_start_loc(tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
    return tokens.shape[1] - 1


def score_answer_choices(args, tokenizer, model, question, option):
    start_loc = get_start_loc(tokenizer, question)

    inputs = tokenizer(question, option, return_tensors="pt", truncation="only_first", max_length=512).to(args.device)
    labels = copy.deepcopy(inputs["input_ids"])
    labels[0, :start_loc+1] = -100

    with torch.no_grad():
        loss = model(**inputs, labels=labels).loss
    return -loss.detach().cpu().item()


def accuracy(l1, l2):
    assert len(l1) == len(l2)
    return sum(1 for x,y in zip(l1,l2) if x == y) / len(l1)

def find_best_match(target, choices):
    scores = [fuzz.ratio(target, c) for c in choices]
    best_score = max(scores)
    return choices[scores.index(best_score)]

def get_valid_prompts(task_name):
    prompts = DatasetTemplates(task_name)
    prompt_names = prompts.all_template_names
    valid_prompt_names = []

    for name in prompt_names:
        if prompts[name].metadata.original_task and prompts[name].metadata.choices_in_prompt:
            valid_prompt_names.append(name)

    return prompts, valid_prompt_names

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def size_limit(text, max_size):
    toks = text.split()
    return " ".join(toks[:max_size])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", type=str, help="GPU id to train models.")
    # parser.add_argument("--model_name", default="gpt-j-6B", type=str, help="model.")
    parser.add_argument("--model_name", default="opt-30b", type=str, help="model.")
    parser.add_argument("--per_gpu_mem", default="40GB", type=str, help="GPU memory on each.")
    parser.add_argument("--use_know", default="no", type=str, help="if we add knowledge to the prompt")
    parser.add_argument("--task", default="agnews", type=str, help="Task.")
    parser.add_argument("--demo_num", default=0, type=int, help="The number of demostration example used per class.")
    args = parser.parse_args()
    random.seed(11)
    
    args.device = "cuda:" + args.gpu_id

    #args.device = "cpu"
    

    print("Load model...")
    tokenizer, model = init_model(args, args.model_name)

    # dataset
    with open("../mc_benchmark.json","r",encoding="utf-8") as f:
        mc_dataset = json.load(f)

    valid_task_names = []

    #valid_task_names += ["super_glue/rte", "super_glue/cb", "anli", "super_glue/wsc.fixed", "winogrande/winogrande_xl", "winogrande/winogrande_debiased", "super_glue/copa", "hellaswag", "super_glue/wic"]
    # valid_task_names += ["mmlu_stem", "mmlu_humanities", "mmlu_social_sciences", "mmlu_other"]
    # valid_task_names += ["super_glue/copa"]
    # valid_task_names += ["super_glue/wic"]
    # valid_task_names = ["agnews","esnli","cikqa","strategyqa","boolq"] # arc will be processed separately
    valid_task_names = []
    valid_task_names.append(args.task)

    results = []


    for task_name in valid_task_names:
        print(task_name)
        
        train_set = mc_dataset[task_name]["train"]
        raw_eval_dataset = mc_dataset[task_name]["test"]
        pred_instance_list = []

        prompt_names = ["None"]

        for prompt_name in prompt_names[:1]:
            # prompt = prompts[prompt_name]
            corrects = []
            predictions = []

            demo_prefix = ""
            demo_template = "Question: $question \n Answer: $ans \n \n"
            demos = demo_loading(train_set, args)

            for i in range(args.demo_num):
                for lb in list(demos.keys()):
                    temp = copy.deepcopy(demo_template)
                    q, t, a = text_loading(demos[lb][i], task_name, args)
                    demo_prefix += temp.replace("$question",q).replace("$ans",t)

            for instance in tqdm(raw_eval_dataset, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
                # prepare the demonstration, in the format of:
                # Question: xxxxx
                # Answer: xxxxx
                #
                # Question: xxxxx
                # Answer:
                prompt = ""
                prompt +=  demo_prefix

                # question_prefix = "Question: $question \n Answer: "
                question, target, answer_choices = text_loading(instance, task_name, args)
                # prompt += question_prefix.replace("$question",question)
                prompt = question

                try:
                    assert target in answer_choices
                except AssertionError:
                    print(f'unmatched target and answer_choices: `{target}` `{answer_choices}`. Using the best match in the answer_choices')
                    target = find_best_match(target, answer_choices)

                correct = answer_choices.index(target)
                corrects.append(correct)

                scores = []
                for option in answer_choices:
                    scores.append(score_answer_choices(args, tokenizer, model, prompt, option))

                prediction = scores.index(max(scores))
                predictions.append(scores.index(max(scores)))

                pred_instance = {"question": question, "target": target, "answer_choices": answer_choices,
                                "pred_scores": scores, "prediction": prediction, "correct": correct,
                                "prompt": prompt_name}

                pred_instance_list.append(pred_instance)

            print(task_name, prompt_name, "accuracy:", accuracy(corrects, predictions))
            results.append("\t".join([task_name, prompt_name, str(accuracy(corrects, predictions))]))

        path = root_dir+"/"+args.use_know+"/"+args.model_name.replace("/","_")+"/"
        os.makedirs(path,exist_ok=True)
        save_to_json(pred_instance_list, path+task_name+".json")
        #  save_jsonl_file(pred_instance_list, f"{root_dir}/{task_name}.jsonl")

    print("\n".join(results))
            
