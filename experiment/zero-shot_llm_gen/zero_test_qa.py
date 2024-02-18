import argparse
from curses import raw
from datasets import load_dataset, load_from_disk
from promptsource.templates import DatasetTemplates
import torch
import random
import copy
import os
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, GPTNeoForCausalLM, OPTForCausalLM, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM
from transformers import GPTJForQuestionAnswering, GPT2Tokenizer, GPTJForCausalLM
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
import json

import string
import re
import argparse
import sys
from _ast import List
from collections import Counter
# from data_utils import load_jsonl_file, save_jsonl_file

root_dir = "./output"


def save_to_json(list, path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(list, f)


def init_model(args, model_name):
    # bert-base-uncased, bert-large-uncased, roberta-base, roberta-large
    if "bert-" in model_name or "roberta-" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)

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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def generate_answer(args, tokenizer, model, question):
    # inputs = tokenizer(question, option, return_tensors="pt", truncation="only_first", max_length=512).to(args.device)
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(args.device)
    original_length = len(input_ids[0])
    offset = 10 if args.task != "triviaqa" else 30
    gen_tokens = model.generate(input_ids,
            do_sample=True,
            temperature=0.1,
            max_length=original_length+offset,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_toks = gen_tokens[0][original_length:]
    gen_text = tokenizer.batch_decode([output_toks], skip_special_tokens=True)[0]

    return gen_text

def max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def size_limit(text, max_size):
    toks = text.split()
    return " ".join(toks[:max_size])


def prompt_decoration(instance,args,type="train"):
    template =  """
    Question: $query
    Answer: $ans
    """
    sent = instance["sent"]
    answer = instance["ans"]
    if "|" in answer:
        answer = answer.split("|")[0]

    if type=="test":
        answer=""

    template = template.replace("$query",sent).replace("$ans",answer)

    if args.use_know == "yes":
        template = "Context: $context \n" + template
        template = template.replace("$context",size_limit(instance["knowledge"], 120))

    return template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", type=str, help="GPU id to train models.")
    parser.add_argument("--model_name", default="opt-30b", type=str, help="model.")
    parser.add_argument("--per_gpu_mem", default="40GB", type=str, help="GPU memory on each.")
    parser.add_argument("--use_know", default="no", type=str, help="if we add knowledge to the prompt")
    parser.add_argument("--task", default="webquestions", type=str, help="Task.")
    parser.add_argument("--demo_num", default=0, type=int, help="The number of demostration example used per class.")
    parser.add_argument("--know_len", default=120, type=int, help="The number of demostration example used per class.")
    args = parser.parse_args()
    random.seed(11)

    args.device = "cuda:" + args.gpu_id

    #args.device = "cpu"
    
    print("Load model...")
    tokenizer, model = init_model(args, args.model_name)

    # dataset
    with open("../benchmark.json","r",encoding="utf-8") as f:
        dataset = json.load(f)

    valid_task_names = []
    valid_task_names.append(args.task) #[ 'webquestions'] #, 'curatedtrec','hotpotqa', 'triviaqa', 'nq'

    results = []


    for task_name in valid_task_names:
        print(task_name)
        train_data = dataset[task_name]["train"]
        raw_eval_dataset = dataset[task_name]["test"]
        pred_instance_list = []

        prompt_names = ["None"]

        for prompt_name in prompt_names[:1]:
            # prompt = prompts[prompt_name]
            f1s = []
            predictions = []

            demo_prefix = ""
            demo_examples = random.sample(train_data, args.demo_num)
            # demo format
            # Context: (if use know)
            # Question:
            # Answer:

            for instance in demo_examples:
                demo_prefix += prompt_decoration(instance,args)

            # for instance in tqdm(raw_eval_dataset, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
            for instance in tqdm(raw_eval_dataset):
                # example_prefix = prompt_decoration(instance,args,"test")
                # question = demo_prefix+example_prefix
                question = instance["sent"]
                if args.use_know == "yes":
                    question = instance["sent"]+" "+ size_limit(instance["knowledge"], 400)

                gold_answers = instance["ans"].split("|")
                # print("----------------------")
                # print(question)

                # if len(f1s) > 5:
                #     break

                pred = generate_answer(args, tokenizer, model, question)

                # print(pred)
                # print(instance["ans"])

                score = max_over_ground_truths(f1_score,pred,gold_answers)

                f1 = score
                f1s.append(f1)

                pred_instance = {"question": question, "target": gold_answers,
                                "prediction": pred, "f1": f1,
                                "prompt": prompt_name}

                pred_instance_list.append(pred_instance)

            print(task_name, prompt_name, "f1", sum(f1s)/len(f1s))
            results.append("\t".join([task_name, prompt_name, str(sum(f1s)/len(f1s))]))

        path = root_dir+"/"+args.use_know+"/"+args.model_name.replace("/","_")+"/"
        os.makedirs(path,exist_ok=True)
        save_to_json(pred_instance_list, path+task_name+".json")
        #  save_jsonl_file(pred_instance_list, f"{root_dir}/{task_name}.jsonl")

    print("\n".join(results))
            
