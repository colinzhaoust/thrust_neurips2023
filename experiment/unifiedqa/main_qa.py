import json
import os
from tqdm import tqdm, trange
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

root_dir = "./preds/"

def run_model(input_string, device, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--task", default=None, type=str, required=True)
parser.add_argument("--use_know", default="no", type=str, help="if we add knowledge to the prompt")
parser.add_argument("--model", default="allenai/unifiedqa-t5-large", type=str, required=False)
args = parser.parse_args()

task = args.task

device = torch.device('cuda')
model_name = args.model # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

pred_collection = {}

with open("./benchmark.json","r",encoding="utf-8") as f:
    dataset = json.load(f)

# for task in ['webquestions', 'curatedtrec', 'nq', 'hotpotqa', 'triviaqa']:
batch_size = 10000

if args.use_know == "yes":
    print("with knowledge setting")

print(task)
if len(dataset[task]["test"]) < batch_size:
    pred_collection[task] = []
    for i, record in enumerate(tqdm(dataset[task]["test"])):
        if args.use_know != "yes":
            sent = record["sent"]
            toks = sent.split(" ")
            if len(toks) > 450:
                toks = toks[:450]
            result = run_model(" ".join(toks),device)

        elif args.use_know == "yes":
            sent = record["sent"]+" "+record["knowledge"]
            toks = sent.split(" ")
            result = run_model(" ".join(toks),device)

        pred_collection[task].extend(result)

    path = root_dir+args.use_know+"/"+args.model.replace("/","_")+"/"
    os.makedirs(path,exist_ok=True)
    with open(path+task+".json","w",encoding="utf-8") as f:
        json.dump(pred_collection[task], f)


else:
    num_batch = len(dataset[task]["test"])//batch_size

    for i in trange(num_batch+1):
        if task == 'triviaqa' and i == 0:
            continue

        batch_start = i*batch_size
        batch_end = min(len(dataset[task]["test"]), (i+1)*batch_size)
        pred_collection[task] = []
        for i, record in enumerate(tqdm(dataset[task]["test"][batch_start:batch_end])):
            sent = record["sent"]
            toks = sent.split(" ")
            if len(toks) > 480:
                toks = toks[:480]

            if args.use_know == "yes":
                result = run_model(record["sent"]+" "+record["knowledge"],device)
            else:
                result = run_model(" ".join(toks), device)
            pred_collection[task].extend(result)

        interval = str(batch_start)+":"+str(batch_end)
        path = root_dir+args.use_know+"/"+args.model.replace("/","_")+"/"
        with open(path+task+"_"+interval+".json","w",encoding="utf-8") as f:
            json.dump(pred_collection[task], f)