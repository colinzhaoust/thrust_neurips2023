import json
from tqdm import tqdm, trange
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration



def text_loading(record, task_name, args):

    if task_name in ["cikqa","strategyqa","boolq"]:
        question = record["sent"].strip() 
        if question[-1] not in [".",",","?"]:
            question += "."

        if args.use_know == "yes":
            question += " "+ record["knowledge"]

        question += " above is correct or wrong? "
            
        if record["option"+record["label"]] in ["correct","similar"]:
            target = "correct"
        elif record["option"+record["label"]] in ["wrong","different"]:
            target = "wrong"

        answer_choices = ["correct","wrong"]

    elif task_name in ["esnli"]:
        question = record["sent"].strip() 
        if question[-1] not in [".",",","?"]:
            question += "."

        if args.use_know == "yes":
            question += " "+ record["knowledge"]

        question += " above sentences are "
            
        if record["option"+record["label"]] in ["correct","similar"]:
            target = "similar"
        elif record["option"+record["label"]] in ["wrong","different"]:
            target = "different"

        answer_choices = ["similar","different"]

    elif task_name == "agnews":
        # change world to political
        answer_choices = ["political news","sports news","business news","technology news"]
        target = record["option"+record["label"]].replace("about ","")
        target += " news"

        if target == "world news":
            target = "political news"

        question = record["sent"].strip()

        if args.use_know == "yes":
            question += " "+ record["knowledge"]

        question += " This example is "

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
        question += " The answer is "

        name1 = record['option1']
        name2 = record['option2']
        name3 = record['option3']

        if "option4" not in record:
            name4 = record['option3']
        else:
            name4 = record['option4']

        if not 'label' in record:
            # This is a dummy label for test prediction.
            # test.jsonl doesn't include the `answer`.
            label = "1"
        else:
            label = record['label']

        target = answer_choices[int(label)-1]

        if args.use_know == "no":
            indexer = [" (A) "," (B) "," (C) "," (D) "]
            for n, name in enumerate([name1,name2,name3,name4]):
                question += indexer[n] + name

        if args.use_know == "yes":
            question += "\n" +record["knowledge"]


    return question, target, answer_choices


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

task = args.task
pred_collection = {}

with open("../mc_benchmark.json","r",encoding="utf-8") as f:
    dataset = json.load(f)

# for task in ['webquestions', 'curatedtrec', 'nq', 'hotpotqa', 'triviaqa']:
batch_size = 10000

if args.with_know == "yes":
    print("with knowledge setting")

print(task)
if len(dataset[task]["test"]) < batch_size:
    pred_collection[task] = []

    for i, record in enumerate(tqdm(dataset[task]["test"])):
        question, target, answer_choices = text_loading(record, args.task, args)

        sent = question
        toks = sent.split(" ")
        if len(toks) > 450:
            toks = toks[:450]
        result = run_model(" ".join(toks))
        if target in result:
            
        pred_collection[task].extend(result)

    with open("./pred_collection_"+task+"_"+ args.with_know+".json","w",encoding="utf-8") as f:
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
            

            if args.with_know == "yes":
                result = run_model(record["sent"]+" "+record["knowledge"])
            else:
                result = run_model(" ".join(toks))
            pred_collection[task].extend(result)

        interval = str(batch_start)+":"+str(batch_end)
        with open("./pred_collection_"+task+"_"+interval+"_"+args.with_know+".json","w",encoding="utf-8") as f:
            json.dump(pred_collection[task], f)