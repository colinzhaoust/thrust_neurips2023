# using a simpler version to creat thrust

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random
import math

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from tqdm import tqdm, trange
import pathlib

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def extract_layer_cls(embeddings,layer):
    rep = []
    this_layer_embeddings = embeddings[layer]
    for emb in this_layer_embeddings:
        rep.append(emb[0])
        # try:
        #     assert len(emb[0]) == 1024
        # except:
        #     print(len(emb[0]))

    return rep

def create_embeddings(tokenizer, model, texts, target_layer):
    device = torch.device('cuda')
    # create tokenized inputs
    batch_size = 39
    target_layer = target_layer

    model.to(device)

    # naive batching
    if len(texts) < batch_size:
        inputs = tokenizer(texts,max_length=80, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            batch_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states
            embeddings = []
            for embedding in batch_embeddings:
                embeddings.append(embedding.detach().cpu().tolist())
            del batch_embeddings
            torch.cuda.empty_cache()
    else:
        embeddings = []
        for i in range(25):
            embeddings.append([])

        num_batch = len(texts)//batch_size

        for i in range(num_batch+1):
            batch_start = i*batch_size
            batch_end = min(len(texts), (i+1)*batch_size)
            batch_texts = texts[batch_start:batch_end]

            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                batch_embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states
                for j, embedding in enumerate(batch_embeddings):
                    # print(len(batch_embeddings)) roberta_large: 25 layers
                    # assert len(batch_embeddings) == 13
                    if target_layer == -1:
                        if j == len(batch_embeddings)-1:
                            embeddings[j].extend(embedding.detach().cpu().tolist())
                    elif j == target_layer:
                        embeddings[j].extend(embedding.detach().cpu().tolist())

                # save cuda memory
                del batch_embeddings
                del inputs
                torch.cuda.empty_cache()

    # 25 * num_example * seq_len * 768 -> num_example * 768
    return extract_layer_cls(embeddings, target_layer)


def extract_texts(dataset):
    texts = []
    # for item in tqdm(dataset):
    for item in dataset:
        sent = item["sent"].replace("?  ?","?").strip()
        toks = sent.split(" ")
        if len(toks) > 480:
            toks = toks[:480]
        texts.append(" ".join(toks))

    return texts


def model_clustering(dataset, model, tokenizer, args):
    clusters = {}

    all_texts = extract_texts(dataset)
    embeddings = create_embeddings(tokenizer, model, all_texts, args.target_layer)

    # clustering the embeddings and redistribute to various clusters
    print("start clustering ...")
    num_c = int(len(all_texts)**(0.25))
    if num_c <= 2:
        num_c = 3

    kmeans = KMeans(n_clusters=num_c, random_state=42)
    distances = kmeans.fit_transform(embeddings)
    for i in range(num_c):
        clusters[i] = [list(kmeans.cluster_centers_[i]),0,0] # centroid, num_examples,centerness

    for i, label in enumerate(kmeans.labels_):
        # calculate the number of examples per cluster
        clusters[label][1] += 1
        clusters[label][2] += distances[i][label] # inertia per cluster
    
    return clusters


def vectorized_knowledge_bottleneck(clusters, test_embeddings, test_data):
    # kb_{example} = \sum_{cluster} (-1)^2(lb=entailed) M/r^2, where M =|cluster| and r= euclidean(example,centorid_cluster)
    # kb_{set} = mean? percentile?

    kb_collection = []

    for i, item_emb in enumerate(test_embeddings):
        thrust_vecs = []
        for k,cluster in clusters.items():
            w = cluster[1]/(euclidean(item_emb, cluster[0])**3)
            thrust_vecs.append(np.multiply(w,np.subtract(item_emb, cluster[0])))
            
        thrust = np.linalg.norm(np.sum(thrust_vecs, axis=0))
        kb_collection.append(thrust)

    return kb_collection


def main():
    mc_results = {}

    with open("./benchmark.json","r",encoding="utf-8") as f:
        dataset = json.load(f)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, type=str, required=True)
    parser.add_argument("--use_know", default="no", type=str, help="if we add knowledge to the prompt")
    parser.add_argument("--model", default="allenai/unifiedqa-t5-large", type=str, required=False)
    parser.add_argument("--target_layer", default=-1, type=int, required=False)
    args = parser.parse_args()

    task = args.task

    clustering = model_clustering
    knowledge_bottleneck = vectorized_knowledge_bottleneck

    # model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
    # tokenizer = tokenizer_class.from_pretrained(model_path,model_max_length=512)
    # model.eval()
    device = torch.device('cuda')
    model_name = args.model # you can specify the model size here
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name).encoder.to(device)
    model = T5ForConditionalGeneration.from_pretrained(model_name).decoder.to(device)
    model.eval() # fix the weight and ignore dropout

    # process the training queries and get the cluster info
    print("loading train data and conducting clustering...")
    train_size = -1
    if train_size == -1:
        train_size = len(dataset[task]["train"])
    clusters = clustering(dataset[task]["train"][:train_size], model, tokenizer, args)

    print("processing the test data...")
    test_data = dataset[task]["test"]

    # batched test data experiment
    batch_size = 10000
    # test_data = dataset[task]["test"][:100]

    if len(test_data) < batch_size:
        all_texts = extract_texts(test_data)
        test_embs = create_embeddings(tokenizer, model, all_texts, args.target_layer)
        kb_list = knowledge_bottleneck(clusters, test_embs,test_data)
    else:
        num_batch = len(test_data)//batch_size
        print("Batches:",num_batch+1)
        kb_list = []
        for i in trange(num_batch+1):
            batch_start = i*batch_size
            batch_end = min(len(test_data), (i+1)*batch_size)
            batch_data = test_data[batch_start:batch_end]
            all_texts = extract_texts(batch_data)
            test_embs = create_embeddings(tokenizer, model, all_texts, args.target_layer)
            kb_list.extend(knowledge_bottleneck(clusters, test_embs, batch_data))

        # interval = str(batch_start)+":"+str(batch_end)
    root_dir = "./newkbs/" + str(args.target_layer) +"/"
    path = root_dir+args.model.replace("/","_")+"/"
    os.makedirs(path,exist_ok=True)
    with open(path+task+".json","w",encoding="utf-8") as f:
        json.dump(kb_list,f)
    

if __name__ == "__main__":
    main()
