# Thrust: Adaptively Propels Large Language Models with External Knowledge
This is the github repo for our NeurIPS 2023 paper "Thrust: Adaptively Propels Large Language Models with External Knowledge". 

The focus of our work is to help conduct efficient usage of external information by only retrieving such knowledge when necessary.

The method is built from the assumption that if a model can solve a task, it should be able to categorize the examples of this task as well. 

We score if the model is familiar with a new example through its relations with multiple clusters of small-sized demo data on the same task.

We report a 26% average performance improvement on 88% evaluated cases.

## Data Format
Check our dataset format in mini-datasets at **./dataset/mini_mc_benchmark** and **./dataset/mini_qa_benchmark** for the multi-choice and QA data used, respectively. Do send an email for the full-size datasets.

    tasks: a string indicating which task the data belongs to. 
    For MC, tasks are: cikqa, esnli, strategyqa, agnews, boolq, arc-easy, arc-hard. 
    For QA, tasks are: hotpotqa, triviaqa, webquestions, curatedtrec, nq. Check our paper for details.
    
        split: train, test, metric
        
            examples: a list of dictionaries of the data involved. Keys are:
            
                "sent": original query;
                
                "knowledge": the corresponding knowledge from multiple sources (e.g., retrievers), Check our paper for details;
                
                "ans": answer to the question. Different datasets can have different answer formats.

## Experiments

We provide the code for the main experiments on how to acquire the model output and thrust scores in **./experiment/** (Fig.4 and Table 1 in the original paper).

Do email me for details about other experiments, e.g., Figure 7 in Appendix.


## Application of Thrust

In our original paper, we present the application to conduct adaptive augmentation of external knowledge on CLS and QA. This application can be extended to other tasks such as ECBD, as discussed in Sec 6.1 of the paper.

On the other hand, since the score is based on model familiarity with the queries, it can also be regarded as an instance-level zero-shot performance estimation without extra training or data generation.

## Citation

      @inproceedings{
        zhao2023thrust,
        title={Thrust: Adaptively Propels Large Language Models with External Knowledge},
        author={Xinran Zhao and Hongming Zhang and Xiaoman Pan and Wenlin Yao and Dong Yu and Jianshu Chen},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023},
        url={https://openreview.net/forum?id=x9FOu3W6iy}
      }


## Others
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](mailto:xinranz3@andrew.cmu.edu), I will respond to that as soon as possible.

Other running scripts will be released soon. Sample code and data are also available in the supplementary materials of the corresponding OpenReview page.
