# ID-XCB: Data-independent Debiasing for cyberbullying detection
Implementation of paper: ID-XCB: Data-independent Debiasing for Fair and Accurate Transformer-based Cyberbullying Detection
<p align=center><img src="Architecture.png" width="500" center/>

## Core idea and implementation
We combine the three training losses to fine-tune a classifier. Adversarial training(Embedding loss) and task training(BCELoss) optimise ID-XCB model parameters on a training dataset and simultaneously enforce the fairness constraints(FC) on a validation set to reduce swear word impact.
<p align=center><img src="code.png" width="600" center/>

## Datasets 
#### Instagram cyberbullying detection datasets （Need to request from the dataset author）
Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han, Qin Lv, and Shivakant Mishra. 2015. **Analyzing labeled cyberbullying incidents on the instagram social network**. In _Socinfo_. Springer, 49–66.
#### Vine cyberbullying detection datasets （Need to request from the dataset author）
Rahat Ibn Rafiq, Homa Hosseinmardi, Richard Han, Qin Lv, Shivakant Mishra, and Sabrina Arredondo Mattson. 2015. **Careful what you share in six seconds: Detecting cyberbullying instances in Vine**. In _ASONAM_. ACM, 617–622.
#### Swear lexicon_1 (Publicly available)
Agrawal S, Awekar A. 2018 Mar 1. **Deep learning for detecting cyberbullying across multiple social media platforms**. In European conference on information retrieval 2018 Mar 1 (pp. 141-153). 
#### Swear lexicon_2 (Publicly available)
We use Google bad words lists[https://code.google.com/archive/p/badwordslist/downloads]
## Experiments
The source code is written in Python and is a Jupyter Notebook. 
#### [1]Install the requirements using the following command:
```bash
pip install -r requirements.txt
```
#### [2] Find the data for experiments in the data fold 
* train.csv--For fine-tuning the task of cyberbullying detection.

* test.csv--For cyberbullying detection test task.

* goemotion.csv--For emotion domain adaptive task.

#### [3] Running
* HDCyberbllying.ipynb--Create HDCyberbllying datasets(train.csv and test.csv)

* EAT.ipynb--Experiments on Roberta, Bert,Distilbert, Electra, XLnet, Mpnet,T5

* LLAMA_3&2_EAT.ipynb--Experiments on LLAMA_3&2

### Python packages version
* pandas==2.0.3
* scikit_learn=1.2.2
* torch==2.2.2
* numpy==1.25.2
* transformers==4.40.0
* datasets==2.18.0
* accelerate==0.29.3
* evaluate==0.4.1
* bitsandbytes==0.43.1
* huggingface_hub==0.22.2
* trl==0.8.6
* peft==0.10.0r,val_dataloader,test_dataloader,X_train,Y_train
