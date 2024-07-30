# ID-XCB: Data-independent Debiasing for cyberbullying detection
Implementation of paper: ID-XCB: Data-independent Debiasing for Fair and Accurate Transformer-based Cyberbullying Detection
<p align=center><img src="Architecture.png" width="500" center/>

## Main idea and implementation
We combine the three training losses to fine-tune a classifier. Adversarial training(**Embedding loss**) and task training(**BCELoss**) optimise ID-XCB model parameters on a training dataset and simultaneously enforce the fairness constraints(**FC**) on a validation set to reduce swear word impact.
<p align=center><img src="code.png" width="600" center/>

## Datasets 
#### [1] Instagram cyberbullying detection datasets （Need to request from the dataset author）
Homa Hosseinmardi, Sabrina Arredondo Mattson, Rahat Ibn Rafiq, Richard Han, Qin Lv, and Shivakant Mishra. 2015. **Analyzing labeled cyberbullying incidents on the instagram social network**. In _Socinfo_. Springer, 49–66.
#### [2] Vine cyberbullying detection datasets （Need to request from the dataset author）
Rahat Ibn Rafiq, Homa Hosseinmardi, Richard Han, Qin Lv, Shivakant Mishra, and Sabrina Arredondo Mattson. 2015. **Careful what you share in six seconds: Detecting cyberbullying instances in Vine**. In _ASONAM_. ACM, 617–622.
#### [3] Swear lexicon_1 (Publicly available)
Agrawal S, Awekar A. 2018 Mar 1. **Deep learning for detecting cyberbullying across multiple social media platforms**. In European conference on information retrieval 2018 Mar 1 (pp. 141-153). 
#### [4] Swear lexicon_2 (Publicly available)
We use Google bad words lists[https://code.google.com/archive/p/badwordslist/downloads]

### Python package version
* pandas==2.0.3
* scikit_learn=1.2.2
* torch==2.3.1+cu121
* numpy==1.25.2
* transformers==4.40.0
* datasets==2.18.0
* python==3.10.12
* json==2.0.9
* nltk==3.8.1
* huggingface_hub==0.22.2
* psutil==5.3.0
