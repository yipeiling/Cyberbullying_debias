# ID-XCB: Data-independent Debiasing for cyberbullying detection
Implementation of paper: ID-XCB: Data-independent Debiasing for Fair and Accurate Transformer-based Cyberbullying Detection
<img src="https://github.com/yipeiling/Cyberbullying-debias/blob/main/Architecture.png" width=40% height=40%>
Architecture.png



## Datasets
#### Cyberbullying detection domain
HDCyberbullying is a celebrity cyberbullying dataset we created, which was extracted from  [Cyberbullying dataset](https://www.kaggle.com/datasets/surekharamireddy/malignant-comment-classification) and [Fake news dataset](https://paperswithcode.com/dataset/fakenewsamt-celebrity)
#### Emotion detection domain
We use [GoEmotions dataset](https://paperswithcode.com/dataset/goemotions)

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
