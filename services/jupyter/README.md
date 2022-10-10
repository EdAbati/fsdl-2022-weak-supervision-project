# FSDL Course 2022 - Weak Supervision and Deep Learning with text data

Team 44

# Weakly supervised dataset with unlabeled data from AG NEWS dataset

https://huggingface.co/datasets/bergr7/weakly_supervised_ag_news

```python
from datasets import load_dataset
dataset = load_dataset("bergr7/weakly_supervised_ag_news")
```

## Source data

AG is a collection of more than 1 million news articles. News articles have been
gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
activity. ComeToMyHead is an academic news search engine which has been running
since July, 2004. The dataset is provided by the academic comunity for research
purposes in data mining (clustering, classification, etc), information retrieval
(ranking, search, etc), xml, data compression, data streaming, and any other
non-commercial activity. For more information, please refer to the link
http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang
(xiang.zhang@nyu.edu) from the dataset above. It is used as a text
classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
LeCun. Character-level Convolutional Networks for Text Classification. Advances
in Neural Information Processing Systems 28 (NIPS 2015).

## Environment

Create a new virtual envinronment(You need to have conda installed):

```bash
conda env create -f environment.yml
```

## How to run Rubrix

> Note - Original Rubrix dataset not available - WIP

Requisites -> Docker

Create container with ElasticSearch for Rubrix:

```Docker
docker run -d --name elasticsearch-for-rubrix -p 9200:9200 -p 9300:9300 -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.10.2
```

Then with the venv activate run:

```bash
python -m rubrix
```

Afterward, you should be able to access the web app at http://localhost:6900/. The default username and password are rubrix and 1234.

![Rules on Rubrix](./img/rules_rubrix.png)
