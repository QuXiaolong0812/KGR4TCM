# KGR4TCM

This repository is the implementation of KGR4TCM (refer to RippleNet):
> KGR4TCM: Knowledge Graph-based Intelligent Recommender System for Traditional Chinese Medicine

KGR4TCM is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.

KGR4TCM is a modified traditional Chinese medicine entity recommendation system based on RippleNet. It is not intended for users who are professional doctors, but for a wide range of general users who are interested in TCM.

KGR4TCM will be submitted to the 9th TCM Information Conference for peer review.

### Files in the folder

- `chinese_medicine_data/`
  - `Entities_ratings.csv`: raw rating file of Book-Crossing dataset;交叉数据集的原始评级文件
  - `item_index2entity_id_rehashed.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;从原始评级文件中的项目索引到KG中的实体ID的映射；
  - `kg_rehashed.txt`: knowledge graph file;
- `src/`: implementations of KGR4TCM.



### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- tensorflow-gpu == 1.4.0
- numpy == 1.14.5
- sklearn == 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py
$ python main.py (note: use -h to check optional arguments)
```
