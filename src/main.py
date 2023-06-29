import argparse
import numpy as np
from data_loader import load_data
from predict import predict
from tool import read_item_index_to_entity_id_file

np.random.seed(555)  # 可复现随机种子

# default settings for KGR4TCM
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='entities', help='which dataset to use')  # 数据集
parser.add_argument('--dim', type=int, default=4, help='dimension of entity and relation embeddings')  # 嵌入维度
parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')  # 跳数
parser.add_argument('--top_k', type=int, default=6, help='the number of recommended items')  # 返回的推荐系数
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')  # kge权重
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # learning rate 学习率
parser.add_argument('--batch_size', type=int, default=2261,
                    help='batch size')  # 批次大小,如果想要一次性返回结果，则batch_size的值最好为中医药实体数量
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')  # 每一跳的记忆大小
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')  # 更新每一跳结果的方式
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')  # 得到user向量的方式
parser.add_argument('--strong_recommend_mode', type=bool, default=False,
                    help='whether using strong recommended mode to recommend')  # 得到user向量的方式
# 后两维的参数都在model函数中发挥作用。

args = parser.parse_args()  # 转载入参数

# 初始化映射关系
item_index2entity_id = dict()
entity_id2item_index = dict()
read_item_index_to_entity_id_file(item_index2entity_id, entity_id2item_index)

inputs = input("")
input_exercises = [n for n in inputs.split()]
input_entities = []
i = 0
while i < len(input_exercises):
    input_entities.append(item_index2entity_id[input_exercises[i]])
    i += 1

datas = load_data(args, input_entities)  # 加载数据，返回图谱信息、ripple涟漪集合、用户交互历史以及预测样本

result = predict(args, datas, entity_id2item_index)  # 开始预测数据
print(result)
