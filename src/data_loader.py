# 尝试在此文件中自定义函数
# data_loader.py train.py and model.py need to be rewritten by Qu Xiaolong

import collections
import os
import numpy as np


def load_data(args, input_entities):
    n_entity, n_relation, kg = load_kg(args)
    user_history_dict = dict()
    user_history_dict[0] = input_entities
    print(user_history_dict)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    predict_data = load_all_recommended_rating(args)

    return predict_data, n_entity, n_relation, ripple_set


def load_all_recommended_rating(args):
    # reading rating file
    rating_file = '../chinese_medicine_data/all_recommended_rating_final'
    if os.path.exists(rating_file + '.npy'):  # 防止二次创建
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_ratings = rating_np.shape[0]
    train_indices = np.random.choice(n_ratings, size=int(n_ratings * 1), replace=False)
    train_data = rating_np[train_indices]

    return train_data


# 加载图谱文件，将加载得到的图谱矩阵通过切片操作再用set()去重取长度得到实体和关系的数量，返回实体数量、关系数量和图谱字典
def load_kg(args):
    # reading kg file
    kg_file = '../chinese_medicine_data/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


# 通过循环加载得到的矩阵构建图谱字典，键为头部实体，值为尾部实体和关系组成的triple
def construct_kg(kg_np):
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))  # 就是按照head建立字典，将尾节点和关系放入到字典中
    return kg


# 将一个创建实例为list的默认字典复制给ripplenet_set，循环用户历史字典，在字典键值对中进行内部跳数（n_hop）循环：
# 如果跳数为0，则最后一跳的尾部实体为用户历史字典中当前键值对的所有值，否则为水波字典中当前键值对列表值中最后一个三元组的第二个元素。
# 紧接着循环尾部实体，将图谱中该实体对应的三元组循环赋值给memories_h,memories_r,memories_t列表。
# 如果头实体的记忆列表中没有元素，则在水波字典中的用户键对应的值列表中追加当前列表的最后一个三元组，否则就为每个用户采样一个固定大小的一跳记忆
# 随机生成一个范围在头实体列表长度大小，尺寸为记忆数目的列表，列表内元素是否重复取决于列表长度和及记忆数目的大小关系，
# 再通过生成的列表筛选得到一组新的三元组列表追加到用户键对应的值列表中，最终返回水波字典。
def get_ripple_set(args, kg, user_history_dict):
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:  # 对于每个用户
        for h in range(args.n_hop):  # 该用户的兴趣在KG多跳hop中
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:  # 如果不传播，上一跳的结果就直接是该用户的历史记录
                tails_of_last_hop = user_history_dict[user]
            else:  # 去除上一跳的记录
                tails_of_last_hop = ripple_set[user][-1][2]

            # 去除上一跳的三元组特征
            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # 为每个用户采样固定大小的邻居
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
