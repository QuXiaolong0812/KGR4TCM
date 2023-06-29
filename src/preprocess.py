# 这个py文件的目的是从原数据集中得到需要的字段，然后对齐item和实体的id（表示了同一个物品）
# 按照对齐的id，对推荐rating和图谱kg都制作一个方便处理的final文件
# 目标文件：
# kg_final.txt：h，r，t
# rating_final.txt：userid，itemid，rating

import argparse
import numpy as np

RATING_FILE_NAME = dict({'entities': 'Entities_ratings.csv'})  # 评分文件，目前我的项目文件中还没考虑
SEP = dict({'entities': ';'})  # 评分文件分割符
THRESHOLD = dict({'entities': 5})  # 按照预期假设，目前是否喜好的阈值是3


# 读取样本-实体文件，循环读出的内容：样本索引等于行数据的第一个元素，
# 顿悟id等于行数据的第二个元素。对新样本索引字典中的样本索引键重新赋值，
# 从i=0开始，每次自增1，直到文件读取到最后一行，对实体索引字典中的顿悟id键采取同样操作。
def read_item_index_to_entity_id_file():
    file = '../chinese_medicine_data/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


# 按顺序流程逐行解释
# 将新的样本字典的值赋值给样本集合，用户正负评级分别初始化为空的字典。
# 打开评级文件，从第二行开始循环，清除两边的空字符再用分隔符分开组成array。
# 旧的样本索引等于array的第二列，如果旧样本索引不在新样本索引则跳出本次循环。
# 样本索引等于新的索引字典中旧索引键的值，旧的用户索引等于整数化array的第一列，评级等于浮点化array的第三列。
# 如果评级大于数据集的阈值，就把旧的用户索引和样本索引作为键值对存入用户正评级字典中，否则存入用户负评级字典。
# 循环正字典的键值对，把旧的用户索引和用户数量（通过循环自加1）作为键值对存入新的用户字典中。
# 将新字典中旧用户索引的键值赋值给用户索引。循环正样本，将用户索引和样本索引写入文件中。
# 用样本集合减去正负样本集合得到无监督的样本。
# 紧接着在无监督样本里随机不重复采样长度为正样本长度的列表，循环写入用户索引和样本索引。
def convert_rating():
    file = '../chinese_medicine_data/'+RATING_FILE_NAME[DATASET]

    print('reading rating file: ' + file + ' ...')
    item_set = set(item_index_old2new.values())
    user_positive_ratings = dict()  # 用户正相关评分
    user_negative_ratings = dict()  # 用户负相关评分

    for line in open(file, encoding='utf-8').readlines()[1:]:   # 先将第一行读出
        array = line.strip().split(SEP[DATASET])

        #  remove prefix and suffix quotation marks for BX dataset
        array = list(map(lambda x: x[1:-1], array))  # 去除引号操作

        item_index_old = array[1]   # 习题ID
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index_new = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:    # 正相关
            if user_index_old not in user_positive_ratings:
                user_positive_ratings[user_index_old] = set()
            user_positive_ratings[user_index_old].add(item_index_new)
        else:
            if user_index_old not in user_negative_ratings:
                user_negative_ratings[user_index_old] = set()
            user_negative_ratings[user_index_old].add(item_index_new)

    print('converting rating file ...')
    writer = open('../chinese_medicine_data/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_positive_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_negative_ratings:
            unwatched_set -= user_negative_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


# 逐行读取图谱文件的内容并按制表符分离得到旧的头部实体，关系和尾部实体。
# 然后往实体索引字典中添加旧头实体-实体数量键值对和旧尾实体-实体数量键值对，
# 往关系索引字典里添加旧关系-关系数量键值对。再分别取出头实体，关系和尾实体写入文件中。
def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)  # 该变量记录当前知识图谱实体数组中已有多少实体数量
    relation_cnt = 0  # 关系数量

    writer = open('../chinese_medicine_data/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'entities':
        files.append(open('../chinese_medicine_data/kg_rehashed.txt', encoding='utf-8'))

    for file in files:
        for line in file:
            array = line.strip().split('\t')    # 分割行字符串，获取头结点、子节点、关系节点
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:     # 节点不在实体数组内，则把实体存储实体数组并同时获得一个编号
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]    # 记住头结点的实体编号

            if tail_old not in entity_id2index:     # 同上
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:   # 对关系节点进行同样的操作
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='entities', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('data preprocess done!')
