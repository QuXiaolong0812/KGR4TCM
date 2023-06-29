import tensorflow as tf
from model import KGR4TCM
from tool import get_top_k_exercises_id


# 预测结果并打印信息
def predict(args, datas, entity_id2item_index):
    predict_data = datas[0]
    n_entity = datas[1]
    n_relation = datas[2]
    ripple_set = datas[3]
    model = KGR4TCM( args, n_entity, n_relation )  # 初始化Ripple对象即实例化模型
    result = []

    if args.strong_recommend_mode:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            scores, predictions = evaluation(sess, args, model, predict_data, ripple_set, args.batch_size)
            temp = get_top_k_exercises_id(args, scores, entity_id2item_index)
            i = 0
            while i < args.top_k:
                result.append(temp[i].exercises_id)
                i += 1
    else:
        for item in ripple_set[0][1][2]:
            result.append(entity_id2item_index[item])
            if len(result) == args.top_k:
                break
    return result

def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):  # 喂入ripple_set每一跳的结果
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    scores = []
    predictions = []
    while start < data.shape[0]:  # 只对测试集进行评估
        scores, predictions = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        start += batch_size
    return scores, predictions


