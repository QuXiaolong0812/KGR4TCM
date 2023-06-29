import tensorflow as tf


class KGR4TCM( object ):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    # 设置样本和标签变量，为头部实体、关系、尾部实体创建三个空的列表。
    # 循环总跳数，为三个空列表提前预留好迭代跳数将时产生的相应数据。
    def _build_inputs(self):
        # 输入有items id，labels和用户每一跳的ripple set记录
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):   # 每一跳的结果
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    # 初始化实体嵌入矩阵和关系嵌入矩阵，初始化的每一层的输出与输入必须是同方差的，并且前向传播与反向传播时梯度也是同方差的。
    # 初始权重值应当为均值为0，的正态分布。
    def _build_embeddings(self):    # 得到嵌入
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        # relation连接head和tail所以维度是self.dim*self.dim
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    # 初始化转换矩阵，创建该矩阵的目的是为了更新每跳结尾的样本嵌入。
    # 将实体嵌入矩阵中的索引为样本变量的项取出赋值给样本嵌入矩阵。
    # 将空列表赋值给头部实体、关系、尾部实体三个嵌入，随即循环总跳数：在每次循环中分别去实体嵌入矩阵和关系嵌入矩阵中寻找索引为记忆实体和记忆关系列表中该跳的内容对应的矩阵添加到实体嵌入和关系嵌入列表。
    # 然后调用"关键处理"方法得到总跳数的响应，再调用预测函数算出模型的分数，再进行sigmoid归一化。
    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        # 更新item嵌入的转换矩阵，这个不一定是必要的，可以使用直接替换或者加和策略。
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]，得到item的嵌入
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):     # 得到每一跳的实体，关系嵌入list
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        # 按公式计算每一跳的结果
        o_list = self._key_addressing()

        # 得到分数
        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    # 循环总跳数：每次循环时，在头部实体嵌入列表的第hop个元素的维度3处增加一个维度然后赋值给扩展的头部实体矩阵，
    # 然后将该矩阵和关系嵌入列表中第hop个元素相乘，并删除维度3的轴处大小为1的维度赋值给实体和关系的乘积。
    # 在样本嵌入列表的第hop个元素的维度2处增加一个维度赋值给样本矩阵，然后将Rh和v相乘并删除上一步增加的维度得到样本v和实体h在关系R的空间中的相似度，
    # 再经过softmax归一化。然后将归一化的概率增加一个维度去和尾部实体嵌入列表的第hop个元素相乘，
    # 然后计算上一步的得到的张量各个维度上元素的总和赋值给样本的hop阶响应，然后调用方法更新样本嵌入，最后把hop阶响应添加到响应列表中。
    def _key_addressing(self):  # 得到olist
        o_list = []
        for hop in range(self.n_hop):   # 依次计算每一跳
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]，计算Rh，使用matmul函数
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]，然后和v内积计算相似度
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]，softmax输出分数
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]，然后分配分数给尾节点得到o
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    #
    def update_item_embedding(self, item_embeddings, o):
        # 计算完hop之后，更新item的Embedding操作，可以有多种策略
        if self.item_update_mode == "replace":  # 直接换
            item_embeddings = o
        elif self.item_update_mode == "plus":   # 加到一起
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":  # 用前面的转换矩阵
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":     # 用矩阵而且再加到一起
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    # 把响应列表里面的元素全部取出拼接成一维向量和样本嵌入矩阵相乘，计算各个维度上元素总和赋值给得分
    def predict(self, item_embeddings, o_list):
        y = o_list[-1]  # -1表示只用olist的最后一个向量
        if self.using_all_hops:     # 或者如果前面参数里面设置了则使用所有向量的相加来代表user
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]  ，user和item算内积得到预测值
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    # 计算给定预测值和真实标签之间的交叉熵损失赋值给基础损失；
    # 迭代相加关系r的指标张量的切片（也就是当前跳）和重构的指标矩阵两者之间的平方误差，这一步操作在代码中的做法是头实体、
    # 关系、尾实体三个嵌入矩阵相乘再做sigmod归一化然后取各个维度上元素的平均值，循环相加总跳数次，最后再乘以图谱的权重；
    # 迭代头实体嵌入、关系嵌入和尾实体嵌入三个列表在当前跳的内容的平方求和再取均值得到三部分的loss，
    # 迭代过程中将三者依次相加，循环结束时即可得到最终的正则化器来防止模型过拟合。最后将三种损失相加。
    def _build_loss(self):  # 损失函数有三部分
        # 1用于推荐的对数损失函数
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        # 2知识图谱表示的损失函数
        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))    # 为hRt的表示是否得当
        self.kge_loss = -self.kge_weight * self.kge_loss

        # 3正则化损失
        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss   # 三者相加

    # 采用tf中引入了二次方梯度校正全局最优点的Adam优化算法，最小化损失函数。
    def _build_train(self):     # 使用adam优化
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''

    # 将优化器、损失函数和数据放入tf中开始训练
    def train(self, sess, feed_dict):   # 开始训练
        return sess.run([self.optimizer, self.loss], feed_dict)

    # 运行会话，计算模型的精准率和召回率
    def eval(self, sess, feed_dict):    # 开始测试
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        # 计算auc和acc
        # auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        # acc = np.mean(np.equal(predictions, labels))
        return scores, predictions

