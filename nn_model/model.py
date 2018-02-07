import tensorflow as tf
import numpy as np


class TFModel:
    def __init__(self, emb_size: int, n_comp: int, logdir: str = None,
                 saved_model_path: str = None, restore: bool = False) -> None:
        assert (saved_model_path is not None) if restore else True,\
            "saved_model_path must be specified only if restore is True!"
        tf.reset_default_graph()



class MultiComp:
    def __init__(self, emb_size: int, n_comp: int, logdir: str = None,
                 saved_model_path: str = None, restore: bool = False) -> None:

        assert (saved_model_path is not None) if restore else True,\
            "saved_model_path must be specified only if restore is True!"

        tf.reset_default_graph()

        self.pos = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='pos_ph')
        self.neg = tf.placeholder(dtype=tf.float32, shape=[None, emb_size], name='neg_ph')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_ph')

        comp_init = np.random.randn(n_comp, emb_size) / np.sqrt(emb_size * n_comp)
        self.sense_comps = tf.Variable(comp_init, dtype=tf.float32, name='senses')

        # Attention
        #         self.centroid = tf.get_variable("centroid", shape=[1, emb_size],
        #                                 initializer=tf.contrib.layers.xavier_initializer())
        #         self.att_cont_weights = tf.matmul(self.pos, self.centroid, transpose_b=True, name='get_logits_context_att') # [None x 1]
        #         self.att_cont_weights = tf.nn.softmax(self.att_cont_weights, axis=0, name='softmax_context_att') # [? x 1]
        #         print(self.att_cont_weights)

        #         mean_cont = tf.reduce_mean(self.pos, axis=0, keep_dims=True)
        #         mean_cont = tf.matmul(self.att_cont_weights, self.pos, transpose_a=True) # [1 x 100]
        mean_cont = tf.reduce_mean(self.pos, axis=0, keep_dims=True)
        mean_cont = tf.nn.l2_normalize(mean_cont, 1)  # 1 x 100

        norm_sens = tf.nn.l2_normalize(self.sense_comps, 1)  # n_comp x 100

        att = tf.reduce_sum(mean_cont * norm_sens, axis=1, keep_dims=True)
        #         noise = tf.random_normal(att.get_shape()) * tf.reduce_mean(att) / 3
        #         att += noise
        self.att = att
        #         self.att = tf.nn.softmax(att, dim=0)
        word_emb = tf.reduce_sum(self.sense_comps * self.att, axis=0, keep_dims=True)

        # Cosine loss
        norm_pos = tf.nn.l2_normalize(self.pos, 1)
        norm_neg = tf.nn.l2_normalize(self.neg, 1)
        norm_word_emb = tf.nn.l2_normalize(word_emb, 1)

        pos_loss = tf.reduce_mean(tf.reduce_sum(norm_pos * norm_word_emb, axis=1))
        neg_loss = tf.reduce_mean(tf.reduce_sum(norm_neg * norm_word_emb, axis=1))

        self.loss = -pos_loss + neg_loss  # + 1e-3*tf.nn.l2_loss(self.sense_comps)
        opt = tf.train.AdamOptimizer(self.lr)
        # opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = opt.minimize(self.loss, name='train_op')

        self.sess = tf.Session()

        if logdir is not None:  # save graph files for tensorboard and start to record changes
            self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

        self.saver = tf.train.Saver()
        if restore:
            self.saver.restore(self.sess, saved_model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train_on_sample(self, pos_samp, neg_samp, learning_rate):
        feed_dict = {self.pos: pos_samp, self.neg: neg_samp, self.lr: learning_rate}
        loss, att, _ = self.sess.run([self.loss, self.att, self.train_op], feed_dict)
        return loss, att

    def predict_on_sample(self, samp):
        return self.sess.run(self.att, {self.pos: samp})

    def get_linear_combination(self, samp):
        att, vecs = self.sess.run([self.att, self.sense_comps], {self.pos: samp})
        return np.sum(att * vecs, axis=0)

    def save_model(self, model_path: str = None):
        self.writer.close()  # close writer for tensorboard
        self.saver.save(self.sess, save_path=model_path)
