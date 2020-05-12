'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/2/7
'''

import os
import numpy as np
import tensorflow as tf

class behavior_cloning(object):

    def __init__(self, state_dim=1, act_dim=1, model_path=None, have_model=False):
        self.model_path = model_path
        self.model_name = 'behavior_cloning'
        self.graph = tf.Graph()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.log_path=self.model_path + "/log"

        self.learning_rate = 1e-3
        self.global_step = 1

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        self.build_network()
        self.def_saver()

        with self.sess.as_default(), self.graph.as_default():

            self.summary = tf.summary.FileWriter(self.log_path, self.sess.graph)

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())


    def def_saver(self):
        with self.sess.as_default(), self.graph.as_default():

            self.all_saver = tf.train.Saver(tf.global_variables())


    def build_network(self):

        with self.sess.as_default(), self.graph.as_default():


            self.state  = tf.placeholder(tf.float32, [None, self.state_dim], name='stack_state')
            self.a_label = tf.placeholder(tf.float32, [None, self.act_dim], name='real_action')

            with tf.name_scope('bc_network'):

                out = self.state

                out = tf.layers.dense(out, 100, activation=tf.nn.relu)
                out = tf.layers.dense(out, 100, activation=tf.nn.relu)

                self.action = tf.layers.dense(out, self.act_dim, activation=tf.nn.tanh)

            with tf.name_scope("train"):

                self.mse_loss = tf.losses.mean_squared_error(self.a_label, self.action)
                tf.summary.scalar('bc_loss', self.mse_loss)
                self.merge = tf.summary.merge_all()
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.mse_loss)


    def train(self, batch):

        with self.sess.as_default(), self.sess.graph.as_default():

            loss, _, summ = self.sess.run([self.mse_loss, self.train_op, self.merge],
                                    feed_dict={self.state: batch["state"],
                                               self.a_label:batch["action"]})
            self.summary.add_summary(summ, global_step=self.global_step)
            self.global_step += 1

        return loss


    def get_loss(self, batch):

        with self.sess.as_default(), self.sess.graph.as_default():

            loss = self.sess.run(self.mse_loss, feed_dict={self.state: batch["state"],
                                                           self.a_label:batch["action"]})

        return loss


    def get_means(self, state):

        with self.sess.as_default(), self.sess.graph.as_default():

            state = np.array(state).reshape([1, -1])

            action = self.sess.run(self.action,feed_dict={
                self.state:state,
            })[0]

        return action


    def save_model(self):

        with self.sess.as_default(), self.sess.graph.as_default():
            self.all_saver.save(self.sess, os.path.join(self.model_path, self.model_name))


    def load_model(self):

        with self.sess.as_default(), self.sess.graph.as_default():
            self.all_saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
