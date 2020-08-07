'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/10/8
'''

import numpy as np
import tensorflow as tf

class existing_pi():

    def __init__(self, model_path='./policy'):

        self.sess = tf.Session()

        with self.sess.as_default():
            saver = tf.train.import_meta_graph(model_path+'.meta')
            saver.restore(self.sess, model_path)

            graph = tf.get_default_graph()
            self.state_input = graph.get_tensor_by_name('policy/obs:0')
            self.deterministic_action = graph.get_tensor_by_name('policy/Net/action/means/Tanh:0')
            self.stochastic_action = graph.get_tensor_by_name('policy/Net/action/clip_by_value:0')


    def get_means(self, state):

        state = np.array(state)
        state = state[np.newaxis, :]

        with self.sess.as_default():
            a = self.sess.run(self.stochastic_action, feed_dict={self.state_input:state})[0]

        return a

    def get_action(self, state):

        state = np.array(state)
        state = state[np.newaxis, :]

        with self.sess.as_default():
            a = self.sess.run(self.deterministic_action, feed_dict={self.state_input:state})[0]

        return a
