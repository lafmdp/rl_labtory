'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/10/8
'''

import numpy as np
import tensorflow as tf

class existing_gail_pi():

    def __init__(self, model_path='./policy'):

        self.sess = tf.Session()

        with self.sess.as_default():
            saver = tf.train.import_meta_graph(model_path+'.meta')
            saver.restore(self.sess, model_path)

            graph = tf.get_default_graph()
            self.state_input = graph.get_tensor_by_name('gail_policy/obs:0')
            self.action = graph.get_tensor_by_name('gail_policy/Net/actor/means/Tanh:0')


    def get_action(self, state):

        state = np.array(state)
        state = state[np.newaxis, :]

        with self.sess.as_default():
            a = self.sess.run(self.action, feed_dict={self.state_input:state})[0]

        return a
