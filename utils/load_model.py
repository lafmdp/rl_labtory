'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/9/4
'''


import numpy as np
import tensorflow as tf


class load_model():

    def __init__(self, model_path='./dense_model/model'):
        self.sess = tf.Session()

        with self.sess.as_default():
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(self.sess, model_path)

            self.graph = tf.get_default_graph()
            self.state_input = self.graph.get_tensor_by_name('policy/obs:0')
            self.deterministic_action = self.graph.get_tensor_by_name('policy/Net/action/Reshape:0')
            self.stochastic_action = self.graph.get_tensor_by_name('policy/Net/action/Reshape_1:0')

    def get_action(self, obs):
        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs).reshape([1, -1])
            feed_dict = {self.state_input: obs}

            actions = self.sess.run(self.deterministic_action, feed_dict=feed_dict)

        return actions[0]

if __name__ == '__main__':
    try:
        model = load_model("C:\\Users\\10719\\Desktop\\mask-hfo-1\\model")
    except Exception as e:
        print("Error occurs while loading!")
        exit()

    print("Successfully loaded model!")
