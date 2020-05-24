'''

  @python version : 3.6.8
  @author : pangjc
  @time : 2019/11/17
'''
import os
import gym
import tensorflow as tf
import  numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from matplotlib import pyplot as plt
import seaborn as sns
fig,ax=plt.subplots()

def update_xml(index):
    os.system('cp ./xml_path/{}/walker2d.xml /home/pangjc/anaconda3/lib/python3.6/site-packages/gym/envs/mujoco/assets/walker2d.xml'.format(index))

    # os.system(
    #     'cp ./xml_path/{}/inverted_pendulum.xml /home/pangjc/anaconda3/lib/python3.6/site-packages/gym/envs/mujoco/assets/inverted_pendulum.xml'.format(
    #         index))

    # os.system(
    #     'cp ./xml_path/{}/hopper.xml /home/pangjc/anaconda3/lib/python3.6/site-packages/gym/envs/mujoco/assets/hopper.xml'.format(
    #         index))

class pi():

    def __init__(self, model_path='./policy'):

        self.sess = tf.Session()

        with self.sess.as_default():
            saver = tf.train.import_meta_graph(model_path+'.meta')
            saver.restore(self.sess, model_path)

            graph = tf.get_default_graph()
            self.state_input = graph.get_tensor_by_name('policy/obs:0')
            self.action = graph.get_tensor_by_name('policy/Net/action/means/Tanh:0')


    def get_action(self, state):

        with self.sess.as_default():

            state = np.array(state)
            state = state[np.newaxis, :]

            with self.sess.as_default():
                a = self.sess.run(self.action, feed_dict={self.state_input:state})[0]

            return a


if __name__ == "__main__":
    env = gym.make('Walker2d-v2')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high
    act_space = env.action_space

    for policy_type in ["policy","expert"]:

        if policy_type == "policy":
            update_xml(1)
        else:
            update_xml(2)

        time.sleep(2)

        if not os.path.exists("./{}_distribute".format(policy_type)):
            os.mkdir("./{}_distribute".format(policy_type))

        po = pi(model_path='./{}/policy'.format(policy_type))

        s_list = []
        game_num = 0

        while True:

            print('%s-%d'%(policy_type, game_num))
            s = env.reset()

            while True:

                a = po.get_action(s)
                # env.render()

                s_, r, done, _ = env.step(a * high[0])

                s_list.append(s)

                s = s_

                if done:

                    game_num += 1

                    break

            if game_num >= 500:

                for state_index in range(obs_dim):
                    this_state = np.array([state[state_index] for state in s_list])

                    ax.hist(this_state, bins=100, histtype="stepfilled", normed=True, alpha=0.6)
                    sns.kdeplot(this_state, shade=True)

                    plt.savefig("./{}_distribute/state[{}].jpg".format(policy_type, state_index))
                    plt.close()

                break



