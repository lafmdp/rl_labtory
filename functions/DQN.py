'''
  Proximal policy optimization method for policy.

  @python version : 3.6.4
  @author : pangjc
  @time : 2019/7/2
'''

import os
import random
import datetime
import numpy as np
import tensorflow as tf

nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class replay_buffer():

    def __init__(self, size=100000, state_size=17):

        self.size = size
        self.batchsize=256
        self.state_size = state_size

        self.recorder = np.array([], dtype=np.float)

    def append(self, batch):

        sample_num = self.recorder.shape[0]
        if  sample_num > self.size:
            index = np.random.choice(sample_num, int(self.size), replace=False).tolist()
            self.recorder = self.recorder[index]
        else:
            pass

        s = np.vstack(batch["state"]).astype(np.float32).reshape((-1,self.state_size))
        a = np.hstack(batch["action"]).astype(np.float32).reshape((-1,1))
        s_ = np.vstack(batch["state_"]).astype(np.float32).reshape((-1, self.state_size))
        r = np.hstack(batch["reward"]).astype(np.float32).reshape((-1,1))
        done = np.hstack(batch["done"]).astype(np.float32).reshape((-1,1))

        recorder = np.hstack((s, s_, a, r, done))

        if sample_num == 0:
            self.recorder = recorder.copy()
        else:
            self.recorder = np.vstack((self.recorder, recorder))

    def sample(self):

        sample_num = self.recorder.shape[0]
        if sample_num < self.batchsize:
            sample_index = np.random.choice(sample_num, self.batchsize, replace=True).tolist()
        else:
            sample_index = np.random.choice(sample_num, self.batchsize, replace=False).tolist()

        ret = {
            "s":self.recorder[sample_index, :self.state_size],
            "s_":self.recorder[sample_index, self.state_size:2*self.state_size],
            "a":self.recorder[sample_index, -3].astype(np.int),
            "r":self.recorder[sample_index, -2],
            "done":self.recorder[sample_index, -1]
        }

        return ret

class policy():

    def __init__(self,
                 state_space=11,
                 action_space=6,
                 have_model=False,
                 model_name='policy',
                 model_path='./Documents/PolicyModel/{}/'.format(nowTime),
                 ):


        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.state_space = state_space
        self.action_space = action_space
        self.model_path = model_path
        self.model_name = model_name
        self.have_model = have_model

        self.epislon = 0.05
        self.learning_rate = 3e-4
        self.gamma = 0.99

        self.build_network()

        self.n_training = 0

        with self.sess.as_default(), self.graph.as_default():

            self.saver = tf.train.Saver(self.get_variables())

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())


    def q_net(self, input, name):
        with tf.variable_scope(name):
            out = tf.layers.dense(input, 128, tf.nn.relu)
            out = tf.layers.dense(out, 256, tf.nn.relu)
            out = tf.layers.dense(out, 128, tf.nn.relu)
            out = tf.layers.dense(out, 64, tf.nn.relu)

            Q = tf.layers.dense(out, self.action_space, None)

        return Q


    def build_network(self):
        with self.sess.as_default(), self.graph.as_default():
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_space], name='obs')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, ], name="reward_placeholder")
            self.state_ = tf.placeholder(dtype=tf.float32, shape=[None, self.state_space], name="state__placeholder")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name="action_placeholder")
            self.done = tf.placeholder(dtype=tf.float32, shape=[None, ], name="done_placeholder")

            with tf.variable_scope("q_function"):
                self.q1 = self.q_net(self.state, "q1")
                self.target_q1 = self.q_net(self.state_, "q1_target")

                self.q2 = self.q_net(self.state, "q2")
                self.target_q2 = self.q_net(self.state_, "q2_target")

                target_index = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32),
                                         tf.cast(tf.argmax(self.target_q1, axis=-1), dtype=tf.int32)], axis=1)

                target = self.rewards + self.gamma * (1-self.done) * tf.gather_nd(self.target_q2, target_index)
                target = tf.stop_gradient(target)

                a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
                q_eval = tf.gather_nd(params=self.q1, indices=a_indices)
                q_eval2 = tf.gather_nd(params=self.q2, indices=a_indices)

                self.loss = tf.reduce_mean((target-q_eval)**2) + \
                            tf.reduce_mean((target - q_eval2) ** 2)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = optimizer.minimize(self.loss)

                self.scope = tf.get_variable_scope().name

            self.q1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_function/q1")
            self.target_q1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_function/q1_target")
            update_target = [tf.assign(t, q*0.01+t*0.99) for t,q in zip(self.target_q1_params, self.q1_params)]

            self.q2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_function/q2")
            self.target_q2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_function/q2_target")
            self.update_target = update_target + [tf.assign(t, q*0.01+t*0.99) for t, q in zip(self.target_q2_params, self.q2_params)]

    def get_action(self, obs):
        import random

        if random.uniform(0,1) < self.epislon:
            ret = {
                "actions": random.randint(0, self.action_space-1)
            }
        else:
            with self.sess.as_default(), self.graph.as_default():
                obs = np.array(obs).reshape((1,-1))

                q1, q2 = self.sess.run([self.q1, self.q2], feed_dict={self.state: obs})

                ret = {
                    "actions": (q1+q2)[0].argmax()
                }

        return ret


    def get_means(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs).reshape((1, -1))

            q1,q2 = self.sess.run([self.q1,self.q2], feed_dict={self.state: obs})

            ret = {
                "actions": (q1+q2)[0].argmax()
            }

        return ret


    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


    def train(self, buffer):
        with self.sess.as_default(), self.graph.as_default():
            # self.sess.run(self.update_target)

            for _ in range(1000):
                sample = buffer.sample()

                loss, _, _ = self.sess.run([self.loss, self.train_op, self.update_target], feed_dict={
                    self.state:sample["s"],
                    self.state_:sample["s_"],
                    self.action:sample["a"].squeeze(),
                    self.done:sample["done"].squeeze(),
                    self.rewards:sample["r"].squeeze()
                })


    def save_model(self):

        with self.sess.as_default(), self.graph.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path, self.model_name))

    def load_model(self):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
