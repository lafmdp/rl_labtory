'''
  Proximal policy optimization method for policy.

  @python version : 3.6.4
  @author : pangjc
  @time : 2019/7/2
'''

import os
import copy
import datetime
import numpy as np
import tensorflow as tf
from utils.sac_core import *

nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

hype_parameters = {
    "gamma": 0.99,
    "need_log": False,
    "init_lr": 1e-3,
    "polyak": 0.995,
    "alpha_auto": True,
    "target_entropy_auto": True
}

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.acts_buf = np.zeros([size, 1], dtype=np.float64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, batch):

        for state, state_,action, reward, done \
                 in zip(batch["state"],
                        batch["state_"],
                        batch["action"],
                        batch["reward"],
                        batch["done"]):

            for s,s_,a,r,d in zip(state,state_,action,reward,done):

                self.obs1_buf[self.ptr] = s
                self.obs2_buf[self.ptr] = s_
                self.acts_buf[self.ptr] = a
                self.rews_buf[self.ptr] = r
                self.done_buf[self.ptr] = d
                self.ptr = (self.ptr + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=1024):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class policy():

    def __init__(self,
                 state_space=11,
                 action_space=6,
                 need_log=False,
                 have_model=False,
                 model_name='policy',
                 log_path='./Documents/PolicyModel/{}/logs/'.format(nowTime),
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
        self.need_log = need_log

        self.gamma = hype_parameters["gamma"]
        self.init_lr = hype_parameters["init_lr"]
        self.polyak = hype_parameters["polyak"]
        self.target_entropy_auto = hype_parameters["target_entropy_auto"]
        self.alpha_auto = hype_parameters["alpha_auto"]

        self.build_graph()

        self.n_training = 0

        with self.sess.as_default(), self.graph.as_default():

            self.saver = tf.train.Saver(self.get_variables())

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope("sac"):

                # Inputs to computation graph
                self.x_ph, self.a_ph,self.x2_ph, self.r_ph, self.d_ph = \
                    placeholders(self.state_space, self.action_space, self.state_space, None, None)

                alpha = "auto"

                # alpha Params
                if self.target_entropy_auto:  # discrete case should be self.target_entropy_auto < ln(act_dim)
                    target_entropy = tf.log(tf.cast(self.action_space, tf.float32)) * 0.5
                else:
                    target_entropy = tf.cast(self.target_entropy_auto, tf.float32)

                log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=target_entropy)

                if self.alpha_auto:  # auto tune alpha
                    alpha = tf.exp(log_alpha)
                else:  # fixed alpha
                    alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=alpha)

                network_params = {
                    'hidden_sizes': [64, 64],
                    'activation': 'relu',
                    'policy': kl_policy
                }

                # Main outputs from computation graph
                with tf.variable_scope('main'):
                    self.mu, self.pi, pi_entropy, self.pi_logits, self.q1_logits, \
                    q2_logits, q1_a, q2_a, q1_pi, self.q2_pi = \
                        a_out_mlp_actor_critic(self.x_ph, self.a_ph, alpha, **network_params)

                # Target value network
                with tf.variable_scope('target'):
                    _, _, pi_entropy_targ, _, _, _, _, _, q1_pi_targ, q2_pi_targ = \
                        a_out_mlp_actor_critic(self.x2_ph, self.a_ph, alpha, **network_params)

                # Min Double-Q:
                min_q_logits = tf.minimum(self.q1_logits, q2_logits)
                min_q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)

                # Targets for Q regression
                q_backup = self.r_ph + self.gamma * (1 - self.d_ph) * tf.stop_gradient(
                    min_q_pi_targ + alpha * pi_entropy_targ)

                # critic losses
                q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a) ** 2)
                q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a) ** 2)
                value_loss = q1_loss + q2_loss

                # kl using cross entropy (D_KL = H(P,Q) - H(P))
                pi_action_probs = tf.nn.softmax(self.pi_logits, axis=-1)
                q_log_action_probs = tf.nn.log_softmax(min_q_logits, axis=-1)
                pi_q_cross_entropy = -tf.reduce_sum(pi_action_probs * q_log_action_probs, axis=-1)
                pi_loss = tf.reduce_mean(pi_q_cross_entropy - alpha * pi_entropy)

                # alpha loss for temperature parameter
                alpha_backup = tf.stop_gradient(target_entropy - pi_entropy)
                alpha_loss = -tf.reduce_mean(log_alpha * alpha_backup)

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr, epsilon=1e-08)
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

                # Value train op
                # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr, epsilon=1e-08)
                with tf.control_dependencies([train_pi_op]):
                    train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q'))

                alpha_optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr, epsilon=1e-08)
                with tf.control_dependencies([train_value_op]):
                    train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=get_vars('log_alpha'))

                # Polyak averaging for target variables
                # (control flow because sess.run otherwise evaluates in nondeterministic order)
                with tf.control_dependencies([train_value_op]):
                    self.target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                                   for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

                self.step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a,
                                 pi_entropy, target_entropy, alpha_loss, alpha,
                                 train_pi_op, train_value_op, train_alpha_op, self.target_update]

    def get_action(self, obs):
        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            a = self.sess.run(self.pi, feed_dict={self.x_ph: obs})[0]

        return a

    def get_means(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            a = self.sess.run(self.mu, feed_dict={self.x_ph: obs})[0]

        return a

    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_return(self, rewards):
        dis_rewards = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            dis_rewards[t] = running_add

        return dis_rewards

    def train(self, batch):
        with self.sess.as_default(), self.graph.as_default():
            values = batch["acts"].astype(np.int32)
            n_values = self.action_space
            action_one_hot = np.eye(n_values)[values].squeeze()

            # print("obs1 shape", batch["obs1"].shape)
            # print("obs2 shape", batch["obs2"].shape)
            # print("rews shape", batch["rews"].shape)
            # print("done shape", batch["done"].shape)

            feed_dict = {self.x_ph: batch['obs1'],
                         self.x2_ph: batch['obs2'],
                         self.a_ph: action_one_hot,
                         self.r_ph: batch['rews'],
                         self.d_ph: batch['done'],
                         }


            self.sess.run(self.step_ops, feed_dict)

    def save_model(self):

        with self.sess.as_default(), self.graph.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path, self.model_name))

    def load_model(self):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
