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
from functions import sac_core
from functions.sac_core import get_vars

nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

hype_parameters = {
    "gamma": 0.99,
    "need_log": False,
    "init_lr": 1e-3,
    "polyak": 0.995
}


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float64)
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
        self.alpha = 0.2
        self.init_lr = hype_parameters["init_lr"]
        self.polyak = hype_parameters["polyak"]

        self.build_graph()

        self.n_training = 0

        with self.sess.as_default(), self.graph.as_default():
            if need_log:
                self.summary = tf.summary.FileWriter(log_path, self.sess.graph)

            self.saver = tf.train.Saver(self.get_variables())

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        ac_kwargs = {
            "hidden_sizes": [100, 100]
        }

        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope("sac"):
                # Inputs to computation graph
                self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = \
                    sac_core.placeholders(self.state_space, self.action_space, self.state_space, None, None)

                # Main outputs from computation graph
                with tf.variable_scope('main'):
                    self.mu, self.pi, logp_pi, q1, q2 = sac_core.mlp_actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

                with tf.variable_scope('main', reuse=True):
                    # compose q with pi, for pi-learning
                    _, _, _, q1_pi, q2_pi = sac_core.mlp_actor_critic(self.x_ph, self.pi, **ac_kwargs)

                    # get actions and log probs of actions for next states, for Q-learning
                    _, pi_next, logp_pi_next, _, _ = sac_core.mlp_actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

                # Target value network
                with tf.variable_scope('target'):
                    # target q values, using actions from *current* policy
                    _, _, _, q1_targ, q2_targ = sac_core.mlp_actor_critic(self.x2_ph, pi_next, **ac_kwargs)



                # Count variables
                var_counts = tuple(sac_core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])

                #print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n' % var_counts)

                # Min Double-Q:
                min_q_pi = tf.minimum(q1_pi, q2_pi)
                min_q_targ = tf.minimum(q1_targ, q2_targ)

                # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
                q_backup = tf.stop_gradient(
                    self.r_ph + self.gamma * (1 - self.d_ph) * (min_q_targ - self.alpha * logp_pi_next))

                # Soft actor-critic losses
                pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)
                q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
                q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
                value_loss = q1_loss + q2_loss

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

                # Value train op
                # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
                value_params = get_vars('main/q')
                with tf.control_dependencies([train_pi_op]):
                    train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

                # Polyak averaging for target variables
                # (control flow because sess.run otherwise evaluates in nondeterministic order)
                with tf.control_dependencies([train_value_op]):
                    target_update = tf.group([tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

                # All ops to call during one training step
                self.step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi,
                                 train_pi_op, train_value_op, target_update]

                # Initializing targets to match main variables
                self.target_init = tf.group([tf.assign(v_targ, v_main)
                                             for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

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


    def get_return(self, rewards):
        dis_rewards = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            dis_rewards[t] = running_add

        return dis_rewards

    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


    def train(self, batch):
        with self.sess.as_default(), self.graph.as_default():
            feed_dict = {self.x_ph: batch['obs1'],
                         self.x2_ph: batch['obs2'],
                         self.a_ph: batch['acts'],
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
