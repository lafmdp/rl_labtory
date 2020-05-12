'''
  PPO algorithm for base policy.
  Based on PPO algorithm (OpenAI)
  This code is adapted from https://github.com/uidilr/ppo_tf/blob/master/ppo.py

  @Author : pangjc

'''

import os
import copy
import numpy as np
import tensorflow as tf

import datetime
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
if not os.path.exists('../Documents/'):
    os.mkdir('../Documents')

class Policy_net:

    def  __init__(self, name: str, sess, state_space, act_space, activation=tf.nn.relu, units=128):
        '''
        Network of PPO algorithm
        :param k: history used
        :param name: string
        :param sess:
        :param ob_space:
        :param act_space:
        :param activation:
        :param units:
        '''
        self.sess = sess

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, state_space], name='obs')
            with tf.variable_scope('policy_net'):
                out = tf.layers.dense(self.obs, 128, tf.nn.relu, trainable=True)
                out = tf.layers.dense(out, 256, tf.nn.relu, trainable=True)
                out = tf.layers.dense(out, 128, tf.nn.relu, trainable=True)
                out = tf.layers.dense(out, 64, tf.nn.relu, trainable=True)
                self.act_probs = tf.layers.dense(inputs=out, units=act_space, activation=tf.nn.softmax, trainable=True)

            with tf.variable_scope('value_net'):
                out2 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu, trainable=True)
                out2 = tf.layers.dense(inputs=out2, units=256, activation=tf.nn.relu, trainable=True)
                out2 = tf.layers.dense(inputs=out2, units=128, activation=tf.nn.relu, trainable=True)
                self.v_preds = tf.layers.dense(inputs=out2, units=1, activation=None, trainable=True)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        obs = np.array(obs)

        if obs.shape[0] != 1:
            obs = obs[np.newaxis, :]

        if stochastic:   # act in a stochastic form
            act_prob, act, v_preds = self.sess.run([self.act_probs, self.act_stochastic, self.v_preds],
                                                    feed_dict={self.obs: obs})
        else:
            act_prob, act, v_preds = self.sess.run([self.act_probs, self.act_deterministic, self.v_preds],
                                                    feed_dict={self.obs: obs})

        return act[0], act_prob[0][act[0]], v_preds[0, 0]

    def get_action_prob(self, obs):

        v, act_prob = self.sess.run([self.v_preds, self.act_probs], feed_dict={self.obs: obs})

        return act_prob[0], v[0,0]

    def get_variables(self):
        with self.sess.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOTrain:

    def __init__(self, name, sess, Policy, action_space, gamma=0.99, clip_value=0.2, c_1=0.5, c_2=0.05, lr=1e-5):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value: to clip ratio
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        self.Policy = Policy
        #self.Old_Policy = Old_Policy
        self.sess = sess
        self.gamma = gamma
        self.lamda = 0.95
        self.batch_size = 256
        self.epoch_num = 10
        self.clip_value = clip_value
        self.c_1 = c_1
        self.c_2 = c_2
        self.adam_lr = lr
        self.adam_epsilon = 1e-6
        gra_clip = 1

        with tf.name_scope(name):

            # inputs for train_op
            with tf.variable_scope('train_inp'):
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
                self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
                self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
                self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                self.mu_prob = tf.placeholder(dtype=tf.float32, shape=[None], name='mu_prob')

            act_probs = self.Policy.act_probs

            # probabilities of actions which agent took with policy
            act = tf.one_hot(indices=self.actions, depth=action_space)
            act_probs = act_probs * act
            act_probs = tf.reduce_sum(act_probs, axis=1)

            with tf.variable_scope('loss'):

                # construct computation graph for loss_clip
                ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                                - tf.log(tf.clip_by_value(self.mu_prob, 1e-10, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                                  clip_value_max=1 + self.clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                loss_clip = -tf.reduce_mean(loss_clip)
                self.sum_clip = tf.summary.scalar('loss_clip', loss_clip)

                # construct computation graph for loss of value function
                v_preds = self.Policy.v_preds
                loss_vf = tf.squared_difference((self.rewards + self.gamma * self.v_preds_next), v_preds)
                loss_vf = tf.reduce_mean(loss_vf)
                self.sum_vf = tf.summary.scalar('value_difference', loss_vf)

                # construct computation graph for loss
                self.total_loss = loss_clip + self.c_1 * loss_vf
                self.sum_loss = tf.summary.scalar('total_loss', self.total_loss)

                self.g = tf.reduce_sum(self.rewards)
                self.sum_g = tf.summary.scalar('return', self.g)

            self.merged = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.total_loss, tvars), gra_clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def train(self, obs, actions, gaes, rewards, v_preds_next, mu_prob):
        summary, _, total_loss = self.sess.run([self.merged, self.train_op, self.total_loss], feed_dict={self.Policy.obs: obs,
                                                                                                       self.actions: actions,
                                                                                                       self.rewards: rewards,
                                                                                                       self.v_preds_next: v_preds_next,
                                                                                                       self.gaes: gaes,
                                                                                                       self.mu_prob: mu_prob})
        return summary

    def get_gaes(self, rewards, v_preds, v_preds_next):

        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
        return gaes

    def get_return(self, rewards):
        dis_rewards = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            dis_rewards[t] = running_add

        return dis_rewards

    def ppo_train(self, observations, actions, rewards, gaes, v_preds_next, mu_prob, verbose=False):
        if verbose:
            print('PPO train now..........')

        #self.assign_policy_parameters()
        for i in range(self.epoch_num):
            actions = actions[:,np.newaxis]
            rewards = rewards[:,np.newaxis]
            gaes = gaes[:,np.newaxis]
            v_preds_next = v_preds_next[:,np.newaxis]
            mu_prob = mu_prob[:, np.newaxis]

            # concat and shuffle
            dataset = np.hstack((observations, actions, rewards, gaes, v_preds_next, mu_prob))
            np.random.shuffle(dataset)

            # recover
            observations = dataset[:, :8]
            actions = dataset[:, -5]
            rewards = dataset[:, -4]
            gaes = dataset[:, -3]
            v_preds_next = dataset[:, -2]
            mu_prob = dataset[:, -1]

            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            gaes = np.squeeze(gaes)
            v_preds_next = np.squeeze(v_preds_next)
            mu_prob = np.squeeze(mu_prob)

            l = len(actions)

            start = 0
            end = start+self.batch_size
            while end < l:
                summary = self.train(observations[start:end],
                                     actions[start:end],
                                     gaes[start:end],
                                     rewards[start:end],
                                     v_preds_next[start:end],
                                     mu_prob[start:end])
                yield summary
                start += self.batch_size
                end += self.batch_size

        if verbose:
            print('PPO train end..........')

class ppo():

    def __init__(self,
                 state_space,
                 action_space,
                 p_type,
                 log_path='./Documents/PolicyModel/{}/logs/'.format(nowTime),
                 model_path='./Documents/PolicyModel/{}/'.format(nowTime),
                 have_model=False,
                 lr=1e-4):

        self.model_path = model_path

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.p_type = p_type
        self.state_dim = state_space
        with self.sess.as_default(), self.graph.as_default():
            self.pi = Policy_net('policy', self.sess, state_space, action_space)
            self.PPOTrain = PPOTrain('train', self.sess, self.pi, action_space=action_space, lr=lr)

            self.n_training = 0

        with self.sess.as_default(),self.graph.as_default():

            self.summary = tf.summary.FileWriter(log_path+p_type, self.sess.graph)

            self.saver = tf.train.Saver(self.pi.get_variables())

            if have_model:
                self.sess.run(tf.global_variables_initializer())
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):

        with self.sess.as_default():
            action,_, value = self.pi.act(state)

        return action

    def get_value(self, state):

        with self.sess.as_default():
            action,_, value = self.pi.act(state)

        return value

    def get_act_v(self, state):
        if state.shape[0] != 1:
            state = state[np.newaxis, :]

        with self.sess.as_default():
            act_prob, v = self.pi.get_action_prob(state)

        return act_prob, v

    def get_action_value(self, state, stochastic=True):

        with self.sess.as_default():
            action, act_prob, value = self.pi.act(state, stochastic)

        return action, act_prob, value

    def train(self, batch):

        with self.sess.as_default():
            with self.graph.as_default():

                # convert list to numpy array
                observation = np.vstack(batch["state"]).astype(np.float32)
                action = np.hstack(batch["action"]).astype(dtype=np.int32)
                gae = np.hstack(batch["gae"]).astype(np.float32)
                reward = np.hstack(batch["reward"]).astype(dtype=np.float32)
                v_pred_next = np.hstack(batch["v_next"]).astype(dtype=np.float32)
                mu_prob = np.hstack(batch["mu_prob"]).astype(np.float32)

                for s in self.PPOTrain.ppo_train(observation, action, reward, gae, v_pred_next, mu_prob):
                    self.summary.add_summary(s, self.n_training)
                    self.n_training += 1

    def save_model(self):
        with self.sess.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path,self.p_type))

    def load_model(self):
        with self.sess.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path,self.p_type))