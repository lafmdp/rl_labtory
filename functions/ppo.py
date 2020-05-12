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

nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "need_log": False,
    "batch_size": 5000,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 3,
    "c_2": 0.01,
    "init_lr": 3e-4,
    "lr_epsilon": 1e-6
}


'''
:param state_space: 状态空间大小
:param action_space:  动作空间大小
:param gamma:  默认0.99
:param lamda:   默认0.95
:param log_path:  tensorboard 文件路径
:param model_path:  tensorflow model 路径
:param use_model:  是否已经有现成模型，初始化时设为False， 使用或训练时设为True
:param need_log:  是否需要log，训练时设为True，使用时设为False
:param batch_size:  训练时一个batch的大小，可以调
:param epoch_num:  一批样本的训练轮数，可以调
:param clip_value:  默认0.2
:param c_1:  value function的系数，可以调
:param c_2:  entropy的系数，可以调
:param lr:  学习率
:param lr_epsilon: Adam的超参
'''


class policy():

    def __init__(self,
                 state_space=11,
                 action_space=6,
                 need_log=False,
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
        self.log_path = self.model_path + '/logs'
        self.need_log = need_log

        self.lamda = hype_parameters["lamda"]
        self.gamma = hype_parameters["gamma"]
        self.batch_size = hype_parameters["batch_size"]
        self.epoch_num = hype_parameters["epoch_num"]
        self.clip_value = hype_parameters["clip_value"]
        self.c_1 = hype_parameters["c_1"]
        self.c_2 = hype_parameters["c_2"]
        self.init_lr = hype_parameters["init_lr"]
        self.lr_epsilon = hype_parameters["lr_epsilon"]

        self.build_graph()

        self.n_training = 0

        with self.sess.as_default(), self.graph.as_default():
            if need_log:
                self.summary = tf.summary.FileWriter(self.log_path, self.sess.graph)

            self.saver = tf.train.Saver(self.get_variables())

            if have_model:
                self.load_model()
            else:
                self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope('policy'):
                self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.state_space], name='obs')
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.noisy_linear_cosine_decay(
                    learning_rate=self.init_lr, decay_steps=100000, global_step=self.global_step,
                    initial_variance=0.01, variance_decay=0.1, num_periods=0.2, alpha=0.05, beta=0.2)
                self.add_global = self.global_step.assign_add(1)

                with tf.variable_scope("Net"):
                    with tf.variable_scope('action'):
                        out = tf.layers.dense(self.obs, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 256, tf.nn.relu)
                        out = tf.layers.dense(out, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 64, tf.nn.relu)

                        self.means = tf.layers.dense(out, self.action_space, tf.nn.tanh, name="means")

                        self.log_vars = tf.constant(-2.2, dtype=tf.float32, shape=[self.action_space, ],
                                                    name='log_variance')

                        sampled_act = (self.means +
                                       tf.exp(self.log_vars / 2.0) *
                                       tf.truncated_normal(shape=(self.action_space,)))

                        self.sampled_act = tf.clip_by_value(sampled_act, -1, 1)

                    with tf.variable_scope('value'):
                        out2 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.nn.relu)
                        out2 = tf.layers.dense(inputs=out2, units=256, activation=tf.nn.relu)
                        out2 = tf.layers.dense(inputs=out2, units=128, activation=tf.nn.relu)
                        self.value = tf.layers.dense(inputs=out2, units=1, activation=None)

                self.scope = tf.get_variable_scope().name

            with tf.name_scope("policy_train"):

                with tf.name_scope('train_input'):
                    self.returns = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward_to_go')
                    self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                    self.old_actions = tf.placeholder(tf.float32, [None, self.action_space], name='action_done')
                    self.old_means = tf.placeholder(tf.float32, [None, self.action_space], name='old_means')
                    self.first_step_return = tf.placeholder(tf.float32, [None], name="first_return")
                    self.trajectory_len = tf.placeholder(tf.float32, [None], name="traj_len")
                    self.batch_lr = tf.placeholder(tf.float32, [None], name="batch_learning_rate")
                    self.batch_actor_loss = tf.placeholder(tf.float32, [None], name="batch_a_loss")
                    self.batch_critic_loss = tf.placeholder(tf.float32, [None], name="batch_c_loss")
                    self.batch_entropy = tf.placeholder(tf.float32, [None], name="batch_entropy")

                with tf.name_scope('loss_and_train'):
                    with tf.name_scope('policy_loss'):
                        self._logprob()
                        ratios = tf.exp(self.logp - self.logp_old)
                        clipped_ratios = tf.clip_by_value(ratios,
                                                          clip_value_min=1 - self.clip_value,
                                                          clip_value_max=1 + self.clip_value,
                                                          name='continuous_ratios')
                        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                        self.actor_loss = -tf.reduce_mean(loss_clip)

                    with tf.name_scope('value_loss'):
                        self.critic_loss = tf.losses.mean_squared_error(self.returns, self.value)

                    with tf.name_scope('entropy_loss'):
                        self.entropy = tf.reduce_sum(
                            0.5 * self.log_vars + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
                        self.entropy_loss = -self.entropy

                    with tf.name_scope('total_loss'):
                        total_loss = self.actor_loss + self.c_1 * self.critic_loss #+ self.c_2 * self.entropy_loss

                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.lr_epsilon)

                    self.train_op = optimizer.minimize(total_loss)

                    if self.need_log:
                        lr_summary = tf.summary.scalar('decay_learning_rate',
                                                       tf.reduce_mean(self.batch_lr))
                        al_summary = tf.summary.scalar('actor_loss',
                                                       tf.reduce_mean(self.batch_actor_loss))
                        cl_summary = tf.summary.scalar('critic_loss',
                                                       tf.reduce_mean(self.batch_critic_loss))
                        ent_summary = tf.summary.scalar('entropy', tf.reduce_mean(self.batch_entropy))
                        ret_summary = tf.summary.scalar('return',
                                                        tf.reduce_mean(self.first_step_return))
                        traj_len_summary = tf.summary.scalar("trajectory_length",
                                                             tf.reduce_mean(self.trajectory_len))

                        self.train_merge = tf.summary.merge(
                            [lr_summary, al_summary, cl_summary, ret_summary, traj_len_summary, ent_summary])

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions
        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.old_actions - self.means) /
                                     tf.exp(self.log_vars), axis=-1)

        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.log_vars)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.old_actions - self.old_means) /
                                         tf.exp(self.log_vars), axis=-1)
        self.logp_old = logp_old

    def get_gaes(self, rewards, v_preds, v_preds_next):

        deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

        return gaes

    def get_return(self, rewards):

        dis_rewards = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * hype_parameters["gamma"] + rewards[t]
            dis_rewards[t] = running_add

        return dis_rewards

    def get_action(self, obs):
        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions, value = \
                self.sess.run([self.sampled_act, self.value], feed_dict={self.obs: obs})

            ret = {
                "actions": actions[0],
                "value": value[0]
            }

        return ret

    def get_means(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions = self.sess.run(self.means, feed_dict={self.obs: obs})

        return actions[0]

    def get_value(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            value = self.sess.run(self.value, feed_dict={self.obs: obs})

        return value[0]

    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def train(self, batch):
        with self.sess.as_default(), self.graph.as_default():

            # convert list to numpy array
            state = np.vstack(batch["state"]).astype(np.float32).squeeze()
            old_action = np.vstack(batch["action"]).astype(dtype=np.float32).squeeze()
            gae = np.vstack(batch["gae"]).astype(np.float32).squeeze()
            ret = np.hstack(batch["return"]).astype(dtype=np.float32).squeeze()
            first_step_return = np.array(batch["sum_reward"])
            trajectory_len = np.array(batch["trajectory_len"])

            traj_len = [len(s) for s in batch["state"]]
            print('Return:')
            print('length:\t%f, Mean:\t%f' % (sum(traj_len)/len(traj_len), first_step_return.mean()))
            print('Max:\t%f, Min:\t%f' % (first_step_return.max(), first_step_return.min()))

            old_mean = self.sess.run(self.means, feed_dict={self.obs: state})
            old_mean = np.array(old_mean, dtype=np.float32).squeeze()

            s_s = self.state_space
            a_s = self.action_space
            print("Sample Nums:",state.shape[0])

            gae = gae[:, np.newaxis]
            ret = ret[:, np.newaxis]

            if a_s == 1:
                old_action = old_action[:, np.newaxis]
                old_mean = old_mean[:, np.newaxis]

            actor_loss = []
            critic_loss = []
            entropy = []
            learning_r = []

            for i in range(self.epoch_num):

                dataset = np.hstack((state, old_action, old_mean, gae, ret))
                np.random.shuffle(dataset)

                states = dataset[:, :s_s]
                old_actions = dataset[:, s_s:s_s + a_s]
                old_means = dataset[:, s_s + a_s:s_s + 2 * a_s]
                gaes = dataset[:, -2]
                rets = dataset[:, -1]

                gaes = np.squeeze(gaes)
                rets = rets[:, np.newaxis]

                if s_s == 1:
                    old_actions = old_actions[:, np.newaxis]

                sample_num = dataset.shape[0]

                start = 0
                end = min(start + self.batch_size, sample_num)

                while start < sample_num:
                    a_loss, c_loss, lr, entropy_loss, _, _ = \
                        self.sess.run([self.actor_loss,
                                       self.critic_loss,
                                       self.learning_rate,
                                       self.entropy_loss,
                                       self.train_op,
                                       self.add_global],
                                      feed_dict={self.obs: states[start:end],
                                                 self.returns: rets[start:end],
                                                 self.gaes: gaes[start:end],
                                                 self.old_actions: old_actions[start:end],
                                                 self.old_means: old_means[start:end]})

                    actor_loss.append(a_loss)
                    critic_loss.append(c_loss)
                    learning_r.append(lr.mean())
                    entropy.append(-entropy_loss)

                    start += self.batch_size
                    end = min(start + self.batch_size, sample_num)

            if self.need_log:
                summary = self.sess.run(self.train_merge, feed_dict={
                    self.trajectory_len: trajectory_len,
                    self.first_step_return: first_step_return,
                    self.batch_actor_loss: np.array(actor_loss, dtype=np.float32),
                    self.batch_critic_loss: np.array(critic_loss, dtype=np.float32),
                    self.batch_lr: np.array(learning_r, dtype=np.float32),
                    self.batch_entropy: np.array(entropy, dtype=np.float32)
                })
                self.summary.add_summary(summary, self.n_training)
                self.n_training += 1

    def save_model(self):

        with self.sess.as_default(), self.graph.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path, self.model_name))

    def load_model(self):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
