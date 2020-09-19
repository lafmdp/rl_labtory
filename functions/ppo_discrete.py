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

nowTime = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')

hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 2,
    "c_2": 0.01,
    "init_lr": 3e-4,
    "lr_epsilon": 1e-6
}

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

        self.lamda = hype_parameters["lamda"]
        self.gamma = hype_parameters["gamma"]
        self.epoch_num = hype_parameters["epoch_num"]
        self.clip_value = hype_parameters["clip_value"]
        self.c_1 = hype_parameters["c_1"]
        self.c_2 = hype_parameters["c_2"]
        self.init_lr = hype_parameters["init_lr"]
        self.lr_epsilon = hype_parameters["lr_epsilon"]

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
            with tf.variable_scope('policy'):
                self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.state_space], name='obs')
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = self.init_lr

                with tf.variable_scope("Net"):
                    with tf.variable_scope('action'):
                        out = tf.layers.dense(self.obs, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 256, tf.nn.relu)
                        out = tf.layers.dense(out, 128, tf.nn.relu)
                        out = tf.layers.dense(out, 64, tf.nn.relu)

                        self.action_probs = tf.layers.dense(out, self.action_space, tf.nn.softmax)

                        self.sthst_action = tf.reshape(tf.multinomial(tf.log(self.action_probs), num_samples=1), shape=[-1])
                        self.detmnst_action = tf.reshape(tf.argmax(self.action_probs, axis=1), shape=[-1])

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
                    self.old_actions = tf.placeholder(tf.int32, [None], name='action_done')
                    self.old_probs = tf.placeholder(tf.float32, [None], name='old_means')
                    self.first_step_return = tf.placeholder(tf.float32, [None], name="first_return")
                    self.trajectory_len = tf.placeholder(tf.float32, [None], name="traj_len")
                    self.batch_lr = tf.placeholder(tf.float32, [None], name="batch_learning_rate")
                    self.batch_actor_loss = tf.placeholder(tf.float32, [None], name="batch_a_loss")
                    self.batch_critic_loss = tf.placeholder(tf.float32, [None], name="batch_c_loss")

                with tf.name_scope('loss_and_train'):
                    with tf.name_scope('policy_loss'):
                        act = tf.one_hot(indices=self.old_actions, depth=self.action_space)
                        self.specific_probs = tf.reduce_sum(self.action_probs * act, axis=-1)

                        ratios = tf.exp(tf.log(self.specific_probs+ 1e-10) - tf.log(self.old_probs + 1e-10) )
                        clipped_ratios = tf.clip_by_value(ratios,
                                                          clip_value_min=1 - self.clip_value,
                                                          clip_value_max=1 + self.clip_value,
                                                          name='clip_ratios')
                        loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                        self.actor_loss = -tf.reduce_mean(loss_clip)

                    with tf.name_scope('value_loss'):
                        self.critic_loss = tf.losses.mean_squared_error(self.returns, self.value)


                    with tf.name_scope('total_loss'):
                        total_loss = self.actor_loss + self.c_1 * self.critic_loss

                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.lr_epsilon)

                    self.train_op = optimizer.minimize(total_loss)


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

            actions, value, action_prob = \
                self.sess.run([self.sthst_action, self.value, self.action_probs], feed_dict={self.obs: obs})

            ret = {
                "actions": actions[0],
                "value": value.tolist()[0][0]
            }

        return ret


    def get_means(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            actions = self.sess.run(self.detmnst_action, feed_dict={self.obs: obs})

        return actions[0]

    def get_value(self, obs):

        with self.sess.as_default(), self.graph.as_default():
            obs = np.array(obs)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            value = self.sess.run(self.value, feed_dict={self.obs: obs})

        return value.tolist()[0][0]

    def get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def train(self, batch):
        with self.sess.as_default(), self.graph.as_default():

            # convert list to numpy array
            state = np.vstack(batch["state"]).astype(np.float32).squeeze()
            old_action = np.hstack(batch["action"]).astype(dtype=np.float32).reshape([-1,1])
            gae = np.hstack(batch["gae"]).astype(np.float32).reshape([-1,1])
            ret = np.hstack(batch["return"]).astype(dtype=np.float32).reshape([-1,1])
            first_step_return = np.array(batch["sum_reward"])
            trajectory_len = np.array(batch["trajectory_len"])



            traj_len = [len(s) for s in batch["state"]]

            print('Return:')
            print('length:\t%f, Mean:\t%f' % (sum(traj_len)/len(traj_len), first_step_return.mean()))
            print('Max:\t%f, Min:\t%f' % (first_step_return.max(), first_step_return.min()))

            old_prob = self.sess.run(self.specific_probs, feed_dict={self.obs: state, self.old_actions: old_action.squeeze()})
            old_prob = np.array(old_prob, dtype=np.float32).reshape([-1,1])

            s_s = self.state_space

            batch_size = state.shape[0] // 10

            actor_loss = []
            critic_loss = []
            learning_r = []

            dataset = np.hstack((state, old_action, old_prob, gae, ret))

            for i in range(self.epoch_num):

                np.random.shuffle(dataset)

                states = dataset[:, :s_s]
                old_actions = dataset[:, -4]
                old_probs = dataset[:, -3]
                gaes = dataset[:, -2]
                rets = dataset[:, -1]

                gaes = np.squeeze(gaes)
                rets = rets[:, np.newaxis]

                sample_num = dataset.shape[0]

                start = 0
                end = min(start + batch_size, sample_num)

                while start < sample_num:
                    if end - start <= 100:
                        break
                    try:
                        a_loss, c_loss, _ = \
                            self.sess.run([self.actor_loss,
                                           self.critic_loss,
                                           self.train_op],
                                          feed_dict={self.obs: states[start:end],
                                                     self.returns: rets[start:end],
                                                     self.gaes: gaes[start:end],
                                                     self.old_actions: old_actions[start:end].squeeze(),
                                                     self.old_probs: old_probs[start:end].squeeze()})

                        actor_loss.append(a_loss)
                        critic_loss.append(c_loss)
                        # learning_r.append(lr.mean())


                    except Exception as e:
                        print(e)
                        print("start:", start)
                        print("end:", end)

                    start += batch_size
                    end = min(start + batch_size, sample_num)

    def save_model(self):

        with self.sess.as_default(), self.graph.as_default():
            self.saver.save(self.sess, save_path=os.path.join(self.model_path, self.model_name))

    def load_model(self):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, save_path=os.path.join(self.model_path, self.model_name))
