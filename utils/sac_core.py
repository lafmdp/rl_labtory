import numpy as np
import os

import tensorflow as tf


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""
def kl_policy(x, act_dim, hidden_sizes, activation, a_bar=None):

    # policy network outputs
    net = mlp(x, list(hidden_sizes), activation, activation)
    if a_bar is not None:
        net = tf.concat([net, a_bar], 1)
    logits = tf.layers.dense(net, act_dim, activation='linear')

    # action and log action probabilites (log_softmax covers numerical problems)
    action_probs = tf.nn.softmax(logits, axis=-1)
    log_action_probs = tf.nn.log_softmax(logits, axis=-1)

    # policy with no noise
    mu = tf.argmax(logits, axis=-1)

    # polciy with noise
    policy_dist = tf.distributions.Categorical(logits=logits)
    pi = policy_dist.sample()

    # entropy over discrete actions
    pi_entropy = -tf.reduce_sum(action_probs * log_action_probs, axis=-1)

    onehot_pi = tf.one_hot(pi, depth=act_dim, axis=-1, dtype=tf.float32)

    return mu, pi, onehot_pi, pi_entropy, logits,

"""
Actor-Critics
"""
def a_out_mlp_actor_critic(x, a, alpha, a_bar = None, hidden_sizes=[400,300], activation=tf.nn.relu, policy=kl_policy):

    act_dim = a.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        mu, pi, onehot_pi, pi_entropy, pi_logits = kl_policy(x, act_dim, hidden_sizes, activation, a_bar=a_bar)

    # vfs
    with tf.variable_scope('q1'):
        q1_logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q1_a  = tf.reduce_sum(tf.multiply(q1_logits, a), axis=1)
        q1_pi = tf.reduce_sum(tf.multiply(q1_logits, onehot_pi), axis=1)

    with tf.variable_scope('q2'):
        q2_logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q2_a  = tf.reduce_sum(tf.multiply(q2_logits, a), axis=1)
        q2_pi = tf.reduce_sum(tf.multiply(q2_logits, onehot_pi), axis=1)

    return mu, pi, pi_entropy, pi_logits, q1_logits, q2_logits, q1_a, q2_a, q1_pi, q2_pi
