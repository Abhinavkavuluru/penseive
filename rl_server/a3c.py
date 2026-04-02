import numpy as np
import tensorflow as tf

# Use TF1 compatibility mode
tf_v1 = tf.compat.v1
tf_v1.disable_eager_execution()


GAMMA = 0.99
A_DIM = 6
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
S_INFO = 4


def _fc(input_tensor, in_dim, out_dim, scope_name, activation=tf.nn.relu):
    """Fully connected layer matching tflearn FullyConnected variable naming."""
    with tf_v1.variable_scope(scope_name):
        W = tf_v1.get_variable('W', [in_dim, out_dim])
        b = tf_v1.get_variable('b', [out_dim])
        out = tf.matmul(input_tensor, W) + b
        if activation is not None:
            out = activation(out)
        return out


def _conv1d(input_tensor, in_channels, out_channels, kernel_size, scope_name):
    """Conv1D layer matching tflearn Conv1D variable naming (uses conv2d internally)."""
    with tf_v1.variable_scope(scope_name):
        W = tf_v1.get_variable('W', [kernel_size, 1, in_channels, out_channels])
        b = tf_v1.get_variable('b', [out_channels])
        # tflearn conv_1d expands dim 1: [batch, seq, ch] → [batch, 1, seq, ch]
        input_4d = tf.expand_dims(input_tensor, 1)
        conv_out = tf.nn.conv2d(input_4d, W, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.nn.relu(conv_out + b)
        # Squeeze height dim and flatten
        out = tf.squeeze(out, axis=1)  # [batch, seq, out_channels]
        # Flatten: seq_dim * out_channels
        shape = out.get_shape().as_list()
        flat_dim = shape[1] * shape[2] if len(shape) == 3 else out_channels
        out = tf.reshape(out, [-1, flat_dim])
        return out


class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf_v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf_v1.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf_v1.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.math.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            axis=1, keepdims=True)),
                       -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.math.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf_v1.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf_v1.variable_scope('actor'):
            inputs = tf_v1.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])

            # FC layers matching tflearn FullyConnected naming
            split_0 = _fc(inputs[:, 0:1, -1], 1, 128, 'FullyConnected')
            split_1 = _fc(inputs[:, 1:2, -1], 1, 128, 'FullyConnected_1')

            # Conv1D layers matching tflearn Conv1D naming (conv2d internally)
            split_2_flat = _conv1d(inputs[:, 2:3, :], self.s_dim[1], 128, 4, 'Conv1D')
            split_3_flat = _conv1d(inputs[:, 3:4, :], self.s_dim[1], 128, 4, 'Conv1D_1')
            split_4_flat = _conv1d(inputs[:, 4:5, :A_DIM], A_DIM, 128, 4, 'Conv1D_2')

            split_5 = _fc(inputs[:, 4:5, -1], 1, 128, 'FullyConnected_2')

            merge_net = tf.concat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)

            dense_net_0 = _fc(merge_net, 768, 128, 'FullyConnected_3')
            out = _fc(dense_net_0, 128, self.a_dim, 'FullyConnected_4', activation=tf.nn.softmax)

            return inputs, out

    def train(self, inputs, acts, act_grad_weights):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf_v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf_v1.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tf.reduce_mean(tf.square(self.td_target - self.out))

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf_v1.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf_v1.variable_scope('critic'):
            inputs = tf_v1.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])

            split_0 = _fc(inputs[:, 0:1, -1], 1, 128, 'FullyConnected')
            split_1 = _fc(inputs[:, 1:2, -1], 1, 128, 'FullyConnected_1')

            split_2_flat = _conv1d(inputs[:, 2:3, :], self.s_dim[1], 128, 4, 'Conv1D')
            split_3_flat = _conv1d(inputs[:, 3:4, :], self.s_dim[1], 128, 4, 'Conv1D_1')
            split_4_flat = _conv1d(inputs[:, 4:5, :A_DIM], A_DIM, 128, 4, 'Conv1D_2')

            split_5 = _fc(inputs[:, 4:5, -1], 1, 128, 'FullyConnected_2')

            merge_net = tf.concat([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)

            dense_net_0 = _fc(merge_net, 768, 128, 'FullyConnected_3')
            out = _fc(dense_net_0, 128, 1, 'FullyConnected_4', activation=None)

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0]

    v_batch = critic.predict(s_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf_v1.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf_v1.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf_v1.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf_v1.summary.merge_all()

    return summary_ops, summary_vars
