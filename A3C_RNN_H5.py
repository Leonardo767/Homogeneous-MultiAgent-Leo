
# coding: utf-8

# # A3C - Multi-agent
#
# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
#
# `tensorboard --logdir=train_RNN`

# In[7]:


from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
# get_ipython().magic('matplotlib inline')
import minecraft_H5 as minecraft
import pickle

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)
# assert len(dev_list) > 1


# ### Helper Functions

# In[8]:

def build_world0():
    world_shape = (10, 5, 10)
    world = np.zeros(world_shape)
    # place source blocks
    for pos in minecraft.SOURCES:
        world[pos[0], pos[1], pos[2]] = -2
        if pos[0]-1 >= 0:
            world[pos[0]-1, pos[1], pos[2]] = 0
        if pos[0]+1 < world_shape[0]:
            world[pos[0]+1, pos[1], pos[2]] = 0
        if pos[2]-1 >= 0:
            world[pos[0], pos[1], pos[2]-1] = 0
        if pos[2]+1 < world_shape[2]:
            world[pos[0], pos[1], pos[2]+1] = 0
    # place agents randomly
    rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
        2), np.random.randint(world_shape[2])
    while not (world[rx, ry, rz] == 0 and ((ry == 0) or (ry > 0 and world[rx, ry-1, rz] == -1))):
        rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
            world_shape[1]), np.random.randint(world_shape[2])
    world[rx, ry, rz] = 1
    return world


def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import imageio
    imageio.mimwrite(fname, images, subrectangles=True)


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def good_discount(x, gamma):
    return discount(x, gamma)

# Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# ## A3C Approach

# ### Implementing the Actor-Critic Network### Implementing the Deep Q-Network

# In[9]:


class ACNet:
    def __init__(self, scope, a_size, trainer):

        with tf.variable_scope(str(scope)+'/qvalues'):
            self.is_Train = True
            # The input size may require more work to fit the interface.
            self.inputs = tf.placeholder(
                shape=[1, 5, 10, 5, 10], dtype=tf.float32)
            self.myinput = tf.transpose(self.inputs, perm=[0, 2, 3, 4, 1])
            self.policy, self.value, self.state_out, self.state_in, self.state_init, self.has_block, _, self.is_built = self._build_net(
                self.myinput)
        with tf.variable_scope(str(scope)+'/qvaluesB'):
            self.inputsB = tf.placeholder(
                shape=[EXPERIENCE_BUFFER_SIZE, 5, 10, 5, 10], dtype=tf.float32)
            self.myinputB = tf.transpose(self.inputsB, perm=[0, 2, 3, 4, 1])
            self.policyB, self.valueB, self.state_outB, self.state_inB, self.state_initB, self.has_blockB, self.validsB, self.is_builtB = self._build_net(
                self.myinputB)
        if(scope != GLOBAL_NET_SCOPE):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(
                self.actions, a_size, dtype=tf.float32)
            self.valids = tf.placeholder(
                shape=[None, a_size], dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.target_has = tf.placeholder(tf.float32, [None])
            self.target_built = tf.placeholder(tf.float32, [None])
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(
                self.policyB * self.actions_onehot, [1])
            self.train_value = tf.placeholder(tf.float32, [None])

            # Loss Functions
            self.value_loss = 0.5 * tf.reduce_sum(self.train_value*tf.square(
                self.target_v - tf.reshape(self.valueB, shape=[-1])))

            # something to encourage exploration
            self.entropy = - \
                tf.reduce_sum(
                    self.policyB * tf.log(tf.clip_by_value(self.policyB, 1e-10, 1.0)))
            self.block_loss = - tf.reduce_sum(self.target_has*tf.log(tf.clip_by_value(tf.reshape(self.has_blockB, shape=[-1]), 1e-10, 1.0))+(
                1-self.target_has)*tf.log(tf.clip_by_value(1-tf.reshape(self.has_blockB, shape=[-1]), 1e-10, 1.0)))
            self.built_loss = - tf.reduce_sum(self.target_built*tf.log(tf.clip_by_value(tf.reshape(self.is_builtB, shape=[-1]), 1e-10, 1.0))+(
                1-self.target_built)*tf.log(tf.clip_by_value(1-tf.reshape(self.is_builtB, shape=[-1]), 1e-10, 1.0)))
            self.policy_loss = - \
                tf.reduce_sum(tf.log(tf.clip_by_value(
                    self.responsible_outputs, 1e-15, 1.0)) * self.advantages)
            self.valid_loss = - tf.reduce_sum(tf.log(tf.clip_by_value(self.validsB, 1e-10, 1.0))
                                              * self.valids+tf.log(tf.clip_by_value(1-self.validsB, 1e-10, 1.0)) * (1-self.valids))
            self.loss = 0.5 * self.value_loss + self.policy_loss + 0.5*self.block_loss + \
                0.5*self.built_loss + 0.5*self.valid_loss - self.entropy * 0.01

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvaluesB')
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(
                self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            global_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvaluesB')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

            self.homogenize_weights = update_target_graph(
                str(scope)+'/qvaluesB', str(scope)+'/qvalues')
        if TRAINING:
            print("Hello World... From  "+str(scope))     # :)

    def _build_net(self, inputs):
        w_init = layers.variance_scaling_initializer()
        conv1 = layers.conv3d(inputs=inputs, padding="SAME", num_outputs=64, kernel_size=[
                              5, 5, 5], stride=2, data_format="NDHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)
        conv2 = layers.conv3d(inputs=conv1, padding="SAME", num_outputs=128, kernel_size=[
                              3, 3, 3], stride=1, data_format="NDHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)
        conv3 = layers.conv3d(inputs=conv2, padding="SAME", num_outputs=256, kernel_size=[
                              3, 3, 3], stride=2, data_format="NDHWC", weights_initializer=w_init, activation_fn=None)
        res = layers.conv3d(inputs=inputs, padding="VALID", num_outputs=256, kernel_size=[
                            4, 2, 4], stride=3, data_format="NDHWC", weights_initializer=w_init, activation_fn=None)
        conv4_in = tf.nn.relu(conv3+res)
        conv4 = layers.conv3d(inputs=conv4_in, padding="SAME", num_outputs=256, kernel_size=[
                              2, 1, 2], stride=2, data_format="NDHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)
        conv5 = layers.conv3d(inputs=conv4, padding="VALID", num_outputs=512, kernel_size=[
                              2, 1, 2], stride=1, data_format="NDHWC", weights_initializer=w_init, activation_fn=None)
        res2 = layers.conv3d(inputs=conv4_in, padding="VALID", num_outputs=512, kernel_size=[
                             3, 2, 3], stride=1, data_format="NDHWC", weights_initializer=w_init, activation_fn=None)
        flat = tf.nn.relu(layers.flatten(conv5+res2))

        h1 = layers.fully_connected(inputs=flat,  num_outputs=512)
        h2 = layers.fully_connected(
            inputs=h1,  num_outputs=512, activation_fn=None)
        h3 = tf.nn.relu(h2+flat)

        # Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(512, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(h3, [0])
        step_size = tf.shape(inputs)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 512])

        policy_layer = layers.fully_connected(inputs=rnn_out, num_outputs=a_size, weights_initializer=normalized_columns_initializer(
            1./float(a_size)), biases_initializer=None, activation_fn=None)
        policy = tf.nn.softmax(policy_layer)
        policy_sig = tf.sigmoid(policy_layer)
        value = layers.fully_connected(inputs=rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(
            1.0), biases_initializer=None, activation_fn=None)
        has_block = layers.fully_connected(inputs=rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(
            1.0), biases_initializer=None, activation_fn=tf.sigmoid)
        is_built = layers.fully_connected(inputs=rnn_out, num_outputs=1, weights_initializer=normalized_columns_initializer(
            1.0), biases_initializer=None, activation_fn=tf.sigmoid)

        return policy, value, state_out, state_in, state_init, has_block, policy_sig, is_built


# ## Worker Agent

# In[10]:


class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1
        self.groupLock = groupLock

        self.nextGIF = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name, a_size, trainer)
        self.copy_weights = self.local_AC.homogenize_weights
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)

    def train(self, rollout, sess, gamma, bootstrap_value):
        #       [s,a,r,s1,d,v[0,0],train_valid,pred_has_block,int(has_block),train_val,int(is_built)]
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        lastDone = rollout[-1, 4]
        values = rollout[:, 5]
        valids = rollout[:, 6]
        pred_has = rollout[:, 7]
        has_blocks = rollout[:, 8]
        train_value = rollout[:, 9]
        is_built = rollout[:, 10]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * \
            self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages, gamma)

#         if not lastDone:
        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(
            advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: np.stack(discounted_rewards),
                     self.local_AC.inputsB: np.stack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.valids: np.stack(valids),
                     self.local_AC.advantages: advantages,
                     self.local_AC.train_value: train_value,
                     self.local_AC.has_blockB: np.reshape(pred_has, [np.shape(pred_has)[0], 1]),
                     self.local_AC.target_has: has_blocks,
                     self.local_AC.target_built: is_built,
                     self.local_AC.state_inB[0]: rnn_state[0],
                     self.local_AC.state_inB[1]: rnn_state[1]}

        v_l, p_l, b_l, valid_l, e_l, g_n, v_n, bp_l, _ = sess.run([self.local_AC.value_loss,
                                                                   self.local_AC.policy_loss,
                                                                   self.local_AC.block_loss,
                                                                   self.local_AC.valid_loss,
                                                                   self.local_AC.entropy,
                                                                   self.local_AC.grad_norms,
                                                                   self.local_AC.var_norms,
                                                                   self.local_AC.built_loss,
                                                                   self.local_AC.apply_grads],
                                                                  feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), b_l / len(rollout), valid_l/len(rollout), e_l / len(rollout), g_n, v_n, bp_l

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def work(self, max_episode_length, gamma, sess, coord, saver):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops
        total_steps = 0

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
                sess.run(self.copy_weights)

                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                if not TRAINING:
                    completedQ[self.metaAgentID], d = False, False
                    completed_time[self.metaAgentID] = np.nan

                # Initial state from the environment
                if FULL_PLAN and random.random() < 0.5:
                    validActions, has_block, is_built = self.env._reset(
                        self.agentID, empty=EMPTY, full=True)
                else:
                    validActions, has_block, is_built = self.env._reset(
                        self.agentID, empty=EMPTY, full=False)
                s = self.env._observe(self.agentID)
                rnn_state = self.local_AC.state_init
                RewardNb = wrong_block = wrong_built = 0

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF and episode_count >= OBSERVE_PERIOD)):
                    saveGIF = True
                    self.nextGIF += 128
                    GIF_episode = int(episode_count)
                    episode_frames = [self.env._render(mode='rgb_array')]

                self.groupLock.release(0, self.name)
                # synchronize starting time of the threads
                self.groupLock.acquire(1, self.name)

                while (not self.env.finished):  # Give me something!
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state, pred_has_block, pred_is_built = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out, self.local_AC.has_block, self.local_AC.is_built],
                                                                                   feed_dict={self.local_AC.inputs: [s], self.local_AC.state_in[0]: rnn_state[0], self.local_AC.state_in[1]: rnn_state[1]})

                    if(not (np.argmax(a_dist.flatten()) in validActions)):
                        episode_inv_count += 1
                    train_valid = np.zeros(a_size)
                    train_valid[validActions] = 1
                    mysum = np.sum(train_valid)

                    valid_dist = np.array([a_dist[0, validActions]])
                    valid_dist /= np.sum(valid_dist)

                    if TRAINING:
                        if (pred_has_block.flatten()[0] < 0.5) == has_block:
                            wrong_block += 1
                            a = np.random.choice(validActions)
                            train_val = 0
                        elif (pred_is_built.flatten()[0] < 0.5) == is_built:
                            wrong_built += 1
                            a = validActions[np.random.choice(
                                range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            train_val = 1.
                        else:
                            a = validActions[np.random.choice(
                                range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            train_val = 1.
                    else:
                        if GREEDY:
                            a = np.argmax(a_dist.flatten())
                        if not GREEDY or a not in validActions:
                            a = validActions[np.random.choice(
                                range(valid_dist.shape[1]), p=valid_dist.ravel())]
                        train_val = 1.

                    _, r, d, validActions, has_block1, is_built1 = self.env._step(
                        (self.agentID, a))

                    if not TRAINING:
                        extraBlocks = max(
                            0, self.env.world.countExtraBlocks(self.env.state_obj))
                        if np.isnan(completed_time[self.metaAgentID]) and completedQ[self.metaAgentID] != is_built1 and is_built1:
                            completed_time[self.metaAgentID] = episode_step_count+1
                            scaffoldings[self.metaAgentID] = extraBlocks
                            blocks_left[self.metaAgentID] = extraBlocks
                        elif is_built1 and not np.isnan(blocks_left[self.metaAgentID]) and extraBlocks < blocks_left[self.metaAgentID]:
                            blocks_left[self.metaAgentID] = extraBlocks
                        completedQ[self.metaAgentID] |= is_built1

                    self.groupLock.release(1, self.name)
                    self.groupLock.acquire(0, self.name)  # synchronize threads

                    # Get common observation for all agents after all individual actions have been performed
                    s1 = self.env._observe(self.agentID)
                    d = self.env.finished

                    if saveGIF:
                        episode_frames.append(
                            self.env._render(mode='rgb_array'))

                    self.groupLock.release(0, self.name)
                    self.groupLock.acquire(1, self.name)  # synchronize threads

                    episode_buffer.append([s, a, r, s1, d, v[0, 0], train_valid, pred_has_block, int(
                        has_block), train_val, int(is_built)])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    has_block = has_block1
                    is_built = is_built1
                    episode_step_count += 1

                    if r > 0:
                        RewardNb += 1
                    if d == True and TRAINING:
                        print('\n{} Goodbye World. We did it!'.format(
                            episode_step_count), end='\n')

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:

                            if d:
                                s1Value = 0.0
                            else:
                                s1Value = sess.run(self.local_AC.value,
                                                   feed_dict={self.local_AC.inputs: np.array([s]), self.local_AC.state_in[0]: rnn_state[0], self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                            v_l, p_l, b_l, valid_l, e_l, g_n, v_n, bp_l = self.train(
                                episode_buffer[-EXPERIENCE_BUFFER_SIZE:], sess, gamma, s1Value)

                            sess.run(self.pull_global)
                            sess.run(self.copy_weights)

                    if episode_step_count >= max_episode_length or d:
                        break

                episode_rewards[self.metaAgentID].append(episode_reward)
                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(
                    np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print(
                        '                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(
                        episode_count, self.agentID, RewardNb), end='\r')

                if TRAINING:
                    episode_count += 1. / num_workers

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 500 == 0:
                            saver.save(sess, model_path+'/model-' +
                                       str(int(episode_count))+'.cptk')
                            print('Saved Model', end='\r')
                        mean_reward = np.mean(
                            episode_rewards[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_length = np.mean(
                            episode_lengths[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_value = np.mean(
                            episode_mean_values[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_invalid = np.mean(
                            episode_invalid_ops[self.metaAgentID][-SUMMARY_WINDOW:])
                        current_learning_rate = sess.run(
                            lr, feed_dict={global_step: episode_count})

                        summary = tf.Summary()
                        summary.value.add(
                            tag='Perf/Learning Rate', simple_value=current_learning_rate)
                        summary.value.add(tag='Perf/Reward',
                                          simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length',
                                          simple_value=mean_length)
                        summary.value.add(
                            tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        summary.value.add(tag='Perf/Block Prediction Accuracy', simple_value=float(
                            episode_step_count-wrong_block)/float(episode_step_count))
                        summary.value.add(tag='Perf/Plan Completion Accuracy', simple_value=float(
                            episode_step_count-wrong_built)/float(episode_step_count))

                        summary.value.add(
                            tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(
                            tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(
                            tag='Losses/Plan Completion Loss', simple_value=bp_l)
                        summary.value.add(
                            tag='Losses/Block Prediction Loss', simple_value=b_l)
                        summary.value.add(
                            tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(
                            tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(
                            tag='Losses/Var Norm', simple_value=v_n)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated ({})'.format(
                                episode_count, self.workerID), end='\r')
                elif not TRAINING and self.workerID == 1:
                    # only care about plan completion if init state didn't contain the completed plan...
                    if episode_buffer[0][-1] == 0:
                        completed[episode_count] = int(
                            completedQ[self.metaAgentID])
                    if not np.isnan(completed_time[self.metaAgentID]):
                        plan_durations[episode_count] = completed_time[self.metaAgentID]
                        rollout = np.array(episode_buffer)
                        place_moves[episode_count] = np.sum(np.asarray(
                            rollout[:completed_time[self.metaAgentID]+1, 1] > 8, dtype=int))
                    len_episodes[episode_count] = episode_step_count
#                     saveGIF &= (episode_step_count < max_episode_length)

                    if not np.isnan(completed_time[self.metaAgentID]):
                        episode_count += 1
                    GIF_episode = int(episode_count)
#                     print('({}) Thread {}: {} steps ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_inv_count))

                self.groupLock.release(1, self.name)
                self.groupLock.acquire(0, self.name)  # synchronize threads

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        def gif_creation(): return make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode,
                                                                                                             episode_step_count, episode_reward), duration=len(images)*time_per_step, true_image=True, salience=False)
                        threading.Thread(target=(gif_creation)).start()
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path, GIF_episode, episode_step_count),
                                 duration=len(images)*time_per_step, true_image=True, salience=False)
                if self.workerID == 1 and SAVE_EPISODE_BUFFER and episode_step_count < max_episode_length:
                    with open('{}/episode_{}.dat'.format(episodes_path, GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


# ## Training

# In[11]:


# Learning parameters
max_episode_length = 512
episode_count = 1024
EPISODE_START = episode_count
gamma = .9  # discount rate for advantage estimation and reward discounting
GRAD_CLIP = 300.0
LR_Q = 2.e-5  # 8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR = True
ADAPT_COEFF = 1.e-3 / \
    20.  # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
a_size = 13  # New approach
EXPERIENCE_BUFFER_SIZE = 512
OBSERVE_PERIOD = 0.  # Period of pure observation (no value learning)
SUMMARY_WINDOW = 25
NUM_META_AGENTS = 1
NUM_THREADS = 4  # int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
EPISODE_SAMPLES = EXPERIENCE_BUFFER_SIZE
load_model = False
RESET_TRAINER = False
model_path = './model_H5'
gifs_path = './gifs_H5'
train_path = 'train_RNN_H5'
# used to export episode_buffers that can be read/played/recorded by the visualizer
episodes_path = 'gifs3D_H5'
GLOBAL_NET_SCOPE = 'global'

# Simulation options
FULL_HELP = False
MAP_ID = 10  # 0: RANDOMIZED_PLAN, other: given map (list in minecraft_SA4H.py)
OUTPUT_GIFS = True
SAVE_EPISODE_BUFFER = False

# Testing
TRAINING = True
GREEDY = False
NUM_EXPS = 20
EMPTY = True and (not TRAINING)
FULL_PLAN = False  # Help training cleanup by forcing all episodes
MODEL_NUMBER = 300000  # to start with the structure already completed
# Should be enabled near the end of training

# Shared arrays for tensorboard
episode_rewards = [[] for _ in range(NUM_META_AGENTS)]
episode_lengths = [[] for _ in range(NUM_META_AGENTS)]
episode_mean_values = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops = [[] for _ in range(NUM_META_AGENTS)]
completedQ = [False for _ in range(NUM_META_AGENTS)]
completed_time = [np.nan for _ in range(NUM_META_AGENTS)]
printQ = False  # (for headless)


# In[ ]:


tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

if not TRAINING:
    blocks_left = np.array([np.nan for _ in range(NUM_EXPS)])
    scaffoldings = np.array([np.nan for _ in range(NUM_EXPS)])
    completed = np.array([np.nan for _ in range(NUM_EXPS)])
    plan_durations = np.array([np.nan for _ in range(NUM_EXPS)])
    len_episodes = np.array([np.nan for _ in range(NUM_EXPS)])
    # Only makes sense for a single agent (comparison with TERMES code)
    place_moves = np.array([np.nan for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists(episodes_path):
        os.makedirs(episodes_path)

# Create a directory to save episode playback gifs to
if OUTPUT_GIFS and not os.path.exists(gifs_path):
    print("Created gifs_path")
    os.makedirs(gifs_path)

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE, a_size,
                           None)  # Generate global network
    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        # we need the +1 so that lr at step 0 is defined
        lr = tf.divide(tf.constant(LR_Q), tf.sqrt(
            tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
    else:
        lr = tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

    num_workers = NUM_THREADS  # Set workers # = # of available CPU threads
    if not TRAINING:
        NUM_META_AGENTS = 1

    gameEnvs, workers, groupLocks = [], [], []
    for ma in range(NUM_META_AGENTS):
        gameEnv = minecraft.MinecraftEnv(
            num_workers, observation_range=-1, observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = [
            "worker_"+str(i) for i in range(ma*num_workers+1, (ma+1)*num_workers+1)]
        groupLock = GroupLock.GroupLock([workerNames, workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for i in range(ma*num_workers+1, (ma+1)*num_workers+1):
            workersTmp.append(Worker(gameEnv, ma, i, a_size, groupLock))
        workers.append(workersTmp)

    if TRAINING:
        global_summary = tf.summary.FileWriter(train_path)
    else:
        global_summary = 0
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write(
                        'model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(
                    learning_rate=lr, use_locking=True)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            for worker in workers[ma]:
                # synchronize starting time of the threads
                groupLocks[ma].acquire(0, worker.name)

                def worker_work(): return worker.work(
                    max_episode_length, gamma, sess, coord, saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        coord.join(worker_threads)

if not TRAINING:
    print('[{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
        1 - np.nanmean(blocks_left /
                       scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
        np.nanmean(completed), np.sqrt(np.nanvar(completed)),
        np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
        np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
        np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
        np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
        np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
        np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
    )


# ## Systematic Testing

# In[ ]:


# maxe_pisode_length     = 2000
# GREEDY                 = False
# NUM_EXPS               = 100
# EMPTY                  = True
# TRAINING               = False

# MODEL_NUMBER           = 300000
# SAVE_EPISODE_BUFFER    = True
# NUM_META_AGENTS        = 1

# if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
#     os.makedirs('gifs3D')

# for num_workers in [1]:#,2,4,8,12,16,3,6,10,14]:
#     for MAP_ID in range(6,7):
#         episode_count = 0

#         tf.reset_default_graph()
#         config = tf.ConfigProto(allow_soft_placement = True)
#         config.gpu_options.allow_growth=True

#         blocks_left    = np.array([np.nan for _ in range(NUM_EXPS)])
#         scaffoldings   = np.array([np.nan for _ in range(NUM_EXPS)])
#         completed      = np.array([np.nan for _ in range(NUM_EXPS)])
#         plan_durations = np.array([np.nan for _ in range(NUM_EXPS)])
#         len_episodes   = np.array([np.nan for _ in range(NUM_EXPS)])
#         place_moves    = np.array([np.nan for _ in range(NUM_EXPS)]) # Only makes sense for a single agent (comparison with TERMES code)
#         mutex = threading.Lock()

#         episodes_path = 'gifs3D/{:d}_{:d}'.format(num_workers,MAP_ID)
#         if SAVE_EPISODE_BUFFER and not os.path.exists(episodes_path):
#             os.makedirs(episodes_path)

#         with tf.device("/gpu:0"):
#             master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None) # Generate global network
#             trainer = tf.contrib.opt.NadamOptimizer(learning_rate=LR_Q, use_locking=True)

#             global_summary = 0
#             saver = tf.train.Saver(max_to_keep=5)

#             with tf.Session(config=config) as sess:
#                 coord = tf.train.Coordinator()
#                 with open(model_path+'/checkpoint', 'w') as file:
#                     file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
#                     file.close()
#                 ckpt = tf.train.get_checkpoint_state(model_path)
#                 saver.restore(sess,ckpt.model_checkpoint_path)

#                 gameEnvs, workers, groupLocks = [], [], []
#                 for ma in range(NUM_META_AGENTS):
#                     gameEnv = minecraft.MinecraftEnv(num_workers, observation_range=-1, observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
#                     gameEnvs.append(gameEnv)

#                     # Create groupLock
#                     workerNames = ["worker_"+str(i) for i in range(ma*num_workers+1,(ma+1)*num_workers+1)]
#                     groupLock = GroupLock.GroupLock([workerNames,workerNames])
#                     groupLocks.append(groupLock)

#                     # Create worker classes
#                     workersTmp = []
#                     for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
#                         workersTmp.append(Worker(gameEnv,ma,i,a_size,groupLock))
#                     workers.append(workersTmp)

#                 # This is where the asynchronous magic happens.
#                 # Start the "work" process for each worker in a separate thread.
#                 worker_threads = []
#                 for ma in range(NUM_META_AGENTS):
#                     for worker in workers[ma]:
#                         groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
#                         worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
#                         t = threading.Thread(target=(worker_work))
#                         t.start()
#                         worker_threads.append(t)
#                 coord.join(worker_threads)

#         print('[{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
#                num_workers, MAP_ID,
#                1 - np.nanmean(blocks_left/scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
#                np.nanmean(completed), np.sqrt(np.nanvar(completed)),
#                np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
#                np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
#                np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
#                np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
#                np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
#                np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
#              )

#         ofp = open('results.txt','a')
#         ofp.write('{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(
#                num_workers, MAP_ID,
#                1 - np.nanmean(blocks_left/scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
#                np.nanmean(completed), np.sqrt(np.nanvar(completed)),
#                np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
#                np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
#                np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
#                np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
#                np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
#                np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
#              )
#         ofp.close()
