# TODO: your agent here!
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
import random

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""

        # states
        states = layers.Input(shape=(self.state_size,), name='states')

        # Hidden layers
        hidden = layers.Dense(units=32, activation='relu')(states)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        hidden = layers.Dense(units=64, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        hidden = layers.Dense(units=32, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)

        # activation
        act = layers.Dense(units=self.action_size, activation='tanh',
                                   name='raw_actions')(hidden)

        # scale
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(act)
        self.model = models.Model(inputs=states, outputs=actions)

        # loss
        gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-gradients * actions)

        opti = optimizers.Adam()
        updates_op = opti.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, gradients, K.learning_phase()],
                                   outputs=[], updates=updates_op)

class Critic:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):

        # input
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # hidden layers for states
        hidden = layers.Dense(units=32, activation='relu')(states)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        hidden = layers.Dense(units=64, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)
        hidden = layers.Dense(units=32, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Dropout(0.3)(hidden)

        # hidden layers for actions
        hidden2 = layers.Dense(units=32, activation='relu')(actions)
        hidden2 = layers.BatchNormalization()(hidden2)
        hidden2 = layers.Dropout(0.3)(hidden2)
        hidden2 = layers.Dense(units=64, activation='relu')(hidden2)
        hidden2 = layers.BatchNormalization()(hidden2)
        hidden2 = layers.Dropout(0.3)(hidden2)
        hidden2 = layers.Dense(units=32, activation='relu')(hidden2)
        hidden2 = layers.BatchNormalization()(hidden2)
        hidden2 = layers.Dropout(0.3)(hidden2)

        finalq = layers.Add()([hidden, hidden2])
        finalq = layers.Activation('relu')(finalq)
        finalq = layers.Dense(units=1, name='q_values')(finalq)

        self.model = models.Model(inputs=[states, actions], outputs=finalq)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        gradients = K.gradients(finalq, actions)

        self.get_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()], outputs=gradients)

class OUNoise:
    #Ornstein-Uhlenbeck Process
    
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

class DDPGProcess():

    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Target
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise
        self.exploration_theta = 0.09
        self.exploration_sigma = 0.12
        self.exploration_mu = 0

        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Mem
        self.batch_size = 64
        self.buffer_size = 100000
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Params
        self.gamma = 0.93
        self.tau = 0.0012

        # Score tracker and learning parameters
        self.score = 0
        self.best_score = -np.inf
        self.count = 0

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):

        self.total_reward += reward
        self.count += 1
        self.memory.add(self.last_state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        self.last_state = next_state

    def act(self, states):
        states = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(states)[0]
        return list(action + self.noise.sample())

    def learn(self, experiences):
        
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,
                                                                                                       self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        gradients = np.reshape(self.critic_local.get_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        
        self.actor_local.train_fn([states, gradients, 1])

        self.update(self.critic_local.model, self.critic_target.model)
        self.update(self.actor_local.model, self.actor_target.model)

        # Score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score

    def update(self, local_model, target_model):

        lweights = np.array(local_model.get_weights())
        tweights = np.array(target_model.get_weights())
        weights = self.tau * lweights + (1 - self.tau) * tweights
        target_model.set_weights(weights)
