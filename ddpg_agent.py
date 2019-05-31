import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE =  1024     # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4        # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 4
ALT_UPSATE_EVERY = 10
HARD_UPDATE_EVERY = 1000
ALPHA = 0.0 # prioritization level (ALPHA=0 is uniform sampling so no prioritization)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        # self.noise = OUNoise(action_size, random_seed, mu=0., theta=0.5, sigma=0.8)
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.t_step = 0
        self.state  = "learning"

    def step(self, state, action, reward, next_state, done, beta=1.):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        if self.t_step % ALT_UPSATE_EVERY == 0:
            if self.state == "updating":
                self.state = "learning"
            else:
                self.state = "updating"



        if self.state == "updating":
            self.memory.add(state, action, reward, next_state, done)
            # print("updating step: {}:".format(self.t_step))
        else:
            # print('\rt_step {}'.format(self.t_step))
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # if self.t_step % UPDATE_EVERY == 0:
                # experiences = self.memory.sample()
                experiences = self.memory.sample(ALPHA, beta)
                self.learn(experiences, GAMMA)
            # print("learning step: {}:".format(self.t_step))

        self.t_step += 1

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()\
                      # / math.log(self.t_step + 1)
            # action += self.noise.sample() / (self.t_step + 1) / 2
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones, weights, indices = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred)
        actor_loss_mean = actor_loss.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss_mean.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # Update priorities based on td error
        self.memory.update_priorities(indices.squeeze().to(device).data.numpy(), torch.mul(actor_loss,actor_loss).squeeze().to(device).data.numpy())
        # self.memory.update_priorities(indices.squeeze().to(device).data.numpy(), actor_loss.squeeze().to(device).data.numpy())

        # ----------------------- update target networks ----------------------- #
        # if self.t_step % HARD_UPDATE_EVERY == 0:
        #     # logger.info('Copying all parameters from local to target')
        #     self._copy_weights(self.critic_local, self.critic_target)
        #     self._copy_weights(self.actor_local, self.actor_target)
        # else:
        #     if self.t_step % UPDATE_EVERY == 0:
        #         self.soft_update(self.critic_local, self.critic_target, TAU)
        #         self.soft_update(self.actor_local, self.actor_target, TAU)

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def _copy_weights(self, source_network, target_network):
        """Copy source network weights to target"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Naive Prioritized Experience Replay buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # By default set max priority level
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done, max_priority)
        self.memory.append(e)

    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""

        # Probabilities associated with each entry in memory
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** alpha
        probs /= probs.sum()

        # Get indices
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False, p=probs)

        # Associated experiences
        experiences = [self.memory[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)
        indices = torch.from_numpy(np.vstack(indices)).long().to(device)
        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, priorities):
        for i, idx in enumerate(indices):
            # A tuple is immutable so need to use "_replace" method to update it - might replace the named tuple by a dict
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#from: https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
#credits to: Jaromir Janisch
"""
The Memory that is used for Prioritized Experience Replay
"""

from sum_tree import SumTree
class Replay_Memory:
    def __init__(self):
        global MEMORY_LEN
        self.tree = SumTree(MEMORY_LEN)

    def add(self, error, sample):
        global MEMORY_BIAS, MEMORY_POW
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.add(priority, sample)

    def sample(self):
        """
         Get a sample batch of the replay memory
        Returns:
         batch: a batch with one sample from each segment of the memory
        """
        global BATCH_SIZE
        batch = []
        #we want one representative of all distribution-segments in the batch
        #e.g BATCH_SIZE=2: batch contains one sample from [min,median]
        #and from [median,max]
        segment = self.tree.total() / BATCH_SIZE
        for i in range(BATCH_SIZE):
            minimum = segment * i
            maximum = segment * (i+1)
            s = random.uniform(minimum, maximum)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        """
         Updates one entry in the replay memory
        Args:
         idx: the position of the outdated transition in the memory
         error: the newly calculated error
        """
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.update(idx, priority)