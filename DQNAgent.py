# DQNAgent.py - Definition of the DQN network and agent

import torch  					# import PyTorch
import torch.nn as nn  			# import neural network module
import torch.optim as optim  	# import optimization module
import random  					# import random utilities
from config import (DQN_FC1_SIZE, DQN_FC2_SIZE, DQN_FC3_SIZE, GAMMA,
                    EPSILON, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE,
                    BATCH_SIZE, TAU, MEMORY_SIZE, DEVICE, STATE_SIZE, ACTION_SIZE)  # import hyperparameters from config


class DQNNetwork(nn.Module): 						 		# define the DQN neural network architecture
    def __init__(self, STATE_SIZE, ACTION_SIZE):  			# network initialization
        super(DQNNetwork, self).__init__()  				# initialize parent nn.Module
        self.fc1 = nn.Linear(STATE_SIZE, DQN_FC1_SIZE)  	# first fully connected layer
        self.fc2 = nn.Linear(DQN_FC1_SIZE, DQN_FC2_SIZE)  	# second fully connected layer
        self.fc3 = nn.Linear(DQN_FC2_SIZE, DQN_FC3_SIZE)  	# third fully connected layer
        self.fc4 = nn.Linear(DQN_FC3_SIZE, ACTION_SIZE)  	# output layer with one value per action

    def forward(self, x):  				# forward pass through the network
        x = torch.relu(self.fc1(x))  	# apply ReLU after first layer
        x = torch.relu(self.fc2(x))  	# apply ReLU after second layer
        x = torch.relu(self.fc3(x))  	# apply ReLU after third layer
        return self.fc4(x)  			# return raw Q-values for each action


class DQNAgent:  							# define the reinforcement learning agent
    def __init__(self):  					# initialize the agent parameters and networks
        self.STATE_SIZE = STATE_SIZE  		# set state dimension
        self.action_size = ACTION_SIZE  	# set action dimension
        self.memory_size = MEMORY_SIZE  	# set replay memory size
        self.memory = []  					# initialize replay memory list
        self.gamma = GAMMA  				# discount factor for future rewards
        self.epsilon = EPSILON  			# initial exploration probability
        self.epsilon_min = EPSILON_MIN  	# minimum exploration probability
        self.epsilon_decay = EPSILON_DECAY  # exploration decay rate
        self.learning_rate = LEARNING_RATE  # learning rate for optimizer
        self.batch_size = BATCH_SIZE  		# training batch size
        self.tau = TAU  					# soft update interpolation factor
        self.device = torch.device(DEVICE)  # choose computation device
        self.mem_index = 0 				 	# index for circular memory buffer
        self.epsilon_min = EPSILON_MIN  	# duplicate assignment for epsilon min
        self.epsilon_decay_episodes = EPSILON_DECAY  # number of episodes for epsilon decay

        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.epsilon_decay_episodes  # epsilon decrement per episode

        self.policy_model = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(self.device)  # policy network that learns how to act
        self.target_model = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(self.device)  # target network used as a stable reference during training
        self.target_model.load_state_dict(self.policy_model.state_dict())  # initialize target network with policy network weights
        self.target_model.eval()  # disable gradient calculation for target network

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)  # Adam optimizer for policy network
        self.loss_fn = nn.MSELoss()  # mean squared error loss function

    def remember(self, state, action, reward, next_state, done):  # store experience in replay memory
        experience = (state, action, reward, next_state, done)  # pack experience tuple

        if len(self.memory) < self.memory_size:  # if memory is not yet full
            # Memory not full -> append experience
            self.memory.append(experience)  # add experience to memory
        else:  # if memory is full
            # Memory full -> overwrite oldest experience using circular index
            self.memory[self.mem_index] = experience  # replace oldest memory entry

        # Update circular index
        self.mem_index = (self.mem_index + 1) % self.memory_size  # move index forward modulo memory size

    def act(self, state):  # choose action using epsilon-greedy policy
        # if random.random() returns a value less than epsilon, choose a random action
        if random.random() < self.epsilon:  # explore with probability epsilon
            return random.randrange(self.action_size)  # return random action index
        # otherwise choose the action with the highest Q-value
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # prepare state tensor for PyTorch
        with torch.no_grad():  # disable gradient computation to save memory
            q_values = self.policy_model(state_t)  # compute Q-values from policy network
        return q_values.argmax().item()  # return action index with maximum Q-value

    def soft_update(self, policy_net, target_net, tau):  # update target network weights toward policy network weights
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):  # iterate over network parameters
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * policy_param.data)  # perform soft update on each parameter

    def replay(self):  # perform a replay training step
        if len(self.memory) < self.batch_size * 10:  # if memory does not have enough experiences
            return 0  # skip training and return zero loss

        batch = random.sample(self.memory, self.batch_size)  # sample a batch of experiences from replay buffer
        states, actions, rewards, next_states, dones = zip(*batch)  # unpack batch into separate lists

        states_t = torch.FloatTensor(states).to(self.device)  # convert states to tensor
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # convert actions to tensor and add action dimension
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # convert rewards to tensor
        next_states_t = torch.FloatTensor(next_states).to(self.device)  # convert next states to tensor
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # convert done flags to tensor

        q_values = self.policy_model(states_t).gather(1, actions_t)  # select Q-values for chosen actions
        next_q_values = self.target_model(next_states_t).max(1)[0].unsqueeze(1)  # compute max Q-value for next states
        target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values  # calculate target Q-values

        loss = self.loss_fn(q_values, target_q_values)  # compute loss between current Q-values and targets

        self.optimizer.zero_grad()  # zero optimizer gradients
        loss.backward()  # backpropagate loss
        self.optimizer.step()  # update policy network weights
        self.soft_update(self.policy_model, self.target_model, self.tau)  # update target network weights smoothly
        return loss.item()  # return scalar loss value

    def agg_epsilon(self):  # update exploration rate epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)  # decrease epsilon but not below minimum
        return self.epsilon  # return the current epsilon value

    def save_model(self, path):  # save the policy model to disk
        """Save the policy network weights (trained model)"""  # docstring for save_model
        torch.save(self.policy_model.state_dict(), path)  # save model state dictionary
        print(f"Neural network saved to {path}")  # print confirmation message

    def act_we(self, state, network):  # choose action using a specified network without exploration
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # prepare state tensor for network evaluation
        network.eval()  # set network to evaluation mode
        with torch.no_grad():  # disable gradient computation during inference
            q_values = network(state_t)  # compute Q-values using provided network
        return q_values.argmax().item(), q_values  # return selected action and Q-value tensor


# Summary of functions:
# - DQNNetwork: defines the DQN neural network with 3 fully connected layers and an output for each action.
# - DQNAgent.__init__: initializes the agent parameters, policy and target networks, optimizer and loss.
# - DQNAgent.remember: stores an experience in the replay memory and overwrites older experiences if memory is full.
# - DQNAgent.act: performs epsilon-greedy action; chooses a random action or the action with the highest Q-value.
# - DQNAgent.soft_update: gradually updates the target network weights towards the policy network weights using the tau factor.
# - DQNAgent.replay: performs a training update using a random batch from memory, calculates the loss and updates the weights.
# - DQNAgent.agg_epsilon: reduces the epsilon value according to the calculated decrement and returns the new value.
# - DQNAgent.save_model: saves the policy model weights to disk.
# - DQNAgent.act_we: evaluates a state on a given network and returns the best action and all associated Q-values.
