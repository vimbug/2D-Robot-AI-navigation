import config  # import configuration settings
from DQNAgent import DQNAgent  # import DQN agent class
from Environment import Environment  # import environment simulator
from robot import Robot  # import robot controller
import numpy as np  # import numpy for arrays
import matplotlib.pyplot as plt  # import matplotlib for plotting
import utils  # import utility functions
import time  # import time for timing

start_time = time.time()  # record start time for training
ep_rewards = np.array([])  # array for total reward per episode
ep_lengths = np.array([])  # array for number of steps per episode
ep_results = np.array([])  # array for episode results (goal or collision)
ep_epsilons = np.array([])  # array for epsilon value per episode
ep_loss = np.array([])  # array for average loss per episode
# Initialize DQN agent
agent = DQNAgent()  # create DQN agent instance

for episode in range(config.NUM_EPISODES):  # loop over episodes

    # Initialize the environment and the robot
    env = Environment()  # create environment instance

    # create the matrix
    logic_matrix = env.generate_matrix()  # generate occupancy matrix from immage

    # generate initial robot position and target
    initial_position = env.generate_valid_position()  # get valid start position
    target_position = env.create_target()  # get valid target position
    robot_instance = Robot(initial_position)  # create robot instance
    done = False  # episode not done
    total_reward = 0  # total reward accumulator
    step_count = 0  # step counter
    loss = 0  # loss value

    while not done and step_count <= config.MAX_STEPS:  # main training loop

        # get current state and sensor values
        state, sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get state vector

        # choose action using epsilon-greedy policy
        action = agent.act(state)  # select action

        # execute the chosen action and move robot
        robot_instance.execute_action(action)  # apply action

        # get new state after action
        new_state, new_sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get new state

        # calculate reward and check episode end conditions
        reward, done, reason = robot_instance.calculate_reward(target_position, new_sensor_value, action, step_count)  # compute reward

        # add experience to replay buffer
        agent.remember(state, action, reward, new_state, done)  # store experience

        # accumulate total reward
        total_reward += reward  # add reward to total

        if len(agent.memory) > (config.BATCH_SIZE * 20) and step_count % 3 == 0:  # if enough memory and time to train
            loss = agent.replay()  # perform training step

        step_count += 1  # increment step count

    # Update episode statistics
    ep_rewards = np.append(ep_rewards, total_reward)  # store total reward
    ep_lengths = np.append(ep_lengths, step_count)  # store episode length
    ep_results = np.append(ep_results, reason)  # store result
    ep_epsilons = np.append(ep_epsilons, agent.epsilon)  # store epsilon
    # ep_loss = np.append(ep_loss, loss)  # commented out loss tracking

    # Save the model periodically after initial episodes
    if episode > (config.NUM_EPISODES - config.NUM_EPISODES_TO_SAVE) and episode % config.SAVE_FREQUENCY == 0:  # if time to save
        filename = f"dqn_model_ep{episode}.pth"  # create filename
        agent.save_model(filename)  # save model

    # Update epsilon for exploration
    agent.agg_epsilon()  # decrease epsilon

    # print progress
    print(f"episode number: {episode} | step: {step_count}| reason: {reason}")  # print episode info

# at the end of training plot the graphs
fig, axs = plt.subplots(5, 1, figsize=(6, 12))  # create subplots
utils.update_plot(axs, ep_rewards, ep_lengths, ep_results, ep_epsilons)  # update plots
end_time = time.time()  # record end time
print(f"Total execution time: {end_time - start_time:.2f} seconds")  # print total time
plt.show()  # display plots
agent.save_model("dqn_model.pth")  # save final model