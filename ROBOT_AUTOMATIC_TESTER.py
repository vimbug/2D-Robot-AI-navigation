"""
This module performs automatic testing on DQN models saved inside a directory.
For each loaded model, it iterates through the test episodes defined in config,
initializes the environment and the robot, executes the actions predicted by
the network, and collects statistics such as average steps, average reward,
and goal achievement rate.

There are no functions explicitly defined in this script: the logic is organized
into three main phases:
  1. Reading .pth files from the model directory
  2. Running test episodes and accumulating statistics
  3. Visualizing the results in a 3D plot

The purpose of this module is to compare different trained networks and determine
which model achieves the best balance between speed (steps), reward, and goal
completion.
"""
import config  # import configuration settings
from DQNAgent import DQNAgent, DQNNetwork  # import the DQN agent and network classes
from Environment import Environment  # import the environment simulator
from robot import Robot  # import the robot controller class
import numpy as np  # numerical computing library
import matplotlib.pyplot as plt  # plotting library
import os  # operating system utilities
import sys  # system-specific parameters and functions
import torch  # PyTorch machine learning library

if len(sys.argv) < 2:  # check for the model directory argument
    print("Error: model directory not provided")  # print error if missing
    sys.exit(1)  # exit the script with error status

NETWORK_DIR = sys.argv[1]  # first command line argument is the network folder

network_files = sorted([  # collect all .pth model files sorted alphabetically
    os.path.join(NETWORK_DIR, f)  # build full file path
    for f in os.listdir(NETWORK_DIR)  # iterate files in given directory
    if f.endswith(".pth")  # filter only PyTorch model files
])

ep_steps_network = np.array([])  # global array for mean steps of each model
ep_rewards_network = np.array([])  # global array for mean rewards of each model
ep_reasons_network = np.array([])  # global array for goal success rate of each model

agent = DQNAgent()  # initialize a DQN agent helper instance
num_network = 0  # counter for processed networks
for network in network_files:  # iterate through each saved network file
    num_network += 1  # increment the network counter
    print(f"Testing network {num_network}/{len(network_files)}: {network}")  # print progress
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose device
    test_network = DQNNetwork(config.STATE_SIZE, config.ACTION_SIZE).to(device)  # create network
    test_network.load_state_dict(torch.load(network, map_location=device))  # load trained weights

    ep_steps = np.array([])  # reset steps array for this network
    ep_rewards = np.array([])  # reset rewards array for this network
    ep_reasons = np.array([])  # reset termination reasons for this network

    for episode in range(config.NUM_TEST_EPISODES):  # run each test episode
        print(f"  Episode {episode+1}/{config.NUM_TEST_EPISODES}", end="\r")  # show episode progress
        env = Environment()  # create a fresh environment instance
        logic_matrix = env.generate_test_matrix("House_test.png")  # generate occupancy matrix
        initial_position = env.generate_valid_position()  # choose valid robot start position
        target_position = env.create_target()  # create the goal target position
        robot_instance = Robot(initial_position)  # initialize the robot with start position
        done = False  # episode completion flag
        total_reward = 0  # accumulate reward for the episode
        step_count = 0  # count steps taken in episode
        loss = 0  # placeholder for compatibility (unused)

        while not done and step_count <= config.MAX_STEPS:  # loop until episode finishes or max steps
            state, sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get current state
            action, Q_value = agent.act_we(state, test_network)  # choose action from network
            robot_instance.execute_action(action)  # apply action to the robot
            step_count += 1  # increment step count
            new_state, new_sensor_value = robot_instance.get_state(target_position, logic_matrix)  # read new state
            reward, done, reason = robot_instance.calculate_reward(target_position, new_sensor_value, action, step_count)  # compute reward
            total_reward += reward  # add reward to total

        ep_rewards = np.append(ep_rewards, total_reward)  # store episode reward
        ep_steps = np.append(ep_steps, step_count)  # store steps taken
        ep_reasons = np.append(ep_reasons, reason)  # store episode end reason

    mean_steps = np.mean(ep_steps)  # compute average steps for this network
    mean_rewards = np.mean(ep_rewards)  # compute average reward for this network
    goal_rate = np.sum(ep_reasons == 'goal') / len(ep_reasons) * 100  # compute goal success rate

    ep_steps_network = np.append(ep_steps_network, mean_steps)  # add mean steps to global results
    ep_rewards_network = np.append(ep_rewards_network, mean_rewards)  # add mean rewards to global results
    ep_reasons_network = np.append(ep_reasons_network, goal_rate)  # add goal rate to global results

fig = plt.figure()  # create a new matplotlib figure
ax = fig.add_subplot(111, projection='3d')  # add a 3D subplot
ax.scatter(ep_steps_network, ep_rewards_network, ep_reasons_network)  # scatter plot results

for i, name in enumerate(network_files):  # annotate each point with the filename
    ax.text(
        ep_steps_network[i],  # x coordinate
        ep_rewards_network[i],  # y coordinate
        ep_reasons_network[i],  # z coordinate
        name,  # label text for the point #type: ignore
        fontsize=8  # set text size
    )

ax.set_xlabel("Mean steps")  # label x axis
ax.set_ylabel("Mean reward")  # label y axis
ax.set_zlabel("Goal rate [%]")  # label z axis #type: ignore

plt.show()  # display the plot window
