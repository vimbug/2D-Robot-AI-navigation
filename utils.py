import math  # import math module for mathematical functions
import numpy as np  # import numpy for numerical operations
import matplotlib.pyplot as plt  # import matplotlib for plotting

def distance_between_two_points(p1, p2):  # calculate Euclidean distance between two points
    # points are x-y
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])  # return hypotenuse distance

def vector_from_points(base_point, arrow_point):  # compute vector from base to arrow point
    # points are x-y
    return (arrow_point[0] - base_point[0], arrow_point[1] - base_point[1])  # return vector tuple

def vector_from_angle(position):  # create unit vector from angle in position
    # position is x-y-theta
    return np.array([math.cos(math.radians(position[2])), math.sin(math.radians(position[2]))])  # return numpy array vector

def distance_to_target(robot_positions, target_position, index=1):  # calculate distance from robot to target
    # position robot is x-y-theta
    # position target is x-y
    return math.hypot(robot_positions[index][0] - target_position[0], robot_positions[index][1] - target_position[1])  # return distance

def add_new_position(robot_positions, new_robot_position):  # update robot position history
    # Adds the new position to the end
    robot_positions.append(new_robot_position)  # append new position to list
    # removes the first row
    robot_positions.pop(0)  # remove oldest position
    return robot_positions  # return updated list

def angle_between(robot_vector, target_vector):  # compute normalized angle between vectors
    dot = (  # calculate dot product
        robot_vector[0] * target_vector[0] +  # x component
        robot_vector[1] * target_vector[1]    # y component
    )  # dot product

    cross = (  # calculate cross product
        robot_vector[0] * target_vector[1] -  # x*y' - y*x'
        robot_vector[1] * target_vector[0]    # y*x' - x*y'
    )  # cross product

    angle = math.atan2(cross, dot)  # angle in radians between -pi and pi

    return -angle / math.pi  # normalize to [-1, 1] where -1 is left, 0 is straight, and 1 is right

def update_plot(axes, episode_rewards, episode_lengths, episode_results, episode_epsilons, window_size=50):  # update training plots
    """
    Updates the training results plots, including:
    - Total reward
    - Episode length
    - Single episode result
    - Percentages of last window_size episodes
    - Epsilon
    """
    for ax in axes:  # clear all axes
        ax.clear()  # clear axis

    # ---- 1. Total reward per episode ----
    if len(episode_rewards) > 0:  # if rewards data exists
        axes[0].plot(episode_rewards, label="Episode Reward")  # plot rewards
        axes[0].set_ylabel("Total Reward")  # set y label
        axes[0].legend()  # show legend

    # ---- 2. Episode length ----
    if len(episode_lengths) > 0:  # if lengths data exists
        axes[1].plot(episode_lengths)  # plot lengths
        axes[1].set_ylabel("Episode Steps")  # set y label

    # ---- 3. Result of each episode ----
    if len(episode_results) > 0:  # if results data exists
        color_map = {'goal': 'blue', 'collision': 'red', 'max steps': 'green'}  # define color mapping
        episode_colors = [color_map[result] for result in episode_results]  # map results to colors
        axes[2].bar(range(len(episode_results)), [1] * len(episode_results), color=episode_colors)  # plot bars
        axes[2].set_ylabel("Episode Result")  # set y label
        axes[2].set_xticks([])  # remove x ticks

    # ---- 4. Percentage Goal / Collision / Max Steps (last window_size episodes) ----
    if len(episode_results) > 0:  # if results data exists
        results_window = list(episode_results[-window_size:])  # get last window results
        results_types = ['goal', 'collision', 'max steps']  # define result types
        counts = [results_window.count(r) for r in results_types]  # count each type
        total = len(results_window)  # total in window

        percentages = [(c / total) * 100 if total > 0 else 0 for c in counts]  # calculate percentages

        axes[3].bar(results_types, percentages, color=["blue", "red", "green"])  # plot percentages
        axes[3].set_ylabel(f"% last {window_size} episodes")  # set y label
        axes[3].set_ylim(0, 100)  # set y limits

    # ---- 5. Epsilon ----
    if len(episode_epsilons) > 0:  # if epsilons data exists
        axes[4].plot(episode_epsilons)  # plot epsilons
        axes[4].set_ylabel("Epsilon")  # set y label

    plt.tight_layout()  # adjust layout
    plt.pause(0.001)  # pause for display
