import config  # import configuration settings
from DQNAgent import DQNAgent, DQNNetwork  # import DQN agent and network classes
from Environment import Environment  # import environment simulator
from robot import Robot  # import robot controller
import pygame as py  # import pygame for rendering
import math  # import math for calculations
import numpy as np  # import numpy for arrays
import torch  # import PyTorch for neural networks
import time  # import time for delays
import sys  # import sys for command line arguments
from astar import AStarPlanner, path_to_waypoints, inflate_obstacles  # import A* path planning utilities

# Check command line arguments
if len(sys.argv) < 2:  # if no model file provided
    print("Error: you must pass the name of the .pth file to load")  # print error message
    sys.exit(1)  # exit program

# Get the file name from command line
model_filename = sys.argv[1]  # get model filename from arguments
print(f"Loading network: {model_filename}")  # print loading message

py.init()  # initialize pygame

# --------Loading images-------
house = py.image.load("House_test.png")  # load house background image
robot_img = py.image.load("robot.png")  # load robot image
# --------color definitions---------

# Color definitions
white = (255, 255, 255)  # white color
black = (0, 0, 0)  # black color
green = (0, 255, 0)  # green color
red   = (255, 0, 0)  # red color
blue  = (0, 0, 255)  # blue color
purple = (128, 0, 128)  # purple color

# -------Window Settings------
W, H = house.get_size()  # get window size from house image
win = py.display.set_mode((W, H))  # create pygame window
py.display.set_caption("Robot RL with PyTorch - Optimized")  # set window title
font = py.font.SysFont("Arial", 24)  # create font for text
clock = py.time.Clock()  # create clock for frame rate
max_dist = math.hypot(W, H)  # calculate maximum distance

# ------Initialize DQN agent------
agent = DQNAgent()  # create DQN agent instance

# -------Load the trained neural network------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device
# create an empty network with the same architecture
test_network = DQNNetwork(config.STATE_SIZE, config.ACTION_SIZE).to(device)  # create network
# load weights into state_dict
test_network.load_state_dict(torch.load(model_filename, map_location=device))  # load trained weights

# ------Main simulation loop------
for episode in range(10):  # run 10 episodes
    # Initialize the environment and the robot
    env = Environment()  # create environment instance
    # create the matrix
    logic_matrix = env.generate_test_matrix("House_test.png")  # generate occupancy matrix
    logic_matrix_path = env.generate_test_matrix("House_test.png")  # generate matrix for path planning
    # generate initial robot position and target
    initial_position = env.generate_valid_position()  # get valid start position
    target_position = env.create_target()  # get valid target position
    # safety radius (e.g. half robot + margin)
    SAFETY_RADIUS = 30  # set safety radius
    inflated_matrix = inflate_obstacles(logic_matrix_path, SAFETY_RADIUS)  # inflate obstacles
    # A* path planning
    planner = AStarPlanner(inflated_matrix)  # create A* planner
    path = planner.find_path(initial_position[:2], target_position)  # find path

    if path is None:  # if no path found
        raise RuntimeError("A* does not find a valid path")  # raise error

    waypoints = path_to_waypoints(path, step=20)  # convert path to waypoints
    waypoints_for_render = waypoints.copy()  # copy for rendering

    waypoints.pop(0)  # remove the first waypoint (robot initial position)

    robot_instance = Robot(initial_position)  # create robot instance

    done = False  # episode not done
    total_reward = 0  # total reward accumulator
    step_count = 0  # step counter
    action_list = [0, 0]  # action history for oscillation detection
    action_count = 0  # oscillation counter
    trajectory = []  # robot trajectory list

    while not done and step_count < 5000:  # main simulation loop
        # ------draw the house--------
        win.fill(white)  # fill window with white
        win.blit(house, house.get_rect(center=(W//2, H//2)))  # draw house image

        # ------draw robot and target--------
        py.draw.circle(win, red, waypoints[-1], 10)  # draw target as red circle
        robot_img_rotated = py.transform.rotate(robot_img, -robot_instance.current_position[2])  # rotate robot image
        robot_rect = robot_img_rotated.get_rect(center=robot_instance.current_position[:2])  # get rotated rect
        win.blit(robot_img_rotated, robot_rect)  # draw rotated robot

        # ------get current state--------
        state, sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get robot state

        # ------waypoint management--------
        current_wp = waypoints[0]  # get current waypoint

        # Calculate distance from current waypoint
        dist_wp = math.hypot(  # calculate distance
            robot_instance.current_position[0] - current_wp[0],  # delta x
            robot_instance.current_position[1] - current_wp[1]  # delta y
        )

        if dist_wp < 30:  # if close to waypoint
            waypoints.pop(0)  # remove reached waypoint

        if not waypoints:  # if no more waypoints
            done = True  # episode done
        else:  # else
            target_position = current_wp  # set target to current waypoint
        # ------draw waypoints--------
        if len(waypoints_for_render) > 1:  # if waypoints to draw
            for (x, y) in waypoints_for_render:  # for each waypoint
                py.draw.circle(win, (0, 200, 0), (x, y), 4)  # draw green circle

        # ------draw lidar lines--------
        for i in range(len(sensor_value)):  # for each sensor
            ang = np.linspace(config.FIRST_RAY_ANGLE, config.LAST_RAY_ANGLE, config.NUM_LIDAR) + robot_instance.current_position[2]  # calculate angles
            # raw distance read by sensor (before clamp)
            dist = sensor_value[i]  # get sensor distance
            # ray drawing: from r_min to contact point
            start = (robot_instance.current_position[0] + config.LIDAR_RADIUS_MIN * math.cos(math.radians(ang[i])),  # start point
                     robot_instance.current_position[1] + config.LIDAR_RADIUS_MIN * math.sin(math.radians(ang[i])))
            end = (robot_instance.current_position[0] + config.LIDAR_RADIUS_MAX * dist * math.cos(math.radians(ang[i])),  # end point
                   robot_instance.current_position[1] + config.LIDAR_RADIUS_MAX * dist * math.sin(math.radians(ang[i])))
            start_int = (int(start[0]), int(start[1]))  # convert to int
            end_int = (int(end[0]), int(end[1]))  # convert to int
            py.draw.line(win, blue, start_int, end_int, 4)  # draw blue line

        # ------agent chooses action--------
        action, Q_value = agent.act_we(state, test_network)  # get action from network
        # ------force oscillations------------
        action_list.pop(0)  # remove old action
        action_list.append(action)  # add new action
        if (action_list[0] == 2 and action_list[1] == 3) or (action_list[0] == 3 and action_list[1] == 2):  # if oscillating
            action_count += 1  # increment counter
        if action_count == 3:  # if oscillated 3 times
            action = 1  # force forward
            action_count = 0  # reset counter
        # ------execute robot action--------
        robot_instance.execute_action(action)  # execute action
        # ------save robot path and draw it--------
        trajectory.append(tuple(robot_instance.current_position[:2]))  # add position to trajectory
        if len(trajectory) > 1:  # if trajectory has points
            py.draw.lines(win, blue, False, trajectory, 2)  # draw trajectory
        # ------update screen--------
        py.display.flip()  # update display
        # ------count steps--------
        step_count += 1  # increment step
        # ------calculate reward--------
        new_state, new_sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get new state
        reward, _, reason = robot_instance.calculate_reward(target_position, new_sensor_value, action, step_count)  # calculate reward
        total_reward += reward  # accumulate reward

        num_features = len(new_state)  # get number of features
        num_lidar = num_features - 2  # assume last 2 are dist, ang
        print("--------------------------------------------------------------------------------------------------")  # print separator
        print(  # print episode info
            f"total_reward {total_reward:.2f} | "  # total reward
            f"Steps: {step_count} | "  # steps
            f"Reward: {reward:.2f} | "  # current reward
            f"reason: {reason} |"  # termination reason
            f" action: {action} | "  # action taken
            f"Q_values: {Q_value}"  # Q values
        )

        # Print LiDAR
        lidar_str = "|".join(  # create lidar string
            [f"S{i+1}: {new_state[i]:.3f}" for i in range(num_lidar)]  # format each sensor
        )

        print(  # print state info
            f"{lidar_str} | "  # lidar values
            f"Dist_norm: {new_state[-2]:.2f} | "  # normalized distance
            f"Ang_norm: {new_state[-1]:.2f}"  # normalized angle
        )
        print("--------------------------------------------------------------------------------------------------")  # print separator

        # ------delay for better visualization--------
        time.sleep(0.1)  # sleep for 0.1 seconds                    

