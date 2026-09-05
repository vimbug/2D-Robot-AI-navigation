import config  # import configuration settings
from DQNAgent import DQNAgent  # import DQN agent class
from Environment import Environment  # import environment simulator
from robot import Robot  # import robot controller
import pygame as py  # import pygame for rendering
import math  # import math for calculations
import numpy as np  # import numpy for arrays

py.init()  # initialize pygame

# --------Loading images-------  # load images section
house = py.image.load("House_test.png")  # load house image
robot_img = py.image.load("robot.png")  # load robot sprite image

# --------color definitions---------  # color definitions section
white = (255, 255, 255)  # white color
black = (0, 0, 0)  # black color
green = (0, 255, 0)  # green color
red = (255, 0, 0)  # red color
blue = (0, 0, 255)  # blue color
purple = (128, 0, 128)  # purple color

# -------Window Settings------  # window settings section
W, H = house.get_size()  # get image dimensions
win = py.display.set_mode((W, H))  # create window
py.display.set_caption("Robot RL with PyTorch - Optimized")  # set window title
font = py.font.SysFont("Arial", 24)  # create font for text
clock = py.time.Clock()  # create clock for frame limiting

# ------Initialize DQN agent------  # initialize agent section
agent = DQNAgent()  # create DQN agent instance (currently unused for manual testing)

# ------Main simulation loop------  # main loop section
for episode in range(1):  # loop for one episode only
    env = Environment()  # create environment instance
    logic_matrix = env.generate_test_matrix("House_test.png")  # generate occupancy matrix from the test image
    initial_position = env.generate_valid_position()  # choose a valid start position
    target_position = env.create_target()  # choose a valid target position

    robot_instance = Robot(initial_position)  # create robot object with initial position
    done = False  # episode end flag
    total_reward = 0  # reward accumulator
    step_count = 0  # step counter

    while not done and step_count < config.MAX_STEPS:  # run until termination or max steps
        win.fill(white)  # clear screen
        win.blit(house, house.get_rect(center=(W // 2, H // 2)))  # draw house image

        # draw the target and robot
        py.draw.circle(win, red, target_position, 10)  # draw target point
        robot_img_rotated = py.transform.rotate(robot_img, -robot_instance.current_position[2])  # rotate robot image
        robot_rect = robot_img_rotated.get_rect(center=robot_instance.current_position[:2])  # compute robot rect
        win.blit(robot_img_rotated, robot_rect)  # draw robot on screen

        state, sensor_value = robot_instance.get_state(target_position, logic_matrix)  # get current state and lidar sensors

        # draw lidar rays
        ray_angles = np.linspace(config.FIRST_RAY_ANGLE, config.LAST_RAY_ANGLE, config.NUM_LIDAR) + robot_instance.current_position[2]
        for i in range(len(sensor_value)):  # draw each lidar ray
            dist = sensor_value[i]
            start = (
                robot_instance.current_position[0] + config.LIDAR_RADIUS_MIN * math.cos(math.radians(ray_angles[i])),
                robot_instance.current_position[1] + config.LIDAR_RADIUS_MIN * math.sin(math.radians(ray_angles[i])),
            )
            end = (
                robot_instance.current_position[0] + config.LIDAR_RADIUS_MAX * dist * math.cos(math.radians(ray_angles[i])),
                robot_instance.current_position[1] + config.LIDAR_RADIUS_MAX * dist * math.sin(math.radians(ray_angles[i])),
            )
            py.draw.line(win, blue, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 4)

        # render info text
        info_text = f"Steps: {step_count}  Total Reward: {total_reward:.2f}"
        info_surface = font.render(info_text, True, black)
        win.blit(info_surface, (10, 10))

        # event handling
        for event in py.event.get():
            if event.type == py.QUIT:
                done = True
            elif event.type == py.KEYDOWN:
                action = None
                if event.key == py.K_UP:
                    action = 1
                elif event.key == py.K_LEFT:
                    action = 2
                elif event.key == py.K_RIGHT:
                    action = 3
                elif event.key == py.K_DOWN:
                    action = 0
                elif event.key == py.K_SPACE:
                    action = 4

                if action is not None:
                    robot_instance.execute_action(action)  # apply the selected action
                    new_state, new_sensor_value = robot_instance.get_state(target_position, logic_matrix)  # recalculate state
                    reward, done, reason = robot_instance.calculate_reward(target_position, new_sensor_value, action, step_count)
                    total_reward += reward
                    step_count += 1

                    print("--------------------------------------------------------------------------------------------------")
                    print(f"Total Reward: {total_reward:.2f} | Steps: {step_count} | Reward: {reward:.2f} | Reason: {reason}")
                    num_features = len(new_state)
                    num_lidar = num_features - 2
                    lidar_str = " | ".join([f"S{i+1}: {new_state[i]:.3f}" for i in range(num_lidar)])
                    print(f"{lidar_str} | Dist_norm: {new_state[-2]:.2f} | Ang_norm: {new_state[-1]:.2f}")
                    print("--------------------------------------------------------------------------------------------------")

        py.display.flip()  # update the display each frame
        clock.tick(30)  # limit frame rate to 30 FPS

print("Test completed.")  # print completion message

# Guide to this script by line number
# 1-7   : import required modules and project components
# 9     : initialize pygame before creating any windows or fonts
# 11-13 : load test image and robot sprite
# 16-22: define drawing colors for pygame
# 24-29: set up the pygame window, title, font, and clock
# 31-32: instantiate the DQN agent object (currently unused in this manual test)
# 35-43: start a single interactive episode and initialize environment, matrix, start, and target
# 44   : instantiate the Robot object with the start position
# 46-49: initialize episode state: done flag, total_reward, and step_count
# 51-72: render the scene, draw the robot, target, and lidar rays
# 74-87: handle pygame events and map keys to robot actions
# 89-103: execute action, update state, get reward, and print debug values
# 105-106: refresh the display and cap the update rate at 30 FPS
# 108   : print completion message after the simulation ends

