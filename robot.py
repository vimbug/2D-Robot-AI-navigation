import numpy as np
import math
import torch

from config import (FIRST_RAY_ANGLE, LAST_RAY_ANGLE, NUM_LIDAR, DELTA_THETA, DELTA_VEL,
                    LIDAR_RADIUS_MIN, LIDAR_RADIUS_MAX, NUM_LIDAR_STEPS, DEVICE,
                    GOAL_DISTANCE, MAX_STEPS, VEL_EXTRA)

from utils import vector_from_points, vector_from_angle, angle_between, distance_to_target, add_new_position

BASE_ANGLES = torch.deg2rad(torch.linspace(FIRST_RAY_ANGLE, LAST_RAY_ANGLE, NUM_LIDAR, device=DEVICE))

class Robot:
	def __init__(self, initial_position, delta_vel=DELTA_VEL, delta_theta=DELTA_THETA, base_angles=BASE_ANGLES,
				 lidar_min_radius=LIDAR_RADIUS_MIN, lidar_max_radius=LIDAR_RADIUS_MAX, num_lidar_steps=NUM_LIDAR_STEPS, device=DEVICE,
				 goal_distance=GOAL_DISTANCE):

		self.positions                	= [initial_position, initial_position]
		self.delta_vel                	= delta_vel
		self.delta_theta              	= delta_theta
		self.base_angles              	= base_angles
		self.lidar_min_radius         	= lidar_min_radius
		self.lidar_max_radius         	= lidar_max_radius
		self.num_lidar_steps          	= num_lidar_steps
		self.device                   	= device
		self.goal_distance              = goal_distance
		self.W                        	= None
		self.H                        	= None
		self.sensor_value_prev 		  	= None 
		self.osc_penalty			  	= 0.0
		self.prev_action              	= None

	@property
	def current_position(self):
		"""Returns the current position of the robot"""
		return self.positions[1]
	@property
	def previous_position(self):
		"""Returns the previous position of the robot"""
		return self.positions[0]

	def calculate_lidar(self, matrix, noise_std=0):				# calculate the sensor values of the robot
		self.H, self.W = matrix.shape							# starting from the image calculate how wide and high it is, adaptive way
		matrix_tensor = torch.as_tensor(matrix, dtype=torch.int8, device=self.device)		# create tensor on gpu from logical matrix
		eff_angles = self.base_angles + torch.deg2rad(torch.tensor([self.positions[1][2]], device=self.device))		# add the robot's angle to the base angles
		cos_angles = torch.cos(eff_angles)		# tensor of cosines of effective angles
		sin_angles = torch.sin(eff_angles)		# tensor of sines of effective angles
		distances = torch.linspace(self.lidar_min_radius, self.lidar_max_radius, self.num_lidar_steps, device=self.device)		# tensor of distances, starting from min_radius to max_radius through a number of steps decided by us
       
		# we have the tensor of cosines of sines and distances now in x and y starting from the robot's position we calculate each sensor
		x_pos = self.positions[1][0] + distances.unsqueeze(0) * cos_angles.unsqueeze(1)
		y_pos = self.positions[1][1] + distances.unsqueeze(0) * sin_angles.unsqueeze(1)
        
		# create mask of valid positions
		valid_mask = (x_pos >= 0) & (x_pos < self.W) & (y_pos >= 0) & (y_pos < self.H)
		x_int = x_pos.clamp(0, self.W-1).long()		# integer x coordinates
		y_int = y_pos.clamp(0, self.H-1).long()		# integer y coordinates
		wall_vals = matrix_tensor[y_int,x_int]		# extract wall values from logical matrix at x and y positions
		wall_vals[~valid_mask] = 0					# mask invalid values
		is_wall = wall_vals.bool()					# extract boolean wall values
		has_wall = is_wall.any(dim=1)				# check if there are walls
		first_wall_idx = torch.where(
			has_wall,
			is_wall.float().argmax(dim=1),
			torch.full((is_wall.size(0),), self.num_lidar_steps-1, device=self.device, dtype=torch.long)
		)#
		sensor_readings = distances[first_wall_idx]
		sensor_readings += torch.normal(0.0, noise_std, size=sensor_readings.size(), device=self.device)
		return sensor_readings.cpu().numpy()

	def execute_action(self, action):
		new_x, new_y, new_theta = self.positions[1][0], self.positions[1][1], self.positions[1][2]
		if action == 0:#slow forward
			new_x += self.delta_vel * math.cos(math.radians(self.positions[1][2]))
			new_y += self.delta_vel * math.sin(math.radians(self.positions[1][2]))
		elif action == 1:#fast forward
			new_x += (self.delta_vel+VEL_EXTRA) * math.cos(math.radians(self.positions[1][2]))
			new_y += (self.delta_vel+VEL_EXTRA) * math.sin(math.radians(self.positions[1][2]))
		elif action == 2:#turn left
			new_theta -= self.delta_theta
		elif action == 3:#turn right
			new_theta += self.delta_theta
		elif action == 4:#backward only for test
			new_x -= (self.delta_vel+4) * math.cos(math.radians(self.positions[1][2]))
			new_y -= (self.delta_vel+4) * math.sin(math.radians(self.positions[1][2]))	

		new_theta = new_theta % 360
		self.positions = add_new_position(self.positions, (new_x, new_y, new_theta))#use the function add_new_position to update the position in the positions matrix
		return 

	def get_state(self, pos_target, matrice):		# get the current state of the robot
		sensor_values = self.calculate_lidar(matrice)		# calculate the sensor values of the robot
		
		# Clamp and normalize sensor values
		sensor_values_clamp = np.clip(sensor_values, self.lidar_min_radius, self.lidar_max_radius)
		sensor_norm = (sensor_values_clamp - self.lidar_min_radius) / (self.lidar_max_radius - self.lidar_min_radius)

		# Calculate relative angle and normalized distance to target
		v_target = vector_from_points((self.positions[1][0], self.positions[1][1]), pos_target)#vector from robot center to target
		v_robot = vector_from_angle(self.positions[1])#robot direction vector
		rel_ang_norm = angle_between(v_robot, v_target)#included angle
		euclid = math.hypot(pos_target[0] - self.positions[1][0], pos_target[1] - self.positions[1][1])#distance between robot and target in straight line
		max_dist = math.hypot(self.W, self.H)#reference size of the room #type: ignore
		dist_norm = euclid / max_dist#normalization

		# Compose state
		state = np.concatenate([sensor_norm, [dist_norm, rel_ang_norm]])#state vector sensor measurements lidar+straight line distance normalized+included angle normalized
		return state, sensor_norm

	def calculate_reward(self, pos_target, sensor_value, action, step):
  
		# === Current sensor values ===
		sx90, sx60, sx30, centro, dx30, dx60, dx90 = sensor_value
  
		# === Robot state vectors ===
		pos_prev 	= self.positions[0]										#previous position
		pos_curr 	= self.positions[1]										#current position
		dist_prev 	= distance_to_target(self.positions, pos_target, 0)		#previous distance to target
		dist_curr 	= distance_to_target(self.positions, pos_target, 1)		#current distance to target
		delta_dist = dist_prev - dist_curr
		
		#=====Vectors and angles===	
		v_robot_prev 	= vector_from_angle(pos_prev)						#previous robot direction vector
		v_robot_curr 	= vector_from_angle(pos_curr)						#current robot direction vector
		v_target_prev	= vector_from_points(pos_prev, pos_target)			#vector from previous robot to target
		v_target_curr 	= vector_from_points(pos_curr, pos_target)			#vector from current robot to target
		ang_prev 		= angle_between(v_robot_prev, v_target_prev)		#previous included angle
		ang_curr 		= angle_between(v_robot_curr, v_target_curr)		#current included angle
		delta_ang   	= abs(ang_prev) - abs(ang_curr)	                    #difference between preavius angle and current angle


		#-----normalized distance and angle---
		max_dist= math.hypot(self.W, self.H)#reference size of the room #type: ignore
		
    	# === Target reached ===
		if dist_curr <= self.goal_distance:
			return 10+((-40*(step/MAX_STEPS)**2)+40) , True,"goal" #exit with positive reward, true indicates episode terminated and "goal" reached
		
  		# === Collision ===
		if min(sensor_value) <= 0.2: 	
			return -20, True,"collision" #exit with negative reward, true indicates episode terminated for "collision"

		rew_dist = (0.1*delta_dist)	#reward for approaching the target				
		rew_ang  = (20*delta_ang)	#reward for reducing the included angle
		rew_sens = 0				#sum of the rewards provided by the sensors

		rew_sens+=-0.1*sx90+0.1
		rew_sens+=-0.1*dx90+0.1	
		rew_sens+=0.2*sx60-0.2
		rew_sens+=0.2*dx60-0.2
		rew_sens+=0.3*sx30-0.3
		rew_sens+=0.3*dx30-0.3
		rew_sens+=0.4*centro-0.4
  
		#---wall side behavior	
		if sx90<1 or dx90<1 or centro<1 or sx60<1 or dx60<1 or sx30<1 or dx30<1:  # Disables the angular reward when any sensor detects a wall closer than 1 unit
			rew_ang=0
			rew_dist=0		
   
		if sx90==1 and dx90==1 and centro==1 and sx60==1 and dx60==1 and sx30==1 and dx30==1: # Removes the sensor-based reward when all sensors measure exactly 1 unit
			rew_sens=0
   
		#---Total reward calculation ---
		rew = rew_dist + rew_ang + rew_sens # Computes the total reward as the sum of distance, angular, and sensor rewards

		return rew, False,"max steps"
