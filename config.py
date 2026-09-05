import random as rd

IMAGES = [
	"House1.png",
	"House2.png",
	"House3.png",
	"House4.png",
	"House5.png",
 	"House6.png",
	"House7.png",
	"House8.png",
	"House9.png"
]

#IMAGE = rd.choice(IMAGES)
IMAGE  		= "House8.png"
TEST_IMAGE 	= "House_test.png"

# Robot parameters
DELTA_VEL     = 1  		# speed increment for "forward" and "backward" actions
DELTA_THETA   = 5  		# angle increment for "turn left" and "turn right" actions			
VEL_EXTRA     = 6  		# speed increment for "fast forward" action

# Lidar parameters
LIDAR_RADIUS_MIN  =  20		# minimum distance of lidar rays
LIDAR_RADIUS_MAX  =  60		# maximum distance of lidar rays
NUM_LIDAR         =  7 		# number of lidar rays
NUM_LIDAR_STEPS   =  15		# number of steps for each lidar ray
MIN_DISTANCE      =  30		# minimum distance from walls for valid robot position
FIRST_RAY_ANGLE   = -90		# first lidar ray
LAST_RAY_ANGLE    =  90		# last lidar ray


# Training parameters
NUM_EPISODES 			= 600	# Number of training episodes
MAX_STEPS   			= 1500	# Maximum number of steps per episode
NUM_TEST_EPISODES 		= 10	# Number of test episodes
NUM_EPISODES_TO_SAVE 	= 400	# Number of episodes after which to save the model
SAVE_FREQUENCY      	= 10	# Frequency (in episodes) of model saving


# === Thresholds ===
GOAL_DISTANCE             = 15 	    # minimum distance to consider the target reached


#DQN neural network parameters
DQN_FC1_SIZE  = 32				# Size of first fully connected layer
DQN_FC2_SIZE  = 64				# Size of second fully connected layer
DQN_FC3_SIZE  = 32				# Size of third fully connected layer
STATE_SIZE    = NUM_LIDAR+2 	# lidar + distance + normalized theta 
ACTION_SIZE   = 4				# Number of possible actions
GAMMA         = 0.99			# Discount factor
EPSILON       = 1.0				# Initial epsilon value for epsilon-greedy
EPSILON_MIN   = 0.01			# Minimum epsilon value
EPSILON_DECAY = 500				# Epsilon decay per episode
BATCH_SIZE    = 64				# Batch size for training
TAU           = 0.01			# Parameter for target network update
MEMORY_SIZE   = 100000			# Size of replay memory
LEARNING_RATE = 0.0001			# Learning rate
DEVICE        = "cuda"			# or "cpu" device for training


