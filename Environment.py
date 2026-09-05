# environment.py - Environment management for robot simulation

import numpy as np  # import numpy for array operations
from PIL import Image  # import PIL for image processing
import random  # import random for random number generation
from config import MIN_DISTANCE, IMAGES, TEST_IMAGE, DELTA_THETA,IMAGES  # import configuration parameters


class Environment:  # define the Environment class
    def __init__(self):  # initialize environment
        self.minimum_distance = MIN_DISTANCE  # set minimum distance from walls
        self.matrix = None  # initialize occupancy matrix
        self.width = None  # initialize width
        self.height = None  # initialize height

    def generate_matrix(self):  # generate occupancy matrix from test image
        image = random.choice(IMAGES)  # get test image path
        img = Image.open(image).convert("L")  # open and convert to grayscale
        arr = np.array(img)  # convert to numpy array

        threshold = 128  # binarization threshold
        binary_matrix = np.where(arr < threshold, 1, 0)  # create binary matrix (1 for walls)

        self.matrix = binary_matrix  # store matrix
        self.height, self.width = binary_matrix.shape  # set height and width
        return binary_matrix  # return the matrix

    def generate_test_matrix(self,image_test):  # generate occupancy matrix from specified image
        image = image_test  # get image path
        img = Image.open(image).convert("L")  # open and convert to grayscale
        arr = np.array(img)  # convert to numpy array

        threshold = 128  # binarization threshold
        binary_matrix = np.where(arr < threshold, 1, 0)  # create binary matrix (1 for walls)

        self.matrix = binary_matrix  # store matrix
        self.height, self.width = binary_matrix.shape  # set height and width
        return binary_matrix  # return the matrix

    def valid_distance_from_wall(self, x, y, step=4):  # check if position is valid (far from walls)
        for dx in range(-self.minimum_distance, self.minimum_distance + 1, step):  # iterate x offsets
            for dy in range(-self.minimum_distance, self.minimum_distance + 1, step):  # iterate y offsets
                nx, ny = x + dx, y + dy  # compute neighbor position
                if (0 <= nx < self.width) and (0 <= ny < self.height):  # check bounds
                    if self.matrix[ny, nx] == 1:  # if wall present #type: ignore
                        return False  # invalid position
        return True  # valid position

    def generate_valid_position(self):  # generate a valid robot starting position
        while True:  # loop until valid position found
            x = random.randint(0, self.width - 1)  # random x coordinate #type: ignore
            y = random.randint(0, self.height - 1)  # random y coordinate #type: ignore
            angle = random.randint(0, 36) * DELTA_THETA  # random angle
            if self.valid_distance_from_wall(x, y):  # check validity
                return (x, y, angle)  # return position tuple

    def create_target(self):  # generate a valid target position
        while True:  # loop until valid position found
            x = random.randint(0, self.width - 1)  # random x coordinate #type: ignore
            y = random.randint(0, self.height - 1)  # random y coordinate #type: ignore
            if self.valid_distance_from_wall(x, y):  # check validity
                return (x, y)  # return position tuple