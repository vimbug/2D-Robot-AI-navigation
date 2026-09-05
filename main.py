import subprocess
import sys

TIPOLOGY = 2  # 0=training, 1=test reward, 2=render, 3=automatic tester
MODEL_NAME = "dqn_model_ep730.pth" #Select the model to render
NETWORK_DIR = "reward 18" #Directory from which to load the networks for testing

def main():
    if TIPOLOGY   == 0:
        subprocess.run([sys.executable, "ROBOT_TRAINING.py"])#Train the robot and save the trained networks
    
    elif TIPOLOGY == 1:
        subprocess.run([sys.executable, "ROBOT_TEST_REWARD.py"])#Run a simulation to evaluate the robot's reward and performance, where a human controls the robot
                
    elif TIPOLOGY == 2:
        subprocess.run([sys.executable, "ROBOT_RENDER.py",MODEL_NAME])#Run rendering and select the network that controls the robot
    
    elif TIPOLOGY == 3:
        subprocess.run([sys.executable, "ROBOT_AUTOMATIC_TESTER.py",NETWORK_DIR])#Generate a visualization that evaluates all saved networks and displays a 3D plot of steps, reward, and goal completion rate

    else:
        print("Invalid TIPOLOGY")
        sys.exit(1)

if __name__ == "__main__":
    main()
