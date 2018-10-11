import numpy as np
import math
from physics_sim import PhysicsSim

class takeoff_task():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5.):
    
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 10

        self.target_pos = np.array([0., 50., 0.])
        
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        p = self.sim.pose
       
        #Reward increased proximity to target position, with a focus on y coordinates as the measure of altitude
        x_reward =  45. - abs( 0.5 * (self.target_pos[0] - p[0]))
        y_reward =  45. - abs(0.5 * (self.target_pos[1] - p[1])) * 8
        z_reward =  45. - abs( 0.5 * self.target_pos[2] - p[2])
        euler_reward = 45. - (0.2 * (abs(p[3]) +  abs(p[4]) + abs(p[5])))
        reward_weights = np.array([1.0, 8.0, 1.0])
        #df_temp["P"].astype(float)
        reward_weights = reward_weights.astype(float)
        
        reward =  1000.0 - (self.target_pos  - p[:3]) * reward_weights
        
        
        #reward = np.negative(reward)
        reward = reward.sum() #x_reward + y_reward + z_reward + euler_reward + 5.
        reward += (500.0 - p[:3]).sum() 
        #reward = abs(reward) 
        #reward = float(reward) * float(-1.0)
        #mod = float(-1.01)
        #reward *= mod
        #reward *= -1.
        #reward = np.negative(reward)
        #reward = 50.101 - reward
        
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) 
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
