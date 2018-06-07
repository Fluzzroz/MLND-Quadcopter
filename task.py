import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent // changed to target only z
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        #default state_size
        #self.state_size = self.action_repeat * len(self.sim.pose)
        
        #simplified state
        self.state_size = self.action_repeat
        
        self.action_low = 0    #default 0
        self.action_high = 900   #default 900
        self.action_size = 1     #default 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #default value
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
               
        #my reward
        reward  = 1 - 0.01 * abs(self.sim.pose[2] - self.target_pos) - 0.01*abs(self.sim.v[2])
               
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            #default pose_all
            #pose_all.append(self.sim.pose)
            
            #simple state
            pose_all.append(self.sim.pose[2])
        
        #default next_state
        #next_state = np.concatenate(pose_all)
        
        #simple next_state
        next_state = np.array(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #default state, uses x,y,z,phi,theta,psi repeated n times
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        
        #simple state
        state = [self.sim.pose[2]] * self.action_repeat
        return state