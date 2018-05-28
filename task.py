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
            target_pos: target/goal (x,y,z) position for the agent // changed to target v and angular_v, so a list of 6 values
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * len(self.sim.pose)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array(
            [0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # default value, only uses x-y-z component of pose, ignores orientation; also target_pos is set to be [x,y,z]
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        # for my goal, I want all speeds to be 0, but do not care about position or orientation
        # so target_pos is set to be [v_x, v_y, v_z, v_phi, v_theta, v_psi]
        # added more base reward because I saw lots of crashes and I think flying longer is better
        # less weight to rotation because remaining at a fixed position even if turning on itself is more important
        # than moving(or falling) perfectly perpendicular to the ground
        reward = 10.0 - 0.2 * (
            abs(self.sim.v - self.target_pos[:3])).sum() - 0.05 * (
                     abs(self.sim.angular_v - self.target_pos[3:])).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(
                rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            # default pose_all
            # pose_all.append(self.sim.pose)
            # my version
            pose_all.append(self.sim.v)
            pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # default state, uses x,y,z,phi,theta,psi repeated n times
        # state = np.concatenate([self.sim.pose] * self.action_repeat)
        # my state, uses [v_x, v_y, v_z, v_phi, v_theta, v_psi] repeated n times
        state = np.concatenate(
            ([self.sim.v] + [self.sim.angular_v]) * self.action_repeat)
        return state