import sys, shutil, csv
import numpy as np
from agents.agent import Actor, Critic, DDPG
from task import Task

num_episodes = 2000

# I'm setting the initial conditions randomly, but fixed across all episodes.
runtime = 5.  # time limit of the episode
init_pose = np.array(
    [0.0, 0.0, 10.0, 0.32, 0.22, 0.58])  # initial pose (x,y,z,phi,theta,psi)
init_velocities = np.array([1.2, 1.7, 1.3])  # initial velocities
init_angle_velocities = np.array([1.2, 1.4, 0.2])  # initial angle velocities
file_output = 'data.txt'  # file name for saved results
target_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, ])  # my target, which stands for [v_x, v_y, v_z, v_phi, v_theta, v_psi]

task = Task(init_pose=init_pose, init_velocities=init_velocities,
            init_angle_velocities=init_angle_velocities, runtime=runtime,
            target_pos=target_pos)
agent = DDPG(task)
best_reward = -np.inf

current_sim_filename = "current_sim.txt"
best_sim_filename = "best_sim.txt"
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3',
          'rotor_speed4']

rewards = []

for i_episode in range(1, num_episodes + 1):
    state = agent.reset_episode()  # start a new episode
    total_reward = 0

    with open(current_sim_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            # running the simulation
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # saving the current simulation on file
            to_write = [task.sim.time] + list(task.sim.pose) + list(
                task.sim.v) + list(task.sim.angular_v)
            writer.writerow(to_write)

            if done:
                # check to see if we found a new best simulation and if so, update
                if total_reward > best_reward:
                    best_reward = total_reward
                    shutil.copyfile(current_sim_filename, best_sim_filename)

                # save and print current results
                rewards.append(total_reward)
                print(
                    "\rEpisode = {:4d},     total_reward = {:9.3f}     (best = {:9.3f})".format(
                        i_episode, total_reward, best_reward), end="",
                    flush=True)
                break
