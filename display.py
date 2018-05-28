import pandas as pd
import matplotlib.pyplot as plt
current_sim_filename = "current_sim.txt"
best_sim_filename = "best_sim.txt"

#sim = pd.read_csv(current_sim_filename)
sim = pd.read_csv(best_sim_filename)
print("Results of the best simulation:")

#4 plots of the quadcopter state across the chosen(best) episode
plt.subplots(figsize=(16,3))
plt.subplot(141)
plt.plot(sim['time'], sim['x'], label='x')
plt.plot(sim['time'], sim['y'], label='y')
plt.plot(sim['time'], sim['z'], label='z')
plt.title("Position (no goal)")
plt.legend()
plt.subplot(142)
plt.plot(sim['time'], sim['x_velocity'], label='v_x')
plt.plot(sim['time'], sim['y_velocity'], label='v_y')
plt.plot(sim['time'], sim['z_velocity'], label='v_z')
plt.title("Velocities (goal: 0)")
plt.legend()
plt.subplot(143)
plt.plot(sim['time'], sim['phi'], label='phi')
plt.plot(sim['time'], sim['theta'], label='theta')
plt.plot(sim['time'], sim['psi'], label='psi')
plt.title("Orientation (no goal)")
plt.legend()
plt.subplot(144)
plt.plot(sim['time'], sim['phi_velocity'], label='v_phi')
plt.plot(sim['time'], sim['theta_velocity'], label='v_theta')
plt.plot(sim['time'], sim['psi_velocity'], label='v_psi')
plt.title("Angular velocities (goal: 0)")
plt.legend()
plt.show()


#plot of the reward evolution for each episode across training
plt.figure(figsize=(8, 6))
plt.plot(rewards)
plt.title("Total reward after each episode")
plt.show()
