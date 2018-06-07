## Udacity.com's Teach a Quadcopter to fly project

I implemented and Actor-Critic algorithm inspired by the DDPG (Deep Deterministic Policy Gradient) in Keras with a Tensorflow backend whose goal is to teach a quadcopter to take off.

The quadcopter environment is a simulation provided by udacity.

The agent controls the rotor speed of all four rotors simultaneously. It is possible to have the agent control all 4 seperately, but I simplified the problem by having them all synchronized.

I limited the action-state space to the z-position. The reward function uses only the z-position and velocity.

Unfortunately I was unable to get the agent to learn a proper solution. The ones it finds are either a crash-as-fast-as-possible or put-the-pedal-to-the-metal types.



More info regarding tuning hyperparameters and understanding the underlying model can be found here:

https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
