import numpy as np
from physics_sim import PhysicsSim


class Takeoff():
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
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 600
        self.action_size = 1
        
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        reward = 0.0
        reward -= (abs(self.sim.pose[2] - self.target_pos[2])) / 3.0
        reward -= (abs(self.sim.angular_v[:3])).sum()
        reward += 3.0*self.sim.v[2]        
        reward += 2.0*self.sim.pose[2]
        
                    
        np.clip (reward, 1, -1)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds*np.ones(4))  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)

            if (self.sim.pose[2] >= self.target_pos[2]):
                reward += 1
                done = True
                
            if done and self.sim.time < self.sim.runtime: 
                reward += -1

        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
