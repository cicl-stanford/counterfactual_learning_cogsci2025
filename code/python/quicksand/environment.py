from utils import *
import inspect

class Environment:
    def __init__(self, world_generator, agent_class):
        if not (isinstance(world_generator, object) or inspect.isclass(type(world_generator))):
            raise TypeError("world must be an instance of a class")
        if not inspect.isclass(agent_class):
            raise TypeError("agent_class must be a class type and not an instance of a class")
        self.world_generator = world_generator
        self.agent_class = agent_class
        self.agent = None
        self.trajectories = []

    def initialize_agent(self, planner_class, agent_params=None, planner_params=None):
        if planner_params is None:
            planner_params = {}
        if agent_params is None:
            agent_params = {}
        if not inspect.isclass(planner_class):
            raise TypeError("planner_class must be a class type and not an instance of a class")
        world_model = self.world_generator.fork(environment_spec=self.world_generator.get_base_spec())
        planner = planner_class(world_model, **planner_params)
        self.agent = self.agent_class(world_model, planner, **agent_params)

    def run_episode(self, start_location, goal_location, simulation_type='observation', wall_locations=None, greedy=True, update_model=True):
        if simulation_type != 'observation':
            last_trajectory = {s : i['stalled'] for _,s,_,_,_,i in self.trajectories[-1]} if simulation_type == 'counterfactual' else None
            simulated_world_model = self.agent.generate_simulated_world_model(self.world_generator.environment_spec, counterfactual=last_trajectory)
            confusion_rate = 1 - self.agent.prob_simulation_noise
            world = simulated_world_model.generate_world(start_location, goal_location, wall_locations, confusion_rate)
            self.agent.reset_episode(start_location, goal_location, wall_locations, model=simulated_world_model)
        else: 
            self.world = world = self.world_generator.generate_world(start_location, goal_location, wall_locations)
            self.agent.reset_episode(start_location, goal_location, wall_locations)
        
        episode_finished = False
        state = start_location
        trajectory = []
        while not episode_finished:
            action = self.agent.planner.choose_action(state, target_policy=greedy)
            next_state, reward, timestep, episode_finished, info = world.transition(state, action)
            if state != start_location: # start location is never quicksand, but that shouldn't reflect learning
                if update_model:
                    self.agent.update_model(state, info['stalled'], simulation_type=simulation_type)
            else:
                self.agent.visited_states.add(start_location)
            trajectory.append((timestep, state, action, reward, next_state, info))
            state = next_state
        if simulation_type == 'observation':
            self.trajectories.append(trajectory)

        return trajectory
    
    def estimate_performance(self, start_location, goal_location, wall_locations=None, num_episodes=30):
        timesteps = []
        self.world = self.world_generator.generate_world(start_location, goal_location, wall_locations)
        self.agent.reset_episode(start_location, goal_location, wall_locations)
        for _ in range(num_episodes):
            self.world = self.world_generator.generate_world(start_location, goal_location, wall_locations)
            state = start_location
            episode_finished = False
            while not episode_finished:
                action = self.agent.planner.choose_action_eval(state)
                next_state, _, timestep, episode_finished, _ = self.world.transition(state, action)
                state = next_state
            timesteps.append(timestep) # last timestep of the episode
        return timesteps