import numpy as np
import copy 

class ModelLearningAgent:
    def __init__(self, world_model, planner, prior_strength=0.01, mixture_weight=0.9,
                 observation_increment=0.5, hypothetical_increment=0.1, counterfactual_increment=0.1):
        self.world_model = world_model
        self.planner = planner
        for state, value in self.world_model.environment_spec['states'].items():
            if value == None:
                self.world_model.environment_spec['states'][state] = (prior_strength, prior_strength)
        self.action_space = self.world_model.action_space
        self.state_space = self.world_model.state_space
        self.mixture_weight = mixture_weight
        
        self.observation_increment = observation_increment
        self.hypothetical_increment = hypothetical_increment
        self.counterfactual_increment = counterfactual_increment

        self.visited_states = set()

    def generate_simulated_world_model(self, ground_truth_spec, counterfactual=None):
        """Generate a simulated world based on the mixing of model and ground truth, optionally applying counterfactual adjustments."""
        simulated_world_model = self.world_model.fork(environment_spec=copy.deepcopy(self.world_model.environment_spec))
        for state, state_model in simulated_world_model.environment_spec['states'].items():
            if counterfactual and state in counterfactual:
                simulated_world_model.environment_spec['states'][state] = int(counterfactual[state])
            elif isinstance(state_model, tuple):
                simulated_world_model.environment_spec['states'][state] = self.mixture_weight * state_model[0] / sum(state_model) + (1 - self.mixture_weight) * ground_truth_spec['states'][state]
        return simulated_world_model
    
    # def run_simulated_episode(self, simulated_world_model, start_location, goal_location, wall_locations, simulation_type, greedy=True):
    #     """Use the simulated world spec to run a learning episode."""
    #     self.reset_episode(start_location, goal_location, wall_locations, model=simulated_world_model)
    #     simulated_world = simulated_world_model.generate_world(start_location, goal_location, wall_locations)
    #     state = start_location
    #     trajectory = []
    #     episode_finished = False
    #     prior_exp = np.zeros((self.world_model.environment_spec['metadata']['width'], self.world_model.environment_spec['metadata']['height']))
    #     for loc, vals in self.world_model.environment_spec['states'].items():
    #         prior_exp[loc] = sum(vals)

    #     while not episode_finished:
    #         action = self.planner.choose_action(state, prior_experience=prior_exp, target_policy=greedy)
    #         next_state, reward, timestep, episode_finished, info = simulated_world.transition(state, action)
    #         if state != start_location:
    #             self.update_model(state, info, simulation_type)
    #         else:
    #             self.visited_states.add(start_location)

    #         trajectory.append((timestep, state, action, reward, next_state, info))
    #         state = next_state
    #     return trajectory

    def update_model(self, state, outcome, simulation_type):
        stalled = outcome # TODO: fix this to not depend on specifics of quicksand world
        if simulation_type == 'observation':
            increment = self.observation_increment
        elif simulation_type == 'hypothetical':
            increment = self.hypothetical_increment
        elif simulation_type == 'counterfactual':
            increment = self.counterfactual_increment
        
        if state not in self.visited_states:
            state_info = self.world_model.environment_spec['states'].get(state)
            if state_info and not isinstance(state_info, str):
                alpha, beta = state_info
                self.world_model.environment_spec['states'][state] = (alpha + increment * int(stalled), beta + increment * int(not stalled))
            self.visited_states.add(state)

    def reset_episode(self, start_location, goal_location, wall_locations=None, model=None):
        self.visited_states.clear()
        self.planner = self.planner.fork(world_model=model if model else self.world_model)
        self.planner.learn_policy(start_location, goal_location, wall_locations)
        
class SimpleModelLearningAgent(ModelLearningAgent):
    def __init__(self, world_model, planner, prior=0.5, mixture_weight=0.9, observation_increment=0.5, hypothetical_increment=0.1, counterfactual_increment=0.1, unsafe_tile_prob_quicksand=0.8):
        self.world_model = world_model
        self.planner = planner
        for state, value in self.world_model.environment_spec['states'].items():
            if value == None:
                self.world_model.environment_spec['states'][state] = prior
        self.action_space = self.world_model.action_space
        self.state_space = self.world_model.state_space
        self.mixture_weight = mixture_weight
        
        self.observation_increment = observation_increment
        self.hypothetical_increment = hypothetical_increment
        self.counterfactual_increment = counterfactual_increment
        self.unsafe_tile_prob_quicksand = unsafe_tile_prob_quicksand

        self.visited_states = set()

    def generate_simulated_world_model(self, ground_truth_spec, counterfactual=None):
        """Generate a simulated world based on the mixing of model and ground truth, optionally applying counterfactual adjustments."""
        simulated_world_model = self.world_model.fork(environment_spec=copy.deepcopy(self.world_model.environment_spec))
        for state, state_model in simulated_world_model.environment_spec['states'].items():
            if counterfactual and state in counterfactual:
                simulated_world_model.environment_spec['states'][state] = int(counterfactual[state])
            elif isinstance(state_model, float):
                simulated_world_model.environment_spec['states'][state] = self.mixture_weight * state_model * self.unsafe_tile_prob_quicksand + (1 - self.mixture_weight) * ground_truth_spec['states'][state]
        return simulated_world_model

    def update_model(self, state, outcome, simulation_type):
        stalled = outcome
        if simulation_type == 'observation':
            increment = self.observation_increment
        elif simulation_type == 'hypothetical':
            increment = self.hypothetical_increment
        elif simulation_type == 'counterfactual':
            increment = self.counterfactual_increment
            
        if state not in self.visited_states:
            prior = self.world_model.environment_spec['states'].get(state)
            if prior and not isinstance(prior, str):
                if stalled or prior == 1:
                    posterior = 1
                else:
                    posterior = (1 - self.unsafe_tile_prob_quicksand) * prior / (1 - self.unsafe_tile_prob_quicksand * prior)
                self.world_model.environment_spec['states'][state] = (posterior - prior) * increment + prior

            self.visited_states.add(state)
            
class BayesianModelLearningAgent(ModelLearningAgent):
    def __init__(self, world_model, planner, 
                 prior=0.5, mixture_weight=0.9,
                 prob_simulation_noise=0.8, # [0.5, 1], is actually probability of not adding noise
                 prob_observation_noise=1, # [0.5, 1], is actually probability of not adding noise
                 unsafe_tile_prob_quicksand=0.8,
                 decay_rate=0.95,
                 # the proportion of the decay rate to retain when updating the model
                 simulation_rememberence_proportion=0.5,
                 observation_rememberence_proportion=0.8,
                 ):
        self.world_model = world_model
        self.planner = planner
        self.prior = prior
        for state, value in self.world_model.environment_spec['states'].items():
            if value == None:
                self.world_model.environment_spec['states'][state] = self.prior
        self.action_space = self.world_model.action_space
        self.state_space = self.world_model.state_space

        self.mixture_weight = mixture_weight
        self.prob_simulation_noise = prob_simulation_noise
        self.prob_observation_noise = prob_observation_noise
        self.unsafe_tile_prob_quicksand = unsafe_tile_prob_quicksand
        self.decay_rate = decay_rate
        self.simulation_rememberence_proportion = simulation_rememberence_proportion
        self.observation_rememberence_proportion = observation_rememberence_proportion

        self.visited_states = set()

    def generate_simulated_world_model(self, ground_truth_spec, counterfactual=None):
        """Generate a simulated world based on the mixing of model and ground truth, optionally applying counterfactual adjustments."""
        simulated_world_model = self.world_model.fork(environment_spec=copy.deepcopy(self.world_model.environment_spec))
        for state, state_model in simulated_world_model.environment_spec['states'].items():
            if counterfactual and state in counterfactual:
                simulated_world_model.environment_spec['states'][state] = int(counterfactual[state])
            elif isinstance(state_model, float):
                # chance of "insight" - drawing from the ground truth for a given tile
                if np.random.rand() < self.mixture_weight:
                    simulated_world_model.environment_spec['states'][state] = state_model * self.unsafe_tile_prob_quicksand
                else:
                    simulated_world_model.environment_spec['states'][state] = ground_truth_spec['states'][state]
                # simulated_world_model.environment_spec['states'][state] = self.mixture_weight * state_model * self.unsafe_tile_prob_quicksand + (1 - self.mixture_weight) * ground_truth_spec['states'][state]
                
        return simulated_world_model

    def update_model(self, state, outcome, simulation_type):
        quicksand = outcome
        p_unsafe_is_quicksand = self.unsafe_tile_prob_quicksand
        prior = self.world_model.environment_spec['states'].get(state)
        if state not in self.visited_states:
            p_noise = self.prob_observation_noise if simulation_type == 'observation' else self.prob_simulation_noise
            if quicksand:
                self.world_model.environment_spec['states'][state] = \
                    (p_noise * prior) / (p_noise * prior + (1 - p_noise) * (1 - prior))
            else:
                self.world_model.environment_spec['states'][state] = \
                    (1 - p_noise * p_unsafe_is_quicksand) * prior / ((1 - p_noise * p_unsafe_is_quicksand) * prior + (1 - (1 - p_noise) * p_unsafe_is_quicksand) * (1 - prior))
                    
    def forget(self, simulation_states=None, observation_states=None):
        simulation_states = simulation_states or list()
        observation_states = observation_states or list()
        for state, posterior in self.world_model.environment_spec['states'].items():
            proportion_retained = (
                self.observation_rememberence_proportion if state in observation_states 
                else self.simulation_rememberence_proportion if state in simulation_states 
                else 0
            )
            decay_rate = self.decay_rate + (1 - self.decay_rate) * proportion_retained

            # constant decay towards 0.5
            self.world_model.environment_spec['states'][state] = self.prior + (posterior - self.prior) * decay_rate
            
            # constant decay towards random value with expected value of 0.5
            # prior = np.random.beta(20, 20)
            # self.world_model.environment_spec['states'][state] = prior + (posterior - prior) * self.decay_rate

        # if state not in self.visited_states:
        #     prior = self.world_model.environment_spec['states'].get(state)
        #     if prior and not isinstance(prior, str):
        #         if quicksand or prior == 1:
        #             posterior = 1
        #         else:
        #             posterior = (1 - self.unsafe_tile_prob_quicksand) * prior / (1 - self.unsafe_tile_prob_quicksand * prior)
        #         self.world_model.environment_spec['states'][state] = (posterior - prior) * increment + prior

        #     self.visited_states.add(state)


            # if is_observation_trial:
            #     if quicksand:
            #         self.world_model.environment_spec['states'][state] = 1
            #     else:
            #         self.world_model.environment_spec['states'][state] = \
            #             (1 - p_unsafe_is_quicksand) * prior / (1 - p_unsafe_is_quicksand * prior)
            # else:
            #     if quicksand:
            #         self.world_model.environment_spec['states'][state] = \
            #             (p_simulation_noise * prior) / (p_simulation_noise * prior + (1 - p_simulation_noise) * (1 - prior))
            #     else:
            #         self.world_model.environment_spec['states'][state] = \
            #             (1 - p_simulation_noise * p_unsafe_is_quicksand) * prior / ((1 - p_simulation_noise * p_unsafe_is_quicksand) * prior + (1 - (1 - p_simulation_noise) * p_unsafe_is_quicksand) * (1 - prior))        
