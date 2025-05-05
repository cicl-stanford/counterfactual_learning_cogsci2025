import math
import random
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from utils import *
from quicksand_tree import *

class Planner(ABC):
    """
    Abstract base class for all planner types. Defines a standard interface
    for planner implementations for the model learning project.
    """
    def __init__(self, world_model, learning_rate=0.1, discount_factor=1, epsilon=0.1,
                 epsilon_decay=0, min_epsilon=0.05, num_episodes=500, reward_func=lambda t: 1):
        """
        Initialize a Planner with a given world model and parameters for learning.
        """
        self.world_model = world_model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        self.reward_func = reward_func
        self.current_episode = 0
        self.logs = []

    def learn_policy(self, start_location, goal_location, wall_locations=None):
        """
        Learn the policy by running a specified number of episodes, handling each episode's lifecycle.
        """
        for _ in range(self.num_episodes):
            self.reset_for_new_episode(start_location, goal_location, wall_locations)
            log = self.run_single_episode(start_location, goal_location, wall_locations)
            self.finalize_episode(log)
            self.current_episode += 1

    def reset_for_new_episode(self, start_location, goal_location, wall_locations=None):
        """
        Reset or initialize the environment and planner state for a new episode.
        This includes resetting the environment state and any planner-specific state that should be reinitialized.
        """
        self.world = self.world_model.generate_world(start_location, goal_location, wall_locations)
        self.episode_finished = False

    def finalize_episode(self, log):
        """
        Finalize the episode by performing tasks such as logging, parameter adjustments, or state cleanup.
        This method can be overridden by subclasses if specific finalization steps are needed.
        """
        self.logs.append(log)
        self.epsilon = max(self.epsilon * (1 - self.epsilon_decay), self.min_epsilon)

    @abstractmethod
    def run_single_episode(self, start_location, goal_location, wall_locations=None):
        """
        Execute the actions and updates necessary to run a single episode.
        """
        pass

    @abstractmethod
    def update_policy_values(self, state, action, reward, next_state):
        """
        Update policy values for a given state-action pair, facilitating learning of the policy.
        """
        pass

    @abstractmethod
    def choose_action(self, state, target_policy=False):
        """
        Select an action based on the current state using the policy defined by the planner.
        """
        pass

@forkable
class QLearningPlanner(Planner):
    """
    A planner that implements Q-learning with an epsilon-greedy policy for action selection.
    """
    def __init__(self, world_model, learning_rate=0.2, discount_factor=0.9,
                 epsilon=0.8, epsilon_decay=0.03, min_epsilon=0.04, init_value=3, num_episodes=200, reward_func=lambda t: 1):
        super().__init__(world_model, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon, num_episodes, reward_func)
        self.q_table = {state: {action: init_value for action in self.world_model.action_space} 
                                                   for state in self.world_model.state_space}

    def run_single_episode(self, start_location, goal_location):
        """
        Run a single episode using the Q-learning algorithm.
        """
        state = start_location
        log = []
        while not self.episode_finished:
            action = self.choose_action(state)
            next_state, reward, timestep, self.episode_finished, info = self.world.transition(state, action)
            reward = reward * self.reward_func(timestep) if self.episode_finished else reward
            self.update_policy_values(state, action, reward, next_state)
            log.append((state, action, reward, info))
            state = next_state
        return log

    def update_policy_values(self, state, action, reward, next_state):
        """
        Update the Q-table based on the reward received and the maximum Q-value of the next state.
        """
        q_value = self.q_table[state][action]
        next_q_values = [self.q_table[next_state][a] for a in self.world_model.action_space]
        self.q_table[state][action] = q_value + self.learning_rate * (
            reward + self.discount_factor * max(next_q_values) - q_value)

    def choose_action(self, state, target_policy=False):
        """
        Select an action using an epsilon-greedy policy based on the current Q-values.
        """
        if bernoulli(self.epsilon) and not target_policy:
            return random.choice(self.world_model.action_space)
        else:
            state_q_values = self.q_table[state]
            max_value = max(state_q_values.values())
            best_actions = [action for action, value in state_q_values.items() if value == max_value]
            return random.choice(best_actions)

@forkable
class NStepQPlanner(Planner):
    """
    A planner implementing n-step Q learning.
    """
    def __init__(self, world_model, learning_rate=0.1, discount_factor=0.99, n=5,
                 epsilon=0.1, epsilon_decay=0.99, min_epsilon=0.01, init_value=0, num_episodes=100, reward_func=lambda t: 1):
        super().__init__(world_model, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon, num_episodes, reward_func)
        self.n = n
        self.q_table = {state: {action: init_value for action in world_model.action_space}
                        for state in world_model.state_space}
        self.steps_buffer = deque([], maxlen=n+1)

    def run_single_episode(self, start_location, goal_location):
        """
        Runs a single episode using the n-step Q learning algorithm, managing a buffer for n-step lookahead.
        """
        state = start_location
        action = self.choose_action(state)
        self.steps_buffer.append((state, action, 0))  # Append initial state and action with dummy reward
        log = []
        
        while not self.episode_finished:
            next_state, reward, timestep, self.episode_finished, info = self.world.transition(state, action)
            reward = reward * self.reward_func(timestep) if self.episode_finished else reward
            next_action = self.choose_action(next_state)
            self.steps_buffer.append((next_state, next_action, reward))

            if len(self.steps_buffer) >= self.n:
                self.update_policy_values()
            log.append((state, action, reward, info))
            state = next_state
            action = next_action
        # Flush the buffer at the end of the episode
        while self.steps_buffer:
            self.update_policy_values()

        return log
    
    def update_policy_values(self):
        """
        Updates Q-values using the n-step Q method from the buffer.
        """
        if not self.steps_buffer:
            return  # Safeguard against calling on empty buffer
        if len(self.steps_buffer) == 1:
            state, action, G = self.steps_buffer.popleft()  # Get the earliest state, action, and dummy reward
        else:       
            state, action, _ = self.steps_buffer.popleft()  # Get the earliest state, action, and dummy reward
            states, actions, rewards = zip(*list(self.steps_buffer))  # Unzip to separate components

            # Calculate the return G up to the lookahead steps or until the buffer is empty
            G = self.q_table[states[-1]][actions[-1]]  # Start from the estimated value of the last state-action pair
            for reward in reversed(rewards):
                G = reward + self.discount_factor * G  # Discounted sum of rewards

        # Update Q value for the starting state-action pair in the buffer
        old_value = self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * (G - old_value)

    def choose_action(self, state, target_policy=False):
        """
        Choose an action based on the current state using an epsilon-greedy policy with a n-step lookahead.
        """
        if bernoulli(self.epsilon) and not target_policy:
            return random.choice(self.world_model.action_space)
        else:
            # Choose the action with the highest Q-value for the current state
            state_q_values = self.q_table[state]
            max_value = max(state_q_values.values())
            best_actions = [action for action, value in state_q_values.items() if value == max_value]
            return random.choice(best_actions)

@forkable
class NStepQUCBPlanner(NStepQPlanner):
    """
    A planner implementing n-step Q learning with Upper Confidence Bound (UCB) action selection.
    """
    def __init__(self, world_model, learning_rate=0.1, discount_factor=0.99, n=5, num_episodes=100, reward_func=lambda t: 1, c=2):
        super().__init__(world_model, learning_rate=learning_rate, discount_factor=discount_factor, n=n,
                         num_episodes=num_episodes, reward_func=reward_func)
        self.c = c
        self.action_count = {state: {action: 1 for action in self.world_model.action_space} 
                                                   for state in self.world_model.state_space}
    
    def choose_action(self, state, prior_experience=None, target_policy=False):
        """
        Choose an action based on the current state using UCB if not using the target policy.
        """
        state_q_values = self.q_table[state]
        if target_policy:
            max_value = max(state_q_values.values())
            best_actions = [action for action, value in state_q_values.items() if value == max_value]
        else:
            # UCB
            state_action_count = self.action_count[state]
            log_total_count = math.log(sum(state_action_count.values()))
            best_ucb_value = float('-inf')
            best_actions = []
            
            # Calculate UCB value for each action
            for action, q_value in state_q_values.items():
                action_count = state_action_count[action]
                ucb_value = q_value + self.c * math.sqrt(log_total_count / action_count)
                if ucb_value > best_ucb_value:
                    best_ucb_value = ucb_value
                    best_actions = [action]
                elif ucb_value == best_ucb_value:
                    best_actions.append(action)

        chosen_action = random.choice(best_actions)
        self.action_count[state][chosen_action] += 1
        return chosen_action

@forkable
class QuicksandValueIterationPlanner():
    def __init__(self, world_model, discount_factor=0.99, theta=0.1, epsilon=0.1):
        self.world_model = world_model
        self.discount_factor = discount_factor
        self.theta = theta
        self.epsilon = epsilon
        self.value_function = defaultdict(float)
        self.policy_iterations = 0
        
    def learn_policy(self, start_location, goal_location, wall_locations=None):
        self.world = self.world_model.generate_world(start_location, goal_location, wall_locations)
        for state in self.world_model.state_space:
            if state == goal_location:
                self.value_function[state] = 0
            else:
                self.value_function[state] = 0

        # Value Iteration Loop
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for state in self.world_model.state_space:
                # TODO: justin think about this change with wall locations more
                if (state == goal_location) or (state in wall_locations):
                    continue
                v = self.value_function[state]
                self.value_function[state] = max(self.compute_state_value(state, wall_locations))
                delta = max(delta, abs(v - self.value_function[state]))
            self.policy_iterations += 1
        # TODO: justin you were tired when you wrote this; think again if it's okay
        # TODO: to just surgeon in negative value for walls post-convergence for action selection
        # TODO: justin you removed this because it made the plots go crazy
        # for state in wall_locations:
        #     self.value_function[state] = -1

    # TODO: justin review wall locations changes after you get some sleep
    def compute_state_value(self, state, wall_locations=None):
        # Compute the value for all actions from a given state
        action_values = []
        for action in self.world_model.action_space:
            next_state, reward = self.model_transition(state, action)
            # TODO: justin the change is that we just don't include walls in the state value calculation (you can't get to it)
            # if next_state in wall_locations:
            if next_state == state: # note: this does both walls and out of bounds, which might be good? 
                continue
            action_value = reward + self.discount_factor * self.value_function[next_state]
            action_values.append(action_value)
        if len(action_values) == 0:
            return [self.value_function[state]]
        return action_values

    def model_transition(self, state, action):
        # Compute the next state and reward given an action
        next_state, *_ = self.world.transition(state, action)
        reward = self.reward_func(state, next_state)
        return next_state, reward
    
    def reward_func(self, state, next_state):
        # Reward function for the Quicksand environment
        if next_state == self.world.goal_location:
            return 10
        elif state == self.world.start_location:
            return -4 # artificial penalty for starting state to avoid infinite loops
        elif not self.world.inbounds(next_state) or self.world.get_grid_element(next_state).is_wall:
            # TODO: justin think about the valid:#ity of penalizing moving into walls when walls are stochastic
            return -4
        else:
            return - self.expected_time(state)
    
    def expected_time(self, state):
        if type(self.world_model.environment_spec['states'][state]) != tuple:
            return 1 + self.world_model.environment_spec['states'][state]
        else: 
            alpha, beta = self.world_model.environment_spec['states'][state]
            return 1 + alpha / (alpha + beta)

    def choose_action(self, state, target_policy=True):
        if not target_policy and random.random() < self.epsilon:
            return random.choice(self.world_model.action_space)
        else:
            # Select the action which has the maximum value from the current state
            if state == self.world.goal_location:
                return None  # No action needed at the goal
            best_actions = []
            max_value = float('-inf')
            for action in self.world_model.action_space:
                next_state, reward = self.model_transition(state, action)
                if next_state == state: # we mask noop actions
                    continue
                value = reward + self.discount_factor * self.value_function[next_state]
                if value > max_value:
                    max_value = value
                    best_actions = [action]
                elif value == max_value:
                    best_actions.append(action)
            return random.choice(best_actions)
        
    def choose_action_eval(self, state):
        # Select the action which has the maximum value from the current state
        if state == self.world.goal_location:
            return None  # No action needed at the goal
        best_actions = []
        max_value = float('-inf')
        for action in self.world_model.action_space:
            next_state, reward = self.model_transition(state, action)
            if next_state == state: # we mask noop actions
                continue
            value = reward + self.discount_factor * self.value_function[next_state]
            if value > max_value:
                max_value = value
                best_actions = [action]
            elif value == max_value:
                best_actions.append(action)
        return random.choice(best_actions)

@forkable
class QuicksandValueIterationPlannerExplore(QuicksandValueIterationPlanner):
    def __init__(self, world_model, discount_factor=0.99, theta=0.1, epsilon=0.1, explore_bias=0.1):
        super().__init__(world_model, discount_factor, theta, epsilon)
        self.explore_bias = explore_bias

    def choose_action(self, state, target_policy=True):
        if not target_policy and random.random() < self.epsilon:
            return random.choice(self.world_model.action_space)
        else:
            # Select the action which has the maximum value from the current state
            if state == self.world.goal_location:
                return None  # No action needed at the goal
            best_actions = []
            max_value = float('-inf')
            for action in self.world_model.action_space:
                next_state, reward = self.model_transition(state, action)
                value = reward + self.discount_factor * self.value_function[next_state]
                if value > max_value:
                    max_value = value
                    best_actions = [action]
                elif value >= max_value * (1 - self.explore_bias):
                    best_actions.append(action)
            return random.choice(best_actions)

@forkable
class QuicksandValueIterationPlannerUCB(QuicksandValueIterationPlanner):
    def __init__(self, world_model, discount_factor=0.99, theta=0.1, epsilon=0.1, c=2):
        super().__init__(world_model, discount_factor, theta, epsilon)
        self.c = c

    def choose_action(self, state, prior_experience=None, target_policy=True):
        # Select the action which has the maximum value from the current state with UCB
        if not target_policy and random.random() < self.epsilon:
            return random.choice(self.world_model.action_space)
        else:
            if state == self.world.goal_location:
                return None  # No action needed at the goal
            best_actions = []
            max_value = float('-inf')
            total_experience = 0
            max_experience = 0
            for action in self.world_model.action_space:
                next_state, reward = self.model_transition(state, action)
                if prior_experience is None:
                    if type(self.world_model.environment_spec['states'][next_state]) == tuple:
                        alpha, beta = self.world_model.environment_spec['states'][next_state]
                        max_experience = max(max_experience, alpha + beta)
                        total_experience += alpha + beta
                else:
                    total_experience += prior_experience[next_state]
                    max_experience = max(max_experience, prior_experience[next_state])

            for action in self.world_model.action_space:
                next_state, reward = self.model_transition(state, action)
                value = reward + self.discount_factor * self.value_function[next_state]
                if type(self.world_model.environment_spec['states'][next_state]) == tuple:
                    current_experience = sum(self.world_model.environment_spec['states'][next_state])
                elif type(self.world_model.environment_spec['states'][next_state]) in (float, int):
                    current_experience = prior_experience[next_state]
                else:
                    current_experience = max_experience
                value += self.c * math.sqrt(math.log(total_experience) / current_experience)

                if value > max_value:
                    max_value = value
                    best_actions = [action]
                elif value == max_value:
                    best_actions.append(action)
            return random.choice(best_actions)
        
@forkable
class SimpleQuicksandValueIterationPlanner(QuicksandValueIterationPlanner):
    def __init__(self, world_model, discount_factor=0.99, theta=0.1, epsilon=0.1):
        super().__init__(world_model, discount_factor, theta, epsilon)
    
    def expected_time(self, state):
        if type(self.world_model.environment_spec['states'][state]) == tuple:
            raise(ValueError("We don't use a beta distribution with the simple setup."))
        else: 
            # note: 0.8 is the probability of quicksand and 
            # self.world_model.environment_spec['states'][state] is probability of a unsafe (0.8) tile
            return 1 + self.world_model.environment_spec['states'][state] * 0.8

@forkable
class NegLogLikelihoodHardmaxPlanner(): # note: hardmaxer
    def __init__(self, world_model, epsilon=0.0):
        self.world_model = world_model
        self.epsilon = epsilon
        self.value_function = defaultdict(float)
        self.optimal_path = None


    def learn_policy(self, start_location, goal_location, wall_locations=None):
        spec = self.world_model.environment_spec
        grid = np.zeros((spec['metadata']['width'], spec['metadata']['height']))
        for loc, prior in spec['states'].items():
            grid[loc] = prior * 0.8
        self.grid = -np.log(1 - grid)
        self.start_location = start_location
        self.goal_location = goal_location
        self.wall_locations = wall_locations

    def choose_action(self, state, target_policy=True):
        if not target_policy and random.random() < self.epsilon:
            action = random.choice(self.world_model.action_space)
            self.optimal_path = None
            return action
        else:
            if state == self.goal_location:
                self.optimal_path = None
                return None  # No action needed at the goal

            if self.optimal_path is None:
                self.optimal_path = best_path(self.grid, state, self.goal_location, walls=self.wall_locations)[1:]
            next_state = self.optimal_path.pop(0)
            action = (next_state[0] - state[0], next_state[1] - state[1])
            return action
        
    def choose_action_eval(self, state):
        if self.optimal_path is None:
            self.optimal_path = best_path(self.grid, state, self.goal_location, walls=self.wall_locations)[1:]
        next_state = self.optimal_path.pop(0)
        if next_state == self.goal_location:
            self.optimal_path = None
        action = (next_state[0] - state[0], next_state[1] - state[1])
        return action
    
@forkable
class NegLogLikelihoodSoftmaxPlanner():
    def __init__(self, world_model, softmax_temp=0.5):
        self.world_model = world_model
        self.softmax_temp = softmax_temp
        self.value_function = defaultdict(float)
        self.optimal_path = None

    def sample_path(self, start_location, goal_location, wall_locations=None):
        spec = self.world_model.environment_spec
        tree = QuicksandTree(
            rows=spec['metadata']['height'],
            cols=spec['metadata']['width'],
            start_location=start_location,
            goal_location=goal_location,
            wall_locations=wall_locations,
            quicksand_probabilities=self.grid
        )
        
        decision_tree = tree.initialize_decision_tree()
        decision_tree = tree.populate_decision_tree(decision_tree)
        unique_terminal_nodes = tree.get_unique_paths(decision_tree.get_terminal_nodes())
        return tree.sample_path(unique_terminal_nodes, self.softmax_temp)[-1].path_trajectory


    def learn_policy(self, start_location, goal_location, wall_locations=None):
        spec = self.world_model.environment_spec
        grid = make_grid_from_spec(spec) * 0.8
        self.grid = -np.log(1 - grid)
        self.start_location = start_location
        self.goal_location = goal_location
        self.wall_locations = wall_locations

    def choose_action(self, state, target_policy=True):
        if state == self.goal_location:
            self.optimal_path = None
            return None  # No action needed at the goal

        if self.optimal_path is None:
            self.optimal_path = self.sample_path(state, self.goal_location, wall_locations=self.wall_locations)[1:]
        next_state = self.optimal_path.pop(0)
        action = (next_state[0] - state[0], next_state[1] - state[1])
        return action
        
    def choose_action_eval(self, state):
        if self.optimal_path is None:
            self.optimal_path = self.sample_path(state, self.goal_location, wall_locations=self.wall_locations)[1:]
        next_state = self.optimal_path.pop(0)
        if next_state == self.goal_location:
            self.optimal_path = None
        action = (next_state[0] - state[0], next_state[1] - state[1])
        return action