from quicksand_module import *
import numpy as np
import scipy.stats as stats

class QuicksandActions:
    # note: (x, y) = (0, 0) is the top-left corner
    UP = (0, -1)  # No change in x, decrease in y
    DOWN = (0, 1)  # No change in x, increase in y
    LEFT = (-1, 0)  # Decrease in x, no change in y
    RIGHT = (1, 0)  # Increase in x, no change in y
    action_space = [UP, DOWN, LEFT, RIGHT]
    
    @staticmethod
    def describe_action(action):
        if action == QuicksandActions.UP:
            return "Move up (dx=0, dy=-1)"
        elif action == QuicksandActions.DOWN:
            return "Move down (dx=0, dy=1)"
        elif action == QuicksandActions.LEFT:
            return "Move left (dx=-1, dy=0)"
        elif action == QuicksandActions.RIGHT:
            return "Move right (dx=1, dy=0)"
        else:
            return "Unknown action"


class QuicksandTree:
    def __init__(self, 
                 rows,
                 cols,
                 start_location,
                 goal_location,
                 wall_locations,
                 quicksand_probabilities # input should be negative log likelihoods
                 ):
        self.rows = rows
        self.cols = cols
        self.start_location = start_location
        self.goal_location = goal_location
        self.wall_locations = wall_locations
        self.quicksand_probabilities = quicksand_probabilities
        
    def initialize_decision_tree(self):
        state = QuicksandTreeState(
            rows=self.rows, 
            cols=self.cols,
            start_location=self.start_location,
            goal_location=self.goal_location,
            wall_locations=self.wall_locations,
            quicksand_probabilities=self.quicksand_probabilities,
            path_trajectory=[self.start_location],
            agent_position=self.start_location
        )
        return QuicksandTreeNode(state, None, [])

    def populate_decision_tree(self, root_node):
        root_node.visit_node()
        return root_node
    
    def get_unique_paths(self, terminal_nodes):
        unique_nodes = {}
        for node in terminal_nodes:
            state_id = node.quicksand_tree_state.id()
            if state_id not in unique_nodes:
                unique_nodes[state_id] = node
        return unique_nodes.values()
    
    def get_best_path(self, terminal_nodes):
        return max(terminal_nodes, key=lambda node: node.quicksand_tree_state.get_expected_reward()).get_path()
    
    def get_path_expected_reward(self, path):
        if len(path) == 0: return 0
        else: return path[-1].get_expected_reward()

    def get_path_likelihood(self, path_end_state, all_end_states, softmax_temp):
        expected_rewards = np.array([state.get_expected_reward() for state in all_end_states])
        max_reward = np.max(expected_rewards)
        exp_values = np.exp((expected_rewards - max_reward) / softmax_temp)
        path_reward = path_end_state.get_expected_reward()
        exp_path_value = np.exp((path_reward - max_reward) / softmax_temp)

        return exp_path_value / np.sum(exp_values)
    
    def get_noisy_path_likelihood(self, path_end_state, all_end_states, log_noise, softmax_temp):
        expected_rewards = np.array([state.get_noisy_expected_reward(log_noise) for state in all_end_states])
        max_reward = np.max(expected_rewards)
        exp_values = np.exp((expected_rewards - max_reward) / softmax_temp)
        path_reward = path_end_state.get_noisy_expected_reward(log_noise)
        exp_path_value = np.exp((path_reward - max_reward) / softmax_temp)

        return exp_path_value / np.sum(exp_values)
    
    def sample_path(self, terminal_nodes, softmax_temp):
        """Sample a path based on the likelihoods of terminal nodes."""
        expected_rewards = np.array([node.quicksand_tree_state.get_expected_reward() for node in terminal_nodes])
        max_reward = np.max(expected_rewards)
        exp_rewards = np.exp((expected_rewards - max_reward) / softmax_temp)
        total_exp_rewards = np.sum(exp_rewards)
        normalized_likelihoods = exp_rewards / total_exp_rewards
        sampled_node = np.random.choice(np.array(list(terminal_nodes)), p=normalized_likelihoods)

        return sampled_node.get_path()

    def sample_noisy_path(self, terminal_nodes, softmax_temp):
        """Sample a path based on the likelihoods of terminal nodes."""
        expected_rewards = np.array([node.quicksand_tree_state.get_noisy_expected_reward() for node in terminal_nodes])
        max_reward = np.max(expected_rewards)
        exp_rewards = np.exp((expected_rewards - max_reward) / softmax_temp)
        total_exp_rewards = np.sum(exp_rewards)
        normalized_likelihoods = exp_rewards / total_exp_rewards
        sampled_node = np.random.choice(np.array(list(terminal_nodes)), p=normalized_likelihoods)

        return sampled_node.get_path()

class QuicksandTreeNode:
    def __init__(self,
                 quicksand_tree_state,
                 parent_node,
                 child_nodes):
        self.quicksand_tree_state = quicksand_tree_state
        self.parent_node = parent_node
        self.child_nodes = child_nodes
        
    def __str__(self):
        return f'QuicksandTreeNode object has quicksand_tree_state: {self.quicksand_tree_state}\
            \nQuickSandTreeNode object has parent_node: {self.parent_node}\
            \nQuickSandTreeNode object has child_nodes: {self.child_nodes}'
            
    def visit_node(self):
        child_states = self.quicksand_tree_state.get_next_states()
        for state in child_states:
            child_node = self.__class__(state, self, [])
            self.child_nodes.append(child_node)
            
        for node in self.child_nodes:
            node.visit_node()

    def count_child_nodes(self):
        n_nodes = 1
        for child_node in self.child_nodes:
            n_nodes += child_node.count_child_nodes()
        return n_nodes

    def count_terminal_nodes(self):
        n_nodes = 0
        for child_node in self.child_nodes:
            n_nodes += child_node.count_terminal_nodes()
        if self.is_terminal_node():
            n_nodes += 1
        return n_nodes
    
    def get_terminal_nodes(self):
        terminal_nodes = []
        for child_node in self.child_nodes:
            terminal_nodes += child_node.get_terminal_nodes()
        if self.is_terminal_node():
            terminal_nodes.append(self)
        return terminal_nodes
    
    def is_terminal_node(self):
        return self.quicksand_tree_state.agent_position == self.quicksand_tree_state.goal_location
    
    def get_path(self):
        path = []
        state = self
        while state.parent_node:
            path.append(state.quicksand_tree_state)
            state = state.parent_node
        path.append(state.quicksand_tree_state)
        path.reverse()
        return path


# TODO: this should eventually be extensible to simulated trials (e.g., add observed path to state for counterfactuals)
class QuicksandTreeState:
    def __init__(self, 
                 # static to a day
                 rows, 
                 cols,
                 start_location,
                 goal_location,
                 wall_locations,
                 quicksand_probabilities,
                 # dynamic
                 path_trajectory,
                 agent_position
                 ):
        self.rows = rows
        self.cols = cols
        self.start_location = start_location
        self.goal_location = goal_location
        self.wall_locations = wall_locations
        self.quicksand_probabilities = quicksand_probabilities
        self.path_trajectory = path_trajectory
        self.agent_position = agent_position
    
    def __str__(self):
        return f'QuicksandTreeState object {[self,]}\
            \n\trows: {self.rows}\
            \n\tcols: {self.cols}\
            \n\tstart_location: {self.start_location}\
            \n\tgoal_location: {self.goal_location}\
            \n\twall_locations: {self.wall_locations}\
            \n\tquicksand_probabilities: {self.quicksand_probabilities}\
            \n\tpath_trajectory: {self.path_trajectory}\
            \n\tagent_position: {self.agent_position}'
    
    def id(self):
        return f'rows: {self.rows}, \
            cols: {self.cols}, \
            start_location: {self.start_location}, \
            goal_location: {self.goal_location}, \
            wall_locations: {self.wall_locations}, \
            quicksand_probabilities: {np.round(self.quicksand_probabilities, 3).tolist()}, \
            path_trajectory: {self.path_trajectory}'
    
    def get_move_options(self):
        if self.agent_position == self.goal_location:
            return []

        potential_moves = [
            QuicksandActions.UP,
            QuicksandActions.DOWN,
            QuicksandActions.LEFT,
            QuicksandActions.RIGHT
        ]

        valid_positions = []

        for move in potential_moves:
            x, y = self.agent_position[0] + move[0], self.agent_position[1] + move[1]
            if (0 <= x < self.cols and 0 <= y < self.rows and 
                (x, y) not in self.wall_locations and 
                (x, y) not in self.path_trajectory):
                valid_positions.append((x, y))

        return valid_positions

    def get_next_states(self):
        child_states = []
        for position in self.get_move_options():
            new_path_trajectory = self.path_trajectory + [position]
            new_agent_position = position
            child_states.append(QuicksandTreeState(
                                rows=self.rows,
                                cols=self.cols,
                                start_location=self.start_location,
                                goal_location=self.goal_location,
                                wall_locations=self.wall_locations,
                                quicksand_probabilities=self.quicksand_probabilities,
                                path_trajectory=new_path_trajectory,
                                agent_position=new_agent_position))
        return child_states

    def get_expected_reward(self):
        return -sum([self.quicksand_probabilities[position[1]][position[0]] for position in self.path_trajectory])
    
    def get_noisy_expected_reward(self, log_noise):
        noise = - np.log(1 - np.random.random() * log_noise)
        reward = self.get_expected_reward()
        return reward + noise
