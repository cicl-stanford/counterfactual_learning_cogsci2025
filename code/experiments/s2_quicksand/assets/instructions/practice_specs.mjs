export const practice_world = {
  'metadata': {'width': 8, 'height': 3, 'update_per_timestep': false, 'geometric': false, 'world_id': 'practice'},
  'states': {
    '(0,0)': 0.0, '(0,1)': 0.0, '(0,2)': 0.0, 
    '(1,0)': 0.0, '(1,1)': 0.0, '(1,2)': 0.0, 
    '(2,0)': 0.0, '(2,1)': 0.0, '(2,2)': 0.0, 
    '(3,0)': 0.8, '(3,1)': 0.8, '(3,2)': 0.8, 
    '(4,0)': 0.8, '(4,1)': 0.8, '(4,2)': 0.8, 
    '(5,0)': 0.0, '(5,1)': 0.8, '(5,2)': 0.0, 
    '(6,0)': 0.8, '(6,1)': 0.0, '(6,2)': 0.8, 
    '(7,0)': 0.0, '(7,1)': 0.0, '(7,2)': 0.0
  }
}

export const planner_practice = {
    start_position: { x: 0, y: 0 },
    goal_position: { x: 7, y: 1 },
    wall_positions: [ { x: 3, y: 0 } ],
};
export const hypothetical_practice = {
  start_position: { x: 0, y: 2 },
  goal_position: { x: 7, y: 0 },
  wall_positions: [ { x: 5, y: 1 } ],
  dist: null
};
export const counterfactual_practice = {
  start_position: { x: null, y: null },
  goal_position: { x: null, y: null },
  wall_positions: null,
  dist: null
};
