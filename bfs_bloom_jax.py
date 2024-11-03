import jax
from jax import jit, vmap, lax, config, profiler, debug
import jax.numpy as jnp
from acmdp_jax import ACMDP
from functools import partial
import time

# ------------------------------
# Bloom Filter
# ------------------------------
num_hashes = 32 # probability of false positives = 2^-32
bits_per_word = 32 #using dtype=jnp.uint32
#bloom_filter_size = 2**32 bits ==> ~2^27 nodes max in bloom filter ==> depth 7 BFS tree
num_words = 2**27 # = bloom_filter_size // bits_per_word
# 4294967291 = (2**32 - 5) is the largest prime number less than 2^32

@jit
def polynomial_hash(arr, b=7, m=4294967291):
    # arr: jax.numpy array of non-negative integers
    arr = arr.astype(jnp.uint32)
    b = jnp.uint32(b)
    m = jnp.uint32(m)
    def hash_fold(carry, x):
        return (carry * b + x) % m, None
    hash_value, _ = jax.lax.scan(hash_fold, jnp.uint32(0), arr)
    return hash_value

@jit
def compute_hashes(state, b1=7, b2=31, m=4294967291):
    # Compute two base hash values using different bases
    arr = (state + 2).astype(jnp.uint32)
    b1 = jnp.uint32(b1)
    b2 = jnp.uint32(b2)
    m = jnp.uint32(m)
    h1 = polynomial_hash(arr, b=b1, m=m)
    h2 = polynomial_hash(arr, b=b2, m=m)
    # Generate multiple hashes
    hashes = (h1 + jnp.arange(num_hashes, dtype=jnp.uint32) * h2) % m
    return hashes

@jit
def insert(states, bloom_filter):
    indices = vmap(compute_hashes)(states) #(batch_size, num_hashes)
    flat_indices = indices.flatten()
    word_indices = flat_indices // bits_per_word #(batch_size * num_hashes,)
    bit_positions = flat_indices % bits_per_word #(batch_size * num_hashes,)
    masks = jnp.left_shift(jnp.uint32(1), bit_positions) #(batch_size * num_hashes,)
    bloom_filter = bloom_filter.at[word_indices].set(bloom_filter[word_indices] | masks)
    return bloom_filter

@jit
def query(states, bloom_filter):
    indices = vmap(compute_hashes)(states) #(batch_size, num_hashes)
    word_indices = indices // bits_per_word #(batch_size, num_hashes)
    bit_positions = indices % bits_per_word #(batch_size, num_hashes)
    masks = jnp.left_shift(jnp.uint32(1), bit_positions) #(batch_size, num_hashes)

    # Retrieve the bits from the bloom_filter
    bits = bloom_filter[word_indices] & masks #(batch_size, num_hashes)

    # Check if all bits are set
    is_present = jnp.all(bits != 0, axis=1) & jnp.all(states != -3, axis=1)
    return is_present

# ------------------------------
# BFS
# ------------------------------
@jit
def path_transition(path_state, action, visited, inverse):
    visit_value = path_state[0] 

    def true_fn(args):
        path_state, action = args

        path = path_state[1:max_depth+1]
        path = jnp.roll(path, 1)
        path = path.at[0].set(action)

        state = path_state[max_depth+1:]
        state = mdp.transition(state, action, inverse)

        query_result = query(state.reshape(1, -1), visited)
        new_path_state = jnp.concatenate([query_result, path, state], axis=0)
        return new_path_state
    
    def false_fn(args):
        path_state, _ = args
        result = jnp.full(path_state.shape, -3, dtype=jnp.int32)
        result = result.at[0].set(1)
        return result
    
    return lax.cond(visit_value == 0, true_fn, false_fn, (path_state, action))

vmap_path_transition = jit(vmap(
    vmap(path_transition, in_axes=(None, 0, None, None)), 
    in_axes=(0, None, None, None)
))

def get_paths(state, path_frontier, max_depth=6):
    frontier_states = path_frontier[:, 1 + max_depth:]
    comparisons = frontier_states == state
    row_matches = jnp.all(comparisons, axis=1)
    matching_indices = jnp.where(row_matches)[0]
    paths = path_frontier[matching_indices][:, 1:max_depth+1]
    path_list = [paths[i] for i in range(paths.shape[0])]
    optimal_paths = []
    for path in path_list:
        optimal_path = path[path != -1]
        optimal_path = optimal_path[::-1].tolist()
        optimal_paths.append(optimal_path)
    return optimal_paths

@partial(jit, static_argnums=(2, 3))
def bfs_jit(start, goal, mdp, max_depth=6):  
    start = start.reshape(1, -1)
    goal = goal.reshape(1, -1)  
    actions = jnp.arange(mdp.n_actions)

    def process_frontier(path_frontier, goal, visited):
        path_frontier = vmap_path_transition(path_frontier, actions, visited, False)
        path_frontier = path_frontier.reshape(-1, path_frontier.shape[2])
        frontier = path_frontier[:, max_depth+1:]
        visited = insert(frontier, visited)
        return path_frontier, query(goal, visited), visited

    # -------- Depth 0 --------
    visited = jnp.zeros(num_words, dtype=jnp.uint32)
    path = jnp.full(max_depth, -1, jnp.int32)   
    visit_value = jnp.array([0], dtype=jnp.int32) 
    path_frontier0 = jnp.concatenate([visit_value, path, start.reshape(-1)]).reshape(1,-1)
    visited = insert(start, visited)
    goal_found0 = query(goal, visited)
    
    # -------- Depth 1 to max_depth=6 -------- 
    path_frontier1, goal_found1, visited = process_frontier(path_frontier0, goal, visited)
    path_frontier2, goal_found2, visited = process_frontier(path_frontier1, goal, visited)
    path_frontier3, goal_found3, visited = process_frontier(path_frontier2, goal, visited)
    path_frontier4, goal_found4, visited = process_frontier(path_frontier3, goal, visited)
    path_frontier5, goal_found5, visited = process_frontier(path_frontier4, goal, visited)
    # path_frontier6, goal_found6, visited = process_frontier(path_frontier5, goal, visited)

    goal_found = jnp.array(
        [goal_found0, 
         goal_found1, 
         goal_found2, 
         goal_found3, 
         goal_found4, 
         goal_found5, 
        #  goal_found6,
        ], 
        dtype=jnp.bool_
    )
    
    return (
        goal_found, 
        path_frontier0, 
        path_frontier1, 
        path_frontier2, 
        path_frontier3, 
        path_frontier4, 
        path_frontier5, 
        # path_frontier6,
    )

def bfs(start, goal, mdp, max_depth=6):
    result = bfs_jit(start, goal, mdp, max_depth)
    goal_found = result[0]
    path_frontiers = result[1:]
    depth = jnp.argmax(goal_found)
    return get_paths(goal, path_frontiers[depth], max_depth)

@partial(jit, static_argnums=(2, 3))
def bfs_bidirectional_jit(start, goal, mdp, max_depth=6):
    start = start.reshape(1, -1)
    goal = goal.reshape(1, -1)
    actions = jnp.arange(mdp.n_actions)

    def process_frontier(path_frontier, visited, inverse):
        path_frontier = vmap_path_transition(path_frontier, actions, visited, inverse)
        path_frontier = path_frontier.reshape(-1, path_frontier.shape[2])
        frontier = path_frontier[:, max_depth+1:]
        visited = insert(frontier, visited)
        return path_frontier, visited

    # -------- Depth 0 --------
    visited_x = jnp.zeros(num_words, dtype=jnp.uint32)
    path_x = jnp.full(max_depth, -1, jnp.int32)   
    visit_value_x = jnp.array([0], dtype=jnp.int32) 
    path_frontier_x0 = jnp.concatenate([visit_value_x, path_x, start.squeeze()]).reshape(1,-1)
    visited_x = insert(start, visited_x)

    visited_y = jnp.zeros(num_words, dtype=jnp.uint32)
    path_y = jnp.full(max_depth, -1, jnp.int32)   
    visit_value_y = jnp.array([0], dtype=jnp.int32) 
    path_frontier_y0 = jnp.concatenate([visit_value_y, path_y, goal.squeeze()]).reshape(1,-1)
    visited_y = insert(goal, visited_y)

    x0_in_visited_y0 = query(path_frontier_x0[:, max_depth+1:], visited_y)
    y0_in_visited_x0 = query(path_frontier_y0[:, max_depth+1:], visited_x)

    # -------- Depth 1 to max_depth --------
    path_frontier_x1, visited_x = process_frontier(path_frontier_x0, visited_x, inverse=False)
    x1_in_visited_y0 = query(path_frontier_x1[:, max_depth+1:], visited_y)
    y0_in_visited_x1 = query(path_frontier_y0[:, max_depth+1:], visited_x)
    
    path_frontier_y1, visited_y = process_frontier(path_frontier_y0, visited_y, inverse=True)
    x1_in_visited_y1 = query(path_frontier_x1[:, max_depth+1:], visited_y)
    y1_in_visited_x1 = query(path_frontier_y1[:, max_depth+1:], visited_x)

    path_frontier_x2, visited_x = process_frontier(path_frontier_x1, visited_x, inverse=False)
    x2_in_visited_y1 = query(path_frontier_x2[:, max_depth+1:], visited_y)
    y1_in_visited_x2 = query(path_frontier_y1[:, max_depth+1:], visited_x)
    
    path_frontier_y2, visited_y = process_frontier(path_frontier_y1, visited_y, inverse=True)
    x2_in_visited_y2 = query(path_frontier_x2[:, max_depth+1:], visited_y)
    y2_in_visited_x2 = query(path_frontier_y2[:, max_depth+1:], visited_x)

    path_frontier_x3, visited_x = process_frontier(path_frontier_x2, visited_x, inverse=False)
    x3_in_visited_y2 = query(path_frontier_x3[:, max_depth+1:], visited_y)
    y2_in_visited_x3 = query(path_frontier_y2[:, max_depth+1:], visited_x)
    
    path_frontier_y3, visited_y = process_frontier(path_frontier_y2, visited_y, inverse=True)
    x3_in_visited_y3 = query(path_frontier_x3[:, max_depth+1:], visited_y)
    y3_in_visited_x3 = query(path_frontier_y3[:, max_depth+1:], visited_x)

    # path_frontier_x4, visited_x = process_frontier(path_frontier_x3, visited_x, inverse=False)
    # x4_in_visited_y3 = query(path_frontier_x4[:, max_depth+1:], visited_y)
    # y3_in_visited_x4 = query(path_frontier_y3[:, max_depth+1:], visited_x)

    # path_frontier_y4, visited_y = process_frontier(path_frontier_y3, visited_y, inverse=True)
    # x4_in_visited_y4 = query(path_frontier_x4[:, max_depth+1:], visited_y)
    # y4_in_visited_x4 = query(path_frontier_y4[:, max_depth+1:], visited_x)

    # path_frontier_x5, visited_x = process_frontier(path_frontier_x4, visited_x, inverse=False)
    # x5_in_visited_y4 = query(path_frontier_x5[:, max_depth+1:], visited_y)
    # y4_in_visited_x5 = query(path_frontier_y4[:, max_depth+1:], visited_x)

    # path_frontier_y5, visited_y = process_frontier(path_frontier_y4, visited_y, inverse=True)
    # x5_in_visited_y5 = query(path_frontier_x5[:, max_depth+1:], visited_y)
    # y5_in_visited_x5 = query(path_frontier_y5[:, max_depth+1:], visited_x)


    path_frontiers = (
        (path_frontier_x0, path_frontier_y0),
        (path_frontier_x1, path_frontier_y1),
        (path_frontier_x2, path_frontier_y2),
        (path_frontier_x3, path_frontier_y3),
        # (path_frontier_x4, path_frontier_y4),
        # (path_frontier_x5, path_frontier_y5),
    )

    in_visited = (
        (x0_in_visited_y0, y0_in_visited_x0),     
        (x1_in_visited_y0, y0_in_visited_x1), 
        (x1_in_visited_y1, y1_in_visited_x1),
        (x2_in_visited_y1, y1_in_visited_x2), 
        (x2_in_visited_y2, y2_in_visited_x2),
        (x3_in_visited_y2, y2_in_visited_x3),
        (x3_in_visited_y3, y3_in_visited_x3),
        # (x4_in_visited_y3, y3_in_visited_x4),
        # (x4_in_visited_y4, y4_in_visited_x4),
        # (x5_in_visited_y4, y4_in_visited_x5),
        # (x5_in_visited_y5, y5_in_visited_x5),
    )

    return path_frontiers, in_visited

def bfs_bidirectional(start, goal, mdp, max_depth=6):
    path_frontiers, in_visited = bfs_bidirectional_jit(start, goal, mdp, max_depth)
    x_in_visited_y = [jnp.any(z[0]) for z in in_visited]
    y_in_visited_x = [jnp.any(z[1]) for z in in_visited]
    contact_ind_x = np.argmax(x_in_visited_y)
    contact_ind_y = np.argmax(y_in_visited_x)
    contact_ind = jnp.minimum(contact_ind_x, contact_ind_y)
    mask_x, mask_y = in_visited[contact_ind]

    ind_x = (contact_ind + 1) // 2
    ind_y = contact_ind // 2

    path_frontier_x = path_frontiers[ind_x][0]
    path_frontier_y = path_frontiers[ind_y][1]

    contact_x = path_frontier_x[mask_x]
    contact_y = path_frontier_y[mask_y]
    optimal_paths = []
    for path_state_x in contact_x:
        state_x = path_state_x[max_depth+1:]
        for path_state_y in contact_y:
            state_y = path_state_y[max_depth+1:]
            if jnp.all(state_x == state_y) and jnp.all(state_x != -3):
                path_x = path_state_x[1:max_depth+1]
                path_x = path_x[path_x != -1][::-1].tolist()
                path_y = path_state_y[1:max_depth+1]
                path_y = path_y[path_y != -1].tolist()
                optimal_paths.append(path_x + path_y)
    return optimal_paths

# ------------------------------
# Misc
# ------------------------------

import numpy as np
np.random.seed(1334)

def random_walk(start, num_steps):
    state = start
    for _ in range(num_steps):
        action = np.random.randint(0, mdp.n_actions)
        state = mdp.transition(state, action)
    return state

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_python(x) for x in obj)
    else:
        return obj

# ------------------------------
# Generate data for pairs connected to terminal state
# ------------------------------
import os
import pickle
from tqdm import tqdm

max_start_radius = 32
max_depth = 3
max_path_length = 2*max_depth
max_relator_length = 256
mdp = ACMDP(max_relator_length)
num_searches = 10

# data will be 2d array (dtype = np.int8) with rows [encoder input | decoder input]

# encoder input will be an array [r_x1, 0, r_x2, 0, r_y1, 0, r_y2, 0] + 2
# of length 4*max_relator_length where:
#   (*) 0 + 2 is the padding token  
#   (*) r_x1, r_x2 are the relators for the x path
#   (*) r_y1, r_y2 are the relators for the y path

# Given an optimal path (a_1, ..., a_n) corresponding to a given encoder input [start] + [goal],
# decoder input will be an array [x_0, x_1, ..., x_k, 0,...,0, x_{k+1}]
# of length (max_path_length + 2) where:
#   (*) 0 is the padding token
#   (*) x_0 = 13 is the start of sequence token
#   (*) x_i = a_i + 1 for 1 <= i <= n
#   (*) 0 <= a_i <= 11 are the actions in the optimal path
#   (*) x_{n+1} = 14 is the end of sequence token

data_train = []
data_val = []
for search_ind in tqdm(range(num_searches)):
    terminal_state = mdp.tuple_to_array(((1),(2)))
    start_radius = np.random.randint(0, max_start_radius)
    seen_pairs = set()
    while True:
        start = random_walk(terminal_state, start_radius)
        distance_to_start = np.random.randint(0, 2*max_depth)
        goal = random_walk(start, distance_to_start)
        start_tuple = mdp.array_to_tuple(start)
        goal_tuple = mdp.array_to_tuple(goal)
        if (start_tuple, goal_tuple) not in seen_pairs: break

    tqdm.write(f"start = {numpy_to_python(mdp.array_to_tuple(start))}")
    tqdm.write(f"goal = {numpy_to_python(mdp.array_to_tuple(goal))}")


    t0 = time.time()
    optimal_paths = bfs_bidirectional(start, goal, mdp, max_depth)
    t1 = time.time()
    tqdm.write(f"bfs time = {t1 - t0} seconds")
    tqdm.write(f"optimal_paths = {optimal_paths}")
    tqdm.write("")

    # find all optimal subpaths in each optimal path

    seen_data = set()
    for path in optimal_paths:
        states = [start]
        for action in path:
            state = mdp.transition(states[-1], action)
            states.append(state)
        if not np.all(states[-1] == goal): 
            tqdm.write("goal not reached")
            break
        for i in range(len(path)):
            for j in range(i+1, len(path)+1):
                subpath = np.array(path[i:j])
                substart = states[i]
                subgoal = states[j]
                substart_tuple = mdp.array_to_tuple(substart)
                subgoal_tuple = mdp.array_to_tuple(subgoal)
                subpath_tuple = tuple(path[i:j])
                if (substart_tuple, subgoal_tuple, subpath_tuple) not in seen_data:
                    seen_data.add((substart_tuple, subgoal_tuple, subpath_tuple))
                    encoder_input = np.concatenate([substart, subgoal], axis=0, dtype=np.int8) + 2
                    decoder_array = np.concatenate(([13], subpath+1, [14]), axis=0, dtype=np.int8)
                    decoder_array = np.pad(decoder_array, (0, max_path_length + 2 - len(decoder_array)), mode='constant', constant_values=0)
                    data_array = np.concatenate([encoder_input, decoder_array], axis=0)
                    if search_ind < 0.9 * num_searches:
                        data_train.append(data_array)
                    else:
                        data_val.append(data_array)

data_train = np.stack(data_train)
data_val = np.stack(data_val)

metadata_train = {
    'dtype': str(data_train.dtype),
    'shape': data_train.shape
}

metadata_val = {
    'dtype': str(data_val.dtype),
    'shape': data_val.shape
}

os.makedirs('data', exist_ok=True)

with open(os.path.join('data', 'metadata_train.pkl'), 'wb') as f:
    pickle.dump(metadata_train, f)
with open(os.path.join('data', 'metadata_val.pkl'), 'wb') as f:
    pickle.dump(metadata_val, f)

data_train.tofile(os.path.join('data', 'data_train.bin'))
data_val.tofile(os.path.join('data', 'data_val.bin'))