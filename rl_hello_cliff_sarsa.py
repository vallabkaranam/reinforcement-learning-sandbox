import random

# --- Constants & Hyperparameters ---
ROWS = 4
COLS = 12
START_POS = (3, 0)
GOAL_POS = (3, 11)
# Cliff is row 3, columns 1 through 10
# Note: In grid coordinates (row, col), cliff cells are (3, 1) to (3, 10)
CLIFF_CELLS = set((3, c) for c in range(1, 11))

# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTIONS = [0, 1, 2, 3]
ACTION_DELTAS = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1)   # LEFT
}
ACTION_SYMBOLS = {0: '^', 1: '>', 2: 'v', 3: '<'}

ALPHA = 0.5
GAMMA = 1.0
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
SLIP_PROB = 0.1
TOTAL_EPISODES = 500
LOG_INTERVAL = 50

def get_next_position(state, action):
    """
    Calculates the next grid position given a state and action,
    clamping to the grid boundaries.
    """
    row, col = state
    dr, dc = ACTION_DELTAS[action]
    next_r = max(0, min(ROWS - 1, row + dr))
    next_c = max(0, min(COLS - 1, col + dc))
    return (next_r, next_c)

def step(state, action):
    """
    Execute one step in the environment.
    Returns: (next_state, reward, done)
    """
    if random.random() < SLIP_PROB:
        action = random.choice(ACTIONS)

    # Determine tentative next position based on action
    next_pos = get_next_position(state, action)

    # Check for Cliff
    if next_pos in CLIFF_CELLS:
        # Falling into the cliff gives -100 reward and ends the episode
        return next_pos, -100, True

    # Check for Goal
    if next_pos == GOAL_POS:
        # Reaching the goal gives -1 reward (standard step cost) and ends the episode
        return next_pos, -1, True

    # Standard step
    return next_pos, -1, False

def choose_action(state, q_table, epsilon):
    """
    Select an action using an epsilon-greedy policy derived from Q-table.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    
    # Greedy selection (with tie-breaking)
    # Find max Q-value for current state
    best_val = -float('inf')
    best_actions = []
    
    for a in ACTIONS:
        val = q_table.get((state, a), 0.0)
        if val > best_val:
            best_val = val
            best_actions = [a]
        elif val == best_val:
            best_actions.append(a)
    
    return random.choice(best_actions)

def run_sarsa():
    """
    Main SARSA training loop.
    """
    # Initialize Q-table: key=(state, action), value=float
    # Only need to store visited state-actions; default is 0.0
    q_table = {}
    
    # To track rolling average return
    recent_returns = [] 

    print(f"Starting SARSA Training: {TOTAL_EPISODES} episodes")
    print(f"Params: Alpha={ALPHA}, Gamma={GAMMA}, Epsilon Schedule: Start={EPSILON_START}, Min={EPSILON_MIN}, Decay={EPSILON_DECAY}, Slip={SLIP_PROB}")
    
    headers = ["Episode", "Return", "Steps", "Epsilon", "Avg Return (Last 50)"]
    print("-" * 75)
    print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<7} | {headers[4]:<20}")
    print("-" * 75)

    for episode in range(1, TOTAL_EPISODES + 1):
        # Calculate epsilon for this episode
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** (episode - 1)))
        
        state = START_POS
        # Choose initial action (SARSA needs A for the first step)
        action = choose_action(state, q_table, epsilon)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            next_state, reward, done = step(state, action)
            total_reward += reward
            steps += 1
            
            # SARSA Update
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            
            old_q = q_table.get((state, action), 0.0)
            
            if done:
                # Target is just Reward since there is no next action/state value
                target = reward
                # Determine next_action just for loop continuity variables, though loop ends
                next_action = None 
            else:
                next_action = choose_action(next_state, q_table, epsilon) # A'
                next_q = q_table.get((next_state, next_action), 0.0)
                target = reward + GAMMA * next_q
            
            # Update Q-value
            new_q = old_q + ALPHA * (target - old_q)
            q_table[(state, action)] = new_q
            
            # Move to next state/action
            state = next_state
            if not done:
                action = next_action

        # Update stats
        recent_returns.append(total_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        # Log progress
        if episode % LOG_INTERVAL == 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            print(f"{episode:<8d} | {total_reward:<8.1f} | {steps:<6d} | {epsilon:<7.3f} | {avg_return:<20.1f}")

    print("-" * 75)
    return q_table

def run_q_learning():
    """
    Main Q-Learning training loop.
    """
    # Initialize Q-table: key=(state, action), value=float
    q_table = {}
    
    # To track rolling average return
    recent_returns = [] 
    
    print(f"\nStarting Q-Learning Training: {TOTAL_EPISODES} episodes")
    print(f"Params: Alpha={ALPHA}, Gamma={GAMMA}, Epsilon Schedule: Start={EPSILON_START}, Min={EPSILON_MIN}, Decay={EPSILON_DECAY}, Slip={SLIP_PROB}")
    
    headers = ["Episode", "Return", "Steps", "Epsilon", "Avg Return (Last 50)"]
    print("-" * 75)
    print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<7} | {headers[4]:<20}")
    print("-" * 75)

    for episode in range(1, TOTAL_EPISODES + 1):
        # Calculate epsilon for this episode
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** (episode - 1)))
        
        state = START_POS
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Q-Learning chooses action based on Current State using Epsilon-Greedy
            action = choose_action(state, q_table, epsilon)
            
            next_state, reward, done = step(state, action)
            total_reward += reward
            steps += 1
            
            # Q-Learning Update
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]
            
            old_q = q_table.get((state, action), 0.0)
            
            if done:
                target = reward
            else:
                # Max Q over all actions in next state
                max_next_q = -float('inf')
                for a in ACTIONS:
                    val = q_table.get((next_state, a), 0.0)
                    if val > max_next_q:
                        max_next_q = val
                target = reward + GAMMA * max_next_q
            
            # Update Q-value
            new_q = old_q + ALPHA * (target - old_q)
            q_table[(state, action)] = new_q
            
            state = next_state
        
        # Update stats
        recent_returns.append(total_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        # Log progress
        if episode % LOG_INTERVAL == 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            print(f"{episode:<8d} | {total_reward:<8.1f} | {steps:<6d} | {epsilon:<7.3f} | {avg_return:<20.1f}")

    print("-" * 75)
    return q_table

def print_policy(q_table):
    """
    Prints the learned greedy policy and key locations.
    """
    print("\nLearned Greedy Policy:")
    print("Layer key: S=Start, G=Goal, X=Cliff, ^>v< = Best Action")
    
    grid_rows = []
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            pos = (r, c)
            
            # Label special cells
            # Although start is also a state, we can show its policy or 'S'
            # Usually nice to show policy everywhere, but Start is special.
            # Let's show Start as 'S' if it hasn't moved, but actually we want to see the policy at S too to know where it goes.
            # But the prompt asks to mark Start=S, Goal=G, Cliff=X.
            # The prompt implies replacing the character with S/G/X.
            if pos == START_POS:
                row_str.append(" S ")
                continue
            if pos == GOAL_POS:
                row_str.append(" G ")
                continue
            if pos in CLIFF_CELLS:
                row_str.append(" X ")
                continue
                
            # Determine best action for this cell
            best_a = 0
            best_val = -float('inf')
            found = False
            for a in ACTIONS:
                if (pos, a) in q_table:
                    found = True
                val = q_table.get((pos, a), 0.0)
                if val > best_val:
                    best_val = val
                    best_a = a
            
            # If no Q-values exist (unvisited), maybe show '?' or just default
            if not found and best_val == 0.0:
                 row_str.append(" . ") # Unvisited
            else:
                 row_str.append(f" {ACTION_SYMBOLS[best_a]} ")

        grid_rows.append("".join(row_str))
    
    print("\n".join(grid_rows))
    print("(Note: '.' means unvisited state)")


def greedy_rollout(q_table):
    """
    Runs one greedy episode (epsilon=0) and logs the path.
    """
    print("\nFinal Greedy Rollout:")
    state = START_POS
    path = [state]
    total_reward = 0
    done = False
    
    # Limit max steps to avoid infinite loops if policy is stuck
    max_steps = 100 
    steps = 0
    
    print(f"Start at {state}")
    
    while not done and steps < max_steps:
        # choose_action with epsilon=0 for greedy
        # Reuse choose_action logic but enforce epsilon=0
        # Manual greedy selection to be safe and explicit
        best_val = -float('inf')
        best_actions = []
        for a in ACTIONS:
            val = q_table.get((state, a), 0.0)
            if val > best_val:
                best_val = val
                best_actions = [a]
            elif val == best_val:
                best_actions.append(a)
        
        # If unvisited, random or default? 
        # Usually unvisited is 0.0.
        if not best_actions:
             action = random.choice(ACTIONS)
        else:
             action = random.choice(best_actions)

        next_state, reward, done = step(state, action)
        
        state = next_state
        path.append(state)
        total_reward += reward
        steps += 1
    
    if done and state == GOAL_POS:
        status = "Goal Reached!"
    elif done: # cliff
        status = "Fell off Cliff!"
    else:
        status = "Max steps exceeded (Stuck?)"
        
    print(f"Path: {path}")
    print(f"Total Return: {total_reward}")
    print(f"Status: {status}")

if __name__ == "__main__":
    # Optional: seeding
    # random.seed(42)
    
    print("=== SARSA ===")
    learned_sarsa_q = run_sarsa()
    print("\n[SARSA] Learned Greedy Policy")
    print_policy(learned_sarsa_q)
    print("\n[SARSA] Final Greedy Rollout")
    greedy_rollout(learned_sarsa_q)
    
    print("\n" + "="*30 + "\n")
    
    print("=== Q-LEARNING ===")
    learned_q_learning_q = run_q_learning()
    print("\n[Q-Learning] Learned Greedy Policy")
    print_policy(learned_q_learning_q)
    print("\n[Q-Learning] Final Greedy Rollout")
    greedy_rollout(learned_q_learning_q)
