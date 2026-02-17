"""
Reinforcement Learning: Tabular Methods Comparison
Environment: Cliff Walking (Sutton & Barto, Example 6.6)

Why Cliff Walking?
This gridworld imposes a high penalty (-100) for falling into the "cliff" region.
This stark penalty highlights the difference between "safe" (on-policy) and
"optimal but risky" (off-policy) learning algorithms.

Algorithms Compared:
1. SARSA (State-Action-Reward-State-Action):
   - On-Policy: Learns the value of the policy being followed, including its exploration steps.
   - Result: Tends to learn a safer path to avoid the cliff, as it accounts for the
     possibility of epsilon-greedy "mistakes" (falling) during training.

2. Q-Learning:
   - Off-Policy: Learns the value of the optimal greedy policy, regardless of the
     agent's current exploration behavior.
   - Result: Tends to learn the optimal path hugging the cliff. However, in a stochastic
     environment (or with epsilon exploration), this path is dangerous and leads to
     frequent falls during training.

3. Expected SARSA:
   - On-Policy (usually): Similar to SARSA but updates based on the *expected* value
     of the next state actions rather than a sampled one.
   - Result: Reduces variance in updates, often leading to more stable learning than
     standard SARSA, while maintaining the risk-aware characteristics of on-policy methods.
"""

import random

# --- Constants & Hyperparameters ---

# Grid Dimensions
ROWS = 4
COLS = 12

# Key Locations
START_POS = (3, 0)
GOAL_POS = (3, 11)

# The "Cliff": A region that sends the agent back to start with a large penalty.
# Located at Row 3, Columns 1 through 10.
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

# Learning Parameters
ALPHA = 0.5  # Learning Rate: How much we accept the new error (TD error) into our value estimate.
GAMMA = 1.0  # Discount Factor: 1.0 means we care about future rewards equally (episodic task).

# Exploration Schedule (Epsilon-Greedy)
# Why Decay?
# - Start High (1.0): Encourage maximizing exploration of the grid early on.
# - End Low (0.01): As the agent learns, we want it to exploit the best policy found.
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Stochasticity
# Why SLIP_PROB?
# Real-world environments are rarely deterministic. Adding a slip probability (noise)
# tests the robustness of the learned policy. A path that is optimal in a deterministic
# world (hugging the cliff) might be disastrously fragile in a stochastic one.
SLIP_PROB = 0.1

TOTAL_EPISODES = 500
LOG_INTERVAL = 50

def get_next_position(state, action):
    """
    Calculates the next grid position given a state and action,
    clamping to the grid boundaries (walls).
    """
    row, col = state
    dr, dc = ACTION_DELTAS[action]
    next_r = max(0, min(ROWS - 1, row + dr))
    next_c = max(0, min(COLS - 1, col + dc))
    return (next_r, next_c)

def step(state, action):
    """
    Execute one step in the environment.
    Simulates the physics of the gridworld, including stochasticity and rewards.
    
    Returns: (next_state, reward, done)
    """
    # Stochasticity: Sometimes the agent "slips" and takes a random action instead.
    # This forces the agent to consider the risk of unintended moves.
    if random.random() < SLIP_PROB:
        action = random.choice(ACTIONS)

    # Determine tentative next position based on (possibly slipped) action
    next_pos = get_next_position(state, action)

    # Check for Cliff
    if next_pos in CLIFF_CELLS:
        # Falling into the cliff is disastrous.
        # -100 Penalty: Strong negative reinforcement signal to avoid this state-action.
        # Done=True: The episode ends (or effectively resets, treated as terminal here).
        return next_pos, -100, True

    # Check for Goal
    if next_pos == GOAL_POS:
        # Reaching the goal is the objective.
        # -1 Reward: Standard step cost. We want to reach the goal *fast*.
        # Done=True: Episode complete.
        return next_pos, -1, True

    # Standard step
    # Safe move, but costs time (-1 per step).
    return next_pos, -1, False

def choose_action(state, q_table, epsilon):
    """
    Select an action using an epsilon-greedy policy.
    
    Why Epsilon-Greedy?
    It balances:
    1. Exploration (Random): Trying new things to find better rewards (prob = epsilon).
    2. Exploitation (Greedy): Using known knowledge to maximize reward (prob = 1-epsilon).
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    
    # Greedy selection (with tie-breaking)
    # We look up Q(state, action) for all actions and pick the max.
    best_val = -float('inf')
    best_actions = []
    
    for a in ACTIONS:
        val = q_table.get((state, a), 0.0) # Default Q-value is 0.0
        if val > best_val:
            best_val = val
            best_actions = [a]
        elif val == best_val:
            best_actions.append(a)
    
    # Randomly break ties to ensure we don't get stuck biasing one direction
    return random.choice(best_actions)

def run_sarsa():
    """
    Main SARSA (State-Action-Reward-State-Action) training loop.
    
    Core Concept: On-Policy Control
    SARSA updates Q-values based on the action the agent *actually took* in the next step.
    If the agent explores (takes a random bad action), SARSA penalizes the *previous*
    state-action pair because it led to that bad situation.
    """
    q_table = {}
    recent_returns = [] 

    print(f"Starting SARSA Training: {TOTAL_EPISODES} episodes")
    print(f"Params: Alpha={ALPHA}, Gamma={GAMMA}, Epsilon Schedule: Start={EPSILON_START}, Min={EPSILON_MIN}, Decay={EPSILON_DECAY}, Slip={SLIP_PROB}")
    
    headers = ["Episode", "Return", "Steps", "Epsilon", "Avg Return (Last 50)"]
    print("-" * 75)
    print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<7} | {headers[4]:<20}")
    print("-" * 75)

    for episode in range(1, TOTAL_EPISODES + 1):
        # Decay epsilon: Explore less as we learn more
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** (episode - 1)))
        
        state = START_POS
        # SARSA requires choosing the first action *before* the loop starts
        action = choose_action(state, q_table, epsilon)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 1. Execute action (Environment Step)
            next_state, reward, done = step(state, action)
            total_reward += reward
            steps += 1
            
            # 2. Get Q(S, A) - The old estimate
            old_q = q_table.get((state, action), 0.0)
            
            if done:
                # Terminal State: No future rewards, target is just the immediate reward.
                target = reward
                next_action = None 
            else:
                # 3. Choose Next Action (A') *On-Policy*
                # We commit to this action for the next step of the loop.
                # If this action is random/bad (due to epsilon), the target reflects that.
                next_action = choose_action(next_state, q_table, epsilon)
                next_q = q_table.get((next_state, next_action), 0.0)
                
                # Target = Reward + Discounted Value of Actual Next Action
                target = reward + GAMMA * next_q
            
            # 4. Update Q-value towards Target
            new_q = old_q + ALPHA * (target - old_q)
            q_table[(state, action)] = new_q
            
            # 5. Move to next state/action
            state = next_state
            if not done:
                action = next_action

        # Logging
        recent_returns.append(total_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        if episode % LOG_INTERVAL == 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            print(f"{episode:<8d} | {total_reward:<8.1f} | {steps:<6d} | {epsilon:<7.3f} | {avg_return:<20.1f}")

    print("-" * 75)
    return q_table

def run_q_learning():
    """
    Main Q-Learning training loop.
    
    Core Concept: Off-Policy Control
    Q-Learning updates Q-values based on the *max* Q-value of the next state,
    effectively assuming the agent will act optimally from then on.
    It ignores the fact that the agent might actually explore (take a random action).
    """
    q_table = {}
    recent_returns = [] 
    
    print(f"\nStarting Q-Learning Training: {TOTAL_EPISODES} episodes")
    print(f"Params: Alpha={ALPHA}, Gamma={GAMMA}, Epsilon Schedule: Start={EPSILON_START}, Min={EPSILON_MIN}, Decay={EPSILON_DECAY}, Slip={SLIP_PROB}")
    
    headers = ["Episode", "Return", "Steps", "Epsilon", "Avg Return (Last 50)"]
    print("-" * 75)
    print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<7} | {headers[4]:<20}")
    print("-" * 75)

    for episode in range(1, TOTAL_EPISODES + 1):
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** (episode - 1)))
        
        state = START_POS
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # 1. Choose Action (Behavior Policy)
            # This controls how we interact with the world, but NOT how we update values.
            action = choose_action(state, q_table, epsilon)
            
            # 2. Execute action
            next_state, reward, done = step(state, action)
            total_reward += reward
            steps += 1
            
            # 3. Calculate Target
            old_q = q_table.get((state, action), 0.0)
            
            if done:
                target = reward
            else:
                # Off-Policy Magic:
                # We calculate max_a Q(S', a) regardless of what action we *actually* take next.
                # This estimates the value of the 'Greedy' policy.
                max_next_q = -float('inf')
                for a in ACTIONS:
                    val = q_table.get((next_state, a), 0.0)
                    if val > max_next_q:
                        max_next_q = val
                target = reward + GAMMA * max_next_q
            
            # 4. Update Q-value
            new_q = old_q + ALPHA * (target - old_q)
            q_table[(state, action)] = new_q
            
            state = next_state
        
        # Logging
        recent_returns.append(total_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        if episode % LOG_INTERVAL == 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            print(f"{episode:<8d} | {total_reward:<8.1f} | {steps:<6d} | {epsilon:<7.3f} | {avg_return:<20.1f}")

    print("-" * 75)
    return q_table

def run_expected_sarsa():
    """
    Main Expected SARSA training loop.
    
    Core Concept: Expected Value Target
    Instead of sampling the next action (SARSA) or taking the max (Q-Learning),
    Expected SARSA calculates the *expectation* of the Q-values under the current policy.
    
    Target = R + Gamma * Sum(Prob(a|s') * Q(s', a))
    
    This reduces variance because the update doesn't depend on the random die roll
    of the next epsilon-greedy action selection.
    """
    q_table = {}
    recent_returns = [] 
    
    print(f"\nStarting Expected SARSA Training: {TOTAL_EPISODES} episodes")
    print(f"Params: Alpha={ALPHA}, Gamma={GAMMA}, Epsilon Schedule: Start={EPSILON_START}, Min={EPSILON_MIN}, Decay={EPSILON_DECAY}, Slip={SLIP_PROB}")
    
    headers = ["Episode", "Return", "Steps", "Epsilon", "Avg Return (Last 50)"]
    print("-" * 75)
    print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<6} | {headers[3]:<7} | {headers[4]:<20}")
    print("-" * 75)

    for episode in range(1, TOTAL_EPISODES + 1):
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** (episode - 1)))
        
        state = START_POS
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done = step(state, action)
            total_reward += reward
            steps += 1
            
            old_q = q_table.get((state, action), 0.0)
            
            if done:
                target = reward
            else:
                # Calculate Expected Value under Epsilon-Greedy Policy
                expected_q = 0.0
                
                # 1. Identify Greedy Actions (actions with max Q-value)
                best_val = -float('inf')
                best_actions = []
                for a in ACTIONS:
                    val = q_table.get((next_state, a), 0.0)
                    if val > best_val:
                        best_val = val
                        best_actions = [a]
                    elif val == best_val:
                        best_actions.append(a)
                
                num_actions = len(ACTIONS)
                num_best = len(best_actions)
                
                # 2. Calculate Probabilities
                # Greedy Action Prob: (1 - epsilon) shared among best actions + epsilon share
                greedy_prob = (1.0 - epsilon) / num_best + (epsilon / num_actions)
                # Non-Greedy Action Prob: pure epsilon share
                non_greedy_prob = epsilon / num_actions
                
                # 3. Sum Prob * Value
                for a in ACTIONS:
                    val = q_table.get((next_state, a), 0.0)
                    if a in best_actions:
                        expected_q += greedy_prob * val
                    else:
                        expected_q += non_greedy_prob * val
                        
                target = reward + GAMMA * expected_q
            
            new_q = old_q + ALPHA * (target - old_q)
            q_table[(state, action)] = new_q
            
            state = next_state
        
        # Logging
        recent_returns.append(total_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        if episode % LOG_INTERVAL == 0:
            avg_return = sum(recent_returns) / len(recent_returns)
            print(f"{episode:<8d} | {total_reward:<8.1f} | {steps:<6d} | {epsilon:<7.3f} | {avg_return:<20.1f}")

    print("-" * 75)
    return q_table

def print_policy(q_table):
    """
    Prints the learned greedy policy covering the grid.
    Visualizes the best action at each state using symbols (^ > v <).
    """
    print("\nLearned Greedy Policy:")
    print("Layer key: S=Start, G=Goal, X=Cliff, ^>v< = Best Action")
    
    grid_rows = []
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            pos = (r, c)
            
            if pos == START_POS:
                row_str.append(" S ")
                continue
            if pos == GOAL_POS:
                row_str.append(" G ")
                continue
            if pos in CLIFF_CELLS:
                row_str.append(" X ")
                continue
                
            # Find best action for visualization
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
            
            if not found and best_val == 0.0:
                 row_str.append(" . ") # Unvisited / Default
            else:
                 row_str.append(f" {ACTION_SYMBOLS[best_a]} ")

        grid_rows.append("".join(row_str))
    
    print("\n".join(grid_rows))
    print("(Note: '.' means unvisited state)")


def greedy_rollout(q_table):
    """
    Simulates one episode using the learned policy greedily (epsilon=0).
    Verifies if the learned policy actually reaches the goal or falls.
    """
    print("\nFinal Greedy Rollout:")
    state = START_POS
    path = [state]
    total_reward = 0
    done = False
    
    # Safety breakout for infinite loops
    max_steps = 100 
    steps = 0
    
    print(f"Start at {state}")
    
    while not done and steps < max_steps:
        # Greedy Action Selection logic (same as choose_action with epsilon=0)
        best_val = -float('inf')
        best_actions = []
        for a in ACTIONS:
            val = q_table.get((state, a), 0.0)
            if val > best_val:
                best_val = val
                best_actions = [a]
            elif val == best_val:
                best_actions.append(a)
        
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
    # Note: We do not set a seed to observe variability across runs,
    # as stochasticity is a key part of this educational script.
    
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
    
    print("\n" + "="*30 + "\n")

    print("=== EXPECTED SARSA ===")
    learned_expected_sarsa_q = run_expected_sarsa()
    print("\n[Expected SARSA] Learned Greedy Policy")
    print_policy(learned_expected_sarsa_q)
    print("\n[Expected SARSA] Final Greedy Rollout")
    greedy_rollout(learned_expected_sarsa_q)
