import gymnasium as gym
import numpy as np
import pygad
import time
import sys
import threading

# --- 1. Architecture Configuration ---
INPUT_SIZE = 8
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4
NUM_GENES = (INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)

# --- 2. Thread-Safe Environment Handling ---
thread_local = threading.local()

def get_env():
    if not hasattr(thread_local, "env"):
        thread_local.env = gym.make("LunarLander-v3")
    return thread_local.env

# --- 3. Neural Network Logic ---
def relu(x):
    return np.maximum(0, x)

def predict(observation, weights):
    idx = 0
    w1_size = INPUT_SIZE * HIDDEN_SIZE
    w1 = weights[idx:idx+w1_size].reshape((HIDDEN_SIZE, INPUT_SIZE))
    idx += w1_size
    b1 = weights[idx:idx+HIDDEN_SIZE]
    idx += HIDDEN_SIZE

    w2_size = HIDDEN_SIZE * OUTPUT_SIZE
    w2 = weights[idx:idx+w2_size].reshape((OUTPUT_SIZE, HIDDEN_SIZE))
    idx += w2_size
    b2 = weights[idx:idx+OUTPUT_SIZE]

    l1 = relu(np.dot(w1, observation) + b1)
    return np.argmax(np.dot(w2, l1) + b2)

# --- 4. Custom Fitness for "Touch-and-Go" ---
def fitness_func(ga_instance, solution, solution_idx):
    env = get_env()
    total_reward = 0
    episodes = 3

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        landed_once = False

        while not done:
            action = predict(obs, solution)
            obs, reward, term, trunc, _ = env.step(action)

            # Custom logic: if legs touch ground (obs[6] or obs[7] are 1.0)
            if (obs[6] == 1.0 or obs[7] == 1.0) and not landed_once:
                landed_once = True
                reward += 100 # Bonus for first contact

            # If it took off again after landing
            if landed_once and obs[1] > 0.2: # Altitude check
                reward += 2 # Reward for maintaining flight after first touch

            total_reward += reward
            done = term or trunc

    return total_reward / episodes

# --- 5. Dashboard & Timing ---
start_time = None

def on_generation(ga_instance):
    global start_time
    if ga_instance.generations_completed == 1:
        start_time = time.time()
        print("\n" * 8)

    elapsed = time.time() - start_time if start_time else 0
    avg_gen = elapsed / ga_instance.generations_completed if ga_instance.generations_completed > 0 else 0
    _, best_fit, _ = ga_instance.best_solution()

    # Move cursor up 9 lines
    sys.stdout.write("\033[9A")

    header = "========================================"
    title  = "        LUNAR LANDER GA STATUS          "

    stats = [
        header,
        title,
        header,
        f" Generation:     {ga_instance.generations_completed:<5} / {ga_instance.num_generations}",
        f" Best Fitness:   {best_fit:<10.2f}",
        f" Elapsed Time:   {elapsed:<10.2f} sec",
        f" Avg/Gen Time:   {avg_gen:<10.2f} sec",
        header,
        " Training in progress... (Ctrl+C to stop) "
    ]

    for line in stats:
        sys.stdout.write(f"\r{line}\033[K\n")

    sys.stdout.flush()

# --- 6. Genetic Algorithm Setup ---
ga_instance = pygad.GA(
    num_generations=150,
    num_parents_mating=12,
    fitness_func=fitness_func,
    sol_per_pop=60,
    num_genes=NUM_GENES,
    init_range_low=-1.0,
    init_range_high=1.0,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="scattered",
    mutation_type="adaptive",
    mutation_probability=[0.3, 0.05],
    keep_elitism=3,
    on_generation=on_generation,
    parallel_processing=["thread", 16]
)

# --- 7. Main Execution ---
if __name__ == "__main__":
    print("[*] Initializing Training...")

    try:
        ga_instance.run()
    except KeyboardInterrupt:
        print("\n[*] Training interrupted by user.")

    best_sol, best_fit, _ = ga_instance.best_solution()
    print(f"\n[*] Training Complete! Best Fitness: {best_fit:.2f}")

    # Testing the best pilot
    test_env = gym.make("LunarLander-v3", render_mode="human")
    while True:
        obs, _ = test_env.reset()
        done = False
        score = 0
        while not done:
            action = predict(obs, best_sol)
            obs, reward, term, trunc, _ = test_env.step(action)
            score += reward
            done = term or trunc

        print(f"Final Flight Score: {score:.2f}")
        if input("[?] Press Enter to restart (q to quit): ").lower() == 'q':
            break

    test_env.close()
