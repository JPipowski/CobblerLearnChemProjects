import os
import random
import string
import matplotlib.pyplot as plt

# Default configuration
DEFAULT_TARGET = "METHINKS IT IS LIKE A WEASEL"
POPULATION_SIZE = 100
MUTATION_RATE = 0.05  # 5% chance per character to mutate
POSSIBLE_CHARS = string.ascii_uppercase + " "


def generate_random_string(length: int) -> str:
    """Generates an initial random string of the target length."""
    return "".join(random.choice(POSSIBLE_CHARS) for _ in range(length))


def calculate_fitness(candidate: str, target: str) -> int:
    """Calculates fitness as the number of matching characters at identical positions."""
    return sum(1 for c, t in zip(candidate, target) if c == t)


def mutate(parent: str, mutation_rate: float) -> str:
    """Mutates characters in the string based on the mutation rate."""
    child_chars = []
    for char in parent:
        if random.random() < mutation_rate:
            child_chars.append(random.choice(POSSIBLE_CHARS))
        else:
            child_chars.append(char)
    return "".join(child_chars)


def plot_fitness(generations: list[int], fitnesses: list[int], target_len: int,
                 filename: str = "fitness_over_time.png"):
    """Generates and saves a plot of fitness over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitnesses, color="#2b5c8f", linewidth=2, label="Best Fitness")
    plt.axhline(y=target_len, color="r", linestyle="--", label=f"Target Score ({target_len})")
    plt.title("Fitness Over Generations (Weasel Program)", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best Fitness Score", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nPlot successfully saved to: {os.path.abspath(filename)}")


def run_evolution():
    # 1. Interactive Text Entry Box (Terminal input)
    user_input = input(f"Enter target phrase (Press ENTER for default '{DEFAULT_TARGET}'): ").strip().upper()
    target = user_input if user_input else DEFAULT_TARGET

    # Restrict character set to uppercase and space for standard simulation matching
    target = "".join(c if c in POSSIBLE_CHARS else " " for c in target)
    target_length = len(target)

    # 2. Setup tracking structures
    best_string_so_far = generate_random_string(target_length)
    best_score_so_far = calculate_fitness(best_string_so_far, target)
    generation = 0

    generation_history = [generation]
    fitness_history = [best_score_so_far]

    log_file_path = "generations_log.txt"

    # 3. Main Evolutionary Loop & Logging
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        header = f"Evolution Run for Target: '{target}'\n" + "=" * 60 + "\n"
        print(header, end="")
        log_file.write(header)

        initial_msg = f"Generation: {generation:4d} | Best Score: {best_score_so_far:2d}/{target_length} | Best String: '{best_string_so_far}'\n"
        print(initial_msg, end="")
        log_file.write(initial_msg)

        while best_score_so_far < target_length:
            generation += 1

            # Generate offspring pool from current best parent
            offspring_pool = [
                mutate(best_string_so_far, MUTATION_RATE)
                for _ in range(POPULATION_SIZE)
            ]

            # Find best offspring in current pool
            best_offspring = max(
                offspring_pool, key=lambda child: calculate_fitness(child, target)
            )
            best_offspring_fitness = calculate_fitness(best_offspring, target)

            # Update state if better or equal
            if best_offspring_fitness >= best_score_so_far:
                best_string_so_far = best_offspring
                best_score_so_far = best_offspring_fitness

            # Record history
            generation_history.append(generation)
            fitness_history.append(best_score_so_far)

            # Log to terminal and file
            log_entry = f"Generation: {generation:4d} | Best Score: {best_score_so_far:2d}/{target_length} | Best String: '{best_string_so_far}'\n"
            print(log_entry, end="")
            log_file.write(log_entry)

        summary_msg = f"\nTarget reached in {generation} generations!\n"
        print(summary_msg)
        log_file.write(summary_msg)

    print(f"Full generation log saved to: {os.path.abspath(log_file_path)}")

    # 4. Save Fitness Plot
    plot_fitness(generation_history, fitness_history, target_length)


if __name__ == "__main__":
    run_evolution()