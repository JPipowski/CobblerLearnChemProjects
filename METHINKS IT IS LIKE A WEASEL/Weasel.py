
import os
import random
import string
import sys
import matplotlib.pyplot as plt

# GitPython import with error handling
try:
    from git import GitCommandError, Repo
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "GitPython is missing. Run 'pip install GitPython' in your environment."
    )

# Configuration
DEFAULT_TARGET = "METHINKS IT IS LIKE A WEASEL"
POPULATION_SIZE = 100
MUTATION_RATE = 0.05  # 5% chance per character to mutate
POSSIBLE_CHARS = string.ascii_uppercase + " "


def generate_random_string(length: int) -> str:
    """Generates an initial random string of the target length."""
    return "".join(random.choice(POSSIBLE_CHARS) for _ in range(length))


def calculate_fitness(candidate: str, target: str) -> int:
    """Calculates fitness as matching characters at identical positions."""
    return sum(1 for c, t in zip(candidate, target) if c == t)


def mutate(parent: str, mutation_rate: float) -> str:
    """Mutates characters in the string based on the mutation rate."""
    return "".join(
        random.choice(POSSIBLE_CHARS) if random.random() < mutation_rate else char
        for char in parent
    )


def plot_fitness(
    generations: list[int],
    fitness: list[int],
    target_len: int,
    filename: str = "fitness_over_time.png",
):
    """Generates and saves a scatter plot of fitness over generations."""
    plt.figure(figsize=(10, 6))

    plt.scatter(
        generations,
        fitness,
        color="#e74c3c",
        alpha=0.7,
        edgecolors="none",
        label="Generation Fitness",
    )
    plt.plot(
        generations,
        fitness,
        color="#2b5c8f",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Progress Trend",
    )

    plt.axhline(
        y=target_len,
        color="g",
        linestyle="-.",
        label=f"Target Score ({target_len})",
    )
    plt.title("Fitness Over Generations (Weasel Program)", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Best Fitness Score", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nPlot successfully saved to: {os.path.abspath(filename)}")


def commit_and_push_to_github(files_to_commit: list[str], commit_message: str):
    """Stages, commits, and pushes specified files to GitHub using GITHUB_TOKEN env variable."""
    try:
        repo_path = os.getcwd()
        repo = Repo(repo_path)

        if repo.bare:
            print("\n[Git Error] Repository path is bare or invalid.")
            return

        print("\n--- Staging and Committing Files ---")

        # Stage files
        repo.index.add(files_to_commit)
        print(f"Staged files: {files_to_commit}")

        # Commit changes
        repo.index.commit(commit_message)
        print(f"Committed with message: '{commit_message}'")

        origin = repo.remote(name="origin")
        original_url = list(origin.urls)[0]

        # Retrieve token from environment variable
        github_token = os.getenv("GITHUB_TOKEN")

        if not github_token:
            print(
                "\n[Git Warning] GITHUB_TOKEN environment variable not set."
            )
            print("Attempting to push using existing local/default Git credentials...")
            origin.push()
            print("Successfully pushed changes to GitHub!")
            return

        modified_url = False
        try:
            # Inject token safely into HTTPS remote URL for this push operation
            if original_url.startswith("https://"):
                clean_url = original_url.split("@")[-1].replace("https://", "")
                auth_url = f"https://x-access-token:{github_token}@{clean_url}"
                origin.set_url(auth_url)
                modified_url = True

            print("Pushing changes to GitHub...")
            origin.push()
            print("Successfully pushed changes to GitHub!")

        finally:
            # Guarantee the token is restored so it is never left in .git/config on disk
            if modified_url:
                origin.set_url(original_url)

    except GitCommandError as git_err:
        print(f"\n[Git Command Error]: {git_err}")
        print(
            "Tip: Ensure your Personal Access Token has the 'repo' scope enabled."
        )
    except Exception as e:
        print(f"\n[Git Automation Error]: {e}")


def run_evolution():
    # 1. Interactive Text Entry Box (Terminal input)
    user_input = input(
        f"Enter target phrase (Press ENTER for default '{DEFAULT_TARGET}'): "
    ).strip().upper()
    target = user_input if user_input else DEFAULT_TARGET

    # Restrict character set to uppercase and space
    target = "".join(c if c in POSSIBLE_CHARS else " " for c in target)
    target_length = len(target)

    # 2. Setup tracking structures
    best_string_so_far = generate_random_string(target_length)
    best_score_so_far = calculate_fitness(best_string_so_far, target)
    generation = 0

    generation_history = [generation]
    fitness_history = [best_score_so_far]

    log_file_path = "generations_log.txt"
    plot_file_path = "fitness_over_time.png"

    # Robust detection of current script path
    script_file_path: str | bytes = (
        os.path.basename(__file__)
        if "__file__" in globals()
        else os.path.basename(sys.argv[0])
    )

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

            # Find best offspring and fitness in a single pass
            best_offspring, best_offspring_fitness = max(
                (
                    (child, calculate_fitness(child, target))
                    for child in offspring_pool
                ),
                key=lambda item: item[1],
            )

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

    # 4. Save Scatter Plot
    plot_fitness(
        generation_history, fitness_history, target_length, filename=plot_file_path
    )

    # 5. Automatically Stage, Commit, and Push to GitHub
    files_to_push = [plot_file_path, log_file_path]
    if os.path.exists(script_file_path) and script_file_path:
        files_to_push.append(script_file_path)

    commit_msg = f"Auto-commit: Ran Weasel evolution for '{target}' (Generations: {generation})"
    commit_and_push_to_github(files_to_push, commit_msg)


if __name__ == "__main__":
    run_evolution()