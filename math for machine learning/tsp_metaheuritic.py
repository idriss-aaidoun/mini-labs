import math
import random
import sys
import time


# ============================================================
# 1. READ THE INSTANCE
# ============================================================

def read_instance(filename):
    """
    Read a TSP instance.

    Format:
        n
        x0 y0
        x1 y1
        ...
        xn-1 yn-1
    """

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])

    points = []

    for i in range(1, n + 1):
        x, y = map(int, lines[i].split())
        points.append((x, y))

    return points


# ============================================================
# 2. DISTANCE BETWEEN TWO CITIES
# ============================================================

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ============================================================
# 3. BUILD DISTANCE MATRIX
# ============================================================

def build_distance_matrix(points):
    n = len(points)

    distances = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])

            distances[i][j] = d
            distances[j][i] = d

    return distances


# ============================================================
# 4. CALCULATE TOUR LENGTH
# ============================================================

def tour_length(tour, distances):
    """
    Calculate the total length of a tour.

    The tour is:
        tour[0] -> tour[1] -> ... -> tour[n-1] -> tour[0]
    """

    total = 0.0
    n = len(tour)

    for i in range(n):
        current_city = tour[i]
        next_city = tour[(i + 1) % n]

        total += distances[current_city][next_city]

    return total


# ============================================================
# 5. NEAREST NEIGHBOR INITIAL SOLUTION
# ============================================================

def nearest_neighbor(distances, start):
    """
    Construct an initial solution using the
    nearest-neighbor heuristic.
    """

    n = len(distances)

    unvisited = set(range(n))

    tour = [start]
    unvisited.remove(start)

    current = start

    while unvisited:

        next_city = min(
            unvisited,
            key=lambda city: distances[current][city]
        )

        tour.append(next_city)
        unvisited.remove(next_city)

        current = next_city

    return tour


# ============================================================
# 6. 2-OPT MOVE
# ============================================================

def two_opt_swap(tour, i, j):
    """
    Reverse the section between i and j.

    Example:

        A B C D E F

    If i = 1 and j = 4:

        A B C D E F
          --------
        A E D C B F
    """

    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


# ============================================================
# 7. FAST 2-OPT LOCAL SEARCH
# ============================================================

def two_opt(tour, distances, time_limit, start_time):
    """
    Improve a tour using 2-opt.

    First-improvement strategy:
    as soon as we find an improving move,
    we apply it and restart the search.
    """

    n = len(tour)

    improved = True

    while improved:

        # Stop if time limit reached
        if time.perf_counter() - start_time >= time_limit:
            break

        improved = False

        for i in range(1, n - 2):

            for j in range(i + 1, n - 1):

                if time.perf_counter() - start_time >= time_limit:
                    return tour

                a = tour[i - 1]
                b = tour[i]

                c = tour[j]
                d = tour[(j + 1) % n]

                # Current edges:
                # a -> b
                # c -> d

                old_cost = (
                    distances[a][b]
                    + distances[c][d]
                )

                # New edges:
                # a -> c
                # b -> d

                new_cost = (
                    distances[a][c]
                    + distances[b][d]
                )

                # Improvement
                if new_cost < old_cost:

                    tour[i:j + 1] = reversed(tour[i:j + 1])

                    improved = True

                    break

            if improved:
                break

    return tour


# ============================================================
# 8. RANDOM PERTURBATION
# ============================================================

def perturb(tour):
    """
    Randomly modify a solution to escape
    from the current local optimum.
    """

    new_tour = tour.copy()

    n = len(new_tour)

    i = random.randint(1, n - 3)
    j = random.randint(i + 1, n - 2)

    new_tour[i:j + 1] = reversed(new_tour[i:j + 1])

    return new_tour


# ============================================================
# 9. SAVE SOLUTION
# ============================================================

def save_solution(filename, tour):
    """
    Save the solution as a permutation of city IDs.
    """

    with open(filename, "w") as f:
        f.write(" ".join(map(str, tour)))


# ============================================================
# 10. MAIN METAHEURISTIC
# ============================================================

def solve(points, time_limit=60.0):
    """
    Multi-start metaheuristic:

        1. Generate an initial solution.
        2. Improve it with 2-opt.
        3. Keep the best solution.
        4. Perturb the best/current solution.
        5. Apply 2-opt again.
        6. Repeat until time limit.
    """

    start_time = time.perf_counter()

    n = len(points)

    distances = build_distance_matrix(points)

    # --------------------------------------------------------
    # Initial solution
    # --------------------------------------------------------

    # Start with a random city
    start_city = random.randrange(n)

    best_tour = nearest_neighbor(distances, start_city)

    best_value = tour_length(best_tour, distances)

    current_tour = best_tour.copy()

    # --------------------------------------------------------
    # Main search
    # --------------------------------------------------------

    while time.perf_counter() - start_time < time_limit:

        # Create a new starting solution

        if random.random() < 0.5:

            # Random nearest-neighbor construction
            start_city = random.randrange(n)

            current_tour = nearest_neighbor(
                distances,
                start_city
            )

        else:

            # Perturb the current best solution
            current_tour = perturb(best_tour)

        # ----------------------------------------------------
        # Local improvement
        # ----------------------------------------------------

        current_tour = two_opt(
            current_tour,
            distances,
            time_limit,
            start_time
        )

        current_value = tour_length(
            current_tour,
            distances
        )

        # ----------------------------------------------------
        # Update best solution
        # ----------------------------------------------------

        if current_value < best_value:

            best_tour = current_tour.copy()
            best_value = current_value

    return best_tour, best_value


# ============================================================
# 11. PROGRAM ENTRY POINT
# ============================================================

def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python tsp_metaheuristic.py Pb1.txt")
        return

    input_file = sys.argv[1]

    # Output file
    output_file = "solution.txt"

    # Time limit
    time_limit = 60.0

    print("Reading instance...")

    points = read_instance(input_file)

    print("Number of cities:", len(points))

    print("Starting optimization...")

    start = time.perf_counter()

    best_tour, best_value = solve(
        points,
        time_limit
    )

    elapsed = time.perf_counter() - start

    # Save final solution
    save_solution(output_file, best_tour)

    print()
    print("========== RESULTS ==========")
    print("Best distance:", best_value)
    print("CPU time:", elapsed)
    print("Solution saved to:", output_file)
    print("=============================")


if __name__ == "__main__":
    main()