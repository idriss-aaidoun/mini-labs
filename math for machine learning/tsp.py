
#!/usr/bin/env python3
# Team: <NAME 1>, <NAME 2>, <NAME 3>, <NAME 4>
#
# TP Metaheuristic 2023 - Traveling Salesman Problem
# Algorithm: nearest-neighbour start -> local search (2-opt + Or-opt, neighbour
# lists, work queue) -> Iterated Local Search with a local double-bridge kick.
#
# Usage: python tsp.py instance.txt [solution.txt] [cpu_seconds]
#   defaults: solution.txt = sol.txt, cpu_seconds = 55 (limit is 60)
 
import sys
import math
import time
import random
import itertools
 
EPS = 1e-9
K_NEIGHBOURS = 10
 
 
def read_instance(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 (PowerShell '>' output)
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    tok = text.split()
    if not tok:
        sys.exit(f"Error: {path} is empty")
    if not tok[0].isdigit():
        sys.exit(f"Error: {path} is not a valid instance (first token must be "
                 f"the number of cities, got {tok[0]!r})")
    n = int(tok[0])
    if len(tok) < 1 + 2 * n:
        sys.exit(f"Error: {path} declares {n} points but has fewer coordinates")
    return [(int(tok[1 + 2 * i]), int(tok[2 + 2 * i])) for i in range(n)]
 
 
def write_solution(path, tour):
    with open(path, "w") as f:
        f.write(" ".join(map(str, tour)) + "\n")
 
 
def tour_length(tour, D):
    return sum(D[tour[i - 1]][tour[i]] for i in range(len(tour)))
 
 
def solve(pts, out_path, cpu_limit):
    n = len(pts)
    t_end = cpu_limit  # compared with time.process_time() (CPU time)
 
    if n <= 3:
        tour = list(range(n))
        write_solution(out_path, tour)
        return tour, 0.0
 
    D = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in pts] for a in pts]
 
    # Tiny instances: exact brute force
    if n <= 9:
        best, bl = None, float("inf")
        for perm in itertools.permutations(range(1, n)):
            t = (0,) + perm
            l = tour_length(t, D)
            if l < bl:
                best, bl = list(t), l
        write_solution(out_path, best)
        return best, bl
 
    k = min(K_NEIGHBOURS, n - 1)
    # (explicitly exclude i itself: duplicate points can have distance 0)
    neigh = [[c for c in sorted(range(n), key=D[i].__getitem__) if c != i][:k]
             for i in range(n)]
 
    # ---- nearest neighbour construction ----
    unvisited = set(range(1, n))
    tour = [0]
    while unvisited:
        last = D[tour[-1]]
        nxt = min(unvisited, key=last.__getitem__)
        tour.append(nxt)
        unvisited.remove(nxt)
    pos = [0] * n
    for i, c in enumerate(tour):
        pos[c] = i
 
    def rebuild_pos():
        for i, c in enumerate(tour):
            pos[c] = i
 
    def reverse(i, j):
        """Reverse cyclic segment from position i to j (inclusive, forward)."""
        inner = (j - i) % n + 1
        if inner * 2 > n:  # reverse the complement instead (same cycle)
            i, j = (j + 1) % n, (i - 1) % n
            inner = n - inner
        for _ in range(inner // 2):
            a, b = tour[i], tour[j]
            tour[i] = b
            pos[b] = i
            tour[j] = a
            pos[a] = j
            i += 1
            if i == n:
                i = 0
            j -= 1
            if j < 0:
                j = n - 1
 
    def try_2opt(a, queue):
        i = pos[a]
        Da = D[a]
        # successor direction
        b = tour[(i + 1) % n]
        dab = Da[b]
        for c in neigh[a]:
            dac = Da[c]
            if dac >= dab:
                break
            d = tour[(pos[c] + 1) % n]
            if c == b or d == a:
                continue
            delta = dac + D[b][d] - dab - D[c][d]
            if delta < -EPS:
                reverse(pos[b], pos[c])
                queue.extend((a, b, c, d))
                return delta
        # predecessor direction
        b = tour[(i - 1) % n]
        dab = Da[b]
        for c in neigh[a]:
            dac = Da[c]
            if dac >= dab:
                break
            d = tour[(pos[c] - 1) % n]
            if c == b or d == a:
                continue
            delta = dac + D[b][d] - dab - D[c][d]
            if delta < -EPS:
                reverse(pos[a], pos[d])
                queue.extend((a, b, c, d))
                return delta
        return 0.0
 
    def try_oropt(a, queue):
        nonlocal tour
        i = pos[a]
        for L in (1, 2, 3):
            if n < L + 4:
                break
            s1 = a
            sL = tour[(i + L - 1) % n]
            p = tour[(i - 1) % n]
            nx = tour[(i + L) % n]
            gain = D[p][s1] + D[sL][nx] - D[p][nx]
            if gain <= EPS:
                continue
            for e, f in ((s1, sL), (sL, s1)):
                for c in neigh[e]:
                    if (pos[c] - i) % n < L:  # c inside the segment
                        continue
                    dec = D[e][c]
                    if dec >= gain:
                        break
                    ic = pos[c]
                    # option A: c, e..f, succ(c)
                    cs = tour[(ic + 1) % n]
                    if (pos[cs] - i) % n >= L:
                        delta = dec + D[f][cs] - D[c][cs] - gain
                        if delta < -EPS:
                            apply_or(i, L, c, e == s1, True)
                            queue.extend((p, nx, s1, sL, c, cs))
                            return delta
                    # option B: pred(c), f..e, c
                    cp = tour[(ic - 1) % n]
                    if (pos[cp] - i) % n >= L:
                        delta = dec + D[cp][f] - D[cp][c] - gain
                        if delta < -EPS:
                            apply_or(i, L, c, e == sL, False)
                            queue.extend((p, nx, s1, sL, c, cp))
                            return delta
        return 0.0
 
    def apply_or(i, L, c, keep_orientation, after):
        """Move segment tour[i..i+L-1] next to c.
        after=True : c, seg(first element = e), succ(c)   -> seg starts with e
        after=False: pred(c), seg(last element = e), c    -> seg ends with e
        keep_orientation is True when e is the segment's natural first (after)
        or natural last (before) element."""
        rot = tour[i:] + tour[:i]
        seg = rot[:L]
        rest = rot[L:]
        kk = rest.index(c)
        if not keep_orientation:
            seg.reverse()
        if after:
            new = rest[:kk + 1] + seg + rest[kk + 1:]
        else:
            new = rest[:kk] + seg + rest[kk:]
        tour[:] = new
        rebuild_pos()
 
    def local_search(queue):
        total = 0.0
        inq = set(queue)
        while queue:
            a = queue.pop()
            inq.discard(a)
            while True:
                tmp = []
                d = try_2opt(a, tmp)
                if d == 0.0:
                    d = try_oropt(a, tmp)
                if d == 0.0:
                    break
                total += d
                for x in tmp:
                    if x not in inq and x != a:
                        inq.add(x)
                        queue.append(x)
        return total
 
    # ---- initial descent ----
    cur = tour_length(tour, D)
    q = list(range(n))
    random.shuffle(q)
    cur += local_search(q)
    best = cur
    write_solution(out_path, tour)
 
    # ---- iterated local search ----
    W = min(n, 50)
    while time.process_time() < t_end:
        saved_tour = tour[:]
        saved_pos = pos[:]
        old = cur
 
        # local double bridge on a window of <= W consecutive cities
        s = random.randrange(n)
        rot = tour[s:] + tour[:s]
        x, y, z = sorted(random.sample(range(1, W + 1), 3))
        r0, rx1, rx = rot[0], rot[x - 1], rot[x]
        ry1, ry = rot[y - 1], rot[y]
        rz1, rz = rot[z - 1], rot[z % n]
        delta = (D[rx1][ry] + D[rz1][rx] + D[ry1][rz]
                 - D[rx1][rx] - D[ry1][ry] - D[rz1][rz])
        tour[:] = rot[:x] + rot[y:z] + rot[x:y] + rot[z:]
        rebuild_pos()
        cur += delta
        cur += local_search([rx1, rx, ry1, ry, rz1, rz])
 
        if cur < old - EPS:  # strict improvement -> keep and save
            if cur < best - EPS:
                best = cur
                write_solution(out_path, tour)
        elif cur > old + EPS:  # worse -> revert
            tour[:] = saved_tour
            pos[:] = saved_pos
            cur = old
        # equal: keep (plateau move)
 
    final = tour_length(tour, D)
    write_solution(out_path, tour)
    return tour, final
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python tsp.py instance.txt [solution.txt] [cpu_seconds]")
        sys.exit(1)
    inst = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "sol.txt"
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 55.0
    random.seed(12345)
    pts = read_instance(inst)
    tour, length = solve(pts, out, limit)
    print(f"{inst}: n={len(pts)} best={length:.3f} cpu={time.process_time():.1f}s")
 
 
if __name__ == "__main__":
    main()

