import random
import sys
 
if len(sys.argv) < 3:
    sys.exit("Usage: python gen.py <n_cities> <output_file>")
 
n = int(sys.argv[1])
random.seed(n)
with open(sys.argv[2], "w", encoding="ascii") as f:
    f.write(f"{n}\n")
    for _ in range(n):
        f.write(f"{random.randint(0, 1000)} {random.randint(0, 1000)}\n")
print(f"Wrote {n} cities to {sys.argv[2]}")
 