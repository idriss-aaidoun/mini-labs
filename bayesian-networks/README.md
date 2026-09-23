# Bayesian Networks

Labs exploring probabilistic graphical models using [`pgmpy`](https://pgmpy.org/), covering network construction, conditional probability distributions (CPDs), and exact inference.

## Notebooks

### `01_sprinkler_pgmpy.ipynb`
Introductory walkthrough of the classic sprinkler network (`Winter → Sprinkler`, `Winter → Rain`, `Sprinkler → WetGrass`, `Rain → WetGrass`, `Rain → Slippery`). Covers:
- Building a `DiscreteBayesianNetwork` from a hand-specified graph and hand-written CPDs
- Validating a model with `check_model()`
- Exact inference with `VariableElimination`, and the two different meanings of "evidence" (parents in a `TabularCPD` vs. observed values in a query)
- Fork vs. collider path semantics, and how observing/not observing an intermediate variable opens or blocks information flow
- **Explaining away**: how two competing causes of the same observed effect become correlated once that effect is observed
- The distinction between *absent evidence* (unknown) and *evidence fixed to a specific value* (known-false), and why confusing the two gives very different — and sometimes wrong — conclusions
- Learning a CPD from data with `BayesianEstimator` and Dirichlet (Laplace) smoothing

### `02_occupancy_practical.ipynb`
Applied practical: a naive-Bayes-structured Bayesian network (`Occupancy → {Temperature, Light, Sound, CO2, Motion}`) fitted on real chronological sensor data to predict room occupancy. Covers:
- Learning all CPDs from a training split with Dirichlet-smoothed `BayesianEstimator`
- Querying occupancy probability under full sensor evidence vs. with one sensor's reading withheld (simulating a sensor outage)
- Evaluating predictions with Brier score, confusion counts, and an asymmetric-cost decision threshold (false-negative costlier than false-positive)
- Interpreting the effect of a missing sensor: isolating the controlled comparison, quantifying the change, and being explicit about what the result does and does not establish (e.g. no claims beyond this specific train/test split, sensor correlations not accounted for)

## Key concepts recap

- **Graph = structure** (who influences whom), **CPD = strength** (how much, quantified as probabilities)
- A variable with *k* parents needs 2^k CPD columns, one per parent configuration; every column must sum to 1
- **Fork** (`A ← X → B`): open when X is *unobserved*, blocked when X *is* observed
- **Collider** (`A → X ← B`): blocked when X is *unobserved*, open when X *is* observed — the mechanism behind explaining away
- **Absent ≠ false**: omitting a variable from evidence marginalizes over it; setting it to a specific value asserts a (possibly wrong) fact
- Dirichlet/Laplace smoothing (`pseudo_counts`) prevents CPD entries from collapsing to a hard 0 purely due to limited training data
