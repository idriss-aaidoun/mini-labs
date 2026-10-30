# mini-labs

Practical notebooks and experiments in **Artificial Intelligence & Data Science**, collected during my exchange year at **IMT Mines Alès**.

This repository follows my coursework: implementing core algorithms, training models, exploring uncertainty, and building agents that use tools. It contains lab exercises, worked examples, course resources, and experiments at different stages of completion.

## Explore the labs

| Area | Location | Topics |
| --- | --- | --- |
| Machine learning fundamentals | [`bases/`](./bases/) | Linear regression with the normal equation and gradient descent, Titanic classification, and a CIFAR-10 CNN |
| Bayesian networks | [`bayesian-networks/`](./bayesian-networks/) | Graph construction, conditional probability distributions, parameter estimation, and exact inference with pgmpy |
| Deep learning | [`deep_learning/`](./deep_learning/) | Neural networks in NumPy, fully connected networks and CNNs in PyTorch, and a Hugging Face practical |
| Agentic AI | [`agentic/`](./agentic/) | Local LLMs with Ollama and LangChain, structured outputs, tool calling, and ReAct loops |
| Decision analysis and uncertainty | [`decision_analysisç/`](./decision_analysis%C3%A7/) | Reliability diagrams, expected calibration error (ECE), temperature scaling, and conformal prediction |
| Optimization and metaheuristics | [`math for machine learning/`](./math%20for%20machine%20learning/) | Traveling Salesman Problem (TSP), nearest-neighbor construction, local search, and iterated search |
| Additional course resources | [`agentic/advenced machine learning/`](./agentic/advenced%20machine%20learning/) | Lecture PDFs, an ensemble-learning assignment, and another copy of the Bayesian-network labs with occupancy data |

A Titanic notebook is also available at the repository root. Folder names above match the current repository, including spaces and spelling.

## Suggested starting points

- **Review the basics:** open [the ML fundamentals folder](./bases/) and its [linear-regression guide](./bases/README.md).
- **Explore probabilistic reasoning:** start with [the sprinkler notebook](./bayesian-networks/01_sprinkler_pgmpy.ipynb), then the occupancy practical. The [Bayesian-network guide](./bayesian-networks/README.md) explains the main concepts.
- **Build an LLM agent:** work through [TD1 — Working with LLMs](./agentic/TD1_Working_with_LLMs.ipynb), followed by [TP1 — ReAct Agent](./agentic/TP1_ReAct_Agent.ipynb).
- **Study model uncertainty:** explore [the decision-analysis notebook](./decision_analysis%C3%A7/TP_decision_analysis.ipynb).
- **Explore optimization code:** inspect [the TSP solver](./math%20for%20machine%20learning/tsp.py).

## Getting started

### Browse without running

Open any `.ipynb` file on GitHub to read its code, Markdown, and saved outputs.

### Run notebooks locally

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/idriss-aaidoun/mini-labs.git
cd mini-labs
python -m venv .venv
```

Activate the environment:

| Platform | Command |
| --- | --- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\\Scripts\\Activate.ps1` |

Install JupyterLab and launch it:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name mini-labs --display-name "Python (mini-labs)"
python -m jupyter lab
```

Select **Python (mini-labs)** as the notebook kernel. Install the packages required by the notebook you choose, following its imports and setup cells. The repository currently has no shared dependency manifest or pinned environment; installing JupyterLab alone does not install every lab's dependencies.

Typical libraries vary by subject: NumPy, pandas, Matplotlib, scikit-learn, pgmpy, TensorFlow/Keras, PyTorch, and LangChain. Separate environments can help when labs need different package versions.

### Run the local LLM labs

The agentic notebooks use **Ollama** with **`qwen2.5:3b`**. Install Ollama, then download the model:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama server is running; use `ollama serve` if it is not already running through the desktop app. Follow TD1's setup cells for the Python dependencies. Local model inference does not require an API key; LangSmith tracing is optional.

### Run a TSP example

From the repository root, run the solver against the included 100-city instance:

```bash
python "math for machine learning/tsp.py" "math for machine learning/test100.txt" tsp_solution.txt 5
```

The arguments are the input instance, output file, and CPU-time budget in seconds. The script writes a tour as zero-based city IDs to `tsp_solution.txt`. The CPU-time budget guides the search; it is not a strict wall-clock timeout.

## Data and reproducibility

- Open notebooks from their own directory when they use relative file paths, and check dataset paths before running all cells.
- Occupancy CSV files are stored under [the nested Bayesian-network data folder](./agentic/advenced%20machine%20learning/bayesian-networks/data/). Adjust paths if running the copy of the lab in the root-level `bayesian-networks/` folder.
- MNIST files are included under `deep_learning/data/MNIST/`; other datasets may need to be downloaded as directed by each notebook.
- `Pb2.txt` through `Pb5.txt` in the optimization folder are currently empty placeholders. Use a populated instance such as `test100.txt` for the TSP example.
- Training results and timings depend on random seeds, hardware, package versions, and the state of the notebook kernel.
- Keep any optional service credentials outside committed notebook cells and outputs.

## About this repository

These materials document a learning process. Some notebooks include exercise prompts, unfinished cells, corrections, or saved outputs from earlier runs. Saved outputs help with browsing, but reproducing them may require additional dependencies, datasets, or local configuration.

## Author

**Idriss Aaidoun**  
ENSA Kénitra — Big Data & AI track  
Exchange student at IMT Mines Alès — Artificial Intelligence & Data Science

