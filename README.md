# Powerformer

Powerformer is a deep learning-based framework designed for robust and precise fault detection and location in power systems. It leverages the expressiveness of Graph Neural Networks (GNNs) and Transformers by framing the electrical grid as a spatio-temporal graph. The model uses a **Graph Transformer** architecture with Laplacian Eigenvector Positional Encodings (PE) to process time-series data of voltages and currents from various nodes and transmission lines (edges).

By capturing both the localized topology of the power grid and the temporal dynamics of electrical features, Powerformer aims to identify abnormal states ("detect" faults such as LG-A, LLLG, etc.) or pinpoint the exact transmission line or node where the fault occurred ("locate").

## Tech Stack

* **Core Frameworks**: PyTorch, PyTorch Geometric (PyG)
* **Data Processing**: Pandas, NumPy
* **Metrics & Evaluation**: Scikit-Learn, Matplotlib
* **Execution & Scheduling**: SLURM (for multi-GPU HPC environments), Distributed Data Parallel (DDP)

## Setup and Installation

### 1. Environment Preparation

It is recommended to use an Anaconda environment.

```bash
# Create and activate a new conda environment
conda create --name powerformer python=3.10
conda activate powerformer
```

### 2. Install Dependencies

Install the required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Configuration

The model behavior, dataset paths, training hyperparameters, and fault types are primarily controlled through `config.yml`. Update it to match your dataset directory (`data_path`) and output directories (`result_dir`).

### 4. Running the Model

For a standard single-GPU or CPU run:

```bash
python main.py --config-file config.yml
```

For running on an HPC cluster via SLURM (with multi-GPU support):

```bash
sbatch run.sh
```

## Directory Tree

```
Powerformer/
├── config.yml            # Centralized configuration for dataset, model parameters, and training loop.
├── data.py               # Data pipeline: reads CSVs, extracts time-series windows, and builds PyG Data objects with Laplacian PE.
├── layers.py             # Model architecture: contains FeedForwardNN, MultiHeadAttentionLayer, GraphTransformerLayer, and the final PowerFormer model.
├── main.py               # Entry point of the application. Handles directory creation, logging setup, and model initialization.
├── model.txt             # Model architecture summary and parameter dump.
├── requirements.txt      # Python package dependencies.
├── run.sh                # SLURM batch script for executing the training process on compute nodes.
├── train.py              # Single-GPU training and evaluation loop, including metric calculations.
├── train_multi_gpu.py    # Multi-GPU Distributed Data Parallel (DDP) training setup and loop.
├── utils.py              # Helper functions: custom loggers, metrics plotting, confusion matrix generation, and checkpoint loading.
└── visualize.ipynb       # Jupyter notebook for exploratory data analysis, plotting time-series, or visualizing model outputs.
```

## Future Scope

1. **Real-time Inference Pipeline**: Integrating the model with a streaming data broker (e.g., Apache Kafka) for real-time, low-latency fault detection in live PMU (Phasor Measurement Unit) data streams.
2. **Hyperparameter Optimization**: Implementing automated hyperparameter tuning using Ray Tune or Optuna to optimize head count, transformer layers, and feature permutations.
3. **Generalization to Larger Topologies**: Testing and adapting positional encodings for dynamic or significantly larger grid topologies, enabling zero-shot transfer to unseen grid layouts.
4. **Enhanced Explainability**: Exploring attention scores to provide operators with human-readable reasoning about *why* a certain fault location was predicted based on specific anomalous voltage/current patterns.
5. **Additional Fault Classifications**: Expanding the `fault` dictionary to handle highly resistive faults and incipient (evolving) faults which present very subtle time-series signatures.
