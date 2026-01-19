<p align="center">
  <img height="300" src="https://raw.githubusercontent.com/4D-Lab/logos/refs/heads/main/frame.png"/>
</p>
______________________________________________________________________

This repository introduces **FRAME**, a framework for learning fragment-based molecular representations to enhance the interpretability of graph neural networks in drug discovery. FRAME represents chemically meaningful fragments as graph nodes and is compatible with several GNN architectures, including GCN, GAT, and AttentiveFP. It also integrates Integrated Gradients to generate more transparent and chemically grounded model explanations.

## ⚙️ **Installation**
1. Clone the repo:

2. Create and activate your `virtualenv` with Python 3.12, for example as described [here](https://docs.python.org/3/library/venv.html).

3. Install [PyTorch **2.8.0**](https://pytorch.org/get-started/locally/) using:

    ```console
    pip install torch==2.8.0 -f https://download.pytorch.org/whl/cu129
    ```

4. Install FRAME using:

    ```console
    python -m pip install .
    ```
    or for development:
    ```console
    python -m pip install -e .
    ```

## 📂 Dataset Requirements
The CSV file used in FRAME **must** include the following columns:

- **`id`** – A unique identifier for each entry.  
- **`smiles`** – The SMILES representation of the molecule.  
- **`label`** – The target value or class associated with each molecule.  
- **`set`** – Indicates the data split for each entry. This column must contain one of the following values:  
  - `train` (training data) 
  - `valid` (data used for early stopping) 
  - `test` (external test data)

Please ensure that all entries follow this structure so the dataset can be correctly loaded and processed by the pipeline.


## 📄 Configuration
All model parameters and runtime settings are defined in a YAML configuration file.  
An example file, [`parameters.yaml`](./parameters.yaml), is provided.

### Example: Defining hyperparameter ranges for tuning
To enable hyperparameter optimization, define parameters using `min` and `max`:

```yaml
Tune:
  hidden_channels:
    min: 64
    max: 128
```

### Example: Setting fixed parameter values
If you want to specify fixed values without optimization, use `value`:

```yaml
Tune:
  hidden_channels:
    value: 64
```

## 🔎 **Usage**
All entry points accept a `-c/--config` parameter pointing to the YAML config file.

- Generate a processed dataset:
```bash
frame_gen -c parameters.yaml
```

- Run Optuna hyperparameter tuning:
```bash
frame_tune -c parameters.yaml
```

- Train a single model using values in the `Tune` section:
```bash
frame_train -c parameters.yaml
```

- Evaluate trained with the test set:
```bash
frame_eval -c parameters.yaml
```

- Explain and run model prediction:
```bash
frame_explain -c parameters.yaml
```
