# **Placeholder**

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

## Quick summary
- Language: Python >= 3.12
- Placeholder
- Placeholder
- Placeholder

## ⚙️ **Installation**
1. Install Python3-tk and python3-dev with the command:

    ```console
    sudo apt install python3-dev python3-tk
    ```

2. Clone the repo:

3. Create and activate your `virtualenv` with Python 3.12, for example as described [here](https://docs.python.org/3/library/venv.html).

4. Install [PyTorch **2.8.0**](https://pytorch.org/get-started/locally/) using:

    ```console
    pip install torch==2.8.0 -f https://download.pytorch.org/whl/cu129
    ```

5. Install Placeholder using:

    ```console
    python -m pip install -e .
    ```
    or for development:
    ```console
    python -m pip install -e .[dev]
    ```

## 📄 Configuration
All runtime options live in a YAML config (an example is provided as `parameters.yaml`). Important sections:
- `Data`: dataset paths, run name, model choice, batch size, epochs, trials, patience, loader type (`default` or `decompose`) and `path_joblib`/`path_csv`.  
- `Tune`: hyperparameter ranges or fixed values used by the tuning and training scripts.

The CSV file must contain the columns: `id`, `smiles`, `label`, and `set` (where `set` should be one of `train`, `test`, or `valid`).

## 🔎 **Usage**
All entry points accept a `-c/--config` parameter pointing to the YAML config file.

- Generate a processed dataset:
```bash
placeholder_gen -c parameters.yaml
```

- Run Optuna hyperparameter tuning:
```bash
placeholder_tune -c parameters.yaml
```

- Train a single model using values in the `Tune` section:
```bash
placeholder_train -c parameters.yaml
```
