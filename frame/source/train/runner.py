import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from frame.source import models
from frame.source import train as train_pkg


def train_one_seed(seed: int, train_data: list, valid_loader: DataLoader,
                   model_name: str, config: dict, epochs: int,
                   patience: int, task: str, grad_clip: float,
                   drop_edge_p: float, mask_feat_p: float,
                   batch_size: int, workers: int):
    """Train a fresh model under a fixed seed and return its best state.

    Reseeds the global RNGs and pins the train ``DataLoader``'s shuffle
    generator so that two calls with the same ``seed`` produce the same
    trajectory. The valid loader is reused across seeds since it does not
    shuffle.

    Args:
        seed: Integer used to reseed Python/NumPy/Torch RNGs and the
            train-loader generator.
        train_data: List of ``Data`` objects belonging to the train split.
        valid_loader: Pre-built validation loader (no shuffle, seed-free).
        model_name: Backbone selector consumed by ``models.select_model``.
        config: Hyperparameter dict consumed by ``models.model_setup``.
        epochs: Maximum number of epochs to train.
        patience: Early-stopping patience on the ``optim`` metric.
        task: ``"classification"`` or ``"regression"``.
        grad_clip: Max-norm gradient clipping value (``None``/0 disables).
        drop_edge_p: Train-time edge drop probability.
        mask_feat_p: Train-time node-feature mask probability.
        batch_size: Train loader batch size.
        workers: Number of DataLoader workers.

    Returns:
        Tuple ``(best_state, results, fit_time, n_params)`` where
        ``best_state`` is the ``state_dict`` of the best epoch on the valid
        ``optim`` metric, ``results`` is the validation metric dict at that
        state, ``fit_time`` is wall-clock seconds spent training, and
        ``n_params`` is the trainable parameter count.
    """
    train_pkg.set_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=workers,
                              generator=generator)

    model, optim, schdlr, lossfn = models.model_setup(model_name, config,
                                                      epochs=epochs)

    best_metric = -float("inf")
    patience_counter = 0
    best_state = None

    start = time.time()
    for _ in tqdm(range(epochs), ncols=120, desc=f"Seed {seed}"):
        _ = train_pkg.train_epoch(model, optim, lossfn, train_loader,
                                  grad_clip_norm=grad_clip,
                                  drop_edge_p=drop_edge_p,
                                  mask_feat_p=mask_feat_p)
        val_metrics = train_pkg.valid_epoch(model, task, valid_loader)
        schdlr.step()

        if val_metrics["optim"] > best_metric:
            patience_counter = 0
            best_metric = val_metrics["optim"]
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    fit_time = time.time() - start

    model.load_state_dict(best_state)
    results = train_pkg.valid_epoch(model, task, valid_loader)

    n_params = sum(int(np.prod(p.size())) for p in model.parameters()
                   if p.requires_grad)
    return best_state, results, fit_time, n_params
