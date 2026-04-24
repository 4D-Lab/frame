import torch
from torch.optim.lr_scheduler import (LinearLR,
                                      SequentialLR,
                                      CosineAnnealingLR)

from frame.source import train
from frame.source.models import pyg_models

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_setup(model_name, config, epochs=100):
    task = config["task"]
    model = select_model(model_name, config)

    base_optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config["learning_rate"],
                                      betas=(config["beta_1"],
                                             config["beta_2"]),
                                      eps=config["eps"],
                                      weight_decay=config["weight_decay"])
    optimizer = train.Lookahead(base_optimizer, k=5, alpha=0.5)

    warmup_epochs = int(config.get("warmup_epochs", 0))
    eta_min = float(config.get("lr_min", 1e-6))
    if warmup_epochs > 0 and warmup_epochs < epochs:
        warmup = LinearLR(optimizer, start_factor=0.1,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=max(1, epochs - warmup_epochs),
                                   eta_min=eta_min)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup, cosine],
                                 milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs),
                                      eta_min=eta_min)

    if task == "classification":
        bce_weight = config["bce_weight"]
        lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=bce_weight).to(device)

    else:
        lossfn = torch.nn.MSELoss()

    return model, optimizer, scheduler, lossfn


def select_model(model_name, config):
    if model_name == "attentive":
        model = pyg_models.GNN_AttentiveFP(config).to(device)
    elif model_name == "gat":
        model = pyg_models.GNN_GAT(config).to(device)
    elif model_name == "sage":
        model = pyg_models.GNN_SAGE(config).to(device)
    elif model_name == "gin":
        model = pyg_models.GNN_GIN(config).to(device)
    elif model_name == "gcn":
        model = pyg_models.GNN_GCN(config).to(device)
    else:
        raise NotImplementedError("Model not available")

    return model


def _cast_value(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return float(val)
    return str(val)


def tune_fixed(params):
    out = {}
    for name, bounds in params["Tune"].items():
        if "value" not in bounds:
            continue
        out[name] = _cast_value(bounds["value"])
    return out


def optuna_suggest(params, trial):
    configs = {}

    for name, bounds in params["Tune"].items():
        if "choices" in bounds:
            configs[name] = trial.suggest_categorical(name, bounds["choices"])
        elif "min" in bounds:
            log = bool(bounds.get("log", False))
            if isinstance(bounds["max"], int) and not log:
                configs[name] = trial.suggest_int(name, bounds["min"],
                                                  bounds["max"])
            else:
                configs[name] = trial.suggest_float(name,
                                                    float(bounds["min"]),
                                                    float(bounds["max"]),
                                                    log=log)
        else:
            configs[name] = _cast_value(bounds["value"])

    return configs
