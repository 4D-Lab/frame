import torch
from placeholder.source import train
from placeholder.source.models import pyg_models

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_setup(model_name, config):
    bce_weight = config["bce_weight"]

    if model_name == "attentive":
        model = pyg_models.GNN_GCN(config).to(device)
    elif model_name == "gat":
        model = pyg_models.GNN_GAT(config).to(device)
    elif model_name == "sage":
        model = pyg_models.GNN_SAGE(config).to(device)
    elif model_name == "gin":
        model = pyg_models.GNN_GIN(config).to(device)
    else:
        raise NotImplementedError("Model not available")

    base_optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config["learning_rate"],
                                      betas=(config["beta_1"],
                                             config["beta_2"]),
                                      eps=config["eps"],
                                      weight_decay=config["weight_decay"])
    optimizer = train.Lookahead(base_optimizer, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=100,
                                                           eta_min=1e-6)

    lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=bce_weight).to(device)

    return model, optimizer, scheduler, lossfn


def optuna_suggest(params, trial):
    configs = {}

    for name, bounds in params["Tune"].items():
        if "min" in bounds:
            if isinstance(bounds["max"], int):
                configs[name] = trial.suggest_int(name, bounds["min"],
                                                  bounds["max"])
            else:
                configs[name] = trial.suggest_float(name, float(bounds["min"]),
                                                    float(bounds["max"]))
        else:
            if isinstance(bounds["value"], int):
                configs[name] = int(bounds["value"])
            else:
                configs[name] = float(bounds["value"])

    return configs
