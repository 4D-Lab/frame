import torch
from frame.source import train
from frame.source.models import pyg_models

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_setup(model_name, config):
    task = config["task"]
    model = select_model(model_name, config)

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
