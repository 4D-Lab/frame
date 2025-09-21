import random

import torch
import numpy as np
from sklearn import metrics
import torch.backends.cudnn as cudnn


random.seed(8)
np.random.seed(8)
torch.manual_seed(8)
if torch.cuda.is_available():
    torch.cuda.manual_seed(8)
    torch.cuda.manual_seed_all(8)
cudnn.deterministic = True
cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, optim, scheduler, lossfn, loader):
    step = 1
    running_loss = 0.0

    model = model.train()
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad(batch)

        # * Make predictions
        out = model(x=batch.x.float(),
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr.float(),
                    batch=batch.batch)

        # * Compute loss
        true = batch.y.float()
        loss = lossfn(torch.squeeze(out), torch.squeeze(true))
        loss.backward()

        # * Update gradients
        optim.step()

        # * Update tracking
        running_loss += loss.detach().item()
        step += 1

    scheduler.step()
    return running_loss / step


@torch.no_grad()
def test_epoch(model, loader):
    model.eval()

    true = []
    pred = []
    label = []

    for batch in loader:
        batch = batch.to(device)

        # * Make predictions
        out = model(x=batch.x.float(),
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr.float(),
                    batch=batch.batch)

        # * Read prediction values
        detach = torch.sigmoid(out).cpu().detach()
        discretized = (detach >= 0.5).int()
        batch_pred = list(torch.ravel(detach).cpu().detach().numpy())
        batch_label = list(torch.ravel(discretized).cpu().detach().numpy())

        batch_true = list(torch.ravel(batch.y).cpu().detach().numpy())

        true = true + batch_true
        pred = pred + batch_pred
        label = label + batch_label

    # * Get metrics
    acc = metrics.accuracy_score(true, label)
    acc_bal = metrics.balanced_accuracy_score(true, label)
    f1 = metrics.f1_score(true, label, zero_division=0)
    prec = metrics.precision_score(true, label, zero_division=0)
    rec = metrics.recall_score(true, label, zero_division=0)
    mcc = metrics.matthews_corrcoef(true, label)
    roc_auc = metrics.roc_auc_score(true, pred)
    avg_prec = metrics.average_precision_score(true, pred)

    result = {"acc": round(acc, 3),
              "acc_bal": round(acc_bal, 3),
              "f1": round(f1, 3),
              "prec": round(prec, 3),
              "rec": round(rec, 3),
              "mcc": round(mcc, 3),
              "avg_prec": round(avg_prec, 3),
              "roc_auc": round(roc_auc, 3)}

    return result
