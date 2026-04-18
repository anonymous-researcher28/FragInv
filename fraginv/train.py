from .models.CrippenNet import CrippenNet, zinc_PARAMS
from .custom_sampler import SubsetWeightedRandomSampler
import torch
from torch import optim, nn
from torch.utils.data import Subset
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import curve_fit
import pickle
from tqdm import tqdm
import torch.nn.functional as F 
from .custom_stream import make_stream_loader
import torch
import torch.nn.functional as F


def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))




def nudge(atom_fea, adj, noise_factor):
    atom_fea = atom_fea + torch.randn(*atom_fea.shape, device=atom_fea.device)*noise_factor
    adj = adj + torch.randn(*adj.shape, device=atom_fea.device)*noise_factor
    return atom_fea, adj

def shuffle(t):
    idx = torch.randperm(t.nelement())
    t = t.view(-1)[idx].view(t.size())
    return t

def prepare_data(data, N, extra_fea_matrix):
    device = data.x.device
    n_onehot = len(extra_fea_matrix)
    atom_fea = torch.zeros(N, n_onehot+1, device=device)
    atom_fea[:data.x.shape[0],:n_onehot] = 1*data.x[:,:n_onehot]
    atom_fea[data.x.shape[0]:, n_onehot] = 1
    atom_fea = add_extra_features(atom_fea, extra_fea_matrix)
    atom_fea = atom_fea.unsqueeze(0)

    adj = torch.zeros(1, N, N, device=device)
    for n,(i,j) in enumerate(data.edge_index.T):
        adj[0,i,j] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
        adj[0,j,i] = data.edge_attr[n,:].matmul(torch.tensor([1,2,3,1.5], device=device))
    return atom_fea, adj



def prepare_target_vector(data, N):
    target = data.atom_class
    new_target = torch.zeros(data.num_graphs, N, target.shape[1], device=target.device)
    switch_points = torch.where(data.batch[1:] - data.batch[:-1] == 1)[0] + 1
    new_target_pieces = torch.tensor_split(target, switch_points.cpu())
    for i, ten in enumerate(new_target_pieces):
        new_target[i,:ten.shape[0],:] = ten
    return new_target



def prepare_data_vector(data, N, extra_fea_matrix, shuffle=False):
    n_onehot = len(extra_fea_matrix)
    atom_fea = 1 * data.x[:, :n_onehot]
    atom_fea = torch.cat([atom_fea, torch.zeros(atom_fea.shape[0], 1, device=atom_fea.device)], dim=1)
    atom_fea = add_extra_features(atom_fea, extra_fea_matrix)
    new_atom_fea = torch.zeros(data.num_graphs, N, atom_fea.shape[1], device=atom_fea.device)

    switch_points = torch.where(data.batch[1:] - data.batch[:-1] == 1)[0] + 1
    switch_points = switch_points.to(atom_fea.device)
    cuts = torch.cat([torch.tensor([0], device=atom_fea.device), switch_points,
                      torch.tensor([atom_fea.shape[0]], device=atom_fea.device)])
    graph_sizes = cuts[1:] - cuts[:-1]  # (B,)

    if shuffle:
        p = torch.randperm(N, device=atom_fea.device)
    else:
        p = torch.arange(N, device=atom_fea.device)

    pieces = torch.tensor_split(atom_fea, switch_points.cpu())
    for i, ten in enumerate(pieces):
        n_i = ten.shape[0]
        new_atom_fea[i, p[:n_i], :] = ten
        if n_i < N:
            new_atom_fea[i, p[n_i:], n_onehot] = 1  # padding marker

    adj = torch.zeros(data.num_graphs, N, N, device=atom_fea.device)
    bond_type = torch.tensor([1, 2, 3, 1.5], device=atom_fea.device)

    nn = data.batch[data.edge_index[0]]  # (E,)
    starts = cuts[nn]
    ij = data.edge_index.T - starts.unsqueeze(1)  # (E, 2) local indices

    valid = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < N) & (ij[:, 1] < N)
    n_i = graph_sizes[nn]
    valid = valid & (ij[:, 0] < n_i) & (ij[:, 1] < n_i)

    if valid.any():
        ij = ij[valid]
        nn_v = nn[valid]
        w = data.edge_attr[valid].matmul(bond_type)  # (E_valid,)
        adj[nn_v, p[ij[:, 0]], p[ij[:, 1]]] = w
        adj[nn_v, p[ij[:, 1]], p[ij[:, 0]]] = w

    return new_atom_fea, adj

def class_stats(loader_train, show=False, device="cpu"):
    class_sum = []
    for batch_idx, data in enumerate(tqdm(loader_train)):
        data = data.to(device=device)
        class_sum.append(torch.sum(data.atom_class,dim=0).unsqueeze(0))
    class_sum = torch.cat(class_sum, dim=0)
    class_sum = torch.sum(class_sum, dim=0)

    print("Number of atom for each class:", class_sum)
    print("Most common class: %1.10f, %d"%((class_sum/torch.sum(class_sum)).max(), class_sum.max()), class_sum.argmax())
    print("Least common class: %1.10f, %d"%((class_sum/torch.sum(class_sum)).min(), class_sum.min()), class_sum.argmin())
    print("Least common non-zero class: %1.10f, %d"%((class_sum[class_sum > 0]/torch.sum(class_sum)).min(), class_sum[class_sum > 0].min()), np.arange(len(class_sum))[class_sum > 0][class_sum[class_sum > 0].argmin()])
    print("Total number of atoms:", torch.sum(class_sum))
    if show:
        plt.figure()
        plt.bar(torch.arange(len(class_sum)), class_sum, color="navy")
        ax = plt.gca()
        ax.set_yscale('log')
        plt.xticks(np.array([28, 41, 56, 57, 62, 65, 66, 67, 68, 69])-0.5)
        ax.tick_params(labelbottom=False)
        plt.show()
    weights = 1/(class_sum + 1)
    return weights


def _pair_prior_loss(motif_pairs, motif_types, motif_res, weight=0.5, eps=1e-9, device="cpu"):
    if motif_res is None or weight <= 0 or not motif_pairs:
        return torch.zeros((), device=device)
    priors = []
    for (u, v) in motif_pairs:
        ma = motif_types[u]
        mb = motif_types[v]
        p = motif_res.pair_prior(ma, mb)
        priors.append(p)
    if not priors:
        return torch.zeros((), device=device)
    priors = torch.tensor(priors, dtype=torch.float32, device=device)
    return weight * (-torch.log(priors.clamp_min(eps)).mean())

def _capacity_loss(motif_degrees, motif_types, motif_res, weight=0.5, device="cpu"):
    if motif_res is None or weight <= 0 or not motif_degrees:
        return torch.zeros((), device=device)
    over = []
    for deg, m in zip(motif_degrees, motif_types):
        cap = motif_res.max_degree(m)
        if deg > cap:
            over.append(deg - cap)
    if not over:
        return torch.zeros((), device=device)
    return weight * torch.tensor(over, dtype=torch.float32, device=device).mean()




import numpy as np 



def prepare_data_vector(data, N, extra_fea_matrix, shuffle=False):
    n_onehot = len(extra_fea_matrix)
    atom_fea = 1 * data.x[:, :n_onehot]
    atom_fea = torch.cat([atom_fea, torch.zeros(atom_fea.shape[0], 1, device=atom_fea.device)], dim=1)
    atom_fea = add_extra_features(atom_fea, extra_fea_matrix)
    new_atom_fea = torch.zeros(data.num_graphs, N, atom_fea.shape[1], device=atom_fea.device)
    switch_points = torch.where(data.batch[1:] - data.batch[:-1] == 1)[0] + 1
    switch_points = switch_points.to(atom_fea.device)
    pieces = torch.tensor_split(atom_fea, switch_points.cpu())

    if shuffle:
        p = torch.randperm(N, device=atom_fea.device)
    else:
        p = torch.arange(N, device=atom_fea.device)

    # place node features graph by graph
    for i, ten in enumerate(pieces):
        n_i = ten.shape[0]
        new_atom_fea[i, p[:n_i], :] = ten
        if n_i < N:
            new_atom_fea[i, p[n_i:], n_onehot] = 1  # padding marker

    adj = torch.zeros(data.num_graphs, N, N, device=atom_fea.device)
    bond_type = torch.tensor([1, 2, 3, 1.5], device=atom_fea.device)

    nn = data.batch[data.edge_index[0]]  # (E,)
    cuts = torch.cat([torch.tensor([0], device=atom_fea.device),
                      switch_points,
                      torch.tensor([atom_fea.shape[0]], device=atom_fea.device)])
    starts = cuts[nn]
    ij_local = data.edge_index.T - torch.stack([starts, starts], dim=1)
    adj[nn, p[ij_local[:,0]], p[ij_local[:,1]]] = data.edge_attr.matmul(bond_type)

    return new_atom_fea, adj
from .custom_stream import SMILESStream
from torch_geometric.loader import DataLoader

def _make_loader_from_config(split_cfg, config, shuffle=True):
    ds = SMILESStream(
        path=split_cfg["path"],
        smiles_col=split_cfg.get("smiles_col"),
        type_list=config.type_list,
        shuffle_buffer=split_cfg.get("shuffle_buffer", 0 if not shuffle else 4096),
        target=split_cfg.get("target", "logp"),
        max_atoms=split_cfg.get("max_atoms")  # optional cap
    )
    loader = DataLoader(ds,
                        batch_size=split_cfg.get("batch_size", 64),
                        num_workers=split_cfg.get("num_workers", 0),
                        persistent_workers=split_cfg.get("num_workers", 0) > 0)
    steps_per_epoch = split_cfg.get("steps_per_epoch", 2000)  # you can tune this
    return loader, steps_per_epoch
def build_stream_loaders(config):
    train_cfg = config.stream["train"]
    val_cfg   = config.stream["val"]

    train_loader = make_stream_loader(train_cfg["path"], train_cfg, shuffle=True)
    val_loader   = make_stream_loader(val_cfg["path"],   val_cfg,   shuffle=False)

    train_steps = int(train_cfg.get("steps_per_epoch", 1500))
    val_steps   = int(val_cfg.get("steps_per_epoch", 200))
    return train_loader, val_loader, train_steps, val_steps


def train_epoch_stream(model, optimizer, device, train_loader, steps_per_epoch, prepare_batch_fn=None, config=None):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in train_loader:
        batch = batch.to(device)
        if prepare_batch_fn is None:
            # Default: model(batch) must return prediction shaped like batch.y
            pred = model(batch)
            target = batch.y.to(device)
        else:
            inputs, target = prepare_batch_fn(batch, config)
            if isinstance(inputs, (list, tuple)):
                pred = model(*inputs)
            else:
                pred = model(inputs)

        loss = F.l1_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1
        if steps >= steps_per_epoch:
            break

    return total_loss / max(steps, 1)


@torch.no_grad()
def eval_epoch_stream(model, device, val_loader, steps_per_epoch, prepare_batch_fn=None, config=None):
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in val_loader:
        batch = batch.to(device)

        if prepare_batch_fn is None:
            pred = model(batch)
            target = batch.y.to(device)
        else:
            inputs, target = prepare_batch_fn(batch, config)
            if isinstance(inputs, (list, tuple)):
                pred = model(*inputs)
            else:
                pred = model(inputs)

        loss = F.l1_loss(pred, target)
        total_loss += float(loss.detach().cpu())
        steps += 1
        if steps >= steps_per_epoch:
            break

    return total_loss / max(steps, 1)
def _crippen_prepare_batch(batch, config):
    X, A = prepare_data_vector(
        data=batch,
        N=config.max_size,
        extra_fea_matrix=config._extra_fea_matrix,
        shuffle=getattr(config, "shuffle", False),
    )
    y = batch.y.reshape(-1, 1)
    return (X, A), y
def build_stream_loaders(config):

    # --- helpers ---
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        try:
            return dict(vars(x))
        except Exception:
            return {}

    def _get(d, key, default=None):
        if isinstance(d, dict):
            return d.get(key, default)
        return getattr(d, key, default)

    stream_cfg = _as_dict(getattr(config, "stream", {}))

    train_cfg = _as_dict(stream_cfg.get("train", _get(config.stream, "train", {})))
    val_cfg   = _as_dict(stream_cfg.get("val",   _get(config.stream, "val",   {})))

    if "path" not in train_cfg or "path" not in val_cfg:
        raise ValueError("config.stream.train.path and config.stream.val.path must be set")

    train_loader = make_stream_loader(train_cfg["path"], train_cfg, shuffle=True)
    val_loader   = make_stream_loader(val_cfg["path"],   val_cfg,   shuffle=False)

    train_steps = int(train_cfg.get("steps_per_epoch", 1500))
    val_steps   = int(val_cfg.get("steps_per_epoch", 200))

    return train_loader, val_loader, train_steps, val_steps
def add_extra_features(features: torch.Tensor, extra_fea_matrix: torch.Tensor) -> torch.Tensor:

    if extra_fea_matrix is None or extra_fea_matrix.numel() == 0:
        return features

    extra_fea_matrix = extra_fea_matrix.to(device=features.device, dtype=features.dtype)

    n_feat = features.size(1)
    n_types, n_extra = extra_fea_matrix.shape

    n_onehot = min(n_feat, n_types)
    if n_onehot == 0:
        return features  # nothing to multiply

    base = features[:, :n_onehot]                 # (N, n_onehot)
    efm  = extra_fea_matrix[:n_onehot, :]         # (n_onehot, n_extra)

    extras = base.matmul(efm)                     # (N, n_extra)
    return torch.cat([features, extras], dim=1)


import torch
from torch import nn
from torch.utils.data import DataLoader

class PropertyPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        # Example: simple MLP or GNN readout on top of a graph encoder
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # e.g. predict logP or activity
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [batch_size]


def train_predictor(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    num_epochs: int,
    device: torch.device,
    ckpt_path: str = "predictor.pt"
):
    criterion = nn.MSELoss()  # or BCEWithLogitsLoss / etc.
    model.to(device)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # adapt to your dataset:
            # batch_x: FP or graph embedding
            # batch_y: target property
            batch_x = batch["x"].to(device)  # shape [B, in_dim]
            batch_y = batch["y"].to(device).float()

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch["x"].to(device)
                batch_y = batch["y"].to(device).float()

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f"[Predictor] Epoch {epoch}/{num_epochs} "
              f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best, saved to {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def train(config, output, motif_res=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output, exist_ok=True)

    mdl_name = getattr(config, "model", "CrippenNet")
    def _build_model():
        # Handle extra feature matrix dimensions safely
        n_onehot = 0
        n_extra = 0
        if getattr(config, "_extra_fea_matrix", None) is not None:
            shp = getattr(config._extra_fea_matrix, "shape", None)
            if shp is not None and len(shp) >= 2:
                n_onehot, n_extra = shp[0], shp[1]
            elif shp is not None and len(shp) == 1:
                n_onehot = shp[0]
                n_extra = 1  # treat 1-D as one extra channel
        pad_col = 1  
        in_dim = n_onehot + n_extra + pad_col

        pred_kind = getattr(config, "predictor", "atomic")
        return CrippenNet(
                orig_atom_fea_len=n_onehot,
                n_conv=getattr(config, "n_conv", 3),
                layer_list=getattr(config, "layer_list", [256, 256]),
                classifier=getattr(config, "atom_class", False),
            ).to(device)

    model = _build_model()
    if getattr(config, "use_pretrained", False) or getattr(config, "transfer_learn", False):
        # Prefer ckpt_path from YAML if present
        ckpt_path = getattr(config, "ckpt_path", None)
        if ckpt_path is None:
            # fallback: assume standard name under the current output
            ckpt_path = os.path.join(output, "model_weights.pth")

        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[train] Loaded pretrained weights: {ckpt_path}", flush=True)
        else:
            raise RuntimeError(f"[train] use_pretrained=True but checkpoint not found at {ckpt_path}")
    if not getattr(config, "streaming", False):
        # For generation configs, we only want a pretrained predictor, no extra training
        print("[train] streaming=False and use_pretrained=True -> returning loaded model (no training).",
              flush=True)
        return model
    lr = getattr(config, "learning_rate", getattr(config, "lr", 1e-3))
    wd = getattr(config, "weight_decay", 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_weights = getattr(config, "loss_weights", {})
    if not isinstance(loss_weights, dict):
        loss_weights = {k: getattr(loss_weights, k) for k in ["pair_prior", "capacity"] if hasattr(loss_weights, k)}
    w_pair = float(loss_weights.get("pair_prior", 0.0))
    w_cap = float(loss_weights.get("capacity", 0.0))

    prepare_batch_fn = getattr(config, "prepare_batch_fn", None)
    if mdl_name == "CrippenNet" and prepare_batch_fn is None:
        prepare_batch_fn = _crippen_prepare_batch

    train_loader, val_loader, train_steps, val_steps = build_stream_loaders(config)

    best_val = float("inf")
    best_path = os.path.join(output, "model_weights_stream_best.pth")
    last_path = os.path.join(output, "model_weights_stream_last.pth")

    num_epochs = int(getattr(config, "num_epochs", 30))
    print(f"[train] Streaming ON — epochs={num_epochs}, steps/epoch: train={train_steps}, val={val_steps}",
          flush=True)

    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = batch.to(device)

            # Forward with optional tensorizer
            if prepare_batch_fn is None:
                pred = model(batch)                 
                target = batch.y.to(device)
            else:
                inputs, target = prepare_batch_fn(batch, config)
                pred = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)

            # Base regression loss
            loss = F.l1_loss(pred, target)

            # Optional motif regularizers if data provides lists (safe no-op otherwise)
            if hasattr(batch, "motif_types_list") and motif_res is not None:
                reg_pair = torch.zeros((), device=device)
                reg_cap = torch.zeros((), device=device)
                for types_i, edges_i, degs_i in zip(
                    getattr(batch, "motif_types_list", []),
                    getattr(batch, "motif_edges_list", []),
                    getattr(batch, "motif_degrees_list", []),
                ):
                    if w_pair and types_i and edges_i:
                        reg_pair += _pair_prior_loss(edges_i, types_i, motif_res, weight=w_pair, device=device)
                    if w_cap and types_i and degs_i:
                        reg_cap += _capacity_loss(degs_i, types_i, motif_res, weight=w_cap, device=device)
                loss = loss + reg_pair + reg_cap

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.detach().cpu())
            steps += 1
            if steps >= train_steps:
                break

        tr_loss /= max(steps, 1)
        va_loss = eval_epoch_stream(
            model=model,
            device=device,
            val_loader=val_loader,
            steps_per_epoch=val_steps,
            prepare_batch_fn=prepare_batch_fn,
            config=config,
        )

        print(f"epoch:{epoch+1:03d}  train_L1:{tr_loss:.4f}  val_L1:{va_loss:.4f}", flush=True)

        # Save checkpoints
        torch.save(model.state_dict(), last_path)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)

    print(f"[train] Done. Best val L1={best_val:.4f}. Saved best to {best_path}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model
