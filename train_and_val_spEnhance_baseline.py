import os
import shutil
from copy import deepcopy
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from utils import load_pickle, save_pickle


matplotlib.use("Agg")


class MetricTracker(pl.Callback):
    def __init__(self):
        self.collection = []
        self._current_epoch = None

    def _serialize_metrics(self, metrics):
        out = {}
        for key, value in metrics.items():
            out[key] = value.item() if isinstance(value, torch.Tensor) else value
        return out

    def _update_epoch_metrics(self, trainer):
        metrics = {}
        metrics.update(self._serialize_metrics(deepcopy(trainer.logged_metrics)))
        metrics.update(self._serialize_metrics(deepcopy(trainer.callback_metrics)))
        epoch = trainer.current_epoch
        if self._current_epoch != epoch or not self.collection:
            self.collection.append(metrics)
            self._current_epoch = epoch
        else:
            self.collection[-1].update(metrics)

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        self._update_epoch_metrics(trainer)

    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        self._update_epoch_metrics(trainer)

    def clean(self):
        keys = set().union(*[set(entry.keys()) for entry in self.collection]) if self.collection else set()
        for entry in self.collection:
            for key in keys:
                if key in entry:
                    if isinstance(entry[key], torch.Tensor):
                        entry[key] = entry[key].item()
                else:
                    entry[key] = float("nan")


def get_model(
    model_class,
    model_kwargs,
    dataset,
    prefix,
    val_dataset=None,
    epochs=None,
    device="cuda",
    load_saved=False,
    **kwargs,
):
    checkpoint_file = prefix + "model.ckpt"
    history_file = prefix + "history.pickle"

    if load_saved and os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f"Model loaded from {checkpoint_file}")
        history = load_pickle(history_file)
    else:
        model = None
        history = []

    if (epochs is not None) and (epochs > 0):
        model, hist, best_ckpt_path = train_model(
            model=model,
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset=dataset,
            epochs=epochs,
            device=device,
            val_dataset=val_dataset,
            prefix=prefix,
            **kwargs,
        )

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            shutil.copy2(best_ckpt_path, checkpoint_file)
            print(f"Best model copied to {checkpoint_file}")
        else:
            tmp_trainer = pl.Trainer(logger=False, enable_checkpointing=False)
            tmp_trainer.save_checkpoint(checkpoint_file)
            print(f"Last-epoch model saved to {checkpoint_file}")

        history += hist
        save_pickle(history, history_file)
        print(f"History saved to {history_file}")
        plot_history(history, prefix)

    return model


def train_model(
    dataset,
    batch_size,
    epochs,
    val_dataset,
    prefix,
    model=None,
    model_class=None,
    model_kwargs=None,
    device="cuda",
    monitor_metric="loss_val",
    early_stop_patience=None,
    min_delta=0.0,
    collate_fn=None,
):
    if model_kwargs is None:
        model_kwargs = {}
    if model is None:
        model = model_class(**model_kwargs)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn) if val_dataset is not None else None

    tracker = MetricTracker()
    accelerator = {"cuda": "gpu", "cpu": "cpu"}[device]

    ckpt_dir = os.path.join(os.path.dirname(prefix), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{" + monitor_metric + ":.4f}-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [tracker, checkpoint_callback]
    if early_stop_patience is not None and val_loader is not None:
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                mode="min",
                patience=early_stop_patience,
                min_delta=min_delta,
                verbose=True,
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        deterministic=True,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    model.train()
    t0 = time()
    trainer.fit(model, train_loader, val_loader)
    print(int(time() - t0), "sec")

    tracker.clean()
    history = tracker.collection

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")
        model = model.__class__.load_from_checkpoint(best_ckpt_path)
    else:
        best_ckpt_path = checkpoint_callback.last_model_path
        if best_ckpt_path:
            print(f"Last checkpoint: {best_ckpt_path}")
            model = model.__class__.load_from_checkpoint(best_ckpt_path)

    return model, history, best_ckpt_path


def plot_history(history, prefix):
    if not history:
        print("History is empty, no plot created.")
        return

    metrics = sorted(set().union(*[entry.keys() for entry in history]))
    metric_pairs = {}

    for metric in metrics:
        if metric.endswith("_train"):
            metric_pairs[metric[:-6]] = {"train": metric}
        elif metric.endswith("_val"):
            base = metric[:-4]
            if base not in metric_pairs:
                metric_pairs[base] = {}
            metric_pairs[base]["val"] = metric

    csv_data = {}
    for base, pair in metric_pairs.items():
        plt.figure(figsize=(10, 5))
        for kind, name in pair.items():
            values = np.array([entry.get(name, np.nan) for entry in history], dtype=float)
            plt.plot(values, label=f"{base}_{kind}", linestyle="-")
            csv_data[f"{base}_{kind}"] = values
        plt.title(f"{base.capitalize()} (Train vs Val)")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        outfile = f"{prefix}{base}_combined_history.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(outfile)

    df = pd.DataFrame(csv_data)
    df.index.name = "epoch"
    csv_file = f"{prefix}metrics_history.csv"
    df.to_csv(csv_file)
    print(f"Saved train and loss metrics to: {csv_file}")
