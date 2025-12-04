
import os
import shutil
from copy import deepcopy
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from train import MetricTracker
from utils import load_pickle, save_pickle


matplotlib.use('Agg')

def get_model(
        model_class, model_kwargs, dataset, prefix, val_dataset=None,
        epochs=None, device='cuda', load_saved=False, **kwargs):
    checkpoint_file = prefix + 'model.ckpt'
    history_file = prefix + 'history.pickle'

    # load model if exists
    if load_saved and os.path.exists(checkpoint_file):
        model = model_class.load_from_checkpoint(checkpoint_file)
        print(f'Model loaded from {checkpoint_file}')
        history = load_pickle(history_file)
    else:
        model = None
        history = []

    # train model
    if (epochs is not None) and (epochs > 0):
        model, hist, best_ckpt_path = train_model(
            model=model,
            model_class=model_class, model_kwargs=model_kwargs,
            dataset=dataset, epochs=epochs, device=device,val_dataset = val_dataset,
            prefix=prefix, **kwargs)
        
        # Prefer the BEST checkpoint if available; otherwise save the last-epoch weights
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            shutil.copy2(best_ckpt_path, checkpoint_file)
            print(f'Best model copied to {checkpoint_file}')
        else:
            # Fallback: serialize the current (last-epoch) model
            tmp_trainer = pl.Trainer(logger=False, enable_checkpointing=False)
            tmp_trainer.save_checkpoint(checkpoint_file)
            print(f'Last-epoch model saved to {checkpoint_file}')

        history += hist
        save_pickle(history, history_file)
        print(f'History saved to {history_file}')
        plot_history(history, prefix)

    return model


def train_model(
        dataset, batch_size, epochs,val_dataset,prefix,
        model=None, model_class=None, model_kwargs={},
        device='cuda', 
        monitor_metric = "loss_val",   # <— what to watch
        early_stop_patience=None,                 # <— early stop patience (epochs)
        min_delta=0.0):
    if model is None:
        model = model_class(**model_kwargs)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
    
    tracker = MetricTracker()
    device_accelerator_dict = {'cuda': 'gpu', 'cpu': 'cpu'}
    accelerator = device_accelerator_dict[device]

    # Save the BEST checkpoint (by validation loss)  
    ckpt_dir = os.path.join(os.path.dirname(prefix), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Use to save model every 10 epoch, only for test purpose 
    # checkpoint_callback = ModelCheckpoint(
    #         dirpath=os.path.join(os.path.dirname(prefix), "checkpoints"),  # directory to save
    #         filename="epoch-{epoch:02d}",     # filename format
    #         save_top_k=-1,                    # save all checkpoints
    #         every_n_epochs=10
    # )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{"+monitor_metric+":.4f}-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    callbacks = [tracker, checkpoint_callback]

    # Add EarlyStopping only if patience is provided and we have validation data
    if early_stop_patience is not None and val_loader is not None:
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            mode="min",
            patience=early_stop_patience,
            min_delta=min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            deterministic=True,
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True)
    model.train()
    t0 = time()
    trainer.fit(model, train_loader, val_loader)
    print(int(time() - t0), 'sec')

    tracker.clean()
    history = tracker.collection

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")
        model = model.__class__.load_from_checkpoint(best_ckpt_path)

    return model, history, best_ckpt_path


def plot_history(history, prefix):
    if not history:
        print("History is empty, no plot created.")
        return

    metrics = history[0].keys()
    metric_pairs = {}

    # Match _train and _val metrics into pairs
    for m in metrics:
        if m.endswith('_train'):
            base = m[:-6]  # remove "_train"
            metric_pairs[base] = {'train': m}
        elif m.endswith('_val'):
            base = m[:-4]  # remove "_val"
            if base not in metric_pairs:
                metric_pairs[base] = {}
            metric_pairs[base]['val'] = m
    # Store CSV data here
    csv_data = {}
    
    # Plot all matched pairs
    for base, pair in metric_pairs.items():
        plt.figure(figsize=(10, 5))
        for kind, name in pair.items():
            values = np.array([entry[name] for entry in history])
            linestyle = '-' 
            plt.plot(values, label=f"{base}_{kind}", linestyle=linestyle)
            # Save to CSV data
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
    
    # Save all collected values to CSV
    df = pd.DataFrame(csv_data)
    df.index.name = "epoch"
    csv_file = f"{prefix}metrics_history.csv"
    df.to_csv(csv_file)
    print(f"Saved train and loss metrics to: {csv_file}")