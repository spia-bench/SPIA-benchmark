"""
Longformer Model Training for TAB Dataset

Train a Longformer-based NER model for text anonymization.

Usage:
    python src/baseline/TAB/longformer_experiments/train_model.py --epochs 2 --output long_model.pt
    python src/baseline/TAB/longformer_experiments/train_model.py --epochs 5 --lr 2e-5 --output model.pt
"""

from typing_extensions import TypedDict, OrderedDict
import torch.nn.functional as F
from typing import List, Any
from transformers import LongformerTokenizerFast
from tokenizers import Encoding
import itertools
from torch import nn
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data_handling import *
from longformer_model import Model
from data_manipulation import training_raw, dev_raw, test_raw
import collections
import random
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Longformer model for TAB dataset NER task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_model.py --epochs 2 --output long_model.pt
    python train_model.py --epochs 5 --lr 3e-5 --output model.pt
        """
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="long_model.pt",
        help="Output model file path (default: long_model.pt)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--tokens_per_batch",
        type=int,
        default=4096,
        help="Max tokens per batch (default: 4096)"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    bert = "allenai/longformer-base-4096"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set = LabelSet(labels=["MASK"])

    training = Dataset(data=training_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=args.tokens_per_batch)
    dev = Dataset(data=dev_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=args.tokens_per_batch)
    test = Dataset(data=test_raw, tokenizer=tokenizer, label_set=label_set, tokens_per_batch=args.tokens_per_batch)

    trainloader = DataLoader(training, collate_fn=TrainingBatch, batch_size=args.batch_size, shuffle=True)
    devloader = DataLoader(dev, collate_fn=TrainingBatch, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test, collate_fn=TrainingBatch, batch_size=args.batch_size)

    model = Model(model=bert, num_labels=len(training.label_set.ids_to_label.values()))
    model = model.to(device)

    if device == 'cuda':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]))

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    print(f"\n{'='*60}")
    print(f"Longformer Training")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    total_val_loss = 0
    total_train_loss, epochs_list = [], []
    for epoch in range(args.epochs):
        epochs_list.append(epoch)
        model.train()
        for X in tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            y = X['labels']
            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        total_train_loss.append(loss.item())
        print(f'Epoch: {epoch + 1}')
        print(f'Training loss: {loss.item():.2f}')

    # Validation
    predictions, true_labels, offsets = [], [], []
    inputs, test_pred, test_true, offsets = [], [], [], []
    for X in tqdm.tqdm(devloader, desc="Validation"):
        model.eval()
        with torch.no_grad():
            y = X['labels']
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            val_loss = criterion(y_pred, y)
            pred = y_pred.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred])
            true_labels.extend(list(p) for p in true)
            total_val_loss += val_loss.item()

    avg_loss = total_val_loss / len(devloader)
    print(f'Validation loss: {avg_loss:.2f}')

    out = []
    # Getting entity level predictions
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i])-count)]
            predictions[i] = predictions[i][:len(offsets[i])]

    l1 = [item for sublist in predictions for item in sublist]
    l2 = [item for sublist in offsets for item in sublist]

    it = enumerate(l1+[0])
    sv = 0

    try:
        while True:
            if sv == 1:
                fi, fv = si, sv
            else:
                while True:
                    fi, fv = next(it)
                    if fv:
                        break
            while True:
                si, sv = next(it)
                if sv == 0 or sv == 1:
                    break
            out.append((l2[fi][0], l2[fi][1], l2[si-1][2]))

    except StopIteration:
        pass

    d = {}
    for i in out:
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1], i[2]))
        else:
            d[i[0]].append((i[1], i[2]))

    # Filter
    out_dev = {}
    for i in d:
        out_dev[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_dev[i] = d[i]

    with open("preds_dev.json", "w") as f:
        json.dump(out_dev, f)

    # Test evaluation
    predictions, true_labels, offsets = [], [], []
    model.eval()
    for X in tqdm.tqdm(testloader, desc="Test"):
        with torch.no_grad():
            y = X['labels']
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            pred = y_pred.argmax(dim=1).cpu().numpy()
            true = y.cpu().numpy()
            offsets.extend(X['offsets'])
            predictions.extend([list(p) for p in pred])
            true_labels.extend(list(p) for p in true)

    out = []
    for i in range(len(offsets)):
        if -1 in offsets[i]:
            count = offsets[i].count(-1)
            offsets[i] = offsets[i][:(len(offsets[i])-count)]
            predictions[i] = predictions[i][:len(offsets[i])]

    l1 = [item for sublist in predictions for item in sublist]
    l2 = [item for sublist in offsets for item in sublist]

    it = enumerate(l1+[0])
    sv = 0

    try:
        while True:
            if sv == 1:
                fi, fv = si, sv
            else:
                while True:
                    fi, fv = next(it)
                    if fv:
                        break
            while True:
                si, sv = next(it)
                if sv == 0 or sv == 1:
                    break
            out.append((l2[fi][0], l2[fi][1], l2[si-1][2]))

    except StopIteration:
        pass

    d = {}
    for i in out:
        if i[0] not in d:
            d[i[0]] = []
            d[i[0]].append((i[1], i[2]))
        else:
            d[i[0]].append((i[1], i[2]))

    # Filter
    out_test = {}
    for i in d:
        out_test[i] = []
        d[i] = list(map(list, OrderedDict.fromkeys(map(tuple, d[i])).keys()))
        out_test[i] = d[i]

    with open("preds_test.json", "w") as f:
        json.dump(out_test, f)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"\nModel saved to: {args.output}")
    print("Training completed!")


if __name__ == "__main__":
    main()
