import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collator import collator
import random
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from model_new import Graphormer
from lr import PolynomialDecayLR
import argparse
import math
import os
import shutil
import pandas as pd
from tqdm import tqdm


def train(args, model, device, loader, optimizer, lr_scheduler):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y_true = batch.y.view(-1)
        loss = F.nll_loss(pred, y_true)
        '''
        for i in torch.split(pred, args.num_data_augment, dim=0):
            loss += 0.05*torch.norm(i-i.mean(dim=0))/args.num_data_augment
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y)
        y_pred.append(pred)

    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    pred_list = []
    for i in torch.split(y_pred, args.num_data_augment, dim=0):
        pred_list.append(i.mean(dim=0).argmax().unsqueeze(0))
    pred = torch.cat(pred_list)
    y_true = y_true.view(-1, args.num_data_augment)[:, 0]
    correct = (pred == y_true).sum()
    acc = correct.item() / len(pred)

    return acc


def init_params(module, n_layers=6):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def random_split(data_list, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(len(data_list))
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * len(data_list))]
    val_idx = all_idx[int(frac_train * len(data_list)):int((frac_train+frac_valid) * len(data_list))]
    test_idx = all_idx[int((frac_train+frac_valid) * len(data_list)):]
    train_list = []
    test_list = []
    val_list = []
    for i in train_idx:
        train_list.append(data_list[i])
    for i in val_idx:
        val_list.append(data_list[i])
    for i in test_idx:
        test_list.append(data_list[i])
    return train_list, val_list, test_list


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of graph transformer')
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.3)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--warmup_updates', type=int, default=60000)
    parser.add_argument('--tot_updates', type=int, default=1000000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--peak_lr', type=float, default=2e-4)
    parser.add_argument('--end_lr', type=float, default=1e-9)
    parser.add_argument('--edge_type', type=str, default='multi_hop')
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--flag', action='store_true')
    parser.add_argument('--flag_m', type=int, default=3)
    parser.add_argument('--flag_step_size', type=float, default=1e-3)
    parser.add_argument('--flag_mag', type=float, default=1e-3)
    parser.add_argument('--num_data_augment', type=int, default=16)
    parser.add_argument('--num_global_node', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--eval_train', type=bool, default=True)
    parser.add_argument('--perturb_feature', type=bool, default=True)
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    data_list = torch.load('./dataset/data.pt')
    feature = torch.load('./dataset/feature.pt')
    train_dataset, test_dataset, valid_dataset = random_split(data_list, frac_train=0.6, frac_valid=0.2,
                                                              frac_test=0.2, seed=args.seed)
    print('dataset load successfully')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature, perturb=args.perturb_feature))
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, collate_fn=partial(collator, feature=feature))
    args.tot_updates = len(train_loader)*args.epochs
    args.warmup_updates = args.tot_updates/10
    args.max_steps = args.tot_updates + 1
    print(args)

    model = Graphormer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        attn_bias_dim=args.attn_bias_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        weight_decay=args.weight_decay,
        ffn_dim=args.ffn_dim,
        dataset_name=args.dataset_name,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        num_data_augment=args.num_data_augment,
        num_global_node=args.num_global_node,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        flag=args.flag,
        flag_m=args.flag_m,
        flag_step_size=args.flag_step_size,
    )
    if not args.test and not args.validate:
        print(model)
    print('Total params:', sum(p.numel() for p in model.parameters()))
    model.apply(init_params)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)

    val_acc_list, test_acc_list = [], []

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer, lr_scheduler)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)

        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        print("train_acc: %f val_acc: %f test_acc: %f" % (train_acc, val_acc, test_acc))
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    print('best validation acc: ', max(val_acc_list))
    print('best test acc: ', max(test_acc_list))
    print('best acc: ', test_acc_list[val_acc_list.index(max(val_acc_list))])


if __name__ == "__main__":
    main()
