import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from fhgnn import FHGNN
from splitters import scaffold_split
from loader import HiMolGraph, MoleculeDataset

criterion = nn.BCEWithLogitsLoss(reduction = "none")


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)

        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss = torch.sum((pred-y)**2)/y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    #Whether y is non-null or not.
    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y**2 > 0
    #Loss matrix
    loss_mat = criterion(pred.double(), (y+1)/2)
    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat)/torch.sum(is_valid)

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    eval_roc = sum(roc_list)/len(roc_list) #y_true.shape[1]

    return eval_roc, loss


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse=np.sqrt(mean_squared_error(y_true,y_scores))
    return mse, mae, rmse


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of FH-GNN')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for the prediction layer')
    parser.add_argument('--dataset', type=str, default = 'bbbp', 
                        help='[bbbp, bace, sider, clintox,tox21, toxcast, esol,freesolv,lipophilicity]')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help = "the path of input CSV file")
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help = "the path to save output model")
    parser.add_argument('--depth', type=int, default=7, help = "the depth of molecule encoder")
    parser.add_argument('--seed', type=int, default=88, help = "seed for splitting the dataset")
    parser.add_argument('--runseed', type=int, default=88, help = "seed for minibatch selection, random initialization")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'bace', 'bbbp', 'sider', 'clintox']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    print('process data')
    dataset = MoleculeDataset(os.path.join(args.data_dir, args.dataset), dataset=args.dataset)

    print("scaffold")
    smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset, '/processed/smiles.csv'), header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, 
                                                                frac_train=0.8,frac_valid=0.1, 
                                                                frac_test=0.1,seed = args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = FHGNN(data_name=args.dataset, atom_fdim=89, bond_fdim=98,fp_fdim=6338, 
                  hidden_size=512, depth=args.depth, device=device, out_dim=num_tasks,)
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.parameters(), "lr":args.lr})
    optimizer = optim.Adam(model_param_group)
    print(optimizer)

    model_save_path = os.path.join(args.save_dir, args.dataset + '.pth')

    # training based on task type
    if task_type == 'cls':
        best_auc = 0
        train_auc_list, test_auc_list = [], []
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            train(model, device, train_loader, optimizer)
            
            print('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = eval(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = eval(args, model, device, val_loader)
            test_auc, test_loss = eval(args, model, device, test_loader)
            test_auc_list.append(float('{:.4f}'.format(test_auc)))
            train_auc_list.append(float('{:.4f}'.format(train_auc)))

            if best_auc < test_auc:
                best_auc = test_auc
                torch.save(model.state_dict(), model_save_path)
            
            print("train_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))
    
    elif task_type == 'reg':
        train_list, test_list = [], []
        best_rmse = 100
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_mse, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_mse, train_mae, train_rmse = 0, 0, 0
            val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)
            
            test_list.append(float('{:.6f}'.format(test_rmse)))
            train_list.append(float('{:.6f}'.format(train_rmse)))
            if test_rmse<best_rmse:
                torch.save(model.state_dict(), model_save_path)
                
            print("train_mse: %f val_mse: %f test_mse: %f" %(train_mse, val_mse, test_mse))
            print("train_mae: %f val_mae: %f test_mae: %f" %(train_mae, val_mae, test_mae))
            print("train_rmse: %f val_rmse: %f test_rmse: %f" %(train_rmse, val_rmse, test_rmse))


if __name__ == "__main__":
    main()