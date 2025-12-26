import os
import argparse
import configparser
import warnings
import time
import datetime
import math
from copy import deepcopy

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from model import TEGTSNet, build_model
from utils import (
    mae_rmse_mape, count_parameters,
    get_adj_from_csv, get_adj_from_npy, cal_lape
)

warnings.filterwarnings('ignore')

# ==== unified device ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(">>> Force using device:", device)

# ---- redirect legacy .cuda() usage to unified device ----
def _tensor_cuda(self, device=None, non_blocking=False):
    return self.to(device)

def _module_cuda(self, device=None):
    return self.to(device)

torch.Tensor.cuda = _tensor_cuda
torch.nn.Module.cuda = _module_cuda
# --------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='conf/PEMSD4_1dim_12.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(args.config, 'r') as f:
    print(f.read())
print('\n')

config.read(args.config)
data_config = config['Data']
training_config = config['Training']

data_path = data_config['graph_signal_matrix_filename']
distance_csv_path = data_config['distance_csv_path']
dataset_name = data_config['dataset_name']
print("dataset_name: ", dataset_name)

num_of_vertices = int(data_config['num_of_vertices'])
time_slice_size = int(data_config['time_slice_size'])

mode = training_config['mode']
L = int(training_config['L'])
K = int(training_config['K'])
d = int(training_config['d'])
learning_rate = float(training_config['learning_rate'])
max_epoch = int(training_config['epochs'])
decay_epoch = int(training_config['decay_epoch'])
batch_size = int(training_config['batch_size'])
num_his = int(training_config['num_his'])
num_pred = int(training_config['num_pred'])
patience = int(training_config['patience'])
in_channels = int(training_config['in_channels'])

# load dataset
data = np.load(data_path)
trainX, trainTE, trainY = data['train_x'], data['trainTE'], data['train_target']
valX, valTE, valY = data['val_x'], data['valTE'], data['val_target']
testX, testTE, testY = data['test_x'], data['testTE'], data['test_target']

print("train: ", trainX.shape, trainY.shape)
print("val: ", valX.shape, valY.shape)
print("test: ", testX.shape, testY.shape)
print("trainTE: ", trainTE.shape)
print("valTE: ", valTE.shape)
print("testTE: ", testTE.shape)

trainX = torch.from_numpy(np.array(trainX, dtype='float64')).float()
trainY = torch.from_numpy(np.array(trainY, dtype='float64')).float()
valX   = torch.from_numpy(np.array(valX, dtype='float64')).float()
valY   = torch.from_numpy(np.array(valY, dtype='float64')).float()
testX  = torch.from_numpy(np.array(testX, dtype='float64')).float()
testY  = torch.from_numpy(np.array(testY, dtype='float64')).float()

trainTE = torch.from_numpy(np.array(trainTE, dtype='int32'))
valTE   = torch.from_numpy(np.array(valTE, dtype='int32'))
testTE  = torch.from_numpy(np.array(testTE, dtype='int32'))

mean_, std_ = torch.mean(trainX.reshape(-1, in_channels), axis=0), torch.std(trainX.reshape(-1, in_channels), axis=0)
print("mean and std in every feature", mean_.shape, mean_, std_)

trainX = (trainX - mean_) / std_
valX   = (valX - mean_) / std_
testX  = (testX - mean_) / std_

model = build_model(config, bn_decay=0.1).to(device)

mean_ = mean_.to(device)
std_  = std_.to(device)

# adjacency -> Laplacian PE
if 'PEMS' in dataset_name or 'JiNan' in dataset_name:
    A, _ = get_adj_from_csv(distance_csv_path, num_of_vertices)
else:
    A = get_adj_from_npy(distance_csv_path)

lpls = cal_lape(A.copy())
lpls = torch.from_numpy(np.array(lpls, dtype='float32')).float().to(device)

parameters = count_parameters(model)
print('trainable parameters: {:,}'.format(parameters))

# train
print('Start training ...')

val_loss_min = float('inf')
wait = 0
best_model = deepcopy(model.state_dict())
best_epoch = -1

num_train = trainX.shape[0]
num_val = valX.shape[0]
num_test = testX.shape[0]
train_num_batch = math.ceil(num_train / batch_size)
val_num_batch = math.ceil(num_val / batch_size)
test_num_batch = math.ceil(num_test / batch_size)

loss_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=0.9)

exp_datadir = "experiments/TEGTSNet/"
os.makedirs(exp_datadir, exist_ok=True)

params_filename = os.path.join(
    exp_datadir,
    f"{dataset_name}_{num_of_vertices}_{num_his}_{num_pred}_{time_slice_size}_{K}_{L}_{d}_best_params"
)

train_time_epochs, val_time_epochs = [], []
total_start_time = time.time()

for epoch_num in range(max_epoch):
    if wait >= patience:
        print(f'early stop at epoch: {epoch_num}, val loss best = {val_loss_min}')
        break

    permutation = torch.randperm(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]

    start_train = time.time()
    model.train()
    train_loss = 0.0

    print(f"epoch {epoch_num} start!")

    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)

        X = trainX[start_idx:end_idx].to(device)
        TE = trainTE[start_idx:end_idx].to(device)
        label = trainY[start_idx:end_idx].to(device)

        optimizer.zero_grad()
        pred = model(X, TE, lpls, mode)  # TEG-TSNet forward
        pred = pred * std_[0] + mean_[0]

        loss_batch = loss_criterion(pred, label)
        train_loss += float(loss_batch) * (end_idx - start_idx)

        loss_batch.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Train batch {batch_idx+1}/{train_num_batch}, loss={loss_batch:.4f}')

        del X, TE, label, pred, loss_batch

    train_loss /= num_train
    end_train = time.time()

    # validation
    print("evaluating on valid set now!")
    val_loss = 0.0
    start_val = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)

            X = valX[start_idx:end_idx].to(device)
            TE = valTE[start_idx:end_idx].to(device)
            label = valY[start_idx:end_idx].to(device)

            pred = model(X, TE, lpls, mode, 'test')
            pred = pred * std_[0] + mean_[0]

            loss_batch = loss_criterion(pred, label)
            val_loss += float(loss_batch) * (end_idx - start_idx)

            del X, TE, label, pred, loss_batch

    val_loss /= num_val
    end_val = time.time()

    train_time_epochs.append(end_train - start_train)
    val_time_epochs.append(end_val - start_val)

    print('%s | epoch: %04d/%d, train: %.1fs, val: %.1fs, lr: %.6f' %
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch_num + 1,
           max_epoch, end_train - start_train, end_val - start_val, optimizer.param_groups[0]['lr']))
    print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

    if val_loss <= val_loss_min:
        wait = 0
        val_loss_min = val_loss
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch_num
    else:
        wait += 1

    scheduler.step()

print("best MAE is %.4f at epoch %d" % (val_loss_min, best_epoch))
print("avg train/epoch:", np.array(train_time_epochs).mean())
print("avg val/epoch:", np.array(val_time_epochs).mean())
print("total train time:", time.time() - total_start_time, "s")

# test
model.load_state_dict(best_model)
model.eval()

trainY_cpu = trainY.cpu()
valY_cpu = valY.cpu()
testY_cpu = testY.cpu()
mean_np = mean_.cpu().numpy()
std_np = std_.cpu().numpy()

with torch.no_grad():
    # predictions
    def _predict_all(X_all, TE_all, num_batch):
        preds = []
        for batch_idx in range(num_batch):
            s = batch_idx * batch_size
            e = min(X_all.shape[0], (batch_idx + 1) * batch_size)
            Xb = X_all[s:e].to(device)
            TEb = TE_all[s:e].to(device)
            pb = model(Xb, TEb, lpls, mode, 'test')
            preds.append(pb.detach().cpu().numpy())
            del Xb, TEb, pb
        return torch.from_numpy(np.concatenate(preds, axis=0))

    trainPred = _predict_all(trainX, trainTE, train_num_batch)
    valPred = _predict_all(valX, valTE, val_num_batch)

    # test + save one-batch intermediates for analysis
    testPred_list = []
    analysis_intermediates = None
    first_test_batch_TE = None

    start_test = time.time()
    for batch_idx in range(test_num_batch):
        s = batch_idx * batch_size
        e = min(num_test, (batch_idx + 1) * batch_size)
        Xb = testX[s:e].to(device)
        TEb = testTE[s:e].to(device)

        if batch_idx == 0:
            pb, inter = model(Xb, TEb, lpls, mode, 'test', return_intermediate=True)
            analysis_intermediates = inter
            first_test_batch_TE = TEb.detach().cpu().numpy()
        else:
            pb = model(Xb, TEb, lpls, mode, 'test', return_intermediate=False)

        testPred_list.append(pb.detach().cpu().numpy())
        del Xb, TEb, pb

    end_test = time.time()
    testPred = torch.from_numpy(np.concatenate(testPred_list, axis=0))

    # denorm
    trainPred = trainPred * std_np[0] + mean_np[0]
    valPred = valPred * std_np[0] + mean_np[0]
    testPred = testPred * std_np[0] + mean_np[0]


# metrics
train_mae, train_rmse, train_mape = mae_rmse_mape(trainPred, trainY_cpu)
val_mae, val_rmse, val_mape = mae_rmse_mape(valPred, valY_cpu)
test_mae, test_rmse, test_mape = mae_rmse_mape(testPred, testY_cpu)

print('testing time: %.1fs' % (end_test - start_test))
print('             LOSS\tMAE\tRMSE\tMAPE')
print('train      %.2f\t%.2f\t%.2f\t%.2f%%' % (train_mae, train_mae, train_rmse, train_mape))
print('val        %.2f\t%.2f\t%.2f\t%.2f%%' % (val_mae, val_mae, val_rmse, val_mape))
print('test       %.2f\t%.2f\t%.2f\t%.2f%%' % (test_mae, test_mae, test_rmse, test_mape))

