import argparse
import glob
import os

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from TimesNet import Model
from filterpy.kalman import KalmanFilter


def load_data(csv_file, window_size):
    df = pd.read_csv(csv_file,
                     usecols=['MODULE_MAIN', 'MODULE_FRONT_RIGHT', 'MODULE_FRONT_LEFT', 'MODULE_HEAD_LEFT',
                              'MODULE_HEAD_RIGHT', 'MODULE_TAIL_LEFT', 'MODULE_TAIL_RIGHT', 'DISTANCE'])

    x_values = df.iloc[:, :-1].values
    y_values = df.iloc[:, -1].values
    x_values = np.mean(x_values.reshape(-1, 4, x_values.shape[1]), axis=1)
    y_values = np.mean(y_values.reshape(-1, 4), axis=1)

    f = KalmanFilter(dim_x=7, dim_z=7)
    f.x = x_values[0]
    f.H = np.eye(7)
    f.Q = np.diag([7] * 7)
    f.R = np.diag([300] * 7)

    for i in range(1, len(x_values)):
        f.predict()
        f.update(z=x_values[i])
        x_values[i] = f.x

    x, y = [], []
    for i in range(len(x_values) - window_size + 1):
        x.append(x_values[i: i + window_size])
        y.append(y_values[i + window_size - 1])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def distance_to_label(distances):
    threshold1 = 4  # 第一类的阈值
    threshold2 = 6  # 第二类的阈值
    threshold3 = 8  # 第三类的阈值

    distances = distances.squeeze()

    # 计算类别比例
    class1 = torch.clamp((distances - threshold1) / (threshold2 - threshold1), 0, 1)  # 类别1比例
    class2 = torch.clamp((distances - threshold2) / (threshold3 - threshold2), 0, 1)  # 类别2比例

    # 生成类别标签
    class0 = 1 - class1  # 类别0的比例
    class1 = class1 * (1 - class2)  # 修正类别1，使其不重叠类别2
    class2 = class2  # 类别2的比例

    # 堆叠类别标签
    labels = torch.stack([class0, class1, class2], dim=1)

    return labels


if __name__ == "__main__":
    configs = argparse.ArgumentParser(description='Bluetooth RSSI Predict istance')
    configs.add_argument('--seq_len', type=int, default=32)
    configs.add_argument('--e_layers', type=int, default=1,
                         help='the layers of TimesBlock')
    configs.add_argument('--top_k', type=int, default=4,
                         help='k denotes how many top period(frequencies) are taken into consideration')
    configs.add_argument('--d_ff', type=int, default=16,
                         help='the mid dimension of two layer Inception Block V1')
    configs.add_argument('--num_kernels', type=int, default=4,
                         help='the number of layers in the Inception Block V1')
    configs.add_argument('--enc_in', type=int, default=7,
                         help='the input feature of TokenEmbedding')
    configs.add_argument('--d_model', type=int, default=32,
                         help='the dimension of feature space')
    configs.add_argument('--dropout', type=float, default=0.1,
                         help='the dropout rate of the network')
    configs.add_argument('--num_class', type=int, default=3)
    configs = configs.parse_args()

    csv_files = glob.glob('../data_val/rssi/*/*/*.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_inputs = []
    all_outputs = []

    for csv_file in csv_files:
        inputs, outputs = load_data(csv_file, window_size=configs.seq_len)
        all_inputs.append(inputs)
        all_outputs.append(outputs)

    all_inputs = torch.cat(all_inputs).to(device)
    all_outputs = torch.cat(all_outputs).to(device)

    train_size = int(all_inputs.shape[0] * 0.8)

    X_train, X_test = all_inputs[:train_size], all_inputs[train_size:]
    y_train, y_test = all_outputs[:train_size], all_outputs[train_size:]

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4096)

    val_csv_files = glob.glob('../data/rssi/*/*.csv')
    X_val = []
    y_val = []
    for csv_file in val_csv_files:
        inputs, outputs = load_data(csv_file, window_size=configs.seq_len)
        X_val.append(inputs)
        y_val.append(outputs)
    X_val = torch.cat(X_val).to(device)
    y_val = torch.cat(y_val).to(device)
    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_set, batch_size=4096)

    model = Model(configs).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 500
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_classification_loss = 0
        epoch_regression_loss = 0
        epoch_loss = 0
        for iter_idx, (data, distance) in enumerate(train_loader):
            p = 0.8  # 有p的概率生成1
            padding_mask = torch.bernoulli(torch.full((data.shape[0], configs.seq_len), p)).bool().cuda()

            optimizer.zero_grad()
            output_classification, output_regression = model(data, padding_mask)

            label = distance_to_label(distance)
            classification_loss = F.cross_entropy(output_classification, label)

            regression_loss = F.mse_loss(output_regression, distance)

            loss = 5 * classification_loss + 0.1 * regression_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            epoch_classification_loss += classification_loss.item()
            epoch_regression_loss += regression_loss.item()
            epoch_loss += loss.item()

        avg_classification_loss = epoch_classification_loss / len(train_loader)
        avg_regression_loss = epoch_regression_loss / len(train_loader)
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch}/{epochs} - regression Loss: {avg_regression_loss:.4f}')
        print(f'\nEpoch {epoch}/{epochs} - classification Loss: {avg_classification_loss:.4f}')
        print(f'\nEpoch {epoch}/{epochs} - Loss: {avg_loss:.4f}')

        model_folder = "TimesNet_models"
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, f'TimesNet_model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_path)

        model.eval()
        test_classification_loss = 0
        test_regression_loss = 0
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, distance in test_loader:
                padding_mask = torch.ones(size=(data.shape[0], configs.seq_len)).bool().cuda()
                output_classification, output_regression = model(data, padding_mask)

                label = distance_to_label(distance)
                classification_loss = F.cross_entropy(output_classification, label)

                regression_loss = F.mse_loss(output_regression, distance)

                loss = 5 * classification_loss + 0.1 * regression_loss
                test_classification_loss += classification_loss.item()
                test_regression_loss += regression_loss.item()
                test_loss += loss.item()

                _, predicted = torch.max(output_classification.data, 1)
                total += label.size(0)
                correct += (predicted == label.argmax(dim=1)).sum().item()

        test_classification_loss = test_classification_loss / len(test_loader)
        test_regression_loss = test_regression_loss / len(test_loader)
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        print('\nTest set: Average regression loss: {:.6f}'.format(test_regression_loss))
        print('\nTest set: Average classification loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_classification_loss, correct, total, accuracy))
        print('\nTest set: Average loss: {:.6f}'.format(test_loss))

        val_classification_loss = 0
        val_regression_loss = 0
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, distance in val_loader:
                padding_mask = torch.ones(size=(data.shape[0], configs.seq_len)).bool().cuda()
                output_classification, output_regression = model(data, padding_mask)

                label = distance_to_label(distance)
                classification_loss = F.cross_entropy(output_classification, label)

                regression_loss = F.mse_loss(output_regression, distance)

                loss = 5 * classification_loss + 0.1 * regression_loss
                val_classification_loss += classification_loss.item()
                val_regression_loss += regression_loss.item()
                val_loss += loss.item()

                _, predicted = torch.max(output_classification.data, 1)
                total += label.size(0)
                correct += (predicted == label.argmax(dim=1)).sum().item()

        val_classification_loss = val_classification_loss / len(val_loader)
        val_regression_loss = val_regression_loss / len(val_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print('\nVal set: Average regression loss: {:.6f}'.format(val_regression_loss))
        print('\nVal set: Average classification loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
            val_classification_loss, correct, total, accuracy))
        print('\nVal set: Average loss: {:.6f}\n'.format(val_loss))

        scheduler.step()

    torch.save(model.state_dict(), 'TimesNet_model.pth')
