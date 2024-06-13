import argparse
import math
import os

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from filterpy.kalman import KalmanFilter

from TSMixer import Model

# 指定文件夹路径
folder_path = '../data/rssi/F2_侧裙PCB翻转灌胶_重庆地面侧裙反转手表WATCH4pro-6c4精标_黄志聪_20231202'

# 获取文件夹中CSV文件的数量
file_count = sum(1 for f in os.listdir(folder_path) if f.endswith('.csv'))

# 初始化一个包含足够多子图的图表
rows = int(math.sqrt(file_count))  # 向下取整
cols = math.ceil(file_count / rows)  # 保证列数足够

configs = argparse.ArgumentParser(description='Bluetooth RSSI Predict istance')
configs.add_argument('--seq_len', type=int, default=32)
configs.add_argument('--e_layers', type=int, default=2,
                     help='the layers of TimesBlock')
configs.add_argument('--enc_in', type=int, default=7,
                     help='the input feature of TokenEmbedding')
configs.add_argument('--d_model', type=int, default=32,
                     help='the dimension of feature space')
configs.add_argument('--dropout', type=float, default=0.1,
                     help='the dropout rate of the network')
configs.add_argument('--num_class', type=int, default=3)
configs = configs.parse_args()

model = Model(configs)
model.load_state_dict(torch.load('./TSMixer_models/TSMixer_model_epoch_24.pth'))
model.eval()


# 创建 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Slider(id='classification_unlock_threshold',
               min=0.9,
               max=1,
               step=0.01,
               value=0.9,
               ),
    dcc.Slider(id='classification_lock_threshold',
               min=0.5,
               max=1,
               step=0.1,
               value=0.5,
               ),
    dcc.Slider(id='regression_threshold_unlock',
               min=0,
               max=6,
               step=1,
               value=4,
               ),
    dcc.Slider(id='regression_threshold_lock',
               min=4,
               max=10,
               step=1,
               value=6),
    dcc.Graph(id='interactive-graph', style={'height': '1500px', 'width': '2500px'})
])


@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('classification_unlock_threshold', 'value'),
     Input('classification_lock_threshold', 'value'),
     Input('regression_threshold_unlock', 'value'),
     Input('regression_threshold_lock', 'value')]
)
def update_figure(classification_unlock_threshold, classification_lock_threshold, regression_threshold_unlock,
                  regression_threshold_lock):
    # 当前子图的位置
    current_row = 1
    current_col = 1

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f for f in os.listdir(folder_path) if f.endswith('.csv')])
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        input_data = torch.tensor(data[['MODULE_MAIN', 'MODULE_FRONT_RIGHT', 'MODULE_FRONT_LEFT', 'MODULE_HEAD_LEFT',
                                        'MODULE_HEAD_RIGHT', 'MODULE_TAIL_LEFT', 'MODULE_TAIL_RIGHT']].values).float()

        input_data = input_data.reshape(input_data.shape[0] // 4, 4, input_data.shape[1])
        input_data = input_data.mean(axis=1, keepdims=True).squeeze().numpy()

        f = KalmanFilter(dim_x=7, dim_z=7)
        f.x = input_data[0]
        f.H = np.eye(7)
        f.Q = np.diag([7] * 7)
        f.R = np.diag([300] * 7)

        for i in range(1, len(input_data)):
            f.predict()
            f.update(z=input_data[i])
            input_data[i] = f.x

        input_data = torch.tensor(input_data)
        filter_data = input_data

        window_size = 32  # 滑动窗口大小，即时间序列长度

        # 生成滑动窗口样本
        tensor_list = []
        for i in range(input_data.shape[0] - window_size + 1):
            tensor_list.append(input_data[i:i + window_size])
        input_data = torch.stack(tensor_list)

        # 获取模型的原始输出和分类结果
        with torch.no_grad():
            padding_masks = torch.ones(size=(input_data.shape[0], input_data.shape[1])).bool()
            classification_output, regression_output = model(input_data, padding_masks)

            probabilities = torch.softmax(classification_output, dim=1)
            max_probabilities, model_classification_output = probabilities.max(dim=1)

            model_regression_output = regression_output.squeeze()
            # model_regression_output = kalman_filter(model_regression_output)

            # reconstruct_output = model_anomaly_detection(input_data, None, None, None)
            # loss = torch.nn.functional.mse_loss(reconstruct_output, input_data, reduction='none')
            # average_loss_per_sample = torch.mean(loss, dim=[1, 2])

        model_regression_output = model_regression_output.numpy()
        f_distance = KalmanFilter(dim_x=2, dim_z=1)
        f_distance.x = np.array([model_regression_output[0], 0])
        f_distance.F = np.array([[1., 1.],
                                 [0., 1.]])
        f_distance.H = np.array([[1., 0.]])
        f_distance.P *= 1000.
        f_distance.R = 5
        from filterpy.common import Q_discrete_white_noise
        f_distance.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
        max_speed = 0.1
        for i in range(1, len(model_regression_output)):
            f_distance.predict()
            f_distance.update(model_regression_output[i])
            if f_distance.x[1] > max_speed:
                f_distance.x[1] = max_speed
            elif f_distance.x[1] < -max_speed:
                f_distance.x[1] = -max_speed
            model_regression_output[i] = f_distance.x[0]
        model_regression_output = torch.tensor(model_regression_output)

        count_open = 0
        count_close = 0
        smoothed_output = np.copy(model_classification_output)

        std = torch.std(input_data, dim=1)
        std = torch.max(std, dim=1).values

        for i in range(0, len(model_classification_output)):
            if (model_classification_output[i] == 0 and model_regression_output[i] <= regression_threshold_unlock
                    and max_probabilities[i] >= classification_unlock_threshold):
                smoothed_output[i] = 0
                count_open = 16
            elif count_open > 0:
                count_open -= 1
                smoothed_output[i] = 1
            elif model_classification_output[i] == 2 and model_regression_output[i] >= regression_threshold_lock\
                    and max_probabilities[i] >= classification_lock_threshold:
                smoothed_output[i] = 2
                count_close = 16
            elif count_close > 0:
                count_close -= 1
                smoothed_output[i] = 1
            else:
                smoothed_output[i] = 1

        smoothed_output[0:32] = [2] * 32
        for i in range(32, len(smoothed_output)):
            if smoothed_output[i] == 1:
                smoothed_output[i] = smoothed_output[i - 1]

        fig.add_trace(go.Scatter(x=data.index, y=data['DISTANCE'][::4], name='DISTANCE', legendgroup='group1'),
                      row=current_row, col=current_col)
        fig.add_trace(go.Scatter(x=data.index[window_size:], y=model_classification_output * 10,
                                 name='Model Classification Output', legendgroup='group2'),
                      row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index[window_size:], y=max_probabilities * 20, name='Model Probability', legendgroup='group3'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index[window_size:], y=model_regression_output,
                       name='Model Regression Output', legendgroup='group4'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index[window_size:], y=smoothed_output * 10, name='Smoothed Model Output', legendgroup='group100'),
            row=current_row, col=current_col)
        fig.add_trace(go.Scatter(x=data.index, y=data['MODULE_MAIN'][::4], name='MODULE_MAIN', legendgroup='group5'),
                      row=current_row, col=current_col)
        fig.add_trace(go.Scatter(x=data.index, y=filter_data[:, 0], name='Filter_MODULE_MAIN', legendgroup='group5'),
                      row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_FRONT_RIGHT'][::4], name='MODULE_FRONT_RIGHT', legendgroup='group6'),
            row=current_row, col=current_col)
        fig.add_trace(go.Scatter(x=data.index, y=filter_data[:, 1], name='Filter_FRONT_RIGHT', legendgroup='group6'),
                      row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_FRONT_LEFT'][::4], name='MODULE_FRONT_LEFT', legendgroup='group7'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=filter_data[:, 2], name='Filter_MODULE_FRONT_LEFT', legendgroup='group7'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_HEAD_LEFT'][::4], name='MODULE_HEAD_LEFT', legendgroup='group8'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=filter_data[:, 3], name='Fileter_MODULE_HEAD_LEFT', legendgroup='group8'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_HEAD_RIGHT'][::4], name='MODULE_HEAD_RIGHT', legendgroup='group9'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=filter_data[:, 4], name='Filter_MODULE_HEAD_RIGHT', legendgroup='group9'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_TAIL_LEFT'][::4], name='MODULE_TAIL_LEFT', legendgroup='group10'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=filter_data[:, 5], name='Filter_MODULE_TAIL_LEFT', legendgroup='group10'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MODULE_TAIL_RIGHT'][::4], name='MODULE_TAIL_RIGHT', legendgroup='group11'),
            row=current_row, col=current_col)
        fig.add_trace(
            go.Scatter(x=data.index, y=filter_data[:, 6], name='Filter_MODULE_TAIL_RIGHT', legendgroup='group11'),
            row=current_row, col=current_col)
        # 更新下一个子图的位置
        current_col += 1
        if current_col > cols:
            current_col = 1
            current_row += 1
    return fig


# 运行 Dash 应用
if __name__ == '__main__':
    app.run_server(debug=True)
