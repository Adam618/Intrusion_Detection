# from __future__ import print_function, division
# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset


# class SeriesDataset(Dataset):
#     def __init__(self, train=True):

#         if train:  # 训练集
#             path_file = "../Dataset/train_set.csv"
#         else:  # 测试集
#             path_file = "../Dataset/test_set.csv"

#         self.data_frame = pd.read_csv(path_file, header=None, index_col=None)
#         # print(self.data_frame.shape)

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         label = self.data_frame.iloc[idx - 1, -1]
#         label = int(label)
#         series = self.data_frame.iloc[idx - 1, :-1]
#         series = np.array(series)
#         series = series.astype(np.float32)
#         series = torch.from_numpy(series)
#         series = series.reshape(1, len(series), -1)

#         return series, label
    
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SeriesDataset(Dataset):
    def __init__(self, train=True):
        if train:  # 训练集
            path_file = "../Dataset/NSL-KDD/KDDTrain+_progressed.csv"
            "KDD/train_set.csv NSL-KDD/KDDTrain+_progressed.csv"
        else:  # 测试集
            path_file = "../Dataset/NSL-KDD/KDDTest+_progressed.csv"
            "KDD/test_set.csv NSL-KDD/KDDTest+_progressed.csv"
        # 读取数据并转换为 numpy 数组
        data_frame = pd.read_csv(path_file, header=None, index_col=None).values
        self.series = data_frame[:, :-1].astype(np.float32)  # 数据部分
        self.labels = data_frame[:, -1].astype(int)  # 标签部分
        print('self.label.shape',self.labels.shape)
        print('self.series.shape',self.series.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        series = self.series[idx]  # 获取数据
        label = self.labels[idx]  # 获取标签

        # 转换为 PyTorch 张量
        series = torch.from_numpy(series).unsqueeze(0)  # 添加一个维度
        label = torch.tensor(label)

        return series, label
