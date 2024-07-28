import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time


file_path = './subset_data1.csv'
dataset = pd.read_csv(file_path)
print(f"subset_data1:{dataset.shape}")

def main():
    file_path = './subset_data1.csv'
    dataset_all = pd.read_csv(file_path)
    drop_name = [str(i) for i in range(31, 42)]
    labels = dataset_all['41']
    dataset = dataset_all.drop(drop_name, axis=1)

    dataset = dataset.values
    labels = labels.values
    # 获取最后一列作为标签

    # 找出所有唯一的标签
    unique_labels = np.unique(labels)
    # print(unique_labels)

    # 创建一个空的列表来存储抽取的数据
    train_dataset = []
    test_dataset = []
    train_labels = []
    test_labels = []

    # 分割数据
    for label in unique_labels:
        # 找到标签对应的索引
        selected_indices = np.where(labels == label)[0]
        selected_data = dataset[selected_indices]
        selected_label = labels[selected_indices]

        # 分割训练和测试集
        train_data, test_data, train_label, test_label = train_test_split(selected_data, selected_label, test_size=0.3)

        train_dataset.append(train_data)
        test_dataset.append(test_data)
        train_labels.append(train_label)
        test_labels.append(test_label)

    # 合并所有训练和测试数据
    train_dataset = np.concatenate(train_dataset, axis=0)
    test_dataset = np.concatenate(test_dataset, axis=0)
    train_labels = np.concatenate(train_labels, axis=0).reshape(-1, 1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1, 1)

    # 左右合并数据和标签
    train_set = np.hstack((train_dataset, train_labels))
    test_set = np.hstack((test_dataset, test_labels))
    print(train_set.shape)
    print(test_set.shape)
    print(train_set.shape[0] + test_set.shape[0])
    # 保存为没有header和index的文件
    pd.DataFrame(train_set).to_csv('train_set.csv', header=False, index=False)
    pd.DataFrame(test_set).to_csv('test_set.csv', header=False, index=False)


if __name__ == '__main__':
    main()
