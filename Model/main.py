import time
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import models
from SeriesDataset import SeriesDataset
from datetime import datetime
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 将父目录添加到系统路径
sys.path.append(parent_dir)

from Dataset.dataset import handle
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'WenQuanYi Zen Hei' 是一种常用的开源中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# info = 'origin'
info = 'boderline_smotenc'
num_classes = 5
batch_size = 512
train_epoch = 50
# origin
# lr = 0.0001
# boderline_smotenc
# lr = 0.000001
lr = 0.001
main_model = 'CNN_LSTM'  # CNN_LSTM
train_test = 'train'  # train / test
seed_value = 42  # 设置随机种子以确保可重复性
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 设置 PYTHONHASHSEED 环境变量
random.seed(seed_value)  # 设置 Python 的随机种子
np.random.seed(seed_value)  # 设置 NumPy 的随机种子
torch.manual_seed(seed_value)  # 设置 PyTorch 的随机种子
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.set_printoptions(threshold=np.inf)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
if train_test == 'train':
    handle('/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTrain+.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTest+.csv', 
        '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTrain+_progressed.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTest+_progressed.csv')
train_set = SeriesDataset(train=True)
test_set = SeriesDataset(train=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
if main_model == 'CNN_LSTM':
    model = models.CNN_LSTM(num_classes=num_classes).to(device)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss()

# model_save_path = os.path.join("./train_model", main_model)
# mkdir(model_save_path)


model_save_path = os.path.join("./train_model", f"{main_model}_{info}")
mkdir(model_save_path)

def save_models(epoch):
    torch.save(model.state_dict(), model_save_path + "/model.model".format(epoch))


def test():
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    for i, (series, labels) in enumerate(test_loader):
        series = series.to(device)
        labels = labels.to(device)
        _, outputs = model(series)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item() * series.size(0)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

    test_acc = test_acc / len(test_set)
    test_loss = test_loss / len(test_set)
    return test_acc, test_loss


def test_model(plot_show):
    file_path = os.path.join(model_save_path, 'model.model')
    labels_list = []
    prediction_list = []
    model.load_state_dict(torch.load(file_path))
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    t_start = time.time()

    for i, (series, labels) in enumerate(test_loader):
        series = series.to(device)
        labels = labels.to(device)
        feature, outputs = model(series)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item() * series.size(0)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

        labels = labels.tolist()
        labels_list = labels_list + labels

        prediction = prediction.tolist()
        prediction_list = prediction_list + prediction

        if i == 0:
            features = feature.cpu().detach().numpy()
        else:
            features = np.append(features, feature.cpu().detach().numpy(), axis=0)

    # Compute the average acc and loss
    test_acc = test_acc / len(test_set)
    test_loss = test_loss / len(test_set)

    # Compute precision, recall, and F1 score
    precision = precision_score(labels_list, prediction_list, average='macro')  # 或使用'micro', 'weighted'等
    recall = recall_score(labels_list, prediction_list, average='macro')
    f1 = f1_score(labels_list, prediction_list, average='macro')

    # Compute accuracy for binary classification (Normal vs Others)
    binary_labels = [0 if label == 0 else 1 for label in labels_list]
    binary_predictions = [0 if pred == 0 else 1 for pred in prediction_list]
    binary_accuracy = accuracy_score(binary_labels, binary_predictions)

    # Compute per-class precision, recall, and F1 score
    class_names = ["Normal", "DOS", "Probing", "R2L", "U2R"]
    #class_names = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Normal", "Reconnaissance",     "Shellcode", "Worms"]
    
    precision_per_class = precision_score(labels_list, prediction_list, average=None)
    recall_per_class = recall_score(labels_list, prediction_list, average=None)
    f1_per_class = f1_score(labels_list, prediction_list, average=None)
    
    cm = confusion_matrix(labels_list, prediction_list)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr_per_class = fp / (fp + tn)
    fpr = fpr_per_class
    
    # Calculate G-means per class
    g_means_per_class = np.sqrt(recall_per_class * (1 - fpr_per_class))
    g_means = g_means_per_class
    
    t_end = time.time()
    run_time = t_end - t_start

    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 15

    print("Precision per class:", precision_per_class)
    print("Recall per class:", recall_per_class)
    print("F1 score per class:", f1_per_class)
    print("False Positive Rate (FPR) per class:", fpr_per_class)
    print("G-Mean per class:", g_means_per_class)
    # print("TSNE features shape:", tsne_result)
    # print("Confusion matrix:\n", confusion)

    if plot_show:
        # 绘制柱状图
        metrics = ['Precision', 'Recall', 'F1 Score','FPR', 'G-Mean']
        per_class_metrics = [precision_per_class, recall_per_class, f1_per_class, fpr_per_class, g_means_per_class]
        for i, metric in enumerate(metrics):
            plt.figure(figsize=(20, 10))
            plt.bar(class_names, per_class_metrics[i], color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel(metric)
            plt.title(f'{metric} per Class')
            plt.ylim(0, 1.1) 
            for j, value in enumerate(per_class_metrics[i]):
                plt.text(j, value + 0.01, f'{value:.2f}', ha='center')
            plt.savefig(os.path.join(model_save_path, f'{metric}_per_Class.png'))
            plt.close()

        '''t-SNE'''
        tsne = TSNE(n_components=2, random_state=0)
        tsne_result = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels_list)
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels_list) if l == label]
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=label)

        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.legend(loc='lower right', bbox_to_anchor=(1.14, 0), frameon=False)
        plt.savefig(os.path.join(model_save_path, 't-SNE.png'))
        plt.close()

        '''混淆矩阵'''
        guess = prediction_list
        fact = labels_list
        classes = list(set(fact))
        classes.sort()
        confusion = confusion_matrix(fact, guess)
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, cmap=plt.cm.GnBu)  # Using Greys colormap
        indices = range(len(confusion))
        plt.xticks(indices, classes, fontsize=12)
        plt.yticks(indices, classes, fontsize=12)
        plt.colorbar()
        # plt.xlabel('预测标签', fontdict={'family': 'SimSun', 'size': 15})  # 宋体
        # plt.ylabel('实际标签', fontdict={'family': 'SimSun', 'size': 15})  # 宋体
        plt.xlabel('预测标签')  
        plt.ylabel('实际标签')  
        threshold = confusion.max() / 2

        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                color = "white" if confusion[second_index][first_index] > threshold else "black"
                plt.text(first_index, second_index, confusion[second_index][first_index],
                         va='center', ha='center', fontsize=12, color=color)

        plt.savefig(os.path.join(model_save_path, 'Confusion_Matrix.png'))
        plt.close()


    return test_acc, test_loss, run_time, precision, recall, f1, fpr, g_means, binary_accuracy


train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []


def train(num_epochs):
    min_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for step, (series, labels) in enumerate(train_loader):
            series = series.to(device)
            labels = labels.to(device)
            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            _, outputs = model(series)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagation loss
            loss.backward()
            # Adjust parameters according to the computed gradients
            optimizer.step()
            train_loss += loss.item() * series.size(0)
            _, prediction = torch.max(outputs, 1)
            train_acc += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

        train_acc = train_acc / len(train_set)
        train_loss = train_loss / len(train_set)
        # Evaluate on the test set
        test_acc, test_loss = test()
        # Save the model
        if test_loss <= min_loss:
            min_loss = test_loss
            save_models(epoch)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(
            f"Epoch {epoch}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc} , Test Loss: {test_loss}")

    print('train_loss_list:', train_loss_list)
    print('train_acc_list:', train_acc_list)
    print('test_loss_list:', test_loss_list)
    print('test_acc_list:', test_acc_list)

    # 寻找测试集的最高准确率和对应的epoch
    best_test_acc = max(test_acc_list)
    best_epoch = len(test_acc_list) - test_acc_list[::-1].index(best_test_acc) - 1  # 找到最后一个最高准确率在列表中的索引值

    print(
        f"Best Epoch {best_epoch}, Train Accuracy: {train_acc_list[best_epoch]} , Train Loss: {train_loss_list[best_epoch]} , Test Accuracy: {test_acc_list[best_epoch]} , Test Loss: {test_loss_list[best_epoch]}")

    # 将训练记录保存到txt文件
    with open(model_save_path + '/训练记录.txt', 'w') as txt_file:
        for epoch, (train_acc, train_loss, test_acc, test_loss) in enumerate(
                zip(train_acc_list, train_loss_list, test_acc_list, test_loss_list), start=0):
            txt_file.write(
                f"Epoch {epoch}, Train Accuracy: {train_acc}, Train Loss: {train_loss}, Test Accuracy: {test_acc}, Test Loss: {test_loss}\n")
        txt_file.write('train_loss_list: ' + str(train_loss_list) + '\n')
        txt_file.write('train_acc_list: ' + str(train_acc_list) + '\n')
        txt_file.write('test_loss_list: ' + str(test_loss_list) + '\n')
        txt_file.write('test_acc_list: ' + str(test_acc_list) + '\n')

    # 将训练记录写入csv文件
    with open(model_save_path + '/训练记录.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # 写入每一行的数据
        writer.writerow(train_loss_list)
        writer.writerow(test_loss_list)
        writer.writerow(train_acc_list)
        writer.writerow(test_acc_list)

    # 设置全局字体为Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    # 创建横坐标 - 迭代次数
    iterations = range(1, len(train_loss_list) + 1)

    # 绘制图表
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_loss_list, 'b--o', label='训练集', linewidth=2.5)  # 蓝色虚线表示训练集
    plt.plot(iterations, test_loss_list, 'r-.s', label='测试集', linewidth=2.5)  # 红色实线表示测试集

    # 设置图表标题和坐标轴标签
    # plt.xlabel('迭代次数', fontname='SimSun', fontsize=14)
    # plt.ylabel('损失值', fontname='SimSun', fontsize=14)
    # plt.legend(prop={'family': 'SimSun'})
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('损失值',  fontsize=14)
    plt.legend()

    plt.savefig(model_save_path + '/损失值迭代曲线.png', dpi=300)

    # 显示图表
    plt.show()

    # 绘制图表
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_acc_list, 'b--o', label='训练集', linewidth=2.5)  # 蓝色虚线表示训练集
    plt.plot(iterations, test_acc_list, 'r-.s', label='测试集', linewidth=2.5)  # 红色实线表示测试集

    # 设置图表标题和坐标轴标签
    # plt.xlabel('迭代次数', fontname='SimSun', fontsize=14)
    # plt.ylabel('准确率', fontname='SimSun', fontsize=14)
    # plt.legend(prop={'family': 'SimSun'})
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('准确率',  fontsize=14)
    plt.legend()

    plt.savefig(model_save_path + '/准确率迭代曲线.png', dpi=300)

    # 显示图表
    plt.show()


if __name__ == "__main__":
    if train_test == 'train':
        train(train_epoch)
    elif train_test == 'test':
        test_acc, test_loss, run_time, precision, recall, f1, fpr, g_means, binary_accuracy = test_model(True)
        print(' Test Accuracy', test_acc, ' Test Loss', test_loss, ' Run Time', run_time, ' Precision', precision, ' Recall', recall, ' F1', f1, ' FPR', fpr, ' G-means', g_means, ' Binary Accuracy', binary_accuracy)
