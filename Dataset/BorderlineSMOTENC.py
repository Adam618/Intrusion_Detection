# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# class BorderlineSMOTENC:
#     def __init__(self, categorical_features, random_state=None, k_neighbors=5, sampling_strategy='auto'):
#         # 现在的逻辑是用字典sampling_strategy传入各个类别的生成样本数，而其他smote库的逻辑是传入生成后的各个类别总样本数
#         self.categorical_features = [int(x) for x in categorical_features]
#         self.random_state = random_state
#         self.k_neighbors = k_neighbors
#         self.sampling_strategy = sampling_strategy

#     def fit_resample(self, X, y):
#         np.random.seed(self.random_state)

#         # 分离数值特征和分类特征
#         continuous_features = [i for i in range(X.shape[1]) if i not in self.categorical_features]
#         X_continuous = X[:, continuous_features]

#         # 找到所有少数类样本
#         class_counts = Counter(y)
#         majority_class = max(class_counts, key=class_counts.get)
        
#         # 按数量对少数类进行排序
#         minority_classes = sorted(
#             [cls for cls in class_counts if cls != majority_class],
#             key=lambda cls: class_counts[cls]
#         )
#         print("majority_class", majority_class)
#         print("minority_classes", minority_classes)

#         synthetic_samples = []
#         synthetic_labels = []

#         max_class_count = class_counts[majority_class]

#         for minority_class in minority_classes:
#             X_minority = X[y == minority_class]
#             minority_count = class_counts[minority_class]

#             # 根据 sampling_strategy 计算需要生成的样本数量
#             if isinstance(self.sampling_strategy, str) and self.sampling_strategy == 'auto':
#                 n_samples_to_generate = max_class_count - minority_count
#             elif isinstance(self.sampling_strategy, dict):
#                 n_samples_to_generate = self.sampling_strategy.get(minority_class, max_class_count - minority_count)
#             elif isinstance(self.sampling_strategy, float):
#                 n_samples_to_generate = int(self.sampling_strategy * max_class_count) - minority_count
#             elif callable(self.sampling_strategy):
#                 n_samples_to_generate = self.sampling_strategy(y, minority_class)
#             else:
#                 raise ValueError("Invalid sampling_strategy. Use 'auto', dict, float, or callable.")

#             if n_samples_to_generate <= 0:
#                 continue  # 无需生成新的样本

#             print(f"Class {minority_class}: n_samples_to_generate = {n_samples_to_generate}")

#             neigh = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
#             neigh.fit(X_continuous)
#             nns = neigh.kneighbors(X_minority[:, continuous_features], return_distance=False)[:, 1:]

#             # 预先计算少数类样本的少数类和多数类邻居
#             dangerous_samples = []
#             num_neighbors_list = []

#             # 遍历所有少数类样本
#             for sample, neighbors in zip(X_minority, nns):
#                 majority_neighbor_count = sum(y[neighbors] == majority_class)
#                 num_neighbors = neighbors[y[neighbors] == minority_class]
                
#                 # 检查是否是“危险”样本
#                 if majority_neighbor_count > self.k_neighbors / 2 and majority_neighbor_count < self.k_neighbors:
#                     dangerous_samples.append((sample, neighbors))
#                 num_neighbors_list.append(neighbors)

#             total_dangerous_samples = len(dangerous_samples)
#             if total_dangerous_samples == 0:
#                 continue  # 如果没有危险样本，则跳过该类

#             # 计算每个危险样本生成的数量
#             samples_to_generate_per_sample = n_samples_to_generate // total_dangerous_samples
#             remainder = n_samples_to_generate % total_dangerous_samples
            
#             # 从少数类样本选择少数类的邻居进行插值
#             for i, (sample, neighbors) in tqdm(enumerate(dangerous_samples), total=total_dangerous_samples, desc=f'Processing class {minority_class}'):
#                 if len(neighbors) == 0:
#                     continue  # 如果没有邻居，则跳过

#                 # 计算当前危险样本需要生成的数量
#                 num_samples_to_generate = samples_to_generate_per_sample
#                 if remainder > 0:
#                     num_samples_to_generate += 1
#                     remainder -= 1

#                 for _ in range(num_samples_to_generate):
#                     # 随机选择一个少数类样本进行插值
#                     num_neighbors = neighbors[y[neighbors] == minority_class]
#                     if len(num_neighbors) == 0:
#                         continue
#                     neighbor = np.random.choice(num_neighbors)
#                     diff = X[neighbor] - sample
#                     gap = np.random.random()

#                     new_sample = sample + gap * diff

#                     # 对分类特征进行多数投票（使用所有邻居）
#                     new_cat_features = []
#                     for feature in self.categorical_features:
#                         feature_values = X[neighbors, feature]
#                         most_common = Counter(feature_values).most_common(1)[0][0]
#                         new_cat_features.append(most_common)

#                     new_sample[self.categorical_features] = new_cat_features
#                     synthetic_samples.append(new_sample)
#                     synthetic_labels.append(minority_class)

#         X_resampled = np.vstack([X, synthetic_samples])
#         y_resampled = np.hstack([y, synthetic_labels])

#         return X_resampled, y_resampled
    



# # class BorderlineSMOTENC:
# #     def __init__(self, categorical_features, random_state=None, k_neighbors=5, sampling_strategy='auto'):
# #         self.categorical_features = categorical_features
# #         self.random_state = random_state
# #         self.k_neighbors = k_neighbors
# #         self.sampling_strategy = sampling_strategy

# #     def fit_resample(self, X, y):
# #         np.random.seed(self.random_state)

# #         # 分离数值特征和分类特征
# #         continuous_features = [i for i in range(X.shape[1]) if i not in self.categorical_features]
# #         X_continuous = X[:, continuous_features]

# #         # 找到所有少数类样本
# #         class_counts = Counter(y)
# #         majority_class = max(class_counts, key=class_counts.get)
        
# #         # 按数量对少数类进行排序
# #         minority_classes = sorted(
# #             [cls for cls in class_counts if cls != majority_class],
# #             key=lambda cls: class_counts[cls]
# #         )

# #         # 计算连续特征的标准差的中值
# #         minority_data = X[y != majority_class, :]
# #         continuous_stds = np.std(minority_data[:, continuous_features], axis=0)
# #         med = np.median(continuous_stds)

# #         synthetic_samples = []
# #         synthetic_labels = []

# #         max_class_count = class_counts[majority_class]

# #         for minority_class in minority_classes:
# #             X_minority = X[y == minority_class]
# #             minority_count = class_counts[minority_class]

# #             # 根据 sampling_strategy 计算需要生成的样本数量
# #             if isinstance(self.sampling_strategy, str) and self.sampling_strategy == 'auto':
# #                 n_samples_to_generate = max_class_count - minority_count
# #             elif isinstance(self.sampling_strategy, dict):
# #                 n_samples_to_generate = self.sampling_strategy.get(minority_class, max_class_count - minority_count)
# #             elif isinstance(self.sampling_strategy, float):
# #                 n_samples_to_generate = int(self.sampling_strategy * max_class_count) - minority_count
# #             elif callable(self.sampling_strategy):
# #                 n_samples_to_generate = self.sampling_strategy(y, minority_class)
# #             else:
# #                 raise ValueError("Invalid sampling_strategy. Use 'auto', dict, float, or callable.")

# #             if n_samples_to_generate <= 0:
# #                 continue  # 无需生成新的样本

# #             # 使用SMOTE-NC的距离计算方法
# #             def smote_nc_distance(X1, X2):
# #                 # 确保 X1 和 X2 是二维的
# #                 if len(X1.shape) == 1:
# #                     X1 = X1.reshape(1, -1)
# #                 if len(X2.shape) == 1:
# #                     X2 = X2.reshape(1, -1)
# #                 continuous_diff = X1[:, continuous_features] - X2[:, continuous_features]
# #                 continuous_dist = np.sqrt(np.sum(continuous_diff ** 2, axis=1))

# #                 categorical_diff = X1[:, self.categorical_features] != X2[:, self.categorical_features]
# #                 categorical_dist = np.sum(categorical_diff, axis=1) * (med ** 2)

# #                 return continuous_dist + categorical_dist



# #             # 自定义距离函数的最近邻算法
# #             neigh = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric=smote_nc_distance)
# #             neigh.fit(X)
# #             nns = neigh.kneighbors(X_minority, return_distance=False)[:, 1:]

# #             # 预先计算少数类样本的少数类和多数类邻居
# #             dangerous_samples = []
# #             num_neighbors_list = []

# #             # 遍历所有少数类样本
# #             for sample, neighbors in zip(X_minority, nns):
# #                 majority_neighbor_count = sum(y[neighbors] == majority_class)
# #                 num_neighbors = neighbors[y[neighbors] == minority_class]
                
# #                 # 检查是否是“危险”样本
# #                 if majority_neighbor_count > self.k_neighbors / 2 and majority_neighbor_count < self.k_neighbors:
# #                     dangerous_samples.append((sample, neighbors))
# #                 num_neighbors_list.append(neighbors)

# #             total_dangerous_samples = len(dangerous_samples)
# #             if total_dangerous_samples == 0:
# #                 continue  # 如果没有危险样本，则跳过该类

# #             # 计算每个危险样本生成的数量
# #             samples_to_generate_per_sample = n_samples_to_generate // total_dangerous_samples
# #             remainder = n_samples_to_generate % total_dangerous_samples
            
# #             # 从少数类样本选择少数类的邻居进行插值
# #             for i, (sample, neighbors) in tqdm(enumerate(dangerous_samples), total=total_dangerous_samples, desc=f'Processing class {minority_class}'):
# #                 if len(neighbors) == 0:
# #                     continue  # 如果没有邻居，则跳过

# #                 # 计算当前危险样本需要生成的数量
# #                 num_samples_to_generate = samples_to_generate_per_sample
# #                 if remainder > 0:
# #                     num_samples_to_generate += 1
# #                     remainder -= 1

# #                 for _ in range(num_samples_to_generate):
# #                     # 随机选择一个少数类样本进行插值
# #                     num_neighbors = neighbors[y[neighbors] == minority_class]
# #                     if len(num_neighbors) == 0:
# #                         continue
# #                     neighbor = np.random.choice(num_neighbors)
# #                     diff = X[neighbor] - sample
# #                     gap = np.random.random()

# #                     new_sample = sample + gap * diff

# #                     # 对分类特征进行多数投票（使用所有邻居）
# #                     new_cat_features = []
# #                     for feature in self.categorical_features:
# #                         feature_values = X[neighbors, feature]
# #                         most_common = Counter(feature_values).most_common(1)[0][0]
# #                         new_cat_features.append(most_common)

# #                     new_sample[self.categorical_features] = new_cat_features
# #                     synthetic_samples.append(new_sample)
# #                     synthetic_labels.append(minority_class)

# #         X_resampled = np.vstack([X, synthetic_samples])
# #         y_resampled = np.hstack([y, synthetic_labels])

# #         return X_resampled, y_resampled


class BorderlineSMOTENC:
    def __init__(self, categorical_features, random_state=None, k_neighbors=5, sampling_strategy='auto'):
        self.categorical_features = [int(x) for x in categorical_features]
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)

        # 分离数值特征和分类特征
        continuous_features = [i for i in range(X.shape[1]) if i not in self.categorical_features]
        X_continuous = X[:, continuous_features]

        # 找到所有少数类样本
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        
        # 按数量对少数类进行排序
        minority_classes = sorted(
            [cls for cls in class_counts if cls != majority_class],
            key=lambda cls: class_counts[cls]
        )
        print("majority_class", majority_class)
        print("minority_classes", minority_classes)

        synthetic_samples = []
        synthetic_labels = []

        max_class_count = class_counts[majority_class]

        for minority_class in minority_classes:
            X_minority = X[y == minority_class]
            minority_count = class_counts[minority_class]

            # 根据 sampling_strategy 计算生成后的总样本数量
            if isinstance(self.sampling_strategy, str) and self.sampling_strategy == 'auto':
                target_samples = max_class_count
            elif isinstance(self.sampling_strategy, dict):
                target_samples = self.sampling_strategy.get(minority_class, max_class_count)
            elif isinstance(self.sampling_strategy, float):
                target_samples = int(self.sampling_strategy * max_class_count)
            elif callable(self.sampling_strategy):
                target_samples = self.sampling_strategy(y, minority_class)
            else:
                raise ValueError("Invalid sampling_strategy. Use 'auto', dict, float, or callable.")

            n_samples_to_generate = target_samples - minority_count
            if n_samples_to_generate <= 0:
                continue  # 无需生成新的样本

            print(f"Class {minority_class}: n_samples_to_generate = {n_samples_to_generate}")

            neigh = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
            neigh.fit(X_continuous)
            nns = neigh.kneighbors(X_minority[:, continuous_features], return_distance=False)[:, 1:]

            # 预先计算少数类样本的少数类和多数类邻居
            dangerous_samples = []
            num_neighbors_list = []

            # 遍历所有少数类样本
            for sample, neighbors in zip(X_minority, nns):
                majority_neighbor_count = sum(y[neighbors] == majority_class)
                num_neighbors = neighbors[y[neighbors] == minority_class]
                
                # 检查是否是“危险”样本
                if majority_neighbor_count > self.k_neighbors / 2 and majority_neighbor_count < self.k_neighbors:
                    dangerous_samples.append((sample, neighbors))
                num_neighbors_list.append(neighbors)

            total_dangerous_samples = len(dangerous_samples)
            if total_dangerous_samples == 0:
                continue  # 如果没有危险样本，则跳过该类

            # 计算每个危险样本生成的数量
            samples_to_generate_per_sample = n_samples_to_generate // total_dangerous_samples
            remainder = n_samples_to_generate % total_dangerous_samples
            
            # 从少数类样本选择少数类的邻居进行插值
            for i, (sample, neighbors) in tqdm(enumerate(dangerous_samples), total=total_dangerous_samples, desc=f'Processing class {minority_class}'):
                if len(neighbors) == 0:
                    continue  # 如果没有邻居，则跳过

                # 计算当前危险样本需要生成的数量
                num_samples_to_generate = samples_to_generate_per_sample
                if remainder > 0:
                    num_samples_to_generate += 1
                    remainder -= 1

                for _ in range(num_samples_to_generate):
                    # 随机选择一个少数类样本进行插值
                    num_neighbors = neighbors[y[neighbors] == minority_class]
                    if len(num_neighbors) == 0:
                        continue
                    neighbor = np.random.choice(num_neighbors)
                    diff = X[neighbor] - sample
                    gap = np.random.random()

                    new_sample = sample + gap * diff

                    # 对分类特征进行多数投票（使用所有邻居）
                    new_cat_features = []
                    for feature in self.categorical_features:
                        # feature_values = X[neighbors, feature]
                        feature_values = X[num_neighbors, feature] # 仅使用少数类邻居效果更好一点+1.6%
                        most_common = Counter(feature_values).most_common(1)[0][0]
                        new_cat_features.append(most_common)

                    new_sample[self.categorical_features] = new_cat_features
                    synthetic_samples.append(new_sample)
                    synthetic_labels.append(minority_class)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, synthetic_labels])

        return X_resampled, y_resampled
