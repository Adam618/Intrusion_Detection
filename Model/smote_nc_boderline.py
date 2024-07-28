# -*- coding: utf-8 -*-

# import numpy as np
# import pandas as pd
# from sklearn.datasets import make_classification
# from sklearn.neighbors import NearestNeighbors
# from collections import Counter


# class BorderlineSMOTENC:
#     def __init__(self, categorical_features, random_state=None, k_neighbors=5, n_samples_multiplier=5):
#         self.categorical_features = categorical_features
#         self.random_state = random_state
#         self.k_neighbors = k_neighbors
#         self.n_samples_multiplier = n_samples_multiplier

#     def fit_resample(self, X, y):
#         np.random.seed(self.random_state)

#         # 分离数值特征和分类特征
#         X_num = X[:, [i for i in range(X.shape[1]) if i not in self.categorical_features]]
#         X_cat = X[:, self.categorical_features]

#         # 找到所有少数类样本
#         class_counts = Counter(y)
#         majority_class = max(class_counts, key=class_counts.get)
#         minority_classes = [cls for cls in class_counts if cls != majority_class]

#         synthetic_samples = []
#         synthetic_labels = []

#         for minority_class in minority_classes:
#             X_minority = X[y == minority_class]

#             neigh = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
#             neigh.fit(X)
#             nns = neigh.kneighbors(X_minority, return_distance=False)[:, 1:]

#             for sample, neighbors in zip(X_minority, nns):
#                 # 识别危险样本：少数类样本的 k 近邻中多数类样本超过一半
#                 majority_neighbor_count = sum(y[neighbors] == majority_class)
#                 if majority_neighbor_count > self.k_neighbors / 2:
#                     num_neighbors = neighbors[y[neighbors] == minority_class]

#                     for _ in range(self.n_samples_multiplier):
#                         for neighbor in num_neighbors:
#                             diff = X[neighbor] - sample
#                             gap = np.random.random()

#                             new_sample = sample + gap * diff

#                             # 对分类特征进行多数投票
#                             new_cat_features = []
#                             for feature in self.categorical_features:
#                                 feature_values = X[neighbors, feature]
#                                 most_common = Counter(feature_values).most_common(1)[0][0]
#                                 new_cat_features.append(most_common)

#                             new_sample[self.categorical_features] = new_cat_features
#                             synthetic_samples.append(new_sample)
#                             synthetic_labels.append(minority_class)

#         X_resampled = np.vstack([X, synthetic_samples])
#         y_resampled = np.hstack([y, synthetic_labels])

#         return X_resampled, y_resampled


# # 示例数据
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
#                            n_redundant=10, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# # 假设第10列和第15列是分类特征
# categorical_features = [10, 15]

# # 实例化并使用BorderlineSMOTENC
# borderline_smote_nc = BorderlineSMOTENC(categorical_features=categorical_features, random_state=42, k_neighbors=5,
#                                         n_samples_multiplier=5)
# X_resampled, y_resampled = borderline_smote_nc.fit_resample(X, y)

# # 将结果转换为DataFrame以便查看
# df_resampled = pd.DataFrame(X_resampled, columns=[f'feature_{i}' for i in range(X.shape[1])])
# df_resampled['target'] = y_resampled

# df_resampled.to_csv("temp.csv",index=False)

# df_resampled.head()

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
import re

class BorderlineSMOTENC:
    def __init__(self, categorical_features, random_state=None, k_neighbors=5, n_samples_multiplier=5):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_samples_multiplier = n_samples_multiplier

    def fit_resample(self, X, y):
        np.random.seed(self.random_state)

        # 分离数值特征和分类特征
        X_num = X[:, [i for i in range(X.shape[1]) if i not in self.categorical_features]]
        X_cat = X[:, self.categorical_features]

        # 找到所有少数类样本
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [cls for cls in class_counts if cls != majority_class]

        synthetic_samples = []
        synthetic_labels = []

        for minority_class in minority_classes:
            X_minority = X[y == minority_class]

            neigh = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
            neigh.fit(X)
            nns = neigh.kneighbors(X_minority, return_distance=False)[:, 1:]

            for sample, neighbors in zip(X_minority, nns):
                # 识别危险样本：少数类样本的 k 近邻中多数类样本超过一半
                majority_neighbor_count = sum(y[neighbors] == majority_class)
                if majority_neighbor_count > self.k_neighbors / 2:
                    num_neighbors = neighbors[y[neighbors] == minority_class]
                    # 对每个少数类样本进行增强
                    for _ in range(self.n_samples_multiplier):
                        for neighbor in num_neighbors:
                            diff = X[neighbor] - sample
                            gap = np.random.random()

                            new_sample = sample + gap * diff

                            # 对分类特征进行多数投票
                            new_cat_features = []
                            for feature in self.categorical_features:
                                feature_values = X[neighbors, feature]
                                most_common = Counter(feature_values).most_common(1)[0][0]
                                new_cat_features.append(most_common)

                            new_sample[self.categorical_features] = new_cat_features
                            synthetic_samples.append(new_sample)
                            synthetic_labels.append(minority_class)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, synthetic_labels])

        return X_resampled, y_resampled

def identify_one_hot_columns(column_names):
    """
    识别哪些列属于同一个原始分类变量的one-hot编码
    """
    one_hot_groups = defaultdict(list)
    pattern = re.compile(r'(\d+)_\d+')
    
    for col in column_names:
        match = pattern.match(col)
        if match:
            original_feature = int(match.group(1))
            one_hot_groups[original_feature].append(col)

    return one_hot_groups

# 示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                           n_redundant=10, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# 假设第10列和第15列是分类特征，并进行one-hot编码
categorical_features = [10, 15]
X[:, categorical_features] = np.random.randint(0, 5, size=(X.shape[0], len(categorical_features)))

# 实例化并使用BorderlineSMOTENC进行数据增强
borderline_smote_nc = BorderlineSMOTENC(categorical_features=categorical_features, random_state=42, k_neighbors=5, n_samples_multiplier=5)
X_resampled, y_resampled = borderline_smote_nc.fit_resample(X, y)

# 对增强后的数据进行one-hot编码
df = pd.DataFrame(X_resampled, columns=[f'feature_{i}' for i in range(X.shape[1])])
encoder = OneHotEncoder(sparse=False)
X_resampled_cat = encoder.fit_transform(df.iloc[:, categorical_features])
X_resampled_num = df.drop(columns=[f'feature_{i}' for i in categorical_features]).values
X_resampled_onehot = np.hstack((X_resampled_num, X_resampled_cat))

# 获取one-hot编码后的列名
encoded_feature_names = encoder.get_feature_names_out([f'feature_{i}' for i in categorical_features])
original_feature_names = [f'feature_{i}' for i in range(X.shape[1]) if i not in categorical_features]
all_feature_names = original_feature_names + list(encoded_feature_names)

# 将结果转换为DataFrame以便查看
df_resampled = pd.DataFrame(X_resampled_onehot, columns=all_feature_names)
df_resampled['target'] = y_resampled

# 保存并查看结果
df_resampled.to_csv("resampled_data.csv", index=False)
print(df_resampled.head())

