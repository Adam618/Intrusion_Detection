import pandas as pd
from time import time
from Dataset.BorderlineSMOTENC import BorderlineSMOTENC
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, KMeansSMOTE

def Flag():
    list_ = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    flag = {name: idx for idx, name in enumerate(list_)}
    return flag

def Label():
    label_list = [
        # nomal
        ['normal'],
        # DOS
        ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'mailbomb', 'processtable', 'udpstorm'],  # 后三个
        # Probing
        ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'],  # 后两个
        # R2L
        ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'named', 'sendmail', 'snmpgetattack', 'snmpguess', 'warezmaster', 'worm', 'xlock', 'xsnoop'],  # named
        # U2R
        ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 'ps', 'rootkit', 'sqlattack', 'xterm']  # httptunnel
    ]
    label = {name: idx for idx, sublist in enumerate(label_list) for name in sublist}
    return label

def apply_one_hot(df_train, df_test, symbolic):
    print("开始独热编码")
    # 添加标识列
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    df_combined = pd.concat([df_train, df_test])

    # columns_name = [str(i) for i in range(df_combined.shape[1])]
    # columns_name[-1] = 'source'  # 修改最后一列的列名为'source'
    # df_combined.columns = columns_name
    # symbolic_name = [str(i) for i in symbolic]
    # get_dummies是把one-hot编码追加到最后一列，并不会改变之前列的顺序
    df_combined = pd.get_dummies(df_combined, columns=symbolic, dtype=int)

    # label_column = df_combined.pop(str(41))
    # df_combined[str(41)] = label_column
    
    # 根据'source'列拆分数据集
    df_train = df_combined[df_combined['source'] == 'train'].drop(columns=['source'])
    df_test = df_combined[df_combined['source'] == 'test'].drop(columns=['source'])
    # print(df_combined)
    print("独热编码完毕")
    return df_train, df_test


# 筛选出连续特征那个 仅对连续特征归一化
def apply_continues_scalar(df_train, df_test, continues):
    print('continues',continues)
    for column in continues:
        df_j_avg = df_train[column].mean()  # 均值
        df_j_std = df_train[column].std()   # 标准差
        
        if df_j_std == 0:
            df_train[column] = 0
            df_test[column] = 0
            continue
        
        # 标准化
        df_train[column] = (df_train[column] - df_j_avg) / df_j_std
        df_test[column] = (df_test[column] - df_j_avg) / df_j_std
    return df_train, df_test


def apply_smote_etc(df_train, df_test, args, symbolic, continues):
    
    # df_train.to_csv('temp.csv',index=False)
    # y_column = df_train.columns[-2]
    # # 映射标签列
    # df_train[y_column] = df_train[y_column]
    # 指定 X 和 y
  
    # print('df_train[y_column].value_counts()', df_train[y_column].value_counts())

    # 根据传入的参数选择 SMOTE 方法
    if args.smote_type == 'smote':
        # smote应该先离散数据one-hot编码、smote、连续变量归一化
        df_train, df_test = apply_one_hot(df_train, df_test, symbolic)
        smote = SMOTE(random_state=args.seed_value, sampling_strategy=args.sampling_strategy)
    elif args.smote_type == 'borderline':
        # borderline_smote应该先离散数据one-hot编码、borderline_smote、连续变量归一化
        df_train, df_test = apply_one_hot(df_train, df_test, symbolic)
        smote = BorderlineSMOTE(random_state=args.seed_value, k_neighbors=args.k_neighbors, sampling_strategy=args.sampling_strategy)
    elif args.smote_type == 'boderline_smotenc':
        # boderline_smotenc应该先boderline_smotenc、连续变量归一化、one-hot编码,boderline_smotenc并不会改变离散变量类别
        smote = BorderlineSMOTENC(symbolic, random_state=args.seed_value, k_neighbors=args.k_neighbors, sampling_strategy=args.sampling_strategy)

    X = df_train.drop(columns=['label']).values
    y = df_train['label'].values
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_resampled_df = pd.DataFrame(X_resampled, columns=df_train.columns.drop('label'))
    y_resampled_df = pd.DataFrame(y_resampled, columns=['label'])
    df_train_resampled = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    # 这里的逻辑是筛选出连续特征，仅对连续特征归一化
    df_train_resampled, df_test = apply_continues_scalar(df_train_resampled, df_test, continues)
   
    if args.smote_type == 'boderline_smotenc':
        df_train_resampled, df_test = apply_one_hot(df_train_resampled, df_test, symbolic)


    

    # elif args.smote_type == 'borderline':
    #     smote = BorderlineSMOTE(random_state=args.seed_value, k_neighbors=args.k_neighbors, sampling_strategy=args.sampling_strategy)
    # elif args.smote_type == 'smotenc':
    #     smote = SMOTENC(categorical_features=symbolic, random_state=args.seed_value, sampling_strategy=args.sampling_strategy)
    # elif args.smote_type == 'none':
    #     return df_train
    # else:
    #     raise ValueError(f"Unknown smote_type: {args.smote_type}")

    # 进行采样
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    # print('df_train.columns.shape', df_train.columns.shape)
    # print('X_resampled.shape',X_resampled.shape)
    # X_resampled_df = pd.DataFrame(X_resampled, columns=df_train.columns.drop(str(y_column)))
    # y_resampled_df = pd.DataFrame(y_resampled, columns=[y_column])
    
    # # 将 X_resampled 和 y_resampled 合并
    # df_train_combined = pd.concat([X_resampled_df, y_resampled_df], axis=1)
    # # 获取原列顺序
    # columns = list(df_train.columns)
    print("增强前训练集维度：", df_train.shape)
    # 调整合并后的 DataFrame 列顺序，确保标签还是倒数第二列
    # df_train = df_train_combined[columns]
    print("增强后训练集维度：", df_train_resampled.shape)

    return df_train_resampled, df_test




def handle(train_file, test_file, save_train_file, save_test_file, args):
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    num_columns = df_train.shape[1]

    # 生成列名列表，如 ['0', '1', '2', ..., 'n-1']
    column_names = [str(i) for i in range(num_columns)]

    # 设置列名
    df_train.columns = column_names
    df_test.columns = column_names

    protocol = Protocol()
    df_train['1'] = df_train['1'].map(protocol)  # 协议
    df_test['1'] = df_test['1'].map(protocol)

    service = Service()
    df_train['2'] = df_train['2'].map(service)  # 服务
    df_test['2'] = df_test['2'].map(service)

    flag = Flag()
    df_train['3'] = df_train['3'].map(flag)  # 连接状态
    df_test['3'] = df_test['3'].map(flag)

    label = Label()
    df_train['41'] = df_train['41'].map(label)  # 标签
    df_test['41'] = df_test['41'].map(label)
    df_train.rename(columns={'41': 'label'}, inplace=True)
    
    df_test.rename(columns={'41': 'label'}, inplace=True)
    # 标注label列

    symbolic = ['1', '2', '3', '6', '11', '20', '21']  # 7个离散型特征的索引(不包括标签列)
    continues = [col for col in df_train.columns if col not in symbolic + ['label']]
    # 这一步涵盖数据增强、连续变量归一化、离散变量one-hot等过程，随着smote类型不同，先后顺序可能不一致
    df_train, df_test = apply_smote_etc(df_train, df_test, args, symbolic, continues)

    label_column = df_train.pop('label')
    df_train['label'] = label_column

    label_column = df_test.pop('label')
    df_test['label'] = label_column
    

    # for j in range(df_train.shape[1]):
    #     if j in symbolic or j == 41:  # 排除离散型和标签
    #         continue
    #     df_j_avg = df_train.iloc[:, j].mean()  # 均值
    #     df_j_std = df_train.iloc[:, j].std()   # 标准差
        
    #     if df_j_std == 0:
    #         df_train.iloc[:, j] = 0
    #         df_test.iloc[:, j] = 0
    #         continue
    #     # 标准化
    #     df_train.iloc[:, j] = (df_train.iloc[:, j] - df_j_avg) / df_j_std
    #     df_test.iloc[:, j] = (df_test.iloc[:, j] - df_j_avg) / df_j_std



    # print("均处理完毕，开始独热编码")

    # # 添加标识列
    # df_train['source'] = 'train'
    # df_test['source'] = 'test'
    # df_combined = pd.concat([df_train, df_test])

    # columns_name = [str(i) for i in range(df_combined.shape[1])]
    # columns_name[-1] = 'source'  # 修改最后一列的列名为'source'
    # df_combined.columns = columns_name
    # symbolic_name = [str(i) for i in symbolic]
    # df_combined = pd.get_dummies(df_combined, columns=symbolic_name,dtype=int)
    # # print(df_combined)
    # print("独热编码完毕")
    

    


    try:
        # df_train.to_csv(save_train_file, index=None, header=None)
        # df_test.to_csv(save_test_file, index=None, header=None)
        df_train.to_csv(save_train_file, index=None)
        df_test.to_csv(save_test_file, index=None)
    except UnicodeEncodeError:
        print('写入错误')

def Protocol():
    protocol = {'tcp': 0, 'udp': 1, 'icmp': 2}
    return protocol

def Service():
    list_ = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
             'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames',
             'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap',
             'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
             'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
             'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i',
             'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    service = {name: idx for idx, name in enumerate(list_)}
    return service



# start = time()
# handle('/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTrain+.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTest+.csv', 
#        '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTrain+_progressed.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTest+_progressed.csv')
# end = time()
# print("共耗时：" + str(round((end - start) / 60, 3)) + " min")















