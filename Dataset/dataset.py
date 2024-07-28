# import pandas as pd
# from time import time

# def handle():
#     datafile = '/root/CNN-LSTM/Dataset/KDDTrain+.csv'
#     savefile = '/root/CNN-LSTM/Dataset/KDDTrain+progressed.csv'
#     df = pd.read_csv(datafile, header=None)

#     protocol = Protocol()
#     df[1] = df[1].map(protocol)#协议
#     service = Service()
#     df[2] = df[2].map(service)#服务
#     flag = Flag()
#     df[3] = df[3].map(flag)#连接状态
#     label = Label()
#     df[41] = df[41].map(label)#标签
#     class_counts = df[41].value_counts()

#     print(df.columns)
#     print("Class Counts:",class_counts)

#     symbolic = [1,2,3,6,11,20,21]         # 7个离散型特征的索引
    
#     for j in range(df.shape[1]):
#         if j in symbolic or j>=31:       # 排除离散型和后十个
#             continue
#         df_j_avg = df[j].mean()         # 均值
#         df_j_mad = df[j].mad()          # 平均绝对偏差
#         if df_j_avg==0 or df_j_mad==0:
#             df[j]=0
#             continue
#         # 标准化
#         df[j] = (df[j]-df_j_avg)/df_j_mad
#         # 归一化
#         df[j] = (df[j]-df[j].min())/(df[j].max() - df[j].min())
#         print(str(j)+" 列处理完毕，剩余 "+str(df.shape[1]-j)+" 列未处理")

#     print("均处理完毕，开始独热编码")

#     columns_name = [str(i) for i in range(df.shape[1])]
#     df[41] = df[41].map(label)
#     df.columns = columns_name
#     symbolic_name = [str(i) for i in symbolic]
#     df_result = pd.get_dummies(df, columns=symbolic_name)
#     print("独热编码完毕")

#     try:
#         df_result.to_csv(savefile, index=None)
#     except UnicodeEncodeError:
#         print('写入错误')


# def Protocol():
#     protocol = {'tcp':0,'udp':1,'icmp':2}
#     return protocol

# def Service():
#     list_ = ['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
#                  'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
#                  'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
#                  'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
#                  'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
#                  'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
#                  'uucp','uucp_path','vmnet','whois','X11','Z39_50']
#     service = {}
#     for i in range(len(list_)):
#         service[list_[i]] = i
#     return service

# def Flag():
#     list_ = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
#     flag = {}
#     for i in range(len(list_)):
#         flag[list_[i]] = i
#     return flag

# def Label():
#     label_list = [
#         # nomal
#         ['normal'],
#         # DOS
#         ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop','apache2','mailbomb','processtable','udpstorm'],#后三个
#         #Probing
#         ['ipsweep', 'nmap', 'portsweep', 'satan','mscan','saint'],#后两个
#         #R2L
#         ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster','named','sendmail','snmpgetattack','snmpguess','warezmaster','worm','xlock','xsnoop'],#named
#         #U2R
#         ['buffer_overflow', 'loadmodule', 'perl', 'rootkit','httptunnel','ps','rootkit','sqlattack','xterm']#httptunnel
#     ]
#     label = {}
#     for i in range(len(label_list)):
#         for j in range(len(label_list[i])):
#             label[label_list[i][j]] = i
#     return label

# start = time()
# handle()
# end = time()
# print("共耗时："+str(round((end-start)/60,3))+" min")




import pandas as pd
from time import time

def handle(train_file, test_file, save_train_file, save_test_file):
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    protocol = Protocol()
    df_train[1] = df_train[1].map(protocol)  # 协议
    df_test[1] = df_test[1].map(protocol)

    service = Service()
    df_train[2] = df_train[2].map(service)  # 服务
    df_test[2] = df_test[2].map(service)

    flag = Flag()
    df_train[3] = df_train[3].map(flag)  # 连接状态
    df_test[3] = df_test[3].map(flag)

    label = Label()
    df_train[41] = df_train[41].map(label)  # 标签
    df_test[41] = df_test[41].map(label)

    symbolic = [1, 2, 3, 6, 11, 20, 21]  # 7个离散型特征的索引

    for j in range(df_train.shape[1]):
        if j in symbolic or j >= 31:  # 排除离散型和后十个
            continue
        df_j_avg = df_train[j].mean()  # 均值
        df_j_mad = (df_train[j] - df_train[j].mean()).abs().mean()  # 平均绝对偏差
        if df_j_avg == 0 or df_j_mad == 0:
            df_train[j] = 0
            df_test[j] = 0
            continue
        # 标准化
        df_train[j] = (df_train[j] - df_j_avg) / df_j_mad
        df_test[j] = (df_test[j] - df_j_avg) / df_j_mad
        # 归一化
        df_train[j] = (df_train[j] - df_train[j].min()) / (df_train[j].max() - df_train[j].min())
        df_test[j] = (df_test[j] - df_train[j].min()) / (df_train[j].max() - df_train[j].min())
        # df_test[j] = (df_test[j] - df_test[j].min()) / (df_test[j].max() - df_test[j].min())
        print(str(j) + " 列处理完毕，剩余 " + str(df_train.shape[1] - j) + " 列未处理")

    print("均处理完毕，开始独热编码")

    # 添加标识列
    df_train['source'] = 'train'
    df_test['source'] = 'test'
    df_combined = pd.concat([df_train, df_test])

    columns_name = [str(i) for i in range(df_combined.shape[1])]
    columns_name[-1] = 'source'  # 修改最后一列的列名为'source'
    df_combined.columns = columns_name
    symbolic_name = [str(i) for i in symbolic]
    df_combined = pd.get_dummies(df_combined, columns=symbolic_name,dtype=int)
    # print(df_combined)
    print("独热编码完毕")
    

    label_column = df_combined.pop(str(41))
    df_combined[str(41)] = label_column
    
    # 根据'source'列拆分数据集
    df_train = df_combined[df_combined['source'] == 'train'].drop(columns=['source'])
    df_test = df_combined[df_combined['source'] == 'test'].drop(columns=['source'])

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

start = time()
handle('/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTrain+.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/原始数据集/KDDTest+.csv', 
       '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTrain+_progressed.csv', '/root/autodl-tmp/7.15_网络入侵检测/CNN-LSTM/Dataset/NSL-KDD/KDDTest+_progressed.csv')
end = time()
print("共耗时：" + str(round((end - start) / 60, 3)) + " min")















