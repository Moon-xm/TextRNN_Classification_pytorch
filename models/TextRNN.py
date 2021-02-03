# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class config(object):
    def __init__(self):
        # 路径类 带*的是运行前的必要文件  未带*文件/文件夹若不存在则训练过程会生成
        self.train_path = 'data/train.txt'  # *
        self.dev_path = 'data/dev.txt'  # *
        self.class_ls_path = 'data/class.txt'  # *
        self.pretrain_dir = 'data/sgns.sogou.char'  # 前期下载的预训练词向量*
        self.test_path = 'data/test.txt'  # 若该文件不存在会加载dev.txt进行最终测试
        self.vocab_path = 'data/vocab.pkl'
        self.model_save_dir = 'checkpoint'
        self.model_save_name = self.model_save_dir + '/TextRNN.ckpt'  # 保存最佳dev acc模型

        # 可调整的参数
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz,  若不存在则后期生成
        # 随机初始化:random
        self.embedding_type = 'embedding_SougouNews.npz'
        self.use_gpu = True  # 是否使用gpu(有则加载 否则自动使用cpu)
        self.batch_size = 128
        self.pad_size = 32  # 句子长度限制  短补(<PAD>)长截
        self.num_epochs = 40  # 训练轮数
        self.num_workers = 0  # 启用多线程
        self.learning_rate = 0.001  # 训练发现0.001比0.01收敛快(Adam)
        self.embedding_dim = 300  # 词嵌入维度
        self.hidden_size = 300  # 隐藏层维度
        self.num_layers = 2  # RNN层数
        self.bidirectional = True  # 双向 or 单向
        self.require_improvement = 1  # 1个epoch若在dev上acc未提升则自动结束

        # 由前方参数决定  不用修改
        self.class_ls = [x.strip() for x in open(self.class_ls_path, 'r', encoding='utf-8').readlines()]
        self.num_class = len(self.class_ls)
        self.vocab_len = 0  # 词表大小(训练集总的字数(字符级)） 在embedding层作为参数 后期赋值
        self.embedding_pretrained = None  # 根据config.embedding_type后期赋值  random:None  else:tensor from embedding_type
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,
                                                          freeze=False)  # 表示训练过程词嵌入向量会更新
        else:
            self.embedding = nn.Embedding(config.vocab_len, config.embedding_dim,
                                          padding_idx=config.vocab_len - 1)  # PAD索引填充
        if config.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.config = config
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, batch_first=True,
                           bidirectional=config.bidirectional)  # (B, 1, S, e_d) -> output, hidden

        self.fc = nn.Linear(config.hidden_size*self.num_directions, config.num_class)  # -> (B, num_class)

    def forward(self, x):
        # 数据预处理时，x被处理成是一个tuple,其形状是: (data, length).
        # 其中data(b_size, seq_len),  length(batch_size)
        # x[0]:(b_size, seq_len)
        x = self.embedding(x[0])  # -> (B, S, e_d)
        h_0, c_0 = self._init_hidden(self.config, batchs=x.size(0))
        out, (hidden, c) = self.rnn(x,(h_0, c_0))  # hidden:(num_layer*nun_directions, B,  hidden_size)
        # output is batch_first but hidden not
        if self.num_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 沿batch_size 方向拼接-> (B, e_b*2)
        else:
            hidden_cat = hidden[-1]
        out = self.fc(hidden_cat)  # (B, e_b*2) -> (B, num_class)
        return out

    def _init_hidden(self, config, batchs):  # 初始化h_0和c_0 与GRU不同的是多了c_0（细胞状态）
        h_0 = torch.zeros(config.num_layers*self.num_directions, batchs,  config.hidden_size)
        c_0 = torch.zeros(config.num_layers*self.num_directions, batchs, config.hidden_size)
        return self._make_tensor(h_0), self._make_tensor(c_0)

    def _make_tensor(self, tensor):
        """
        函数说明： 将传入的tensor转移到cpu或gpu内

        Parameter：
        ----------
            tensor - 需转换的张量
            config.device - cpu or cuda:0
        Return:
        -------
            tensor_ret - 转换好的LongTensor类型张量
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2021-2-2 21:53:20
        """
        tensor_ret = tensor.to(self.config.device)
        return tensor_ret
