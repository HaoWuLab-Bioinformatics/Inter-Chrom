import torch.nn.functional as F
import torch
from torch import nn
import numpy as np

EPS = 1e-7
AdaptiveAvgPool1d_size = 768
embedding_matrix = np.load('DNABERT_embedding_matrix.npy')


class ECA(nn.Module):
    def __init__(self, k_size, seed=None):
        super(ECA, self).__init__()
        torch.manual_seed(seed=seed)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(-1, -2)
        b, c, _, = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2))
        x = x * y.expand_as(x)
        x = torch.sum(x, dim=-1)
        return x


class AttLayer(nn.Module):
    def __init__(self, attention_dim, kernel_num):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim
        # self.dense = nn.Linear(64, self.attention_dim)
        self.W = nn.Parameter(torch.randn(kernel_num, self.attention_dim))
        self.b = nn.Parameter(torch.randn(self.attention_dim))
        self.u = nn.Parameter(torch.randn(self.attention_dim, 1))
        self.act = nn.Tanh()

    def forward(self, x, mask=None):
        uit = torch.tanh(torch.add(torch.matmul(x, self.W), self.b))
        # uit = self.dense(x)
        ait = torch.matmul(uit, self.u)
        ait = torch.squeeze(ait, -1)
        ait = torch.exp(ait)

        if mask is not None:
            ait = torch.mul(ait, mask.float())
        ait1 = ait / torch.add(torch.sum(ait, dim=1, keepdim=True), torch.tensor(1e-7))

        ait1 = ait1.unsqueeze(-1)
        weighted_input = torch.mul(x, ait1)
        output = torch.sum(weighted_input, dim=1)

        return output


class LLC_genomics(nn.Module):
    def __init__(self, latent_dim1=256, latent_dim2=128, genomics_dim=100, input_dim=10000, seed=None):
        super(LLC_genomics, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        stride = 3
        kernel_size = 7
        kernel_num = 16
        pooling_size = 5
        # convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=kernel_num, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(in_channels=kernel_num, out_channels=kernel_num, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv1d(in_channels=kernel_num, out_channels=kernel_num, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)

        # Dropout layer
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.2)

        for i in range(3):
            input_dim = ((input_dim - kernel_size)//stride + 1)//pooling_size

        self.dense1 = nn.Linear(input_dim * kernel_num, latent_dim1)
        self.dense2 = nn.Linear(latent_dim1 * 2, 256)
        self.dense3 = nn.Linear(256, 64)
        self.dense4 = nn.Linear(64, 1)
        self.act = nn.Sigmoid()

        self.linear1 = nn.Linear(genomics_dim, latent_dim2)
        self.linear2 = nn.Linear(latent_dim2, 128)
        self.linear3 = nn.Linear(128, 1)
        # self.linear4 = nn.Linear(32, 1)
        # self.merge1_dense = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(input_dim * 64 * 2, latent_dim1),
        #     nn.ReLU()
        # )
        # Merge2
        self.merge2_dense = nn.Sequential(
            nn.BatchNorm1d(genomics_dim),
            # nn.Dropout(0.2),
            nn.Linear(genomics_dim, latent_dim2),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        merge = self.linear3(x)
        merge = self.act(merge)
        return merge

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to:
        bs, feat, img_len
        '''
        return x.transpose(-2, -1).contiguous()


class InterChrom(nn.Module):
    def __init__(self, input_dim=500, seed=None):
        super(InterChrom, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        stride = 1
        kernel_size = 40
        pooling_size = 20
        eca_k_size = 3
        embedding_dim = AdaptiveAvgPool1d_size
        for i in range(1):
            input_dim = ((input_dim - kernel_size)//stride + 1)//pooling_size

        self.dropout = nn.Dropout(0.5)
        self.embedding_x1 = nn.Embedding(4096, embedding_dim)
        self.embedding_x1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_x1.weight.requires_grad = True

        self.embedding_y1 = nn.Embedding(4096, embedding_dim)
        self.embedding_y1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_y1.weight.requires_grad = True

        self.embedding_x2 = nn.Embedding(4096, embedding_dim)
        self.embedding_x2.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_x2.weight.requires_grad = True

        self.embedding_y2 = nn.Embedding(4096, embedding_dim)
        self.embedding_y2.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_y2.weight.requires_grad = True

        self.embedding_x3 = nn.Embedding(4096, embedding_dim)
        self.embedding_x3.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_x3.weight.requires_grad = True

        self.embedding_y3 = nn.Embedding(4096, embedding_dim)
        self.embedding_y3.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_y3.weight.requires_grad = True

        self.embedding_x4 = nn.Embedding(4096, embedding_dim)
        self.embedding_x4.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_x4.weight.requires_grad = True

        self.embedding_y4 = nn.Embedding(4096, embedding_dim)
        self.embedding_y4.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding_y4.weight.requires_grad = True

        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)

        self.bn = nn.BatchNorm1d(embedding_dim*2)
        self.dropout = nn.Dropout(0.5)
        self.dropout03 = nn.Dropout(0.3)

        self.dense = nn.Linear(embedding_dim, 1)
        self.sig = nn.Sigmoid()

        self.dense11 = nn.Linear(768 * 2, 1024)
        self.dense12 = nn.Linear(1024, 256)
        self.dense13 = nn.Linear(256, 32)
        self.dense14 = nn.Linear(32, 1)

        self.dense21 = nn.Linear(768 * 2, 1024)
        self.dense22 = nn.Linear(1024, 256)
        self.dense23 = nn.Linear(256, 32)
        self.dense24 = nn.Linear(32, 1)

        self.dense31 = nn.Linear(768 * 2, 1024)
        self.dense32 = nn.Linear(1024, 256)
        self.dense33 = nn.Linear(256, 32)
        self.dense34 = nn.Linear(32, 1)

        self.dense41 = nn.Linear(768 * 2, 1024)
        self.dense42 = nn.Linear(1024, 256)
        self.dense43 = nn.Linear(256, 32)
        self.dense44 = nn.Linear(32, 1)

        self.merge_dense1 = nn.Linear(32*4, 32)
        self.merge_dense2 = nn.Linear(32, 1)

        self.ecax1 = ECA(k_size=eca_k_size)
        self.ecax2 = ECA(k_size=eca_k_size)
        self.ecax3 = ECA(k_size=eca_k_size)
        self.ecax4 = ECA(k_size=eca_k_size)
        self.ecay1 = ECA(k_size=eca_k_size)
        self.ecay2 = ECA(k_size=eca_k_size)
        self.ecay3 = ECA(k_size=eca_k_size)
        self.ecay4 = ECA(k_size=eca_k_size)

    def forward(self, x, y):
        x1 = x[:, 0, :]
        y1 = y[:, 0, :]
        x1 = self.embedding_x1(x1)
        x1 = self.ecax1(x1)
        y1 = self.embedding_y1(y1)
        y1 = self.ecay1(y1)
        merge1 = torch.cat([x1, y1], dim=1)
        merge1 = self.act(self.dense11(merge1))
        merge1 = self.dropout03(merge1)
        merge1 = self.act(self.dense12(merge1))
        merge1 = self.dropout03(merge1)
        merge1 = self.act(self.dense13(merge1))
        merge1 = self.dropout03(merge1)

        x2 = x[:, 1, :]
        y2 = y[:, 1, :]
        x2 = self.embedding_x2(x2)
        x2 = self.ecax2(x2)
        y2 = self.embedding_y2(y2)
        y2 = self.ecay2(y2)
        merge2 = torch.cat([x2, y2], dim=1)
        merge2 = self.act(self.dense21(merge2))
        merge2 = self.dropout03(merge2)
        merge2 = self.act(self.dense22(merge2))
        merge2 = self.dropout03(merge2)
        merge2 = self.act(self.dense23(merge2))
        merge2 = self.dropout03(merge2)

        x3 = x[:, 2, :]
        y3 = y[:, 2, :]
        x3 = self.embedding_x3(x3)
        x3 = self.ecax3(x3)
        y3 = self.embedding_y3(y3)
        y3 = self.ecay3(y3)
        merge3 = torch.cat([x3, y3], dim=1)
        merge3 = self.act(self.dense31(merge3))
        merge3 = self.dropout03(merge3)
        merge3 = self.act(self.dense32(merge3))
        merge3 = self.dropout03(merge3)
        merge3 = self.act(self.dense33(merge3))
        merge3 = self.dropout03(merge3)

        x4 = x[:, 3, :]
        y4 = y[:, 3, :]
        x4 = self.embedding_x4(x4)
        x4 = self.ecax4(x4)
        y4 = self.embedding_y4(y4)
        y4 = self.ecay4(y4)
        merge4 = torch.cat([x4, y4], dim=1)
        merge4 = self.act(self.dense41(merge4))
        merge4 = self.dropout03(merge4)
        merge4 = self.act(self.dense42(merge4))
        merge4 = self.dropout03(merge4)
        merge4 = self.act(self.dense43(merge4))
        merge4 = self.dropout03(merge4)

        merge = torch.cat([merge1, merge2, merge3, merge4], dim=1)
        merge = self.act(self.merge_dense1(merge))
        merge = self.sig(self.merge_dense2(merge))

        return merge

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to:
        bs, feat, img_len
        '''
        return x.transpose(-2, -1).contiguous()