import math
import torch
from torch import autograd

from torch.nn.parameter import Parameter


import torch.nn as nn
import torch.nn.functional as F
'''
parameter将一个不可训练的类型tensor转换成可训练的类型parameter，
并将其绑定到这个module里面，所以经过类型转换这个变成了模型的一部分，
成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是为了想
让某些变量在学习过程中不断的修改其值以达到最优解。
'''

class GraphAttentionLayer(nn.Module): # Module类的单继承
    """
        参数：
        in_features:输入特征，每个输入样本的大小
        out_features:输出特征，每个输出样本的大小
        bias:偏置，如果设置为false，则层将不会学习加法偏置。默认值true
        属性：
        weight:形状模块的可学习权重
        bias:形状模块的可学习偏差
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
            # super函数用于调用父类的方法
            # super().__init__()表示子类既能重写__init__()方法又能调用父类的方法
        self.in_features = in_features  # 输入特征大小
        self.out_features = out_features # 输入特征大小
        self.dropout = dropout
        self.alpha = alpha #学习因子
        self.concat = concat

        ######################参数定义##########################
        # 两组参数W和a需要训练
        self.W = Parameter(torch.zeros(size=(in_features, out_features))) #建立全为0的矩阵，大小为（输入维度，输出维度）
            # 先转化为张量，再转化为可训练的Parameter对象
            # Parameter用于将参数自动加入到参数列表中
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # 将W初始化
        #xavier初始化 ,让输入的样本空间和输出的类别空间的方差相当，这样才是好的初始化，不然错误的初始化，会使模型训练的很慢。

        self.a = Parameter(torch.zeros(size=(2*out_features,1))) # a是只有一列的向量
        # print(self.a.shape) torch.Size([16,1])
        nn.init.xavier_uniform_(self.a.data,gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,h,adj): # h node features
        # h: (N, in_features) in_features=1433?
        # adj: sparse matrix with shape(N, N) 2708*2708
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(Wh.shape) torch.Size([2708, 7]) 7是label的个数
        # N = Wh.size()[0] #节点个数2708个
        # print(N) 2708 nodes的个数
        # a_input = torch.cat([Wh.repeat(1,N).view(N*N,-1), Wh.repeat(N,1)],dim=1).view(N, -1, 2 * self.out_features) #Whi Whj向量拼接
        # view(2,3)就和reshape一样，view(-1)将高维展成一维，从行的顺序开始展开，[........],view(3,-1) -1代表是一个不用计算，自己计算列为多少，自动分类为3行x列
        # a_input是有Whi和Whj concat得到的对应论文里的Whi||Whj
        e = self._prepare_attention_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        # one_like(e)构建维度和e一样大小的全1矩阵

        attention = torch.where(adj>0, e, zero_vec)  # where 满足adj>0,输出e，否则输出zero_vec
        '''
        utils.py中的adj邻接矩阵：两个节点有边，则为1，否则为0。 adj大小[2708,2708],2708节点
        print(adj) 注意adj经过归一化的
        当adj>0 则两节点有边
        当adj<0 则两节点没边，让边的权重最小为-9e15，即该边权重很小，不用关注不用attention
        '''
        attention = F.softmax(attention, dim=1) #得到的eij经过softmanx 对应论文中公式2
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = F.dropout(attention,self.dropout, training=self.training)
        # dropout里面有权重a和W，所以Dropout是指在模型训练时,随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        # 训练时用dropout(令trainning=True)，评估时关掉dropout(令trainning=False)

        h_prime = torch.matmul(attention, Wh) #对应论文公式4
        if self.concat:
            return F.elu(h_prime) # elu激活函数 非线性
        else:
            return h_prime
    # 向量拼接函数
    def _prepare_attention_mechanism_input(self,Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh,self.a[:self.out_features,:]) #N*1
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) #N*1
        e = Wh1 + Wh2.T  #N*N
        return self.leakyrelu(e)

    def __repr__(self):  # 打印输出
        return self.__class__.__name__ + '(' \
               + str(self.in_features + '->' \
                     + str(self.out_features) + ')')
        # 打印形式 GraphConvolution(输入特征->输出特征)


# 稀疏矩阵的layer层
# 稀疏layer只是为了能加快GAT的求解，当batch_size = 1时使用。
# 但是该layer模块只是定义了一层的attention机制。感觉cora数据集更加合适系数版本的注意力层。
# 稀疏版本的运行速度是非稀疏版本的10倍以上。
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

# 稀疏矩阵GAT Layer
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


    def __repr__(self):  # 打印输出
        return self.__class__.__name__ + '(' \
                   + str(self.in_features + '->' \
                         + str(self.out_features) + ')')
            # 打印形式 GraphConvolution(输入特征->输出特征)

