import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块并简写为 nn
import math  # 导入数学库，用于开平方等基础函数

class MultiHeadSelfAttention(nn.Module):  # 定义多头自注意力模块类
    def __init__(self, d_model, num_heads):  # 初始化方法，传入总维度与头数
        super(MultiHeadSelfAttention, self).__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 保存头数
        self.d_model = d_model  # 保存总特征维度
        # 确保总维度能被头数整除
        assert d_model % num_heads == 0  # 断言检查，不满足会报错
        self.d_k = d_model // num_heads  # 每个头分到的维度
        # 定义 Wq, Wk, Wv 和最终输出的 Wo 线性层
        self.W_q = nn.Linear(d_model, d_model)  # 定义 Q 的线性层
        self.W_k = nn.Linear(d_model, d_model)  # 定义 K 的线性层
        self.W_v = nn.Linear(d_model, d_model)  # 定义 V 的线性层
        self.W_o = nn.Linear(d_model, d_model)  # 定义输出映射的线性层

    def forward(self, x):  # 前向传播函数
        # x 的形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()  # 获取输入张量形状
        # 1. 线性变换得到 Q, K, V
        Q = self.W_q(x)  # 计算 Query，形状 [batch_size, seq_len, d_model]
        K = self.W_k(x)  # 计算 Key
        V = self.W_v(x)  # 计算 Value
        # 2. 拆分成多个头 (Reshape + Transpose)
        # 期望形状: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 分头并交换维度
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 分头并交换维度
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 分头并交换维度
        # 3. 计算 Scaled Dot-Product Attention
        # K.transpose(-2, -1) 形状为 [batch_size, num_heads, d_k, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # 点积后缩放
        # 4. Softmax 归一化得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)  # 在最后一维做 softmax
        # 5. 权重乘以 V
        output = torch.matmul(attention_weights, V)  # 得到每个头的输出
        # 6. 把多个头拼接回去 (Concat)
        # transpose后需要 contiguous() 才能用 view 改变形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # 合并头部
        # 7. 经过最后一层线性映射
        return self.W_o(output)  # 输出映射回 d_model 维度

# ----- 测试复现 -----
if __name__ == "__main__":  # 脚本直接运行时才执行下面测试代码
    # 模拟输入：Batch大小为2，句子长度为5，每个词由64维向量表示
    simulated_input = torch.randn(2, 5, 64)  # 生成随机输入张量
    # 实例化一个 8 头的自注意力层
    mha = MultiHeadSelfAttention(d_model=64, num_heads=8)  # 创建多头注意力层
    # 前向传播
    output = mha(simulated_input)  # 执行前向传播得到输出
    print("输入形状:", simulated_input.shape)  # 打印输入形状
    print("输出形状:", output.shape)  # 打印输出形状（应与输入一致）