import torch  # 导入 PyTorch 主库，提供张量与基础计算功能
import torch.nn as nn  # 导入神经网络模块，简写为 nn 便于调用
import torch.nn.functional as F  # 导入函数式接口，包含激活/softmax 等

class MiniSelfAttention(nn.Module):  # 定义自注意力模块类，继承 nn.Module
    def __init__(self, d_model):  # 初始化方法，d_model 表示特征维度
        super().__init__()  # 调用父类初始化，注册参数与子模块
        self.d_model = d_model  # 保存模型维度，供后续计算使用
        # 对应 Q, K, V 的三个线性变换矩阵（全连接层）
        self.q_linear = nn.Linear(d_model, d_model)  # 定义 Q 的线性层
        self.k_linear = nn.Linear(d_model, d_model)  # 定义 K 的线性层
        self.v_linear = nn.Linear(d_model, d_model)  # 定义 V 的线性层

    def forward(self, x):  # 前向传播函数，x 是输入张量
        # 假设输入 x 的形状是: [batch_size, seq_len, d_model]
        # 例如: [2个句子, 每句5个词, 每个词用8维向量表示]
        batch_size, seq_len, d_model = x.size()  # 解包张量形状，便于后续使用
        # 1. 线性变换得到 Q, K, V
        Q = self.q_linear(x)  # 对输入做线性变换得到 Query
        K = self.k_linear(x)  # 对输入做线性变换得到 Key
        V = self.v_linear(x)  # 对输入做线性变换得到 Value
        # 2. 计算 Q 和 K 的点积 (注意转置 K 的最后两个维度)
        # K.transpose(-2, -1) 的形状从 [B, N, D] 变成 [B, D, N]
        # scores 形状变为: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))  # 批量矩阵乘法得到相关性
        # 3. 缩放 (除以 根号下 d_k)
        d_k = d_model  # 设定缩放因子 d_k，这里等于特征维度
        scores = scores / (d_k ** 0.5)  # 缩放以稳定 softmax 的数值
        # 4. Softmax 归一化得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # 在最后一维做归一化
        # 5. 权重乘以 V 得到最终输出
        # output 形状: [batch_size, seq_len, d_model]
        output = torch.matmul(attention_weights, V)  # 加权求和得到输出表示
        return output, attention_weights  # 返回输出和注意力权重

# --- 导师带你测试运行 ---
if __name__ == "__main__":  # 脚本直接运行时才执行下面测试代码
    # 模拟输入：2个样本，每个序列长5，特征维度是8
    test_input = torch.randn(2, 5, 8)  # 生成随机输入张量
    # 实例化我们的自注意力层
    attention_layer = MiniSelfAttention(d_model=8)  # 创建自注意力层实例
    # 前向传播
    out, weights = attention_layer(test_input)  # 调用模型得到输出与权重
    print("输入形状:", test_input.shape)  # 打印输入张量的形状
    print("输出形状:", out.shape)  # 打印输出张量的形状
    print("注意力权重矩阵形状 (每个词对其他词的权重):", weights.shape)  # 打印权重形状