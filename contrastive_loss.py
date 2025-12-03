import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPositiveInfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, features, labels=None):
        """
        Args:
            features: 形状为 [batch_size, feature_dim] 的特征向量
            labels: 形状为 [batch_size] 的标签，用于识别相同类别的样本作为正样本对
        Returns:
            loss: 标量损失值
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 如果没有提供标签，则使用增强视图作为正样本对
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        else:
            # 创建标签匹配矩阵，相同类别的样本对设为True
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
            # 移除对角线上的自身匹配
            mask.fill_diagonal_(False)
        
        # 计算特征之间的相似度矩阵
        features = F.normalize(features, dim=1)  # L2归一化
        similarity_matrix = torch.matmul(features, features.T)
        
        # 应用温度系数
        similarity_matrix = similarity_matrix / self.temperature
        
        # 为每个样本选择一个正样本
        positive_mask = mask.float()
        # 确保每行至少有一个正样本
        positive_samples = torch.where(
            positive_mask.sum(1) > 0,
            positive_mask,
            torch.eye(batch_size, device=device)
        )
        
        # 计算正样本的损失
        exp_logits = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算每个样本的正样本损失均值
        mean_log_prob_pos = (positive_samples * log_prob).sum(1) / (positive_samples.sum(1) + 1e-8)
        
        # 计算最终的损失
        loss = -mean_log_prob_pos.mean()
        
        return loss



def get_contrastive_loss(loss_type='multi_positive', temperature=0.1):
    """
    获取对比学习损失函数
    Args:
        loss_type: 'multi_positive' 或 'supcon'
        temperature: 温度参数
    """
    if loss_type == 'multi_positive':
        return MultiPositiveInfoNCE(temperature=temperature)

    else:
        raise ValueError(f'Unknown loss type: {loss_type}') 