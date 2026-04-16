import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.stats import spearmanr,pearsonr

# 1. 稀疏MoE模块（保持不变）
class Sparse_MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        # 筛选Top-K专家
        gate_logits = self.gate(x_flat)
        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
        gate_weights = torch.softmax(top_k_values, dim=1)
        
        # 专家计算与加权
        expert_outputs = []
        for i in range(self.num_experts):
            mask = (top_k_indices == i).float()
            expert_out = self.experts[i](x_flat)
            weighted_out = expert_out.unsqueeze(1) * mask.unsqueeze(-1)
            expert_outputs.append(weighted_out)
        
        # 聚合输出
        moe_out = torch.sum(torch.stack(expert_outputs), dim=0)
        moe_out = torch.sum(moe_out * gate_weights.unsqueeze(-1), dim=1)
        return moe_out.view(batch_size, seq_len, self.d_model)

# 2. 调整为双输入的MLP特征交互层
class MLPInteractionLayer(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.4):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim else 2 * d_model
        
        # MLP结构：输入为两个特征的拼接（2*d_model）
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(self.hidden_dim, 2 * d_model),  # 输出两个特征的交互结果
            nn.LayerNorm(2 * d_model)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        x1_flat = x1.squeeze(1)  # [batch, d_model]
        x2_flat = x2.squeeze(1)
        combined = torch.cat([x1_flat, x2_flat], dim=1)  # [batch, 2*d_model]
        
        mlp_out = self.mlp(combined)
        
        # 拆分回两个分支
        x1_out = mlp_out[:, :self.d_model].unsqueeze(1)
        x2_out = mlp_out[:, self.d_model:].unsqueeze(1)
        
        return x1_out, x2_out

# 3. 双输入主模型（MLP交互+MoE）
class CombinedMLPMoEModel(nn.Module):
    def __init__(self, input_dims, d_model=64, hidden_dim_mlp=None, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # 两个特征分支的投影层
        self.proj_x1 = nn.Linear(input_dims[0], d_model)
        self.proj_x2 = nn.Linear(input_dims[1], d_model)
        
        # MLP特征交互层（双输入版本）
        self.mlp_interaction = MLPInteractionLayer(d_model, hidden_dim_mlp, dropout)
        
        # 稀疏MoE增强
        self.moe = Sparse_MoE(d_model, num_experts, top_k)
        
        # 特征融合与回归头
        self.fusion_fc = nn.Linear(d_model * 2, d_model)  # 改为2*d_model输入
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(d_model, 1)

    def forward(self, x1, x2):
        # 步骤1：特征投影+增加序列维度
        x1_proj = self.proj_x1(x1).unsqueeze(1)  # [batch, 1, d_model]
        x2_proj = self.proj_x2(x2).unsqueeze(1)
        
        # 步骤2：MLP特征交互
        x1_mlp, x2_mlp = self.mlp_interaction(x1_proj, x2_proj)
        
        # 步骤3：MoE增强+特征融合
        x1_moe = self.moe(x1_mlp).squeeze(1)  # [batch, d_model]
        x2_moe = self.moe(x2_mlp).squeeze(1)
        
        fused = torch.cat([x1_moe, x2_moe], dim=1)  # [batch, 2*d_model]
        fused = self.fusion_fc(fused)
        fused = self.bn(fused)
        fused = self.dropout(fused)
        
        # 回归输出
        out = self.reg_head(fused)  # [batch, 1]
        return out

# 4. 双输入数据集类
class FeatureDataset(Dataset):
    def __init__(self, x1_feats, x2_feats, target):
        self.x1 = torch.tensor(x1_feats.values, dtype=torch.float32)
        self.x2 = torch.tensor(x2_feats.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.target[idx]

# 5. 双输入数据加载工具
def load_single_feat(feat_path, meta_df):
    if feat_path.endswith('.pkl'):
        embed = pd.read_pickle(feat_path)
        embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
        embed_df.rename(columns={'index': 'ID'}, inplace=True)
    elif feat_path.endswith('.pt'):
        embed = torch.load(feat_path, weights_only=True)
        embed_df = pd.DataFrame.from_dict(embed).T.reset_index()
        embed_df.rename(columns={'index': 'ID'}, inplace=True)
    elif feat_path.endswith('.csv'):
        embed_df = pd.read_csv(feat_path)
    else:
        raise ValueError('仅支持 .pkl, .pt, .csv 格式')
    
    aligned_feat = meta_df[['ID']].merge(embed_df, how='inner', on='ID')
    return aligned_feat.drop(columns=['ID'])

def load_two_feats(feat_paths, meta_path):
    meta_df = pd.read_csv(meta_path)
    if 'ID' not in meta_df.columns or 'target' not in meta_df.columns:
        raise ValueError('元数据必须包含 ID 和 target 列')
    
    x1_feat = load_single_feat(feat_paths[0], meta_df)
    x2_feat = load_single_feat(feat_paths[1], meta_df)
    
    assert len(x1_feat) == len(x2_feat) == len(meta_df), \
        "特征文件与元数据样本数量不匹配（请检查ID对齐）"
    return x1_feat, x2_feat, meta_df['target']

# 6. 双输入特征归一化
def scale_two_features(train_x1, train_x2, test_x1, test_x2):
    scaler_x1 = MinMaxScaler(feature_range=(0, 1))
    scaler_x2 = MinMaxScaler(feature_range=(0, 1))
    
    train_x1_scaled = scaler_x1.fit_transform(train_x1)
    train_x2_scaled = scaler_x2.fit_transform(train_x2)
    
    test_x1_scaled = scaler_x1.transform(test_x1)
    test_x2_scaled = scaler_x2.transform(test_x2)
    
    return (
        pd.DataFrame(train_x1_scaled), pd.DataFrame(train_x2_scaled),
        pd.DataFrame(test_x1_scaled), pd.DataFrame(test_x2_scaled)
    )

# 7. 训练与评估函数（调整为双输入）
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=300):
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1_batch, x2_batch, batch_target in train_loader:  # 移除x3
            x1_batch, x2_batch = x1_batch.to(device), x2_batch.to(device)
            batch_target = batch_target.to(device)
            
            outputs = model(x1_batch, x2_batch)  # 双输入
            loss = criterion(outputs.squeeze(), batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于Epoch {epoch+1}（验证集损失连续{patience}轮未下降）")
                    model.load_state_dict(torch.load('best_model.pth'))
                    return

def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x1_batch, x2_batch, batch_target in data_loader:  # 移除x3
            x1_batch, x2_batch = x1_batch.to(device), x2_batch.to(device)
            batch_target = batch_target.to(device)
            outputs = model(x1_batch, x2_batch)  # 双输入
            loss = criterion(outputs.squeeze(), batch_target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x1_batch, x2_batch, batch_target in data_loader:  # 移除x3
            x1_batch, x2_batch = x1_batch.to(device), x2_batch.to(device)
            outputs = model(x1_batch, x2_batch)  # 双输入
            
            all_preds.extend(outputs.cpu().numpy().squeeze())
            all_targets.extend(batch_target.numpy())
    return np.array(all_preds), np.array(all_targets)

# 8. 结果保存（保持不变）
def save_results(r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test, rho_train, rho_test, pearson_train ,pearson_test, method):
    return pd.DataFrame({
        'method': [method],
        'r2_train': [r2_train],
        'mae_train': [mae_train],
        'rmse_train': [rmse_train],
        'spearman_train': [rho_train],
        'pearson_train':[pearson_train],
        'r2_test': [r2_test],
        'mae_test': [mae_test],
        'rmse_test': [rmse_test],
        'spearman_test': [rho_test],
        'pearson_test':[pearson_test]
    })

# 9. 双输入回归主流程
def run_regression(train_x1, train_x2, train_target, test_x1, test_x2, test_target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 拆分验证集
    from sklearn.model_selection import train_test_split
    train_x1, val_x1, train_x2, val_x2, train_target, val_target = train_test_split(
        train_x1, train_x2, train_target, test_size=0.2, random_state=42
    )
    
    # 构建数据集
    train_dataset = FeatureDataset(train_x1, train_x2, train_target)
    val_dataset = FeatureDataset(val_x1, val_x2, val_target)
    test_dataset = FeatureDataset(test_x1, test_x2, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化双输入模型
    input_dims = [train_x1.shape[1], train_x2.shape[1]]  # 仅两个输入维度
    model = CombinedMLPMoEModel(
        input_dims=input_dims,
        d_model=64,
        hidden_dim_mlp=128,
        num_experts=6,
        top_k=2,
        dropout=0.5
    ).to(device)
    
    # 损失函数与优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    # 训练
    print("\n开始训练...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=70)
    
    # 评估
    y_pred_train, y_true_train = evaluate_model(model, train_loader, device)
    y_pred_val, y_true_val = evaluate_model(model, val_loader, device)
    y_pred_test, y_true_test = evaluate_model(model, test_loader, device)
    
    # 计算指标
    metrics_train = {
        'r2': metrics.r2_score(y_true_train, y_pred_train),
        'mae': metrics.mean_absolute_error(y_true_train, y_pred_train),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_train, y_pred_train)),
        'spearman': spearmanr(y_true_train, y_pred_train)[0],
        'pearson': pearsonr(y_true_train, y_pred_train.squeeze())[0]
    }
    metrics_test = {
        'r2': metrics.r2_score(y_true_test, y_pred_test),
        'mae': metrics.mean_absolute_error(y_true_test, y_pred_test),
        'rmse': np.sqrt(metrics.mean_squared_error(y_true_test, y_pred_test)),
        'spearman': spearmanr(y_true_test, y_pred_test)[0],
        'pearson': pearsonr(y_true_test, y_pred_test.squeeze())[0]
    }
    
    # 打印结果
    print(f"\n验证集指标 - R²: {metrics.r2_score(y_true_val, y_pred_val):.3f}, MAE: {metrics.mean_absolute_error(y_true_val, y_pred_val):.3f}")
    print(f"训练集指标 - R²: {metrics_train['r2']:.3f}, MAE: {metrics_train['mae']:.3f}, "
          f"RMSE: {metrics_train['rmse']:.3f}, Spearman: {metrics_train['spearman']:.3f}")
    print(f"测试集指标 - R²: {metrics_test['r2']:.3f}, MAE: {metrics_test['mae']:.3f}, "
          f"RMSE: {metrics_test['rmse']:.3f}, Spearman: {metrics_test['spearman']:.3f},pearson:{metrics_test['pearson']:.3f}")
    
    return (metrics_train['r2'], metrics_train['mae'], metrics_train['rmse'],
            metrics_test['r2'], metrics_test['mae'], metrics_test['rmse'],
            metrics_train['spearman'], metrics_test['spearman'],
            metrics_train['pearson'], metrics_test['pearson'])

# 10. 双输入文件入口流程
def run_regression_on_files(train_feat_paths, train_meta_path, test_feat_paths, test_meta_path):
    method = "MLP+MoE(2branches)"  # 更新方法名
    
    # 加载数据
    print("加载训练数据...")
    train_x1, train_x2, train_target = load_two_feats(train_feat_paths, train_meta_path)  # 双输入
    print("加载测试数据...")
    test_x1, test_x2, test_target = load_two_feats(test_feat_paths, test_meta_path)  # 双输入
    
    # 特征归一化
    print("特征归一化...")
    scaled = scale_two_features(train_x1, train_x2, test_x1, test_x2)  # 双输入
    train_x1_scaled, train_x2_scaled, test_x1_scaled, test_x2_scaled = scaled
    
    # 运行回归
    metrics = run_regression(
        train_x1_scaled, train_x2_scaled, train_target,
        test_x1_scaled, test_x2_scaled, test_target
    )
    
    # 保存结果
    return save_results(*metrics, method=method)

# 11. 双输入命令行入口
def main():
    parser = argparse.ArgumentParser(description="双分支特征回归模型（MLP+MoE）")
    # 训练集参数（仅x1和x2）
    parser.add_argument("--train_x1", required=True, help="训练集x1特征文件路径")
    parser.add_argument("--train_x2", required=True, help="训练集x2特征文件路径")
    parser.add_argument("--train_meta", required=True, help="训练集元数据文件路径（含ID和target）")
    # 测试集参数（仅x1和x2）
    parser.add_argument("--test_x1", required=True, help="测试集x1特征文件路径")
    parser.add_argument("--test_x2", required=True, help="测试集x2特征文件路径")
    parser.add_argument("--test_meta", required=True, help="测试集元数据文件路径（含ID和target）")
    # 输出参数
    parser.add_argument("-o", "--output", required=True, help="结果输出CSV文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行并保存结果
    results = run_regression_on_files(
        train_feat_paths=[args.train_x1, args.train_x2],  # 双输入路径
        train_meta_path=args.train_meta,
        test_feat_paths=[args.test_x1, args.test_x2],  # 双输入路径
        test_meta_path=args.test_meta
    )
    results.to_csv(args.output, index=False)
    print(f"\n结果已保存至: {args.output}")

if __name__ == "__main__":
    main()