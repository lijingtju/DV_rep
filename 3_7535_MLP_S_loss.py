import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import joblib

# 读取特征和目标数据
X = pd.read_csv("8group_feas.csv") #SHAPE(25118, 6511)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)

csv_data = pd.read_csv('H1N1_smiles_stand_label.csv')
y = csv_data['rvi_H1N1'].values

# 先划分出测试集，再从剩余数据中划分出验证集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 定义加权均方误差损失函数
class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=2, beta=0.1):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        mse = nn.MSELoss(reduction='none')(y_pred, y_true)
        weights = torch.where(y_true > 0.5, self.alpha, self.beta)
        weighted_mse = mse * weights
        return weighted_mse.mean()

# 定义MLP模型
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 回归任务输出1个连续值
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
input_dim = X_train.shape[1]
model = MLPRegressor(input_dim)

# 定义损失函数和优化器
criterion = WeightedMSELoss(alpha=103, beta=2)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# 设置早停参数和模型保存路径
best_rmse = float('inf')
patience = 10
counter = 0
model_path = "best_model.pth"  # 保存模型路径

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train)
    train_loss = criterion(y_pred_train, y_train)  # 计算训练集的 Loss
    train_loss.backward()
    optimizer.step()

    # 每个 epoch 结束时评估验证集性能
    model.eval()
    with torch.no_grad():
        # 计算验证集 RMSE
        y_pred_val = model(X_val)
        val_mse = criterion(y_pred_val, y_val)
        y_pred_val_np = y_pred_val.numpy()
        y_val_np = y_val.numpy()
        results_df = pd.DataFrame({
            'y_pred': y_pred_val_np.flatten(),  # 将数组展平以确保只有一维数据
            'y_true': y_val_np.flatten()  # 将数组展平以确保只有一维数据
        })
        # 保存 DataFrame 为 CSV 文件
        results_df.to_csv('val_results.csv', index=False)
        val_rmse = torch.sqrt(val_mse)

    # 打印训练集 Loss 和验证集 RMSE
    print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Validation RMSE: {val_rmse.item()}')

    # 检查是否需要提前停止训练并保存最优模型
    if val_rmse.item() < best_rmse:
        best_rmse = val_rmse.item()
        counter = 0  # 重置计数器
        # 保存最优模型
        torch.save(model.state_dict(), model_path)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# 加载保存的最优模型
best_model = MLPRegressor(input_dim)
best_model.load_state_dict(torch.load(model_path))
best_model.eval()

# 使用最优模型预测测试集并计算 RMSE
with torch.no_grad():
    y_pred_test = best_model(X_test)
    test_mse = criterion(y_pred_test, y_test)
    y_pred_test_np = y_pred_test.numpy()
    y_test_np = y_test.numpy()
    results_df = pd.DataFrame({
        'y_test_pred': y_pred_test_np.flatten(),  # 将数组展平以确保只有一维数据
        'y_test_true': y_test_np.flatten()  # 将数组展平以确保只有一维数据
    })
    # 保存 DataFrame 为 CSV 文件
    results_df.to_csv('test_results.csv', index=False)


    # 计算预测值大于 0.6 的样本的掩码
    high_pred_mask = y_pred_test_np.flatten() > 0.4
    y_pred_high = y_pred_test_np[high_pred_mask]
    y_test_high = y_test_np[high_pred_mask]

    # 计算预测值大于 0.6 的样本中，真实值也大于 0.6 的样本比例
    if len(y_pred_high) > 0:
        high_true_mask = y_test_high.flatten() > 0.4
        high_true_ratio = high_true_mask.sum() / len(y_pred_high)
        print(high_true_mask.sum(), len(y_pred_high))
    else:
        high_true_ratio = float('nan')  # 如果没有预测值大于 0.6 的样本，返回 NaN

    # 计算测试集 RMSE
    test_rmse = torch.sqrt(test_mse)
    print(f'Best model Test RMSE: {test_rmse.item()}')
    print(f'Ratio of True > 0.6 among Predicted > 0.6: {high_true_ratio}')




