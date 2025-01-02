import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
import os

# 固定随机种子
def set_seed(seed=32):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(32)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN(nn.Module):
    def __init__(self, input_dim):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(10),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_predict = self.regressor(features)
        domain_predict = self.domain_classifier(features)
        return class_predict, domain_predict

def evaluate_model_and_metrics(model, data_loader, scaler_labels):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch[:2]  # 只获取前两个元素
            outputs, _ = model(data)
            actuals.extend(scaler_labels.inverse_transform(labels.cpu().numpy()).flatten())
            predictions.extend(scaler_labels.inverse_transform(outputs.cpu().numpy()).flatten())

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return mae, mse, r2, actuals, predictions

# 加载数据集
source_data_excel = pd.read_excel('D:\\data\\SD1.xlsx', header=None)
target_data_excel = pd.read_excel('D:\\data\\TD1.xlsx', header=None)

# 数据预处理
scaler_features = MinMaxScaler()
X_source = scaler_features.fit_transform(source_data_excel.iloc[:, :-1])
X_target = scaler_features.transform(target_data_excel.iloc[:, :-1])

scaler_labels = MinMaxScaler()
y_source = scaler_labels.fit_transform(source_data_excel.iloc[:, -1].values.reshape(-1, 1))
y_target = scaler_labels.transform(target_data_excel.iloc[:, -1].values.reshape(-1, 1))

# 合并源域和目标域数据
X = np.concatenate((X_source, X_target))
y = np.concatenate((y_source, y_target)).ravel()  # 将 y 转换为一维数组

# 将连续标签转换为离散标签
y_target_discrete = pd.cut(y_target.flatten(), bins=10, labels=False)

# 创建分层K折交叉验证对象
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 分裂节点的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点的最小样本数
    'bootstrap': [True, False]  # 是否使用自助法
}

rf = RandomForestRegressor(random_state=42)

# 使用网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
grid_search.fit(X, y)  # X 和 y 是你的数据集和标签

# 获取最佳参数
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# 使用最佳参数训练模型
best_rf = grid_search.best_estimator_

# 四折交叉验证
fold = 1
for train_index, test_index in kf.split(X_target, y_target_discrete):
    print(f"Fold {fold}")

    # 初始化损失记录
    fold_losses = []

    # 目标数据的训练集和验证集
    X_train_target, X_test_target = X_target[train_index], X_target[test_index]
    y_train_target, y_test_target = y_target[train_index], y_target[test_index]

    # 使用最佳随机森林提取特征
    X_source_features = best_rf.apply(X_source)
    X_train_target_features = best_rf.apply(X_train_target)
    X_test_target_features = best_rf.apply(X_test_target)

    # 转换为PyTorch张量
    X_train_source_tensor = torch.tensor(X_source_features, dtype=torch.float32)
    y_train_source_tensor = torch.tensor(y_source, dtype=torch.float32)
    X_train_target_tensor = torch.tensor(X_train_target_features, dtype=torch.float32)
    y_train_target_tensor = torch.tensor(y_train_target, dtype=torch.float32)
    X_test_target_tensor = torch.tensor(X_test_target_features, dtype=torch.float32)
    y_test_target_tensor = torch.tensor(y_test_target, dtype=torch.float32)

    # 创建域标签
    domain_labels_source = torch.zeros(X_train_source_tensor.shape[0], dtype=torch.long)
    domain_labels_target = torch.ones(X_train_target_tensor.shape[0], dtype=torch.long)

    # 合并源域数据和目标数据的训练集
    X_train_tensor = torch.cat((X_train_source_tensor, X_train_target_tensor))
    y_train_tensor = torch.cat((y_train_source_tensor, y_train_target_tensor))
    domain_labels = torch.cat((domain_labels_source, domain_labels_target))

    # 定义数据集和数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, domain_labels)
    test_dataset = TensorDataset(X_test_target_tensor, y_test_target_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    model = DANN(input_dim)

    criterion_regress = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 增大学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # 学习率调度器

    # 在训练之前，添加一个损失列表
    losses = []

    epochs = 1000  # 增加训练轮数
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, labels, domains in train_loader:
            optimizer.zero_grad()
            predict, domain_predict = model(data)

            regress_loss = criterion_regress(predict, labels)
            domain_loss = criterion_domain(domain_predict, domains)

            total_loss = regress_loss + 0.01 * domain_loss  # 调整正则化参数
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        losses.append(running_loss / len(train_loader))
        fold_losses.append(running_loss / len(train_loader))  # 添加到每一折的损失记录

        scheduler.step()  # 更新学习率

        if epoch % 500 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # 训练完成后绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存损失记录为Excel文件
    losses_df = pd.DataFrame({'Epoch': range(1, epochs + 1), 'Loss': losses})

    # 确保 D 盘的目标路径存在
    output_directory = 'D:\\data'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        file_path = os.path.join(output_directory, f'train_losses_fold_{fold}.xlsx')
        losses_df.to_excel(file_path, index=False)
        print(f"损失值已保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

    # 计算训练集评估指标
    train_mae, train_mse, train_r2, train_actuals, train_predictions = evaluate_model_and_metrics(model, train_loader, scaler_labels)
    print(f'Fold {fold} - Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}')

    # 保存训练集结果到 DataFrame
    train_results = pd.DataFrame({
        'Actual': train_actuals,
        'Predicted': train_predictions
    })

    # 保存到 Excel 文件
    train_results.to_excel(f'D:\\train_results_fold_{fold}.xlsx', index=False)  # 保存训练集结果

    # 计算验证集评估指标
    test_mae, test_mse, test_r2, test_actuals, test_predictions = evaluate_model_and_metrics(model, test_loader, scaler_labels)
    print(f'Fold {fold} - Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}')

    # 保存测试集结果到 DataFrame
    test_results = pd.DataFrame({
        'Actual': test_actuals,
        'Predicted': test_predictions
    })

    # 保存到 Excel 文件
    test_results.to_excel(f"D:\\test_results_fold_{fold}.xlsx", index=False)  # 保存验证集结果

    fold += 1
