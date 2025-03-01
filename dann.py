import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
import os
from utils import set_seed, evaluate_model_and_metrics  # 导入辅助函数

# Gradient Reversal Layer 定义
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

# DANN 模型定义
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

# 主函数
def main():
    # 设置随机种子
    set_seed(32)

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

    # 合并源域和目标域数据用于随机森林训练
    X = np.concatenate((X_source, X_target))
    y = np.concatenate((y_source, y_target)).ravel()

    # 将连续标签转换为离散标签用于交叉验证
    y_target_discrete = pd.cut(y_target.flatten(), bins=10, labels=False)

    # 创建分层 K 折交叉验证对象
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # 定义随机森林参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)

    # 输出最佳参数并获取最佳模型
    print("Best parameters found: ", grid_search.best_params_)
    best_rf = grid_search.best_estimator_

    # 四折交叉验证
    fold = 1
    for train_index, test_index in kf.split(X_target, y_target_discrete):
        print(f"Fold {fold}")
        fold_losses = []

        # 目标域数据划分
        X_train_target, X_test_target = X_target[train_index], X_target[test_index]
        y_train_target, y_test_target = y_target[train_index], y_target[test_index]

        # 使用最佳随机森林提取特征
        X_source_features = best_rf.apply(X_source)
        X_train_target_features = best_rf.apply(X_train_target)
        X_test_target_features = best_rf.apply(X_test_target)

        # 转换为 PyTorch 张量
        X_train_source_tensor = torch.tensor(X_source_features, dtype=torch.float32)
        y_train_source_tensor = torch.tensor(y_source, dtype=torch.float32)
        X_train_target_tensor = torch.tensor(X_train_target_features, dtype=torch.float32)
        y_train_target_tensor = torch.tensor(y_train_target, dtype=torch.float32)
        X_test_target_tensor = torch.tensor(X_test_target_features, dtype=torch.float32)
        y_test_target_tensor = torch.tensor(y_test_target, dtype=torch.float32)

        # 创建域标签
        domain_labels_source = torch.zeros(X_train_source_tensor.shape[0], dtype=torch.long)
        domain_labels_target = torch.ones(X_train_target_tensor.shape[0], dtype=torch.long)

        # 合并训练数据
        X_train_tensor = torch.cat((X_train_source_tensor, X_train_target_tensor))
        y_train_tensor = torch.cat((y_train_source_tensor, y_train_target_tensor))
        domain_labels = torch.cat((domain_labels_source, domain_labels_target))

        # 定义数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, domain_labels)
        test_dataset = TensorDataset(X_test_target_tensor, y_test_target_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 初始化模型
        input_dim = X_train_tensor.shape[1]
        model = DANN(input_dim)

        # 定义损失函数和优化器
        criterion_regress = nn.MSELoss()
        criterion_domain = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

        # 训练循环
        epochs = 1000
        losses = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data, labels, domains in train_loader:
                optimizer.zero_grad()
                predict, domain_predict = model(data)
                regress_loss = criterion_regress(predict, labels)
                domain_loss = criterion_domain(domain_predict, domains)
                total_loss = regress_loss + 0.01 * domain_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            losses.append(running_loss / len(train_loader))
            fold_losses.append(running_loss / len(train_loader))
            scheduler.step()

            if epoch % 500 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Over Epochs - Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 保存损失记录
        losses_df = pd.DataFrame({'Epoch': range(1, epochs + 1), 'Loss': losses})
        output_directory = 'D:\\data'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        try:
            file_path = os.path.join(output_directory, f'train_losses_fold_{fold}.xlsx')
            losses_df.to_excel(file_path, index=False)
            print(f"Loss values saved to {file_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

        # 评估训练集和测试集
        train_mae, train_mse, train_r2, train_actuals, train_predictions = evaluate_model_and_metrics(model, train_loader, scaler_labels)
        print(f'Fold {fold} - Train MAE: {train_mae:.4f}, Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}')
        test_mae, test_mse, test_r2, test_actuals, test_predictions = evaluate_model_and_metrics(model, test_loader, scaler_labels)
        print(f'Fold {fold} - Test MAE: {test_mae:.4f}, Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}')

        # 保存结果
        train_results = pd.DataFrame({'Actual': train_actuals, 'Predicted': train_predictions})
        test_results = pd.DataFrame({'Actual': test_actuals, 'Predicted': test_predictions})
        train_results.to_excel(f'D:\\train_results_fold_{fold}.xlsx', index=False)
        test_results.to_excel(f'D:\\test_results_fold_{fold}.xlsx', index=False)

        fold += 1

if __name__ == "__main__":
    main()