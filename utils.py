import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def set_seed(seed=32):
    """Fix random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model_and_metrics(model, data_loader, scaler_labels):
    """Evaluate model performance and compute metrics."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in data_loader:
            data, labels = batch[:2]  # 只获取前两个元素（数据和标签）
            outputs, _ = model(data)
            actuals.extend(scaler_labels.inverse_transform(labels.cpu().numpy()).flatten())
            predictions.extend(scaler_labels.inverse_transform(outputs.cpu().numpy()).flatten())

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return mae, mse, r2, actuals, predictions

