# 在GPU上计算OA, kappa, precision, recall, F1, miou, 每个类别的iou
import torch
from torchmetrics.functional.classification import multiclass_cohen_kappa
from torchmetrics.functional.classification import multiclass_precision
from torchmetrics.functional.classification import multiclass_recall
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional.classification import binary_cohen_kappa
from torchmetrics.functional.classification import binary_precision
from torchmetrics.functional.classification import binary_recall
from torchmetrics.functional.classification import binary_f1_score


def all_index(predictions, labels, num_classes, is_val=False):
    '''
    OA, kappa, precision, recall, F1, miou, ious
    '''
    #oa, kappa, precision, recall, f1
    predictions = predictions.flatten()
    labels = labels.flatten()
    OA = (predictions == labels).float().mean().item()

    miou, ious = mean_iou(predictions, labels, num_classes)

    if is_val:
        return OA, miou, ious
    else:
        if num_classes == 2:
            kappa = binary_cohen_kappa(predictions, labels).item()
            precision = binary_precision(predictions, labels).item()
            recall = binary_recall(predictions, labels).item()
            F1 = binary_f1_score(predictions, labels).item()
        else:
            kappa = multiclass_cohen_kappa(predictions, labels, num_classes).item()
            precision = multiclass_precision(predictions, labels, num_classes, "macro").item()
            recall = multiclass_recall(predictions, labels, num_classes, "macro").item()
            F1 = multiclass_f1_score(predictions, labels, num_classes, "macro").item()
        return OA, kappa, precision, recall, F1, miou, ious

        
def mean_iou(predictions, labels, num_classes):
    # 计算平均 Intersection over Union（mIOU）
    # 初始化混淆矩阵
    confusion_mat = torch.zeros((num_classes, num_classes), device=predictions.device)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mat[i, j] = torch.sum((labels == i) & (predictions == j))
    # 计算每个类别的 IOU
    ious = []
    for i in range(num_classes):
        intersection = confusion_mat[i, i]
        union = torch.sum(confusion_mat[i, :]) + torch.sum(confusion_mat[:, i]) - intersection
        if union == 0:
            iou = torch.tensor(0.0)  # 如果该类别在真实标签中不存在，则 IOU 为 0
        else:
            iou = intersection / union
        ious.append(iou.item())
    # 计算 mIOU
    mean_iou = torch.mean(torch.tensor(ious)).item()
    return mean_iou, ious

if __name__ == '__main__':
    # 示例用法
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = torch.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]], device=model_device)  # 模型预测结果
    labels = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], device=model_device)       # 真实标签
    print(predictions.device, labels.device)
    num_classes = 2  # 类别数量

    OA, kappa, precision, recall, F1, miou, ious = all_index(predictions, labels, num_classes)
    result = [OA, kappa, precision, recall, F1, miou, ious]
    name_list = ['OA', 'kappa', 'precision', 'recall', 'F1', 'miou', 'ious']
    for i in range(len(name_list)):
        device = '-'
        if torch.is_tensor(result[i]):
            device = result[i].device
        print(f'{name_list[i]}: {device}, {result[i]}')