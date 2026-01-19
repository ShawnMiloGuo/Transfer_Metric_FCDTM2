import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def all_index(predictions, labels, num_classes, is_val=False):
    '''
    OA, kappa, precision, recall, F1, miou, ious
    '''
    #oa, kappa, precision, recall, f1
    predictions = predictions.flatten()
    labels = labels.flatten()
    OA = accuracy_score(labels, predictions)
    miou, ious = mean_iou(predictions, labels, num_classes)

    if is_val:
        return OA, miou, ious
    else:
        kappa = cohen_kappa_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        F1 = f1_score(labels, predictions, average='macro')
        return OA, kappa, precision, recall, F1, miou, ious


def pixel_accuracy(predictions, labels):
    # 计算像素准确度
    # 将预测结果和真实标签展开成一维数组
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    # 计算相同位置的像素是否一致
    correct_pixels = np.sum(predictions_flat == labels_flat)
    total_pixels = len(predictions_flat)
    # 计算像素准确度
    pixel_acc = correct_pixels / total_pixels
    return pixel_acc

def mean_iou(predictions, labels, num_classes):
    # 计算平均 Intersection over Union（mIOU）
    # 初始化混淆矩阵
    confusion_mat = confusion_matrix(labels.flatten(), predictions.flatten(), labels=list(range(num_classes)))
    # 计算每个类别的 IOU
    ious = []
    for i in range(num_classes):
        intersection = confusion_mat[i, i]
        union = np.sum(confusion_mat[i, :]) + np.sum(confusion_mat[:, i]) - intersection
        if union == 0:
            iou = 0  # 如果该类别在真实标签中不存在，则 IOU 为 0
        else:
            iou = intersection / union
        ious.append(iou)
    # 计算 mIOU
    mean_iou = np.mean(ious)
    return mean_iou, ious

if __name__ == '__main__':
    # 示例用法
    predictions = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1]])  # 模型预测结果
    labels = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])       # 真实标签
    num_classes = 2  # 类别数量

    # 计算像素准确度
    acc = pixel_accuracy(predictions, labels)
    print("像素准确度:", acc)

    # 计算平均 Intersection over Union
    iou = mean_iou(predictions, labels, num_classes)
    print("平均 Intersection over Union:", iou)
