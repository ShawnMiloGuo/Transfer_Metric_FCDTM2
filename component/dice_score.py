import torch
from torch import Tensor
import torch.nn.functional as F


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

if __name__ == "__main__":
    masks_pred = torch.tensor([[[[0]], [[10]], [[5]]]], dtype=torch.float32)
    true_masks = torch.tensor([[[2]]], dtype=torch.long)
    print(f'masks_pred: \n{masks_pred}')
    print(f'true_masks: \n{true_masks}')

    # masks_pred = torch.argmax(masks_pred,keepdim=True)
    masks_pred = F.softmax(masks_pred, dim=1).float()
    true_masks = F.one_hot(true_masks, 3).permute(0, 3, 1, 2).float()

    print(f'masks_pred: \n{masks_pred}')
    print(f'true_masks: \n{true_masks}')

    print(masks_pred.shape)
    print(true_masks.shape)
    dice = dice_loss(masks_pred,
                    true_masks,
                    multiclass=True
                        )
    print(dice)