import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt






def dice_coefficient(tp,fp,fn,tn,smooth=1e-6):

    tp=tp.sum()
    fp=fp.sum()
    fn=fn.sum()

    return ((2*tp+smooth)/(2*tp+fp+fn+smooth)).item()



def intersectionAndUnion(pred, target, args, cfg):
    nclass = args.nclass

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # 'K' classes, pred and target sizes are N or N * L or 
    # N * H * W, each value in range 0 to K - 1.
    assert pred.ndim in [1, 2, 3]
    assert pred.shape == target.shape
    pred = pred.reshape(pred.size).copy()
    target = target.reshape(target.size)
    intersection = pred[np.where(pred == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(nclass + 1))
    area_pred, _ = np.histogram(pred, bins=np.arange(nclass + 1))
    area_target, _ = np.histogram(target, bins=np.arange(nclass + 1))
    area_union = area_pred + area_target - area_intersection
    return area_intersection, area_union


def get_eval_scores(intersection, union, cfg, smooth=1e-10):
    iou_class = intersection.sum / (union.sum + smooth)
    # mIoU = np.mean(iou_class)

    mean_i = intersection.avg
    mean_u = union.avg

    var_i = intersection.var
    var_u = union.var

    iou_class_var = (mean_i / (mean_u + smooth))**2

    tmp1 = (var_i + smooth / (mean_i + smooth))**2
    tmp2 = (var_u + smooth / (mean_u + smooth))**2
    iou_class_var = iou_class_var * (tmp1 + tmp2)

    # mIoU_var = np.mean(iou_class_var)

    class_weights = np.array(cfg['class_weights'])
    iou_weights = class_weights / sum(class_weights)
    wIoU = sum(iou_class * iou_weights)

    # wIoU_var = sum(iou_class_var * iou_weights)

    scores = {
        'eval/wIoU': wIoU,
        # 'eval/wIoU_var': wIoU_var,
    }

    for i, iou in enumerate(iou_class):
        scores[f'eval/idx_{i}_iou'] = iou

    return scores



def visualise_eval(img, target, pred, idx, epoch, args, cfg):
    img_np = img.detach().cpu().numpy()
    target_np = F.one_hot(target, args.nclass).detach().cpu().numpy()
    pred_np = F.one_hot(pred, args.nclass).detach().cpu().numpy()

    for i in range(img_np.shape[0]):
        img = img_np[i].transpose(1,2,0)
        target = 255 * target_np[i]
        pred = 255 * pred_np[i]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Val Epoch {epoch}')
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(target, cmap='gray')
        axs[1].set_title('True Mask')
        axs[1].axis('off')

        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')


        plt.close(fig)
