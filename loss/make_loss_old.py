# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def embed_ifrc_loss(feat_ori, feat_inf, cfg, patch_mask=None):  # by zzw
    # patchembed_ori is detached from graph
    if cfg.MODEL.IFRC_TARGET == "feat":
        x_ori = feat_ori[:, :cfg.MODEL.IFRC_HEAD_NUM*64]
        x_inf = feat_inf[:, :cfg.MODEL.IFRC_HEAD_NUM*64]
    elif cfg.MODEL.IFRC_TARGET == 'cls_token':
        x_ori = feat_ori[:, 0]
        x_inf = feat_inf[:, 0]
    elif cfg.MODEL.IFRC_TARGET == "embed":
        x_ori = feat_ori[:, 1:]
        x_inf = feat_inf[:, 1:]
    elif cfg.MODEL.IFRC_TARGET == "masked_embed":
        x_ori = feat_ori[:, 1:][patch_mask]
        x_inf = feat_inf[:, 1:][patch_mask]

    if cfg.MODEL.IFRC_LOSS_TYPE == 'mse':
        loss = F.mse_loss(x_ori, x_inf)
    elif cfg.MODEL.IFRC_LOSS_TYPE == 'smoothl1':
        loss = F.smooth_l1_loss(x_ori, x_inf)
    elif cfg.MODEL.IFRC_LOSS_TYPE == 'l2dist':
        # a = a.contiguous()
        # b = b.contiguous()
        loss = torch.cdist(x_ori.contiguous(), x_inf.contiguous(), p=2).mean()
    else:
        raise NotImplementedError("Inference type: {} is not implemented".format(cfg.MODEL.IFRC_LOSS_TYPE))
    return loss


def rgb_ifrc_loss(target, pred, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


'''
def rgb_avg_ifrc_loss(target, pred, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss'''

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 2) +
                     epsilon, 0.5).unsqueeze(2).expand_as(feature)
    return torch.div(feature, norm)

def orthonomal_loss(w):
    B, K, C = w.shape
    w_norm = featureL2Norm(w)
    WWT = torch.matmul(w_norm, w_norm.transpose(1, 2))
    return F.mse_loss(WWT - torch.eye(K).unsqueeze(0).cuda(), torch.zeros(B, K, K).cuda(), size_average=False) / (K*K)



def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        if cfg.MODEL.ZZWEXP:
            def loss_func(scores, feats, target, occ_pred=None, patch_mask=None, head_suppress = None, attns=None ,patchembeds=None, pretext_tgt=None, pretext_pred=None,):
                B, C = feats["occ"][0].shape
                H = 12

                if cfg.MODEL.OCC_AUG:
                    # ID loss
                    if isinstance(scores["occ"], list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in scores["occ"][1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = ID_LOSS + 0.5 * F.cross_entropy(scores["occ"][0], target)
                    else:
                        ID_LOSS = F.cross_entropy(scores["occ"], target)
                    # Triplet loss
                    if isinstance(feats["occ"], list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feats["occ"][1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = TRI_LOSS + 0.5 * triplet(feats["occ"][0], target)[0]
                    else:
                        TRI_LOSS = triplet(feats["occ"], target)[0]

                    # weighted sum
                    LOSS = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                           cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    return LOSS, ID_LOSS, TRI_LOSS, None, None

                # ID loss
                if cfg.MODEL.TWO_BRANCHED:
                    if isinstance(scores["occ"], list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in scores["occ"][1:]+scores["ori"][1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.25 * F.cross_entropy(scores["occ"][0], target) + 0.25 * F.cross_entropy(scores["ori"][0], target)
                    else:
                        ID_LOSS = 0.5 * F.cross_entropy(scores["occ"], target) + 0.5 * F.cross_entropy(scores["ori"], target)
                else:
                    if isinstance(scores["occ"], list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in scores["occ"][1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = ID_LOSS + 0.5 * F.cross_entropy(scores["occ"][0], target)
                    else:
                        ID_LOSS = F.cross_entropy(scores["occ"], target)

                # Triplet loss
                if cfg.MODEL.TWO_BRANCHED:
                    if isinstance(feats["occ"], list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feats["occ"][1:]+feats["ori"][1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.25 * triplet(feats["occ"][0], target)[0] + 0.25 * triplet(feats["ori"][0], target)[0]
                    else:
                        TRI_LOSS = 0.5 * triplet(feats["occ"], target)[0] + 0.5 * triplet(feats["ori"], target)[0]
                else:
                    if isinstance(feats["occ"], list):
                        TRI_LOSS = [triplet(feats, target, head_suppress=head_suppress)[0] for feats in feats["occ"][1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = TRI_LOSS + 0.5 * triplet(feats["occ"][0], target, head_suppress=head_suppress)[0]
                    else:
                        TRI_LOSS = triplet(feats["occ"], target)[0]

                # Inference loss
                if cfg.MODEL.TWO_BRANCHED and cfg.MODEL.IFRC:
                    if cfg.MODEL.IFRC_TARGET == 'embed' or cfg.MODEL.IFRC_TARGET == 'masked_embed' or cfg.MODEL.IFRC_TARGET == 'cls_token':
                        IFRC_LOSS = embed_ifrc_loss(patchembeds["ori"], patchembeds["occ"], cfg, patch_mask)
                    elif cfg.MODEL.IFRC_TARGET == 'feat':
                        IFRC_LOSS = embed_ifrc_loss(feats["ori"][0], feats["occ"][0], cfg, patch_mask)
                    else:
                        raise NotImplementedError("pretext type: {} is not implemented".format(cfg.MODEL.IFRC_TARGET))
                else:
                    IFRC_LOSS = 0

                # Occlude Aware loss
                if occ_pred is not None and patch_mask is not None and cfg.MODEL.OCC_AWARE:
                    B = patch_mask.shape[0]
                    if cfg.MODEL.TWO_BRANCHED:
                        occ_pred_occ = occ_pred[:B].reshape((-1, 2))
                        occ_pred_ori = occ_pred[B:].reshape((-1, 2))
                        occ_target_occ = patch_mask.long().reshape((-1))
                        occ_target_ori = torch.zeros_like(occ_target_occ).reshape((-1))
                        OCC_LOSS = F.cross_entropy(occ_pred_occ, occ_target_occ) + F.cross_entropy(occ_pred_ori, occ_target_ori) * 0.1
                    else:
                        occ_pred_occ = occ_pred.reshape((-1, 2))
                        occ_target_occ = patch_mask.long().reshape((-1))
                        OCC_LOSS = F.cross_entropy(occ_pred_occ, occ_target_occ)
                else:
                    OCC_LOSS = 0

                # Head Diversity loss
                if cfg.MODEL.HEAD_ENHANCE:
                    feat_global = feats["occ"][0]
                    feat_global = feat_global.reshape((feat_global.shape[0], 12, -1))
                    HEAD_DIV_LOSS = orthonomal_loss(feat_global)
                    for feat in feats["occ"][1:]:
                        feat = feat.reshape((feat.shape[0], 12, -1))
                        HEAD_DIV_LOSS += orthonomal_loss(feat) * 0.25

                    if attns is not None:
                        HEAD_DIV_LOSS += orthonomal_loss(attns["occ"]) # not good
                else:
                    HEAD_DIV_LOSS = 0


                # triplet loss divided by heads
                TRI_LOSS_LIST = []
                if cfg.MODEL.HEAD_SUP:
                    with torch.no_grad():
                        for h in range(12):
                            head_feat = feats["occ"][0].reshape(B, H, -1)[:, h, :]
                            TRI_LOSS_LIST.append(triplet(head_feat, target)[0])
                        TRI_LOSS_DIV = torch.stack(TRI_LOSS_LIST)
                else:
                    TRI_LOSS_DIV = None


                # weighted sum
                LOSS = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                    cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                    cfg.MODEL.OCC_LOSS_WEIGHT * OCC_LOSS + \
                    cfg.MODEL.IFRC_LOSS_WEIGHT * IFRC_LOSS + \
                    cfg.MODEL.HEAD_DIV_LOSS_WEIGHT * HEAD_DIV_LOSS


                return LOSS, ID_LOSS, TRI_LOSS, OCC_LOSS, IFRC_LOSS, TRI_LOSS_DIV
        else:
            def loss_func(score, feat, target, target_cam):
                if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        if isinstance(score, list):
                            ID_LOSS = [xent(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                        else:
                            ID_LOSS = xent(score, target)
    
                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]
    
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                   cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    else:
                        if isinstance(score, list):
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                            ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                        else:
                            ID_LOSS = F.cross_entropy(score, target)
    
                        if isinstance(feat, list):
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                        else:
                                TRI_LOSS = triplet(feat, target)[0]
    
                        LOSS = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                        return LOSS, ID_LOSS, TRI_LOSS
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                          'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion