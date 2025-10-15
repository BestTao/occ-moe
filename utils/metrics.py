import torch
import numpy as np
import os
from utils.reranking import re_ranking
# from openTSNE import TSNE
from matplotlib import pyplot as plt


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def euclidean_distance_(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat


def weighted_euclidean_distance(qf, gf, occ_score):
    """

    Args:
        qf: [num_query, dim*5]
        gf: [num_gallery, dim*5]
        occ_score: [num_query+num_gallery, 4]
        pid: for debug

    Returns:
        dist_mat: [num_query, num_gallery]
    """
    num_query = qf.shape[0]
    num_gallery = gf.shape[0]
    qfs = qf.reshape(num_query, 5, -1).permute(1, 0, 2)
    gfs = gf.reshape(num_gallery, 5, -1).permute(1, 0, 2)
    dist_mats = [euclidean_distance_(qfs[i], gfs[i]) for i in range(5)]

    # occ score
    non_occ_mean = occ_score[:, :, 0].reshape((occ_score.shape[0], 4, -1)).mean(dim=-1)*2
    weight_soft = non_occ_mean.softmax(dim=-1)*2

    dist_thre = 0.07
    dist_mat = dist_mats[0]
    for i in range(4):
        scale_mat = weight_soft[:num_query, i].unsqueeze(-1).expand(num_query, num_gallery) + \
                    weight_soft[num_query:, i].unsqueeze(-1).expand(num_gallery, num_query).t()
        dist_mats[i+1] = (dist_mats[i+1]-dist_thre)*scale_mat[i] + dist_thre
        dist_mat = dist_mat + dist_mats[i+1]
    return dist_mats[4]

def head_weighted_euclidean_distance(qf, gf):
    """

    Args:
        qf: [num_query, dim*5]
        gf: [num_gallery, dim*5]

    Returns:
        dist_mat: [num_query, num_gallery]
    """
    head_num = 12
    part_head_num = 6
    num_query = qf.shape[0]
    num_gallery = gf.shape[0]
    qfs = qf.reshape(num_query, 5, head_num, -1).permute(2, 0, 1, 3).reshape(head_num, num_query, -1)
    gfs = gf.reshape(num_gallery, 5, head_num, -1).permute(2, 0, 1, 3).reshape(head_num, num_gallery, -1)
    dist_mats = [euclidean_distance_(qfs[i], gfs[i]) for i in range(head_num)]

    part_head_coff = 1.2

    dist_mat = torch.zeros_like(dist_mats[0])
    for i in range(head_num):
        if i < part_head_num:
            dist_mat = dist_mat + dist_mats[i]*part_head_coff
        else:
            dist_mat = dist_mat + dist_mats[i]
    return dist_mat

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    idx5 = np.zeros((num_q, 5), dtype=np.int32)
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        idx5[q_idx] = indices[q_idx][keep][:5]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, idx5


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            # distmat = euclidean_distance(qf, gf)
            distmat = head_weighted_euclidean_distance(qf, gf)
            # print('=> Computing DistMat with weighted_euclidean_distance')
        cmc, mAP, idx5 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf, idx5

    # def save_npy(self, name):
    #     feats = torch.cat(self.feats, dim=0)
    #     if self.feat_norm:
    #         print("The test feature is normalized")
    #         feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
    #     pids = np.asarray(self.pids)
    #     camids = np.asarray(self.camids)
    #     print(str(self.num_query)+"querys")
    #     np.save("./feats/"+name+"_feat.npy", feats)
    #     np.save("./feats/"+name+"_pid.npy", pids)
    #     np.save("./feats/"+name+"_camid.npy", camids)

    def save_npy(self, name):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        
        # 确保目录存在
        save_dir = "./feats/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 递归创建目录
            
        # 保存文件
        pids = np.asarray(self.pids)
        camids = np.asarray(self.camids)
        np.save(os.path.join(save_dir, f"{name}_feat.npy"), feats)
        np.save(os.path.join(save_dir, f"{name}_pid.npy"), pids)
        np.save(os.path.join(save_dir, f"{name}_camid.npy"), camids)
