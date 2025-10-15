# -*- coding=utf-8 -*-
import torch
from augmentation.augmentation_types import *
from detectron2.model_zoo import get_config
from detectron2.engine.defaults import DefaultPredictor
import cv2
import torchvision.transforms as T
import numpy as np
from PIL import Image

class OccAugment():
	def __init__(self, cfg, train_loader):
		self.id_img_masks = {}
		if OcclusionType.INSTANCE_MASK in cfg.INPUT.OCC_TYPES:
			print("开始 OcclusionType.INSTANCE_MASK")
			seg_pred = DefaultPredictor(get_config(cfg.INPUT.SEG_CFG, trained=True))
			self.id_img_masks = self.init_masks(seg_pred, train_loader)
			self.mask_transforms = T.Compose([
				T.Resize(cfg.INPUT.SIZE_TRAIN),
				T.ToTensor(),
				T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
			])

			self.mask_resize = T.Resize((cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16))

	def init_masks(self, seg_pred, train_loader, score_thre=0.6, area_thre=1024, inter_thre=196):
		id_img_masks = {}
		for n_iter, (_, pids, _, _, img_paths) in enumerate(train_loader):
			if (n_iter + 1) % 10 == 0:
				print("preparing person masks: ", n_iter + 1, "/", len(train_loader))
				# break # for debug
			for i in range(len(img_paths)):
				cv_img = cv2.imread(img_paths[i])
				inputs = cv2.resize(cv_img, (128, 256))
				outputs = seg_pred(inputs)

				# filter masks
				predictions = outputs["instances"]
				scores = predictions.scores if predictions.has("scores") else None
				classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
				masks = predictions.pred_masks if predictions.has("pred_masks") else None
				if masks is not None:
					masks = [masks[i] for i in range(len(masks)) if scores[i] >= score_thre and classes[i] == 0]
					# print("old length: ", len(masks), end='')
					# combine masks that has intersection
					del_flag = [False] * len(masks)
					for j in range(len(masks) - 1):
						for k in range(j + 1, len(masks)):
							if not del_flag[k]:
								inter = masks[j] & masks[k]
								if inter.sum() >= inter_thre:
									masks[j] = masks[j] | masks[k]
									del_flag[k] = True
					masks = [masks[j] for j in range(len(masks)) if not del_flag[j] and masks[j].sum() >= area_thre]
					if len(masks) == 0:
						continue
					max_mask = masks[0]
					for j in range(1, len(masks)):
						if masks[j].sum() > max_mask.sum():
							max_mask = masks[j]
					# save masks
					mask_dict = {
						"mask": max_mask,
						"path": img_paths[i]
					}
					if pids[i].item() not in id_img_masks.keys():
						id_img_masks.update({
							pids[i].item(): [mask_dict]
						})
					else:
						id_img_masks[pids[i].item()].append(mask_dict)
		return id_img_masks

	def do_augment(self, imgs, vids, cfg):
		out = imgs.clone()
		patch_mask = torch.zeros((imgs.shape[0], imgs.shape[2] * imgs.shape[3] // 16 // 16))
		occ_types = cfg.INPUT.OCC_TYPES
		occ_ratios = cfg.INPUT.OCC_TYPES_RATIO
		print(len(occ_types))
		print(len(occ_ratios))
		assert len(occ_types) == len(occ_ratios)
		for idx in range(imgs.shape[0]):
			occ_type_rand = torch.rand(size=(1,))[0]
			occ_ratio_sum = 0
			for i, occ_type in enumerate(occ_types):
				occ_ratio_sum += occ_ratios[i]
				if occ_type_rand <= occ_ratio_sum:
					patch_mask[idx] = self.gen_img_occ(imgs, vids, cfg, vids[idx], out[idx], occ_type)
					break

		return out, patch_mask == 1

	def gen_img_occ(self, imgs, vids, cfg, pid, img, occ_type):
		H = cfg.INPUT.SIZE_TRAIN[0]
		W = cfg.INPUT.SIZE_TRAIN[1]
		patch_mask = torch.zeros(H * W // 16 // 16)


		if occ_type == OcclusionType.INSTANCE_MASK:
			# select a mask from different id
			other_ids = [k for k in self.id_img_masks.keys() if k != pid]
			if len(other_ids) == 0:
				return patch_mask
			other_id = np.random.choice(other_ids)
			mask_dict = np.random.choice(self.id_img_masks[other_id])
			instance_img = Image.open(mask_dict["path"]).convert('RGB')
			instance_img = self.mask_transforms(instance_img).to("cuda")
			instance_mask = mask_dict["mask"]
			img_mask = torch.zeros_like(instance_mask)
			# random x&y shift
			dx = torch.randint(low=0, high=W, size=(1,))[0] - int(W / 2)
			dy = torch.randint(low=int(H / 4), high=int(H / 4 * 3), size=(1,))[0]
			ori_x1 = max(0, dx)
			ori_x2 = min(W, W + dx)
			ori_y1 = max(0, dy)
			ori_y2 = min(H, H + dy)
			msk_x1 = max(0, -1 * dx)
			msk_x2 = min(W, W - dx)
			msk_y1 = max(0, -1 * dy)
			msk_y2 = min(H, H - dy)
			m = img_mask[ori_y1:ori_y2, ori_x1:ori_x2] = instance_mask[msk_y1:msk_y2, msk_x1:msk_x2]
			img[:, ori_y1:ori_y2, ori_x1:ori_x2][:, m] = instance_img[:, msk_y1:msk_y2, msk_x1:msk_x2][:, m]
			img_mask = self.mask_resize(torch.unsqueeze(img_mask, 0))
			patch_mask[:] = torch.reshape(img_mask, patch_mask.shape)

		elif occ_type in (OcclusionType.IMG_RECT, OcclusionType.IMG_BLOCK):
			other_idxes = [idx for idx in range(len(vids)) if vids[idx] != pid]
			if len(other_idxes) == 0:
				return patch_mask
			idx_occ = np.random.choice(other_idxes)
			if occ_type == OcclusionType.IMG_BLOCK:
				mask1, mask2, patch_mask[:] = self.gen_blockwise_mask(W, H, masking_ratio=cfg.MODEL.OCC_RATIO,
				                                                        ulrd=cfg.MODEL.OCC_ULRD,
				                                                        margin=cfg.MODEL.OCC_MARGIN,
				                                                        align_bound=cfg.MODEL.OCC_ALIGN_BOUND)
			else:
				mask1, mask2, patch_mask[:] = self.gen_rectangle_mask(W, H, align_bottom=cfg.MODEL.OCC_ALIGN_BTM)
			img[:, mask1] = imgs[idx_occ, :, mask2]
		return patch_mask

	def gen_blockwise_mask(self, w, h, masking_ratio=(0.2,), min_size=4, aspect_ratio=0.3, margin=0.3, ulrd=(0.25, 0.25, 0.25, 0.25), align_bound=True):
		patch_size = 16
		pw = w // patch_size
		ph = h // patch_size
		mask = torch.zeros((h, w), dtype=torch.float32)
		patch_mask = torch.zeros((ph, pw), dtype=torch.float32)
		bound = [w, 0, h, 0]  # x1, x2, y1, y2
		if len(masking_ratio) == 2:  # [min, max]
			m_ratio = masking_ratio[0] + torch.rand(1)*masking_ratio[1]-masking_ratio[0]
		else:
			m_ratio = masking_ratio[0]
		ulrd_flag = torch.rand(size=(1,))[0]
		if ulrd_flag <= ulrd[0]:
			direction = "up"
		elif ulrd_flag <= ulrd[0] + ulrd[1]:
			direction = "left"
		elif ulrd_flag <= ulrd[0] + ulrd[1] + ulrd[2]:
			direction = "right"
		else:
			direction = "down"
		while patch_mask.mean() < m_ratio and min_size < int(np.ceil(ph*pw*(m_ratio-patch_mask.mean()))):
			block_size = torch.randint(low=min_size, high=int(ph*pw*(m_ratio-patch_mask.mean()))+1, size=(1,))[0]
			block_w, block_h = -1, -1
			# generate block
			while not ((0 < block_w < pw and 0 < block_h < ph - int(margin*ph) and (direction == "up" or direction == "down")) or
			           (0 < block_w < pw - int(margin*pw) and 0 < block_h < ph and (direction == "left" or direction == "right"))):
				hw_flag = torch.rand(size=(1,))[0] > 0.5
				ratio = (1-torch.rand(size=(1,))*(1-aspect_ratio)) if hw_flag else 1/(1-torch.rand(size=(1,))*(1-aspect_ratio))  # [0.3, 1/0.3]
				block_w = int(np.ceil(np.sqrt(block_size*ratio)))
				block_h = int(np.ceil(np.sqrt(block_size/ratio)))
			# choose position
			if direction == "up":
				x1 = torch.randint(low=0, high=pw - block_w + 1, size=(1, ))[0]
				y1 = torch.randint(low=0, high=ph - block_h + 1 - int(margin*ph), size=(1, ))[0]
			elif direction == "left":
				x1 = torch.randint(low=0, high=pw - block_w + 1 - int(margin*pw), size=(1, ))[0]
				y1 = torch.randint(low=0, high=ph - block_h + 1, size=(1, ))[0]
			elif direction == "right":
				x1 = torch.randint(low=int(margin*pw), high=pw - block_w + 1, size=(1, ))[0]
				y1 = torch.randint(low=0, high=ph - block_h + 1, size=(1, ))[0]
			else:
				x1 = torch.randint(low=0, high=pw - block_w + 1, size=(1, ))[0]
				y1 = torch.randint(low=int(margin*ph), high=ph - block_h + 1, size=(1, ))[0]
			patch_mask[y1:y1+block_h, x1:x1+block_w] = 1
			mask[y1*patch_size:(y1+block_h)*patch_size, x1*patch_size:(x1+block_w)*patch_size] = 1
			# update boundary
			bound[0] = min(x1*patch_size, bound[0])
			bound[1] = max((x1+block_w)*patch_size, bound[1])
			bound[2] = min(y1*patch_size, bound[2])
			bound[3] = max((y1+block_h)*patch_size, bound[3])
		if align_bound:
			if direction == "up" and bound[2] > 0:
				x_shift = 0
				y_shift = 0 - bound[2]
			elif direction == "left" and bound[0] > 0:
				x_shift = 0 - bound[0]
				y_shift = 0
			elif direction == "right" and bound[1] < w:
				x_shift = w - bound[1]
				y_shift = 0
			elif direction == "down" and bound[3] < h:
				x_shift = 0
				y_shift = h - bound[3]
			else:
				x_shift = 0
				y_shift = 0
			mask = torch.roll(mask, [y_shift, x_shift], [0, 1])
			patch_mask = torch.roll(patch_mask, [y_shift//patch_size, x_shift//patch_size], [0, 1])
			bound[0] += x_shift
			bound[1] += x_shift
			bound[2] += y_shift
			bound[3] += y_shift
		mask_bias = torch.zeros_like(mask)
		x_bias = torch.randint(low=0 - bound[0], high=w + 1 - bound[1], size=(1,))[0]
		y_bias = torch.randint(low=0 - bound[2], high=h + 1 - bound[3], size=(1,))[0]
		mask_bias[bound[2]+y_bias:bound[3]+y_bias, bound[0]+x_bias:bound[1]+x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
		patch_mask = torch.flatten(patch_mask)
		return mask == 1, mask_bias == 1, patch_mask

	def gen_rectangle_mask(self, w, h, min_occ_ratio=(0.2, 0.2), max_occ_ratio=(1.0, 0.5), margin_top=0.3, align_bottom=True):
		patch_size = 16
		pw = w // patch_size
		ph = h // patch_size
		mask = torch.zeros((h, w), dtype=torch.float32)
		patch_mask = torch.zeros((ph, pw), dtype=torch.float32)
		x1 = torch.randint(low=0, high=int(pw*(1-min_occ_ratio[0])) + 1, size=(1,))[0]
		x2 = torch.randint(low=x1+int(min_occ_ratio[0]*pw), high=min(x1+int(max_occ_ratio[0]*pw), pw) + 1, size=(1,))[0]
		if align_bottom:
			y1 = torch.randint(low=int((1-max_occ_ratio[1])*ph), high=int((1-min_occ_ratio[1])*ph), size=(1,))[0]
			y2 = ph
		else:
			y1 = torch.randint(low=int(margin_top * ph), high=int(ph*(1-min_occ_ratio[1])) + 1, size=(1,))[0]
			y2 = torch.randint(low=y1+int(min_occ_ratio[1]*pw), high=min(y1+int(max_occ_ratio[1]*ph), ph) + 1, size=(1,))[0]
		bound = [x1 * patch_size, x2 * patch_size, y1 * patch_size, y2 * patch_size]
		patch_mask[y1:y2, x1:x2] = 1
		mask[bound[2]:bound[3], bound[0]:bound[1]] = 1
		mask_bias = torch.zeros_like(mask)
		x_bias = torch.randint(low=0 - bound[0], high=w + 1 - bound[1], size=(1,))[0]
		y_bias = torch.randint(low=0 - bound[2], high=h + 1 - bound[3], size=(1,))[0]
		mask_bias[bound[2] + y_bias:bound[3] + y_bias, bound[0] + x_bias:bound[1] + x_bias] = mask[bound[2]:bound[3], bound[0]:bound[1]]
		# print(mask.mean(), mask_bias.mean())
		patch_mask = torch.flatten(patch_mask)
		return mask == 1, mask_bias == 1, patch_mask


def make_aug(cfg, train_loader):
	aug = OccAugment(cfg, train_loader)
	return aug