import logging
import time
import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from log_handler.json_logger import JsonLogger
from log_handler import log_config
# from datasets import DataStat
from augmentation.augmentation_types import AugmentationType


# 注意，带有 save_images 的均被注释掉了，正式使用不能注释

def patchify(imgs):
	"""
	imgs: (N, 3, H, W)
	x: (N, L, patch_size**2 *3)
	"""
	p = 16
	assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

	h = imgs.shape[2] // p
	w = imgs.shape[3] // p
	x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
	x = torch.einsum('nchpwq->nhwpqc', x)
	x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
	return x


def unpatchify(x, H, W):
	"""
	x: (N, L, patch_size**2 *3)
	imgs: (N, 3, H, W)
	"""
	p = 16
	h = H // p
	w = W // p
	assert h * w == x.shape[1]

	x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
	x = torch.einsum('nhwpqc->nchpwq', x)
	imgs = x.reshape(shape=(x.shape[0], 3, H, W))
	return imgs


def patchify_rgb_avg(imgs, avg_size=16):
	"""
	imgs: (N, 3, H, W)
	x: (N, L, (16/a)**2*3)
	"""
	p = 16
	a = avg_size
	assert imgs.shape[3] % p == 0 and imgs.shape[2] % p == 0

	h = imgs.shape[2] // p
	w = imgs.shape[3] // p
	x = imgs.reshape(shape=(imgs.shape[0], 3, h, p // a, a, w, p // a, a))
	x = torch.einsum('nchpawqb->nhwabpqc', x)
	x = x.reshape(shape=(imgs.shape[0], h * w, a ** 2, (p // a) ** 2 * 3))
	x = torch.mean(x, 2)
	return x


def unpatchify_rgb_avg(x, H, W, avg_size):
	"""
	x: (N, L, (16/a)**2*3)
	imgs: (N, 3, H, W)
	"""
	p = 16
	a = avg_size
	h = H // p
	w = W // p
	assert h * w == x.shape[1]
	x = x.reshape(shape=(x.shape[0], h, w, p // a, 1, p // a, 1, 3))
	x = x.repeat(1, 1, 1, 1, a, 1, a, 1)
	x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
	x = torch.einsum('nhwpqc->nchpwq', x)
	imgs = x.reshape(shape=(x.shape[0], 3, H, W))
	return imgs


def unpatchify_mask(m, H, W):
	"""
	Args:
		m: (N, L)
		H: int
		W: int

	Returns:
		mask: (N, 3, H, W)
	"""
	p = 16
	h = H // p
	w = W // p
	assert h * w == m.shape[1]

	m = m.reshape(shape=(m.shape[0], h, w, 1, 1, 1))
	m = m.repeat(1, 1, 1, p, p, 3)
	m = torch.einsum('nhwpqc->nchpwq', m)
	mask = m.reshape(shape=(m.shape[0], 3, H, W))
	return mask


def shift(img, max_shift_ratio=0.33):
	out = torch.zeros_like(img)
	w = img.shape[3]
	h = img.shape[2]
	shift_direc = torch.rand(img.shape[0])
	shift_ratio = torch.rand(img.shape[0])
	for idx in range(img.shape[0]):
		if shift_direc[idx] < 0.25:  # shift up
			dist = int(shift_ratio[idx] * max_shift_ratio * h)
			out[idx, :, 0:h - dist - 1, :] = img[idx, :, dist:h - 1, :]
		elif shift_direc[idx] < 0.5:  # shift right
			dist = int(shift_ratio[idx] * max_shift_ratio * w)
			out[idx, :, :, dist:w - 1] = img[idx, :, :, 0:w - dist - 1]
		elif shift_direc[idx] < 0.75:  # shift down
			dist = int(shift_ratio[idx] * max_shift_ratio * h)
			out[idx, :, dist:h - 1, :] = img[idx, :, 0:h - dist - 1, :]
		else:  # shift left
			dist = int(shift_ratio[idx] * max_shift_ratio * w)
			out[idx, :, :, 0:w - dist - 1] = img[idx, :, :, dist:w - 1]
	return out


def rollout(attentions, discard_ratio, head_fusion):
	"""

	Args:
		attentions: [blocks*[B, H, 2+P, 2+P]]
		discard_ratio:
		head_fusion:

	Returns:
		mask: [B, P]
	"""
	result = torch.eye(attentions[0].size(-1))
	with torch.no_grad():
		for attention in attentions:  # blocks
			if head_fusion == "mean":
				attention_heads_fused = attention.mean(axis=1)
			elif head_fusion == "max":
				attention_heads_fused = attention.max(axis=1)[0]
			elif head_fusion == "min":
				attention_heads_fused = attention.min(axis=1)[0]
			else:
				raise Exception("Attention head fusion type Not supported")

			# Drop the lowest attentions, but
			# don't drop the class token
			'''
			flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
			_, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
			indices = indices[indices != 0]
			flat[0, indices] = 0
			'''
			flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
			for i in range(flat.shape[0]):
				_, indices = flat[i].topk(int(flat[i].size(-1) * discard_ratio), -1, False)
				indices = indices[indices != 0]
				indices = indices[indices != 1]
				flat[i, indices] = 0

			I = torch.eye(attention_heads_fused.size(-1))
			a = (attention_heads_fused + 1.0 * I) / 2
			a_sum = a.sum(dim=-1).unsqueeze(-1)
			a = a / a_sum

			result = torch.matmul(a, result)
	mask = result[:, 0, -128:]
	'''
	if occ_token:
		mask = result[:, 0, 2:]
	else:
		mask = result[:, 0, 1:]
		'''
	'''
	# Look at the total attention between the class token,
	# and the image patches
	mask = result[0, 0 , 1 :]
	# In case of 224x224 image, this brings us from 196 to 14
	width = int(mask.size(-1)**0.5)
	mask = mask.reshape(width, width).numpy()
	mask = mask / np.max(mask)'''
	return mask


def hook_out_attention(attns_hook):
	"""

	Args:
		attns_hook: [blocks*[B, H, 1+P, 1+P]]

	Returns:
		res: [B, H, P]
	"""
	result = torch.eye(attns_hook[0].size(-1))
	with torch.no_grad():
		for attn_layer in attns_hook:
			result = torch.matmul(attn_layer, result)
	res = result[:, :, 0, -128:]

	return res.to("cuda")

	# return attns_hook[-1][:, :, 0, -128:] # last layer


def visualize_matches(idx5, val_dataset, path_list, root, id_annotation=False):
	query_root = "../../datasets/Occluded_Duke/query/"
	gallery_root = "../../datasets/Occluded_Duke/bounding_box_test/"
	num_query = idx5.shape[0]
	for q_idx in range(num_query):
		q_id = val_dataset[q_idx][1]
		q_img = cv2.resize(cv2.imread(query_root + path_list[q_idx]), (128, 256))
		g_ids = [val_dataset[num_query + g_idx][1] for g_idx in idx5[q_idx]]
		g_imgs = [cv2.resize(cv2.imread(gallery_root + path_list[num_query + g_idx]), (128, 256)) for g_idx in
		          idx5[q_idx]]
		if id_annotation:
			output = np.ones((300, 868, 3)) * 255
		else:
			output = np.ones((256, 868, 3)) * 255
		cv2.putText(output, "{:0>4d}".format(q_id), (20, 295), 0, 1, (255, 255, 255))
		for m_idx in range(5):
			if id_annotation:
				cv2.putText(output, "{:0>4d}".format(g_ids[m_idx]), (128 * (m_idx + 1) + 20, 295), 0, 1,
				            (255, 255, 255))
			if g_ids[m_idx] == q_id:
				cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 255, 0), 4)
			else:
				cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 0, 255), 4)

		output[:256, :128] = q_img
		output[:256, 228:] = np.concatenate(g_imgs, axis=1)
		cv2.imwrite(root + '/match_results/{:0>5d}.jpg'.format(q_idx + 1), output)


def draw_patch_attn(patchembed, H=256, W=128):
	"""

	Args:
		patchembed: (B, (1+P), D)
		H:
		W:

	Returns:
		attn_maps: (B, 3, H, W)

	"""
	resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)
	patch_size = 16
	cls_token = patchembed[:, 0, :].unsqueeze(1)
	image_embed = patchembed[:, 1:, :]
	attn = (cls_token @ image_embed.transpose(-2, -1))
	attn = attn.softmax(dim=-1).squeeze(1)
	max = attn.max(dim=1)[0].unsqueeze(1)
	norm_attn = attn / max * 2 - 1
	attn_maps = norm_attn.reshape(shape=(norm_attn.shape[0], H // patch_size, W // patch_size)).unsqueeze(1).repeat(1,
	                                                                                                                3,
	                                                                                                                1,
	                                                                                                                1)
	attn_maps = resize(attn_maps)
	return attn_maps


def visualize_occ_label(patch_mask, H=256, W=128):
	"""

	Args:
		occ_pred: (B, P, 2)

	Returns:
		occ_map: (B, 3, H, W)

	"""
	resize = T.Resize((H, W), interpolation=T.InterpolationMode.NEAREST)
	patch_size = 16
	occ_label = patch_mask.float() * 2 - 1
	occ_label = occ_label.reshape(shape=(occ_label.shape[0], H // patch_size, W // patch_size)).unsqueeze(1).repeat(1,
	                                                                                                                3,
	                                                                                                                1,
	                                                                                                                1)
	occ_label = resize(occ_label)
	return occ_label



def update_sample_head_salience(sample_head_salience, path, ch_attn):
	B, C = ch_attn.shape
	H = 12
	head_attn = torch.mean(ch_attn.reshape(B, H, -1), dim=2)
	for i, p in enumerate(path):
		sample_head_salience.update({path[i]: head_attn[i]})


def query_sample_head_salience(sample_head_sailence, path):
	pass


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             augment,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
	log_period = cfg.SOLVER.LOG_PERIOD
	checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
	eval_period = cfg.SOLVER.EVAL_PERIOD

	device = "cuda"
	epochs = cfg.SOLVER.MAX_EPOCHS
	# log_json = False
	hook_attn_from_model = True
	iter_save_log = 1
	H = 12

	logger = logging.getLogger("transreid.train")
	if log_config.ENABLE and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
		json_logger = JsonLogger(cfg=cfg)
		logger.info('start training')
		max_rank1 = 0
	_LOCAL_PROCESS_GROUP = None
	# model = torch.nn.parallel.DataParallel(model, device_ids=[0,1])

	if device:
		model.to(local_rank)
		if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
			if dist.get_rank() == 0:
				print('Using {} GPUs for training'.format(torch.cuda.device_count()))
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
			                                                  find_unused_parameters=True)

	if hook_attn_from_model:
		# append hook
		attns_hook = []  # raw data hook out from model
		modules = {}

		def get_attn(module, input, output):
			attns_hook.append(output.cpu())

		for name, module in model.named_modules():
			if 'attn_drop' in name and not 'occ_block' in name:
				module.register_forward_hook(get_attn)
				modules[name] = module

	if cfg.MODEL.SAMPLE_HEAD_SUP:
		sample_head_salience = {}
	loss_meter = AverageMeter()
	loss_ifrc_meter = AverageMeter()
	loss_occ_meter = AverageMeter()
	loss_id_meter = AverageMeter()
	loss_moe_meter= AverageMeter()
	loss_tri_meter = AverageMeter()
	loss_id_dec_meter = AverageMeter()
	loss_tri_dec_meter = AverageMeter()
	acc_meter = AverageMeter()

	evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
	scaler = amp.GradScaler()
	#
	paras = {}
	for idx, para in enumerate(model.named_parameters()):
		paras[idx] = para
	print("开始训练-------")
	# train
	for epoch in range(1, epochs + 1):
		start_time = time.time()
		loss_meter.reset()
		loss_ifrc_meter.reset()
		loss_occ_meter.reset()
		loss_id_meter.reset()
		loss_tri_meter.reset()
		loss_id_dec_meter.reset()
		loss_tri_dec_meter.reset()
		loss_moe_meter.reset()
		acc_meter.reset()
		evaluator.reset()
		scheduler.step(epoch)
		model.train()
		# freeze occ pred
		if cfg.MODEL.HEAD_SUP:
			if epoch == 1:
				head_mask = torch.ones(H, dtype=torch.bool, requires_grad=False).to(device)
			head_total_loss = torch.zeros(H, requires_grad=False)
		else:
			head_mask = None
		print("进入第一个epoch内部的开始")
		for n_iter, (img, vid, target_cam, target_view, path) in enumerate(train_loader):
			optimizer.zero_grad()
			optimizer_center.zero_grad()
			if hook_attn_from_model:
				attns_hook = []
			img = img.to(device)
			if len(cfg.INPUT.AUG_TYPES) != 0:
				img_occ, patch_mask = augment.do_augment(img, vid, cfg)
				patch_mask = patch_mask.to(device)
			else:
				patch_mask = None
				img_occ = img.clone()
			target = vid.to(device)
			target_cam = target_cam.to(device)
			target_view = target_view.to(device)
			with amp.autocast(enabled=True):
				if cfg.MODEL.ZZWEXP:
					# 请修改注释
					scores, feats, occ_pred, sp_attn, ch_attn,metrics_occ,metrics_ori,experts_attn_occ,experts_attn_ori = model(img_occ, img, cam_label=target_cam,
					                                                  view_label=target_view, head_suppress=None)
					# scores, feats, occ_pred, sp_attn, ch_attn= model(img_occ, img, cam_label=target_cam,
					# 								view_label=target_view, head_suppress=None)
					score = scores["occ"]
					if hook_attn_from_model:
						# for visualize
						attns_map_occ = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean'
						                        )
						if cfg.MODEL.TWO_BRANCHED:
							attns_map_ori = rollout(attns_hook[16:28], discard_ratio=0.5, head_fusion='mean'
							                        )
						# for training
						attns_data = hook_out_attention(attns_hook[:12]).to("cuda")
						attns = {"occ": attns_data}
					else:
						attns = None

					if cfg.MODEL.SAMPLE_HEAD_SUP:
						update_sample_head_salience(sample_head_salience, path, ch_attn["occ"].to("cpu"))
					# loss, l_id, l_tri, l_occ, l_ifrc = loss_fn(scores, feats, target, occ_pred=occ_pred, patch_mask=patch_mask, attns=attns)
					# 请修改注释

					loss, l_id, l_tri, l_occ, l_ifrc, l_tri_div,moe_loss= \
						loss_fn(scores, feats, target, occ_pred=occ_pred, patch_mask=patch_mask,head_suppress=head_mask, attns=None,metrics_occ=metrics_occ,metrics_ori=metrics_ori)
					# loss, l_id, l_tri, l_occ, l_ifrc, l_tri_div= loss_fn(scores, feats, target, occ_pred=occ_pred, patch_mask=patch_mask,
					# 	        head_suppress=head_mask, attns=None)
				else:  # origin transreid
					score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
					if hook_attn_from_model:
						attns_map_ori = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean')
					loss, l_id, l_tri = loss_fn(score, feat, target, target_cam)  # target_cam is not used

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			# for name, param in model.named_parameters():
			# 	if param.grad is not None:
			# 		print(
			# 			f"层名: {name}\n"
			# 			f"梯度均值: {param.grad.mean().item():.6f}\n"
			# 			f"梯度最大值: {param.grad.max().item():.6f}\n"
			# 			f"梯度最小值: {param.grad.min().item():.6f}\n"
			# 			"------------------------"
			# 		)
			# 	else:
			# 		print(f"层名: {name} | 无梯度")

			scaler.step(optimizer)
			scaler.update()
			# compute loss_div sum
			if cfg.MODEL.HEAD_SUP:
				head_total_loss += l_tri_div.cpu()

			if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
				for param in center_criterion.parameters():
					param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
				scaler.step(optimizer_center)
				scaler.update()
			if isinstance(score, list):
				acc = (score[0].max(1)[1] == target).float().mean()
			else:
				acc = (score.max(1)[1] == target).float().mean()

			loss_meter.update(loss.item(), img.shape[0])
			if cfg.MODEL.ZZWEXP and not cfg.MODEL.OCC_AUG:
				if cfg.MODEL.IFRC:
					loss_ifrc_meter.update(l_ifrc.item(), img.shape[0])

			loss_id_meter.update(l_id.item(), img.shape[0])
			loss_moe_meter.update(moe_loss.item(), img.shape[0])
			loss_tri_meter.update(l_tri.item(), img.shape[0])
			acc_meter.update(acc, 1)

			torch.cuda.synchronize()

			# save images

			if log_config.ENABLE and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0) and n_iter < iter_save_log:
				if cfg.MODEL.ZZWEXP:
					if epoch == 1 and n_iter == 0:
						json_logger.save_images(img, 1, '-o')
						if AugmentationType.OCCLUSION in cfg.INPUT.AUG_TYPES:
							json_logger.save_images(img_occ, 1, '-m')
							occ_label = visualize_occ_label(patch_mask)
							json_logger.save_images(occ_label, 1, '-gt')
					elif epoch % 5 == 0 and epoch > 60:
						json_logger.save_images(img, epoch, '-o', id_plus=img.shape[0] * n_iter)
						json_logger.save_images(img_occ, epoch, '-m', id_plus=img_occ.shape[0] * n_iter)
						if hook_attn_from_model:
							json_logger.visualize_attn(img, attns_map_ori, epoch, '-o', prefix='img',
							                           id_plus=img.shape[0] * n_iter)
							json_logger.visualize_expert_attention(img, experts_attn_ori, epoch, '-eo', prefix='img',
							                           id_plus=img.shape[0] * n_iter)
							if cfg.MODEL.TWO_BRANCHED:
								json_logger.visualize_attn(img_occ, attns_map_occ, epoch, '-m', prefix='img',
								                           id_plus=img_occ.shape[0] * n_iter)
								json_logger.visualize_expert_attention(img, experts_attn_occ, epoch, '-em', prefix='img',
							                           id_plus=img.shape[0] * n_iter)

				else:  # baseline
					if epoch == 1:
						json_logger.save_images(img, 1, '-o')
					elif epoch % 10 == 0 and epoch > 60:
						if hook_attn_from_model:
							json_logger.visualize_attn(img, attns_map_ori, epoch, '-o', prefix='img')

			if (n_iter + 1) % log_period == 0 and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
				logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
				            .format(epoch, (n_iter + 1), len(train_loader),
				                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

		if cfg.MODEL.HEAD_SUP:
			# get most salient head id
			least_loss_head = torch.argsort(head_total_loss)[:3]
			if epoch > 40:
				head_mask[least_loss_head] = False

		# append epoch state to json logger

		if log_config.ENABLE and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
			json_logger.append_state({"Epoch": epoch,
			                          "Accuracy": float(acc_meter.avg),
			                          "Learning Rate": scheduler._get_lr(epoch)[0],
			                          "Loss_Total": loss_meter.avg,
			                          "Loss_ID": loss_id_meter.avg,
									  "Loss_moe":loss_moe_meter.avg,
			                          "Loss_Triplet": loss_tri_meter.avg}, new_epoch=True)
			if cfg.MODEL.ZZWEXP:
				if cfg.MODEL.IFRC:
					json_logger.append_state({"Loss_Inference": loss_ifrc_meter.avg})

			json_logger.dump_log()
		end_time = time.time()
		time_per_batch = (end_time - start_time) / (n_iter + 1)
		if cfg.MODEL.DIST_TRAIN:
			pass
		else:
			logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
			            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

		# save checkpoint

		if epoch % checkpoint_period == 0:
			if cfg.MODEL.DIST_TRAIN:
				if dist.get_rank() == 0:
					torch.save(model.state_dict(),
					           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
			else:
				torch.save(model.state_dict(),
				           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

		# evaluation
		# if epoch == 1: # for debug
		if epoch % eval_period == 0:
			print("下面是推理--------")
			if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
				model.eval()
				for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
					with torch.no_grad():
						attns_hook = []
						img = img.to(device)
						camids = camids.to(device)
						target_view = target_view.to(device)
						if cfg.MODEL.ZZWEXP:
							print("即将进入推理模型")
							feat, occ_pred, _ = model(img, cam_label=camids, view_label=target_view)
						else:
							feat = model(img, cam_label=camids, view_label=target_view)
						if hook_attn_from_model:
							attns_map = rollout(attns_hook[:12], discard_ratio=0.5, head_fusion='mean'
							                    )
						evaluator.update((feat, vid, camid))
						# save images
						if log_config.ENABLE and n_iter < iter_save_log:
							json_logger.save_images(img, epoch, '-o', prefix='query', id_plus=img.shape[0] * n_iter)
							if hook_attn_from_model:
								json_logger.visualize_attn(img, attns_map, epoch, '-o', prefix='query',
								                           id_plus=img.shape[0] * n_iter)

				cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
				logger.info("Validation Results - Epoch: {}".format(epoch))
				logger.info("mAP: {:.1%}".format(mAP))
				for r in [1, 5, 10]:
					logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
				torch.cuda.empty_cache()
			'''
			# TODO: combine reduplicative code
			if cfg.MODEL.DIST_TRAIN:
				if dist.get_rank() == 0:
					model.eval()
					for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
						with torch.no_grad():
							img = img.to(device)
							camids = camids.to(device)
							target_view = target_view.to(device)
							feat,_ = model(img, cam_label=camids, view_label=target_view)

							if cfg.MODEL.OCCDECODER and cfg.MODEL.USE_DECODER_FEAT == 'glb':
								evaluator.update((feat[:, :768], vid, camid))
							else:
								evaluator.update((feat, vid, camid))
							# evaluator.update((feat, vid, camid))
					cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
					logger.info("Validation Results - Epoch: {}".format(epoch))
					logger.info("mAP: {:.1%}".format(mAP))
					for r in [1, 5, 10]:
						logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
					torch.cuda.empty_cache()
			else:
				model.eval()
				for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
					with torch.no_grad():
						img = img.to(device)
						camids = camids.to(device)
						target_view = target_view.to(device)
						feat, _ = model(img, cam_label=camids, view_label=target_view)
						# TODO: check it
						if cfg.MODEL.OCCDECODER and cfg.MODEL.USE_DECODER_FEAT == 'glb':
							evaluator.update((feat[:, :768], vid, camid))
						else:
							evaluator.update((feat, vid, camid))
				cmc, mAP, _, _, _, _, _, _ = evaluator.compute()
				logger.info("Validation Results - Epoch: {}".format(epoch))
				logger.info("mAP: {:.1%}".format(mAP))
				for r in [1, 5, 10]:
					logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
				torch.cuda.empty_cache()'''
			if log_config.ENABLE and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
				max_rank1 = max(max_rank1, float(cmc[0]))
				if epoch == epochs:  # last epoch
					json_logger.append_state({"MAX_Rank-1": max_rank1})
				json_logger.append_state({"Rank-1": float(cmc[0]), "Rank-5": float(cmc[4]), "Rank-10": float(cmc[9]), \
				                          "mAP": float(mAP)}, dump=True)


def do_inference(cfg,
                 model,
                 val_loader,
                 val_dataset,
                 num_query):
    device = "cuda"
    # log_json = False
    attn_hook = True
    logger = logging.getLogger("transreid.train")
    if log_config.ENABLE and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
        json_logger = JsonLogger(cfg=cfg)
    logger.info("Enter inferencing")
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    attns = []

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    # 新增：用于收集MoE层贡献数据的变量
    layer_contributions = {}  # 存储各层的贡献数据
    num_batches_to_collect = 10  # 收集数据的批次数量
    # 新增：用于存储槽余弦相似性矩阵
    slot_cosine_sim_matrices = []    
    # append hook
    modules = {}

    def get_attn(module, input, output):
        attns.append(output.cpu())

    for name, module in model.named_modules():
        if attn_hook and 'attn_drop' in name and not 'occ_block' in name:
            module.register_forward_hook(get_attn)
            modules[name] = module
    # 新增：获取核心专家数量
    try:
        if isinstance(model, nn.DataParallel):
            core_experts = model.module.base.core_experts
        else:
            core_experts = model.base.core_experts
    except:
        core_experts = 32
        print(f"使用默认核心专家数量: {core_experts}")
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        attns = []
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, occ_pred, _, all_moe_data = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            
            # 新增：收集MoE层贡献数据
            if n_iter < num_batches_to_collect and all_moe_data is not None and isinstance(all_moe_data, list):
                for layer_idx, layer_data in enumerate(all_moe_data):
                    if not isinstance(layer_data, dict):
                        continue
                      
                    # 获取贡献数据
                    expert_contributions = layer_data.get('expert_contributions')
                    if expert_contributions is not None:
                        # 计算核心专家和通用专家的总贡献
                        core_contrib = expert_contributions[:core_experts].sum().item()
                        univ_contrib = expert_contributions[core_experts:].sum().item()
                        
                        # 存储贡献数据
                        if layer_idx not in layer_contributions:
                            layer_contributions[layer_idx] = []
                        layer_contributions[layer_idx].append((core_contrib, univ_contrib))
            
            if log_config.ENABLE and n_iter == 0:
                json_logger.save_images(img, 0, '-o', prefix='query')
                if attn_hook:
                    attn_mask = rollout(attns[:12], discard_ratio=0.5, head_fusion='mean')
                    json_logger.visualize_attn(img, attn_mask, 0, '-o', prefix='query')

              
                # 新增：可视化MoE专家注意力和贡献
                if all_moe_data is not None and isinstance(all_moe_data, list):
                    # 获取模型配置参数
                    try:
                        core_experts = model.base.core_experts
                    except:
                        core_experts = 32
                        print(f"使用默认核心专家数量: {core_experts}")
                    
                    print(f"找到 {len(all_moe_data)} 层MoE数据")
                    
                    # 遍历所有MoE层
                    for layer_idx, layer_data in enumerate(all_moe_data):
                        # 确保layer_data是字典
                        if not isinstance(layer_data, dict):
                            print(f"警告: 跳过无效的MoE数据类型: {type(layer_data)}")
                            continue                      
                        print(f"处理第 {layer_idx} 层MoE数据")

                        # 可视化专家注意力
                        expert_attentions = layer_data.get('expert_attentions')
                        if expert_attentions is not None:
                            # print(f"  专家注意力数据形状: {expert_attentions.shape}")
                            json_logger.visualize_expert_attention(
                                img, 
                                expert_attentions, 
                                0, 
                                f'-layer{layer_idx}', 
                                prefix='query_expert_attn'
                            )
                        else:
                            print("  警告: 此层无专家注意力数据")
                        
                        # 可视化专家贡献
                        expert_contributions = layer_data.get('expert_contributions')
                        if expert_contributions is not None:
                            print(f"  专家贡献数据形状: {expert_contributions.shape}")
                            json_logger.visualize_expert_contributions(
                                expert_contributions, 
                                0, 
                                f'-layer{layer_idx}',
                                core_experts=core_experts
                            )
                        else:
                            print("  警告: 此层无专家贡献数据")
                        # 新增：处理槽余弦相似性
                        dispatch_weights = layer_data.get('dispatch_weights')
                        print(f"  分发权重形状: {dispatch_weights.shape}")
                        if dispatch_weights is not None:
                            print(f"  分发权重形状: {dispatch_weights.shape}")
                            
                            # 计算槽余弦相似性
                            cosine_sim = json_logger.calculate_slot_cosine_similarity(dispatch_weights, num_slots=48)
                            if cosine_sim is not None:
                                slot_cosine_sim_matrices.append(cosine_sim)
                                print(f"  槽余弦相似性矩阵形状: {cosine_sim.shape}")


						
            if cfg.DATASETS.NAMES == 'occ_reid':
                imgpath_ = [p[:3] + "/" + p for p in imgpath]
                img_path_list.extend(imgpath_)
            else:
                img_path_list.extend(imgpath)

    # 新增：计算平均贡献并生成表格和图表
    if layer_contributions:
        # 计算每层的平均贡献
        avg_layer_contributions = {}
        for layer_idx, contribs in layer_contributions.items():
            core_avg = sum(c[0] for c in contribs) / len(contribs)
            univ_avg = sum(c[1] for c in contribs) / len(contribs)
            avg_layer_contributions[layer_idx] = (core_avg, univ_avg)
        
        # 打印贡献表格
        print("\n" + "="*80)
        print("Table: Contribution values of core and universal experts across network layers")
        print("="*80)
        print("MoE Layer\tCore Expert Contribution\tUniversal Expert Contribution")
        print("-"*80)
        
        # 假设MoE层从第7层开始（根据你的配置调整）
        moe_layers = [7, 8, 9, 10, 11, 12]  # 根据你的实际配置调整
        
        for i, layer_num in enumerate(moe_layers):
            if i in avg_layer_contributions:
                core_contrib, univ_contrib = avg_layer_contributions[i]
                print(f"{layer_num}\t\t{core_contrib:.2f}\t\t\t\t{univ_contrib:.2f}")
        
        print("="*80)
        
        # 生成贡献趋势图
        json_logger.plot_contribution_trend(avg_layer_contributions, moe_layers)
    # 新增：绘制槽余弦相似性热力图
    if slot_cosine_sim_matrices:
        layer_names = [7, 8, 9, 10, 11, 12]  # MoE层从第7层开始
        
        # 确保层数和矩阵数匹配
        if len(slot_cosine_sim_matrices) == len(layer_names):
            json_logger.plot_slot_cosine_similarity(slot_cosine_sim_matrices, layer_names, 0)
            json_logger.analyze_slot_similarity(slot_cosine_sim_matrices, layer_names)
        else:
            print(f"警告: 层数不匹配。期望 {len(layer_names)} 层，实际 {len(slot_cosine_sim_matrices)} 层")
    cmc, mAP, _, _, _, _, _, idx5 = evaluator.compute()
    evaluator.save_npy("no_inf")
    # evaluator.draw_tsne_fig(8, 20, "./figs/tsne.jpg")
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # if cfg.DATASETS.NAMES == 'occ_duke':
    #     ds = DataStat(set="query")
    #     ds.summarize(idx5, img_path_list)

    assert idx5.shape[0] == num_query, "test error"
    if log_config.ENABLE:
        json_logger.append_state({"Rank-1": float(cmc[0]), "Rank-5": float(cmc[4]), "Rank-10": float(cmc[9]), \
                                  "mAP": float(mAP)}, dump=True, new_epoch=True)
        json_logger.visualize_matches(idx5, val_dataset, img_path_list)
    return cmc[0], cmc[4]