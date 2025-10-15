# -*- coding=utf-8 -*-
import os
import json
import datetime
import numpy as np
import cv2
import wandb
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
JSON_LOG_DIR = "../logs/"
if not os.path.exists(JSON_LOG_DIR):
	os.mkdir(JSON_LOG_DIR)
import torch
import torch.nn.functional as F
class JsonLogger:
	"""
		json format:
		{
			"run_name": str
			"config": dict
			"states": [dict]
		}
	"""
	def __init__(self, cfg=None, log_path=None, test=False):
		if cfg is not None:
			self.transform = T.Compose([
				T.Normalize(mean=[-1*cfg.INPUT.PIXEL_MEAN[i]/cfg.INPUT.PIXEL_STD[i] for i in range(3)], std=[1/x for x in cfg.INPUT.PIXEL_STD]),
				T.ToPILImage()
			])
			self.log_dir = cfg.OUTPUT_DIR
			if not os.path.exists(self.log_dir):
				os.mkdir(self.log_dir)
		else:
			self.log_dir = JSON_LOG_DIR
		if log_path is not None:
			self.run_name = log_path.split('/')[-2]
			self.json_path = JSON_LOG_DIR + log_path
			self.load_log()
		else:
			assert cfg is not None
			ls = os.listdir(self.log_dir)
			if not test:
				run_ls = [d for d in ls if d.startswith("run_")]
				self.run_name = "run_{:0>3d}_".format(len(run_ls)+1)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
				os.mkdir(os.path.join(self.log_dir, self.run_name))
				self.json_log = dict()
				self.json_log["run_name"] = self.run_name
				self.json_log["config"] = cfg
				self.json_log["states"] = []
				self.json_path = os.path.join(self.log_dir, self.run_name, "log.json")
				self.dump_log()
			else:
				test_ls = [d for d in ls if d.startswith("test_")]
				self.run_name = "test_{:0>3d}_".format(len(test_ls)+1)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
				os.mkdir(os.path.join(self.log_dir, self.run_name))
				self.json_log = dict()
				self.json_log["run_name"] = self.run_name
				self.json_log["config"] = cfg
				self.json_log["states"] = []
				self.json_path = os.path.join(self.log_dir, self.run_name, "log.json")
				self.dump_log()


	def load_log(self):
		with open(self.json_path, "r") as f:
			self.json_log = json.load(f)

	def dump_log(self):
		with open(self.json_path, "w") as f:
			json.dump(self.json_log, f)

	def append_state(self, state_dict, new_epoch=False, dump=False):
		if new_epoch:
			self.json_log["states"].append(state_dict)
		else:
			self.json_log["states"][-1].update(state_dict)
		if dump:
			self.dump_log()

	def save_images(self, imgs, epoch, suffix, prefix='img', id_plus=0):
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, "imgs")):
			os.mkdir(os.path.join(self.log_dir, self.run_name, "imgs"))
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, 'imgs/epoch_{:0>3d}/'.format(epoch))):
			os.mkdir(os.path.join(self.log_dir, self.run_name, 'imgs/epoch_{:0>3d}/'.format(epoch)))
		for i, img in enumerate(imgs):
			path = os.path.join(self.log_dir, self.run_name,
			                    'imgs/epoch_{:0>3d}/'.format(epoch)+prefix+'{:0>5d}{}.jpg'.format(i+id_plus, suffix))
			img = self.transform(img)
			img.save(path)

	def visualize_attn(self, imgs, attn, epoch, suffix, prefix='img', id_plus=0):
		"""

		Args:
			imgs: [batch_size, C, H, W]
			attn: [batch_size, patches]
			epoch: str

		Returns:

		"""
		attn_size = (16, 8)
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, "attns")):
			os.mkdir(os.path.join(self.log_dir, self.run_name, "attns"))
		if not os.path.exists(os.path.join(self.log_dir, self.run_name, 'attns/epoch_{:0>3d}/'.format(epoch))):
			os.mkdir(os.path.join(self.log_dir, self.run_name, 'attns/epoch_{:0>3d}/'.format(epoch)))
		####### 原来的 #######
		for i, img in enumerate(imgs):
			name = os.path.join(self.log_dir, self.run_name,
			                    'attns/epoch_{:0>3d}/{}{:0>5d}{}.jpg'.format(epoch, prefix, i+id_plus, suffix))
			img = self.transform(img)
			np_img = np.array(img)[:, :, ::-1]
			# 1. 重塑注意力图为指定尺寸
			mask = attn[i].reshape(attn_size).numpy()
			max_mask = np.max(mask)
			mask = mask / max_mask
			# mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), interpolation=cv2.INTER_NEAREST)
			mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), interpolation=cv2.INTER_LINEAR)
			img = np.float32(img) / 255
			heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
			heatmap = np.float32(heatmap) / 255
			cam = heatmap + np.float32(img)
			cam = cam / np.max(cam)
			masked_img = np.uint8(255 * cam)
			cv2.imwrite(name, masked_img)
		####### 原来的 #######

		# for i, img in enumerate(imgs):
		# 	name = os.path.join(self.log_dir, self.run_name,
		# 						f'attns/epoch_{epoch:03d}/{prefix}{i+id_plus:05d}{suffix}.jpg')
		# 	img = self.transform(img)
		# 	np_img = np.array(img)[:, :, ::-1].copy()  # 添加.copy()避免原始数据被修改
			
		# 	# 1. 重塑注意力图
		# 	mask = attn[i].reshape(attn_size).numpy()
			
		# 	# 修复1：优化归一化处理 - 使用对比度增强的归一化
		# 	mask = mask - np.min(mask)  # 确保最小值为0
		# 	if np.max(mask) > 0:
		# 		# 使用非线性增强突出高响应区域
		# 		enhanced_mask = np.power(mask, 0.5)  # 伽马校正增强对比度
		# 		mask = enhanced_mask / np.max(enhanced_mask)
			
		# 	# 2. 插值处理
		# 	mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), 
		# 					interpolation=cv2.INTER_CUBIC)  # 改用立方插值更清晰
			
		# 	# 修复2：调整高斯模糊 - 仅轻微平滑
		# 	mask = cv2.GaussianBlur(mask, (5, 5), 0)  # 减小核大小 (原为15)
			
		# 	# 3. 准备图像
		# 	img_float = np.float32(np_img) / 255
			
		# 	# 修复3：使用更合适的颜色映射
		# 	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)  # HOT色图更突出红色
			
		# 	# 4. 融合处理
		# 	heatmap = np.float32(heatmap) / 255
		# 	alpha = 0.6  # 提高热力图可见度
		# 	cam = cv2.addWeighted(img_float, 1-alpha, heatmap, alpha, 0)
			
		# 	# 修复4：添加颜色条参考
		# 	# 创建颜色条参考
		# 	colorbar = np.zeros((np_img.shape[0], 20, 3), dtype=np.float32)
		# 	for j in range(np_img.shape[0]):
		# 		colorbar[j, :, :] = j / np_img.shape[0]
		# 	colorbar = cv2.applyColorMap(np.uint8(255 * colorbar), cv2.COLORMAP_HOT)
		# 	cam = np.hstack((cam, np.float32(colorbar)/255))
			
		# 	# 5. 保存结果
		# 	masked_img = np.uint8(255 * cam)
		# 	cv2.imwrite(name, masked_img)

		# ####### 新的 #######
		# for i, img in enumerate(imgs):
		# 	name = os.path.join(self.log_dir, self.run_name,
		# 						f'attns/epoch_{epoch:03d}/{prefix}{i+id_plus:05d}{suffix}.jpg')
		# 	img = self.transform(img)
		# 	np_img = np.array(img)[:, :, ::-1].copy()  # 添加.copy()避免原始数据被修改
			
		# 	# 1. 重塑注意力图
		# 	mask = attn[i].reshape(attn_size).numpy()
			
		# 	# 2. 优化归一化处理 - 使用百分位数避免极端值
		# 	min_val = np.percentile(mask, 5)   # 取5%分位数
		# 	max_val = np.percentile(mask, 95)  # 取95%分位数
		# 	mask = (mask - min_val) / (max_val - min_val + 1e-8)
		# 	mask = np.clip(mask, 0, 1)  # 限制在0-1范围内
			
		# 	# 3. 插值处理 (保留平滑效果)
		# 	mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), 
		# 					interpolation=cv2.INTER_CUBIC)
			
		# 	# 4. 轻微平滑 (保留平滑效果)
		# 	mask = cv2.GaussianBlur(mask, (5, 5), 0)
			
		# 	# 5. 准备图像
		# 	img_float = np.float32(np_img) / 255
			
		# 	# 6. 使用JET色图 (原始论文的颜色)
		# 	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
		# 	heatmap = np.float32(heatmap) / 255
			
		# 	# 7. 优化融合方式 (结合两者的优点)
		# 	# 原始论文使用直接相加：cam = heatmap + img_float
		# 	# 新方法使用加权融合：cam = alpha*heatmap + (1-alpha)*img_float
		# 	# 这里使用折中方案：增强热力图但不完全覆盖原图
		# 	enhanced_heatmap = 0.7 * heatmap + 0.3 * img_float
		# 	cam = 0.6 * enhanced_heatmap + 0.4 * img_float
			
		# 	# 8. 可选：添加颜色条参考
		# 	if True:  # 可以根据需要关闭
		# 		colorbar = np.zeros((np_img.shape[0], 20, 3), dtype=np.float32)
		# 		for j in range(np_img.shape[0]):
		# 			colorbar[j, :, :] = j / np_img.shape[0]
		# 		colorbar = cv2.applyColorMap(np.uint8(255 * colorbar), cv2.COLORMAP_JET)
		# 		cam = np.hstack((cam, np.float32(colorbar)/255))
			
		# 	# 9. 保存结果
		# 	masked_img = np.uint8(255 * cam)
		# 	cv2.imwrite(name, masked_img)	
		# ####### 新的 #######
		'''
		# print(attn.min(), attn.max())
		# attn = (attn - attn.min()) / (attn.max() - attn.min())
		for b in range(attn.shape[1]):
			blk_root = JSON_LOG_DIR+self.run_name+'/attns/epoch_{:0>3d}/block_{:0>2d}/'.format(epoch, b)
			if not os.path.exists(blk_root):
				os.mkdir(blk_root)
			for i in range(attn.shape[0]):
				for h in range(attn.shape[2]):
					# print(attn[i][b][h].min(), attn[i][b][h].max())
					attn[i][b][h] = (attn[i][b][h] - attn[i][b][h].min()) / (attn[i][b][h].max() - attn[i][b][h].min())
					img = attn[i][b][h].reshape(attn_size).cpu().numpy() * 255
					img = img.astype(np.uint8)
					img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
					cv2.imwrite(blk_root + prefix + "{:0>3d}_h{:0>2d}{}.jpg".format(i, h, suffix), img)'''
	# def visualize_expert_attention(self, imgs, attn_weights, epoch, suffix, 
	# 							prefix='expert', id_plus=0, expert_idx=0, 
	# 							head_idx=0, slot_idx=0):
	# 	"""
	# 	可视化SoftMoE专家分发权重
		
	# 	Args:
	# 		imgs: [batch_size, C, H, W] 原始图像
	# 		attn_weights: 包含dispatch_weights和combine_weights的元组
	# 		epoch: 当前epoch
	# 		expert_idx: 要可视化的专家索引
	# 		head_idx: 要可视化的注意力头索引
	# 		slot_idx: 要可视化的槽索引
	# 	"""
	# 	dispatch_weights= attn_weights
	# 	attn_size = (16, 8)  # 假设16x8的patch网格
		
	# 	# 创建保存目录
	# 	save_dir = os.path.join(self.log_dir, self.run_name, f"experts/epoch_{epoch:03d}/")
	# 	os.makedirs(save_dir, exist_ok=True)
		
	# 	for i, img in enumerate(imgs):
	# 		# 文件名包含专家、头、槽信息
	# 		name = os.path.join(save_dir, 
	# 						f'{prefix}{i+id_plus:05d}_e{expert_idx}_h{head_idx}_s{slot_idx}{suffix}.jpg')
			
	# 		img = self.transform(img)
	# 		np_img = np.array(img)[:, :, ::-1].copy()
			
	# 		# 1. 提取当前专家的分发权重
	# 		# dispatch_weights形状: [b, h, n, e, s]
	# 		expert_attn = dispatch_weights[i, head_idx, :, expert_idx, slot_idx].detach().cpu().numpy()
			
	# 		# 2. 移除CLS token的权重 (索引0)
	# 		patch_attn = expert_attn[1:]  # 只取图像patch部分
			
	# 		# 3. 重塑为网格形状
	# 		mask = patch_attn.reshape(attn_size)
			
	# 		# 4. 归一化处理
	# 		mask = mask - np.min(mask)
	# 		if np.max(mask) > 0:
	# 			# 使用非线性增强突出高响应区域
	# 			enhanced_mask = np.power(mask, 0.5)  # 伽马校正增强对比度
	# 			mask = enhanced_mask / np.max(enhanced_mask)
			
	# 		# 5. 插值到原始图像尺寸
	# 		mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]), 
	# 						interpolation=cv2.INTER_CUBIC)
			
	# 		# 6. 轻微平滑
	# 		mask = cv2.GaussianBlur(mask, (5, 5), 0)
			
	# 		# 7. 准备图像
	# 		img_float = np.float32(np_img) / 255
			
	# 		# 8. 生成热力图 - 使用JET色图(蓝-绿-黄-红)
	# 		heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	# 		heatmap = np.float32(heatmap) / 255
			
	# 		# 9. 融合图像和热力图
	# 		alpha = 0.6  # 热力图透明度
	# 		cam = cv2.addWeighted(img_float, 1-alpha, heatmap, alpha, 0)
			
	# 		# 10. 添加信息文本
	# 		font = cv2.FONT_HERSHEY_SIMPLEX
	# 		text = f"Expert {expert_idx} | Head {head_idx} | Slot {slot_idx}"
	# 		cam = cv2.putText(cam, text, (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
			
	# 		# 11. 保存结果
	# 		cv2.imwrite(name, np.uint8(255 * cam))

	def aggregate_expert_attention(self, dispatch_weights, combine_weights):
		"""
		聚合每个专家的注意力分布
		"""
		# 确保张量维度一致
		if self.multi_head:
			# 处理多头情况
			# 聚合分发权重：平均所有槽的权重
			expert_dispatch = dispatch_weights.mean(dim=-1)  # [B, H, N, E]
			expert_dispatch = expert_dispatch.permute(0, 2, 1, 3)  # [B, N, H, E]
			
			# 聚合合并权重：按专家分组
			# [B, H, N, E*S] -> [B, H, N, E, S]
			expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
			expert_combine = expert_combine.sum(dim=-1)  # [B, H, N, E]
			expert_combine = expert_combine.permute(0, 2, 1, 3)  # [B, N, H, E]
			
			# 组合两种权重
			expert_attentions = (expert_dispatch + expert_combine) / 2
			expert_attentions = expert_attentions.mean(dim=2)  # [B, N, E]
		else:
			# 处理单头情况
			# 聚合分发权重：平均所有槽的权重
			expert_dispatch = dispatch_weights.mean(dim=-1)  # [B, N, E]
			
			# 聚合合并权重：按专家分组
			# [B, N, E*S] -> [B, N, E, S]
			expert_combine = combine_weights.unflatten(-1, (self.num_experts, self.slots_per_expert))
			expert_combine = expert_combine.sum(dim=-1)  # [B, N, E]
			
			# 组合两种权重
			expert_attentions = (expert_dispatch + expert_combine) / 2
		
		return expert_attentions.permute(0, 2, 1)  # [B, E, N]
	def visualize_expert_attention(self, imgs, expert_attentions, epoch, suffix, prefix='expert', id_plus=0):
		"""
		可视化不同专家的注意力分布
		imgs: 原始图像 [batch, C, H, W]
		expert_attentions: [batch, num_experts, num_tokens]
		"""
		B, C, H, W = imgs.shape
		patch_size = 16
		grid_h, grid_w = H // patch_size, W // patch_size
		
		# 只可视化前10个样本
		for i in range(min(B, 10)):
			img = imgs[i].permute(1, 2, 0).cpu().numpy()
			img = (img - img.min()) / (img.max() - img.min())  # 归一化
			img = np.uint8(255 * img)
			# 确保张量维度正确
			attentions = expert_attentions[i]  # [num_experts, num_tokens]
			if len(attentions.shape) == 3:  # 如果有多余的维度
				attentions = attentions.squeeze(0)
			
			num_experts = attentions.shape[0]
			
			# 创建专家注意力图网格
			cols = min(8, num_experts)
			rows = (num_experts + cols - 1) // cols
			fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
			
			for expert_idx in range(num_experts):
				if rows > 1:
					ax = axes[expert_idx//cols, expert_idx%cols]
				else:
					ax = axes[expert_idx] if cols > 1 else axes
				
				# 获取当前专家的注意力
				# 跳过第一个token (CLS token)
				attn = attentions[expert_idx][1:].reshape(grid_h, grid_w).cpu().numpy()
				
				# 创建热力图
				heatmap = cv2.resize(attn, (W, H))
				heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
				# 应用阈值：将小于0.2的值设为0
				heatmap[heatmap < 0.1] = 0
		
				heatmap = np.uint8(255 * heatmap)
				heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
				
				# 覆盖到原始图像
				overlayed = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
				ax.imshow(overlayed)
				ax.set_title(f'Expert {expert_idx}', fontsize=10)
				ax.axis('off')
			
			# 隐藏多余的子图
			for j in range(num_experts, rows*cols):
				if rows > 1:
					axes[j//cols, j%cols].axis('off')
				else:
					axes[j].axis('off')
			
			plt.tight_layout()
			save_dir = os.path.join(self.log_dir, self.run_name, f"attns/epoch_{epoch:03d}")
			os.makedirs(save_dir, exist_ok=True)
			save_path = os.path.join(save_dir, f'{prefix}{i+id_plus}{suffix}.jpg')
			plt.savefig(save_path, bbox_inches='tight', dpi=150)
			plt.close()  
	def plot_slot_cosine_similarity(self, cosine_sim_matrices, layer_names, epoch=0):
		"""
		绘制槽余弦相似性热力图网格
		cosine_sim_matrices: 列表，包含各层的余弦相似性矩阵
		layer_names: 列表，包含各层的名称
		epoch: 训练周期（用于文件名）
		"""
		try:
			num_layers = len(cosine_sim_matrices)
			
			# 创建子图
			fig, axes = plt.subplots(2, 3, figsize=(18, 12))
			axes = axes.flatten()
			
			# 设置全局颜色映射
			vmin = min([matrix.min().item() for matrix in cosine_sim_matrices])
			vmax = max([matrix.max().item() for matrix in cosine_sim_matrices])
			
			for i, (cosine_sim, layer_name) in enumerate(zip(cosine_sim_matrices, layer_names)):
				ax = axes[i]
				
				# 转换为numpy数组
				if isinstance(cosine_sim, torch.Tensor):
					cosine_sim = cosine_sim.cpu().numpy()
				
				# 绘制热力图
				im = ax.imshow(cosine_sim, cmap='viridis', vmin=vmin, vmax=vmax)
				
				# 设置标题和标签
				ax.set_title(f'Layer {layer_name}', fontsize=14)
				ax.set_xlabel('Slot Index', fontsize=12)
				ax.set_ylabel('Slot Index', fontsize=12)
				
				# 添加颜色条
				cbar = plt.colorbar(im, ax=ax)
				cbar.set_label('Cosine Similarity', fontsize=12)
				
				# 添加对角线标注
				for j in range(cosine_sim.shape[0]):
					ax.text(j, j, f'{cosine_sim[j, j]:.2f}', 
							ha='center', va='center', fontsize=8,
							bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
			
			# 调整布局
			plt.tight_layout()
			
			# 保存图像
			save_dir = os.path.join(self.log_dir, self.run_name, f"slot_similarity/epoch_{epoch:03d}")
			os.makedirs(save_dir, exist_ok=True)
			save_path = os.path.join(save_dir, 'slot_cosine_similarity_grid.png')
			plt.savefig(save_path, bbox_inches='tight', dpi=300)
			plt.close()
			
			print(f"槽余弦相似性热力图已保存至: {save_path}")
			
		except Exception as e:
			print(f"绘制槽余弦相似性热力图时出错: {str(e)}")
			import traceback
			traceback.print_exc()

	def analyze_slot_similarity(self, cosine_sim_matrices, layer_names):
		"""
		分析槽相似性模式
		"""
		print("\n" + "="*60)
		print("槽相似性分析")
		print("="*60)
		
		for i, (cosine_sim, layer_name) in enumerate(zip(cosine_sim_matrices, layer_names)):
			if isinstance(cosine_sim, torch.Tensor):
				cosine_sim = cosine_sim.cpu().numpy()
			
			# 计算对角线平均值（自相似性）
			diag_mean = np.diag(cosine_sim).mean()
			
			# 计算非对角线平均值（槽间相似性）
			mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
			off_diag_mean = cosine_sim[mask].mean()
			
			# 计算相似性差异
			similarity_diff = diag_mean - off_diag_mean
			
			print(f"层 {layer_name}:")
			print(f"  自相似性平均值: {diag_mean:.4f}")
			print(f"  槽间相似性平均值: {off_diag_mean:.4f}")
			print(f"  相似性差异: {similarity_diff:.4f}")
			
			# 检查是否接近理想模式（对角线为1，其他接近0）
			if diag_mean > 0.9 and off_diag_mean < 0.2:
				print("  ✅ 槽模式良好: 高自相似性，低槽间相似性")
			else:
				print("  ⚠️ 槽模式不理想: 自相似性不足或槽间相似性过高")
			
			print()
				
	def plot_contribution_trend(self, avg_contributions, moe_layers):
		"""
		绘制专家贡献趋势图
		avg_contributions: 字典，键为层索引，值为(core_contrib, univ_contrib)元组
		moe_layers: MoE层对应的实际层号列表
		"""
		try:
			core_values = []
			univ_values = []
			layers = []
			
			for i, layer_num in enumerate(moe_layers):
				if i in avg_contributions:
					core_contrib, univ_contrib = avg_contributions[i]
					core_values.append(core_contrib)
					univ_values.append(univ_contrib)
					layers.append(layer_num)
			
			plt.figure(figsize=(10, 6))
			plt.plot(layers, core_values, 'o-', label='Core Experts', linewidth=2, markersize=8)
			plt.plot(layers, univ_values, 's-', label='Universal Experts', linewidth=2, markersize=8)
			
			plt.xlabel('Network Layer', fontsize=12)
			plt.ylabel('Contribution Value', fontsize=12)
			plt.title('Contribution of Core and Universal Experts across Network Layers', fontsize=14)
			plt.legend(fontsize=12)
			plt.grid(True, linestyle='--', alpha=0.7)
			
			# 添加数值标签
			for i, (core, univ) in enumerate(zip(core_values, univ_values)):
				plt.annotate(f'{core:.2f}', (layers[i], core), textcoords="offset points", 
							xytext=(0,10), ha='center', fontsize=9)
				plt.annotate(f'{univ:.2f}', (layers[i], univ), textcoords="offset points", 
							xytext=(0,-15), ha='center', fontsize=9)
			
			plt.tight_layout()
			
			# 保存图像
			save_dir = os.path.join(self.log_dir, self.run_name, "contribution_analysis")
			os.makedirs(save_dir, exist_ok=True)
			save_path = os.path.join(save_dir, 'expert_contributions_across_layers.png')
			plt.savefig(save_path, bbox_inches='tight', dpi=300)
			plt.close()
			
			print(f"专家贡献趋势图已保存至: {save_path}")
			
		except Exception as e:
			print(f"绘制贡献趋势图时出错: {str(e)}")	

	def visualize_expert_contributions(self, expert_contributions, epoch, suffix, core_experts=32):
		"""
		可视化专家贡献分布
		expert_contributions: [num_experts] 每个专家的平均贡献
		core_experts: 核心专家的数量
		"""
		try:
			# 确保输入是有效的
			if expert_contributions is None:
				print("警告: expert_contributions 为 None，跳过可视化")
				return
				
			# 转换为NumPy数组
			if isinstance(expert_contributions, torch.Tensor):
				expert_contributions = expert_contributions.detach().cpu().numpy()
			elif not isinstance(expert_contributions, np.ndarray):
				print(f"警告: 不支持的数据类型 {type(expert_contributions)}，跳过可视化")
				return
			
			# 确保是一维数组
			if expert_contributions.ndim > 1:
				print(f"警告: 专家贡献数据维度为 {expert_contributions.shape}，将取平均值")
				expert_contributions = expert_contributions.mean(axis=0)
			
			num_experts = len(expert_contributions)
			
			# 确保核心专家数量有效
			if core_experts > num_experts:
				print(f"警告: core_experts({core_experts}) > num_experts({num_experts})，将使用默认值")
				core_experts = min(32, num_experts)
			
			# 创建索引
			indices = np.arange(num_experts)
			
			# 分离核心专家和通用专家
			core_contrib = expert_contributions[:core_experts]
			occ_contrib = expert_contributions[core_experts:]
			
			# 确保贡献值非负
			core_contrib = np.maximum(core_contrib, 0)
			occ_contrib = np.maximum(occ_contrib, 0)
			
			plt.figure(figsize=(12, 6))
			
			# 绘制柱状图
			plt.bar(indices[:core_experts], core_contrib, color='dodgerblue', alpha=0.8, label='Core Experts')
			plt.bar(indices[core_experts:], occ_contrib, color='darkorange', alpha=0.8, label='Occ Experts')
			
			# 添加分隔线和标注
			if core_experts < num_experts:
				plt.axvline(x=core_experts-0.5, color='red', linestyle='--', linewidth=1.5)
				plt.text(core_experts//2, np.max(expert_contributions)*0.85, 'Core Experts', 
						ha='center', fontsize=12, fontweight='bold')
				plt.text(core_experts + len(occ_contrib)//2, np.max(expert_contributions)*0.85, 'Occ Experts', 
						ha='center', fontsize=12, fontweight='bold')
			
			# 设置标题和标签
			plt.title('Expert Contribution Distribution', fontsize=14, fontweight='bold')
			plt.xlabel('Expert Index', fontsize=12)
			plt.ylabel('Contribution Ratio', fontsize=12)
			plt.xticks(fontsize=10)
			plt.yticks(fontsize=10)
			
			# 添加网格和图例
			plt.grid(axis='y', linestyle='--', alpha=0.5)
			plt.legend(fontsize=12)
			
			# 调整布局
			plt.tight_layout()
			
			# 保存图像
			save_dir = os.path.join(self.log_dir, self.run_name, f"attns/epoch_{epoch:03d}")
			os.makedirs(save_dir, exist_ok=True)
			save_path = os.path.join(save_dir, f'expert_contributions{suffix}.png')
			plt.savefig(save_path, bbox_inches='tight', dpi=150)
			plt.close()
			
			print(f"成功保存专家贡献图: {save_path}")
			
		except Exception as e:
			print(f"可视化专家贡献时出错: {str(e)}")
			print(f"专家贡献数据: {expert_contributions}")
			if hasattr(expert_contributions, 'shape'):
				print(f"数据形状: {expert_contributions.shape}")
	def calculate_slot_cosine_similarity(self, dispatch_weights, num_slots=48):
		"""
		计算槽之间的余弦相似性矩阵
		dispatch_weights: [batch_size, num_heads, num_tokens, num_slots]
		num_slots: 要分析的槽数量
		"""
		try:
			# 合并多头维度：对头维度取平均
			# [batch_size, num_heads, num_tokens, num_slots] -> [batch_size, num_tokens, num_slots]
			avg_dispatch = dispatch_weights.mean(dim=1)
			
			# 取前num_slots个槽
			# [batch_size, num_tokens, num_slots] -> [batch_size, num_tokens, num_slots]
			selected_slots = avg_dispatch[:, :, :num_slots]
			
			# 重塑为向量形式: [batch_size * num_tokens, num_slots]
			slot_vectors = selected_slots.reshape(-1, num_slots)
			
			# 计算余弦相似性矩阵
			# 使用PyTorch的余弦相似度函数
			slot_vectors_norm = F.normalize(slot_vectors, p=2, dim=1)
			cosine_sim = torch.mm(slot_vectors_norm, slot_vectors_norm.t())
			
			# 只取槽与槽之间的相似性（不是所有token之间的相似性）
			# 我们需要的是槽表示之间的相似性，而不是token之间的相似性
			# 上面的方法计算的是所有token表示之间的相似性，这是不正确的
			
			# 正确的方法：计算每个槽的平均表示，然后计算槽之间的相似性
			# 首先计算每个槽的平均表示
			slot_means = slot_vectors.mean(dim=0)  # [num_slots]
			
			# 归一化槽平均表示
			slot_means_norm = F.normalize(slot_means.unsqueeze(0), p=2, dim=1)  # [1, num_slots]
			
			# 计算槽之间的余弦相似性
			cosine_sim = torch.mm(slot_means_norm.t(), slot_means_norm)  # [num_slots, num_slots]
			
			return cosine_sim
			
		except Exception as e:
			print(f"计算槽余弦相似性时出错: {str(e)}")
			import traceback
			traceback.print_exc()
			return None
	def visualize_matches(self, idx5, val_dataset, path_list, id_annotation=False):
		visualize_root = os.path.join(self.log_dir, self.run_name, 'match_results')
		print(f"输出路径：{visualize_root}")
		if not os.path.exists(visualize_root):
			os.mkdir(visualize_root)
		# query_root = "../../datasets/Occluded_Duke/query/"
		# gallery_root = "../../datasets/Occluded_Duke/bounding_box_test/"
		# query_root = "../../datasets/Occluded_ReID/occluded_body_images/"
		# gallery_root = "../../datasets/Occluded_ReID/whole_body_images/"
		num_query = idx5.shape[0]
		for q_idx in range(num_query):
			if not os.path.exists(path_list[q_idx]):
				print(path_list[q_idx], "doesn't exist!")
			q_id = val_dataset[q_idx][1]
			q_img = cv2.resize(cv2.imread(path_list[q_idx]), (128, 256))
			g_ids = [val_dataset[num_query+g_idx][1] for g_idx in idx5[q_idx]]
			g_imgs = [cv2.resize(cv2.imread(path_list[num_query+g_idx]), (128, 256)) for g_idx in idx5[q_idx]]
			if id_annotation:
				output = np.ones((300, 808, 3)) * 255
			else:
				output = np.ones((256, 808, 3)) * 255
			if id_annotation:
				cv2.putText(output, "{:0>4d}".format(q_id), (20, 295), 0, 1, (255, 255, 255))
			cv2.rectangle(q_img, (0, 0), (127, 255), (127, 127, 127), 2)
			for m_idx in range(5):
				if id_annotation:
					cv2.putText(output, "{:0>4d}".format(g_ids[m_idx]), (128*(m_idx+1)+20, 295), 0, 1, (255, 255, 255))
				if g_ids[m_idx] == q_id:
					cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 255, 0), 4)
				else:
					cv2.rectangle(g_imgs[m_idx], (0, 0), (127, 255), (0, 0, 255), 4)

			output[:256, :128] = q_img
			output[:256, 168:] = np.concatenate(g_imgs, axis=1)
			cv2.imwrite(visualize_root + '{:0>5d}.jpg'.format(q_idx+1), output)

	def wandb_sync(self):
		wandb.init(project="zzw_reid", entity="mega_z", name=self.run_name)
		wandb.config.dataset = self.json_log["config"]["DATASETS"]["NAMES"]
		wandb.config.base_lr = self.json_log["config"]["SOLVER"]["BASE_LR"]
		wandb.config.batch_size = self.json_log["config"]["SOLVER"]["IMS_PER_BATCH"]
		if "mae" in self.json_log["config"]["MODEL"]["PRETRAIN_PATH"]:
			wandb.config.pretrain = "MAE"
		wandb.config.id_loss_weight = self.json_log["config"]["MODEL"]["ID_LOSS_WEIGHT"]
		wandb.config.triplet_loss_weight = self.json_log["config"]["MODEL"]["TRIPLET_LOSS_WEIGHT"]
		if self.json_log["config"]["MODEL"]["ZZWTRY"] or self.json_log["config"]["MODEL"]["ZZWEXP"]:
			wandb.config.branch_blocks = self.json_log["config"]["MODEL"]["BRANCH_BLOCKS"]
			wandb.config.occ_decoder = self.json_log["config"]["MODEL"]["OCCDECODER"]

			wandb.config.occlude_type = self.json_log["config"]["MODEL"]["OCC_TYPE"]
			if self.json_log["config"]["MODEL"]["OCC_TYPE"] == 'img_block':
				wandb.config.occlude_ratio = self.json_log["config"]["MODEL"]["OCC_RATIO"]
				# wandb.config.align_bottom_occlude = self.json_log["config"]["MODEL"]["OCC_ALIGN_BTM"]
				wandb.config.align_bound = self.json_log["config"]["MODEL"]["OCC_ALIGN_BOUND"]
				wandb.config.occlude_ulrd = self.json_log["config"]["MODEL"]["OCC_ULRD"]
				wandb.config.patch_align_occlude = self.json_log["config"]["MODEL"]["PATCH_ALIGN_OCC"]

			if self.json_log["config"]["MODEL"]["OCCDECODER"]:
				wandb.config.use_decoder_feat = self.json_log["config"]["MODEL"]["USE_DECODER_FEAT"]

			wandb.config.occlusion_aware = self.json_log["config"]["MODEL"]["OCC_AWARE"]
			if self.json_log["config"]["MODEL"]["OCC_AWARE"]:
				wandb.config.fix_alpha = self.json_log["config"]["MODEL"]["FIX_ALPHA"]
				wandb.config.occlusion_loss_weight = self.json_log["config"]["MODEL"]["OCC_LOSS_WEIGHT"]

			wandb.config.inference = self.json_log["config"]["MODEL"]["IFRC"]
			if self.json_log["config"]["MODEL"]["IFRC"]:
				wandb.config.inference_loss_weight = self.json_log["config"]["MODEL"]["IFRC_LOSS_WEIGHT"]
				wandb.config.inference_loss_type = self.json_log["config"]["MODEL"]["IFRC_LOSS_TYPE"]
				wandb.config.pretext = self.json_log["config"]["MODEL"]["PRETEXT"]
				if self.json_log["config"]["MODEL"]["PRETEXT"] == 'rgb_avg':
					wandb.config.pretext_rgb_pix = self.json_log["config"]["MODEL"]["PRETEXT_RGB_PIX"]

		for step in self.json_log["states"]:
			step.pop("Epoch")
			wandb.log(step)



if __name__ == "__main__":
	json_logger = JsonLogger(log_path="occ_duke/run_036_fix_0.9/log.json")
	json_logger.wandb_sync()
