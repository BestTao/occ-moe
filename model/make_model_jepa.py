import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
	deit_small_patch16_224_TransReID, OccDecoder
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import numpy as np

def shuffle_unit(features, shift, group, begin=1):
	batchsize = features.size(0)
	dim = features.size(-1)
	# Shift Operation
	feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
	x = feature_random
	# Patch Shuffle Operation
	try:
		x = x.view(batchsize, group, -1, dim)
	except:
		x = torch.cat([x, x[:, -2:-1, :]], dim=1)
		x = x.view(batchsize, group, -1, dim)

	x = torch.transpose(x, 1, 2).contiguous()
	x = x.view(batchsize, -1, dim)

	return x


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)

	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)


def add_noise_2_suppress_head(feat, head_suppress):
	B, C = feat.shape
	H = 12
	feat_head_div = feat.reshape(B, H, -1)
	feat_noise = feat_head_div[:, head_suppress, :]
	rand_noise = torch.rand(feat_noise.shape, device=feat_noise.device, requires_grad=False)
	feat_head_div[:, head_suppress, :] = feat_noise.detach() + rand_noise
	return feat_head_div.reshape(B, C)


class Backbone(nn.Module):
	def __init__(self, num_classes, cfg):
		super(Backbone, self).__init__()
		last_stride = cfg.MODEL.LAST_STRIDE
		model_path = cfg.MODEL.PRETRAIN_PATH
		model_name = cfg.MODEL.NAME
		pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT

		if model_name == 'resnet50':
			self.in_planes = 2048
			self.base = ResNet(last_stride=last_stride,
			                   block=Bottleneck,
			                   layers=[3, 4, 6, 3])
			print('using resnet50 as a backbone')
		else:
			print('unsupported backbone! but got {}'.format(model_name))

		if pretrain_choice == 'imagenet':
			self.base.load_param(model_path)
			print('Loading pretrained ImageNet model......from {}'.format(model_path))

		self.gap = nn.AdaptiveAvgPool2d(1)
		self.num_classes = num_classes

		self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
		self.classifier.apply(weights_init_classifier)

		self.bottleneck = nn.BatchNorm1d(self.in_planes)
		self.bottleneck.bias.requires_grad_(False)
		self.bottleneck.apply(weights_init_kaiming)

	def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
		x = self.base(x)
		global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
		global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

		if self.neck == 'no':
			feat = global_feat
		elif self.neck == 'bnneck':
			feat = self.bottleneck(global_feat)

		if self.training:
			if self.cos_layer:
				cls_score = self.arcface(feat, label)
			else:
				cls_score = self.classifier(feat)
			return cls_score, global_feat
		else:
			if self.neck_feat == 'after':
				return feat
			else:
				return global_feat

	def load_param(self, trained_path):
		param_dict = torch.load(trained_path)
		if 'state_dict' in param_dict:
			param_dict = param_dict['state_dict']
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
		print('Loading pretrained model from {}'.format(trained_path))

	def load_param_finetune(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
		print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
	def __init__(self, num_classes, camera_num, view_num, cfg, factory):
		super(build_transformer, self).__init__()
		last_stride = cfg.MODEL.LAST_STRIDE
		model_path = cfg.MODEL.PRETRAIN_PATH
		model_name = cfg.MODEL.NAME
		pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.in_planes = 768

		print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

		if cfg.MODEL.SIE_CAMERA:
			camera_num = camera_num
		else:
			camera_num = 0
		if cfg.MODEL.SIE_VIEW:
			view_num = view_num
		else:
			view_num = 0

		self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
		                                                camera=camera_num, view=view_num,
		                                                stride_size=cfg.MODEL.STRIDE_SIZE,
		                                                drop_path_rate=cfg.MODEL.DROP_PATH,
		                                                drop_rate=cfg.MODEL.DROP_OUT,
		                                                attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
		if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
			self.in_planes = 384
		if pretrain_choice == 'imagenet':
			self.base.load_param(model_path)
			print('Loading pretrained ImageNet model......from {}'.format(model_path))

		self.gap = nn.AdaptiveAvgPool2d(1)

		self.num_classes = num_classes
		self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
		if self.ID_LOSS_TYPE == 'arcface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Arcface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'cosface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Cosface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'amsoftmax':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = AMSoftmax(self.in_planes, self.num_classes,
			                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'circle':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = CircleLoss(self.in_planes, self.num_classes,
			                             s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		else:
			self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier.apply(weights_init_classifier)

		self.bottleneck = nn.BatchNorm1d(self.in_planes)
		self.bottleneck.bias.requires_grad_(False)
		self.bottleneck.apply(weights_init_kaiming)

	def forward(self, x, label=None, cam_label=None, view_label=None):
		global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

		feat = self.bottleneck(global_feat)

		if self.training:
			if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
				cls_score = self.classifier(feat, label)
			else:
				cls_score = self.classifier(feat)

			return cls_score, global_feat  # global feature for triplet loss
		else:
			if self.neck_feat == 'after':
				# print("Test with feature after BN")
				return feat
			else:
				# print("Test with feature before BN")
				return global_feat

	def load_param(self, trained_path):
		param_dict = torch.load(trained_path)
		for i in param_dict:
			self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
		print('Loading pretrained model from {}'.format(trained_path))

	def load_param_finetune(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
		print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):  # with jpm
	def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
		super(build_transformer_local, self).__init__()
		model_path = cfg.MODEL.PRETRAIN_PATH
		pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.in_planes = 768

		print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

		if cfg.MODEL.SIE_CAMERA:
			camera_num = camera_num
		else:
			camera_num = 0

		if cfg.MODEL.SIE_VIEW:
			view_num = view_num
		else:
			view_num = 0

		self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
		                                                local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
		                                                stride_size=cfg.MODEL.STRIDE_SIZE,
		                                                drop_path_rate=cfg.MODEL.DROP_PATH)

		if pretrain_choice == 'imagenet':
			self.base.load_param(model_path)
			print('Loading pretrained ImageNet model......from {}'.format(model_path))

		block = self.base.blocks[-1]
		layer_norm = self.base.norm
		self.b1 = nn.Sequential(
			copy.deepcopy(block),
			copy.deepcopy(layer_norm)
		)
		self.b2 = nn.Sequential(
			copy.deepcopy(block),
			copy.deepcopy(layer_norm)
		)

		self.num_classes = num_classes
		self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
		if self.ID_LOSS_TYPE == 'arcface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Arcface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'cosface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Cosface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'amsoftmax':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = AMSoftmax(self.in_planes, self.num_classes,
			                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'circle':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = CircleLoss(self.in_planes, self.num_classes,
			                             s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		else:
			self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier.apply(weights_init_classifier)
			self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_1.apply(weights_init_classifier)
			self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_2.apply(weights_init_classifier)
			self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_3.apply(weights_init_classifier)
			self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_4.apply(weights_init_classifier)

		self.bottleneck = nn.BatchNorm1d(self.in_planes)
		self.bottleneck.bias.requires_grad_(False)
		self.bottleneck.apply(weights_init_kaiming)
		self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_1.bias.requires_grad_(False)
		self.bottleneck_1.apply(weights_init_kaiming)
		self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_2.bias.requires_grad_(False)
		self.bottleneck_2.apply(weights_init_kaiming)
		self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_3.bias.requires_grad_(False)
		self.bottleneck_3.apply(weights_init_kaiming)
		self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_4.bias.requires_grad_(False)
		self.bottleneck_4.apply(weights_init_kaiming)

		self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
		print('using shuffle_groups size:{}'.format(self.shuffle_groups))
		self.shift_num = cfg.MODEL.SHIFT_NUM
		print('using shift_num size:{}'.format(self.shift_num))
		self.divide_length = cfg.MODEL.DEVIDE_LENGTH
		print('using divide_length size:{}'.format(self.divide_length))
		self.rearrange = rearrange

	def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'

		features, _ = self.base(x, cam_label=cam_label, view_label=view_label)  # [64, 129, 768]

		# global branch
		b1_feat = self.b1(features)
		global_feat = b1_feat[:, 0]

		# JPM branch
		feature_length = features.size(1) - 1  # 128
		patch_length = feature_length // self.divide_length  # 32
		token = features[:, 0:1]

		if self.rearrange:
			x = shuffle_unit(features, self.shift_num, self.shuffle_groups)  # num: 5, groups: 2
		else:
			x = features[:, 1:]
		# lf_1
		b1_local_feat = x[:, :patch_length]
		b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
		local_feat_1 = b1_local_feat[:, 0]

		# lf_2
		b2_local_feat = x[:, patch_length:patch_length * 2]
		b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
		local_feat_2 = b2_local_feat[:, 0]

		# lf_3
		b3_local_feat = x[:, patch_length * 2:patch_length * 3]
		b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
		local_feat_3 = b3_local_feat[:, 0]

		# lf_4
		b4_local_feat = x[:, patch_length * 3:patch_length * 4]
		b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
		local_feat_4 = b4_local_feat[:, 0]

		feat = self.bottleneck(global_feat)

		local_feat_1_bn = self.bottleneck_1(local_feat_1)
		local_feat_2_bn = self.bottleneck_2(local_feat_2)
		local_feat_3_bn = self.bottleneck_3(local_feat_3)
		local_feat_4_bn = self.bottleneck_4(local_feat_4)

		if self.training:
			if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
				cls_score = self.classifier(feat, label)
			else:
				cls_score = self.classifier(feat)
				cls_score_1 = self.classifier_1(local_feat_1_bn)
				cls_score_2 = self.classifier_2(local_feat_2_bn)
				cls_score_3 = self.classifier_3(local_feat_3_bn)
				cls_score_4 = self.classifier_4(local_feat_4_bn)
			return [cls_score, cls_score_1, cls_score_2, cls_score_3,
			        cls_score_4
			        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
			            local_feat_4]  # global feature for triplet loss
		else:
			if self.neck_feat == 'after':
				return torch.cat(
					[feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
			else:
				return torch.cat(
					[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

	def load_param(self, trained_path):
		param_dict = torch.load(trained_path)
		for i in param_dict:
			self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
		print('Loading pretrained model from {}'.format(trained_path))

	def load_param_finetune(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
		print('Loading pretrained model for finetuning from {}'.format(model_path))


class SeModule(nn.Module):
	def __init__(self, in_channel, reduction=16):
		super(SeModule, self).__init__()
		self.pool = nn.AdaptiveAvgPool2d(output_size=1)
		self.fc = nn.Sequential(
			nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
			nn.ReLU(),
			nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		channel_mask = self.fc(x)
		return channel_mask * x, channel_mask[:, 0, 0:]


class build_transformer_exp(nn.Module):  # ablation
	def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, patch_size=16, test_only=False,
	             embed_dim=768):
		super(build_transformer_exp, self).__init__()
		model_path = cfg.MODEL.PRETRAIN_PATH
		pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.occ_aug = cfg.MODEL.OCC_AUG
		self.two_branched = cfg.MODEL.TWO_BRANCHED
		self.inference = cfg.MODEL.IFRC
		self.test_only = test_only  # for
		print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
		if self.training:
			self.img_size = cfg.INPUT.SIZE_TRAIN
		else:
			self.img_size = cfg.INPUT.SIZE_TEST
		self.patch_num = int(self.img_size[0] * self.img_size[1] / (patch_size ** 2))
		self.occ_aware = cfg.MODEL.OCC_AWARE
		# backbone
		if cfg.MODEL.SIE_CAMERA:
			camera_num = camera_num
		else:
			camera_num = 0

		if cfg.MODEL.SIE_VIEW:
			view_num = view_num
		else:
			view_num = 0
		self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
		                                                local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
		                                                stride_size=cfg.MODEL.STRIDE_SIZE,
		                                                drop_path_rate=cfg.MODEL.DROP_PATH,
		                                                occ_aware=cfg.MODEL.OCC_AWARE,
		                                                occ_block_depth=cfg.MODEL.EXTRA_OCC_BLOCKS,
		                                                fix_alpha=cfg.MODEL.FIX_ALPHA)

		# self.channel_attn = SeModule(embed_dim)
		if pretrain_choice == 'imagenet':
			self.base.load_param(model_path)
			print('Loading pretrained ImageNet model......from {}'.format(model_path))

		# head
		if self.two_branched:
			self.head_ori = build_transformer_head(self.base, num_classes, cfg, rearrange)
		self.head_occ = build_transformer_head(self.base, num_classes, cfg, rearrange)
		print(f"self.head_occ的结构:{self.head_occ}")
		self.my_head_occ = my_build_transformer_head(self.base, num_classes, cfg, rearrange)

# 第一个x是遮挡的，下面分支的；x_ori是原图不带遮挡的上分支
	def forward(self,  x, x_ori=None, cam_label=None, view_label=None,
	            head_suppress=None):  # label is unused if self.cos_layer == 'no'
		if self.occ_aware:
			mid_feature = 3
		else:
			mid_feature = 0

		if self.training:
			# feature extraction
			if self.two_branched:
				print(f"训练x_ori:{x_ori}")
				features_ori, occ_pred_ori, _ = self.base(x_ori, cam_label=cam_label, view_label=view_label,
				                                          mid_feature=mid_feature, occ_fix=False)  # [64, 129, 768]
				# features, ch_attn_ori = self.channel_attn(features_ori)
				ch_attn_occ = None
				ch_attn_ori = None
				if self.inference:
					score_ori, feat_ori, attn_ori = self.head_ori(features_ori)

				# 训练阶段： 
				else:
					# 我认为的Holistic Head:
					score_ori, feat_ori, attn_ori = self.head_occ(features_ori)

				features, occ_pred_occ, _ = self.base(x, cam_label=cam_label, view_label=view_label,
				                                      mid_feature=mid_feature, occ_fix=self.occ_aware)  # [64, 129, 768]
				# features, ch_attn_occ = self.channel_attn(features)
				print("训练--------------")
				print(f"self.base后的features:{features.shape}") # ([64, 129, 768])
				print(f"self.base后的occ_pred_occ:{occ_pred_occ}") # None

				# 我认为的Repairing Head:              
				# score_occ, feat_occ, attn_occ = self.head_occ(features)

				# 修改后：
				score_occ, feat_occ, attn_occ = self.my_head_occ(features,feat_ori)

				print(f"self.head_occ后的score_occ全局:{score_occ[0].shape}") # ([64, 702])
				print(f"self.head_occ后的score_occ局部:{score_occ[1].shape}") # ([64, 702])
				print(f"self.head_occ后的feat_occ全局:{feat_occ[0].shape}") # ([64, 768])
				print(f"self.head_occ后的feat_occ局部:{feat_occ[1].shape}") # ([64, 768])
				print(f"self.head_occ后的attn_occ:{attn_occ.shape}") # ([64, 12, 128]) 12为12层注意力吗？
				print("--------------训练")				
			# 若不采用双分支：		
			else: 
				if x_ori is not None:  # for data aug
					flag = torch.rand((x.shape[0])).cuda()
					x_mask = (flag > 0.5).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
					x_input = x * x_mask + x_ori * ~x_mask
				else:
					x_input = x
				features, occ_pred_occ, attn_base = self.base(x_input, cam_label=cam_label, view_label=view_label,
				                                   mid_feature=mid_feature, occ_fix=False)  # [64, 129, 768]
				# features, ch_attn_occ = self.channel_attn(features)
				score_occ, feat_occ, attn_occ = self.head_occ(features, head_suppress=head_suppress)
				ch_attn_occ = None
				score_ori, feat_ori, attn_ori, ch_attn_ori = None, None, None, None
			# occ aware
			# 我认为的后续用于遮挡感知预测器
			if self.occ_aware and not self.occ_aug:
				if self.two_branched:
					occ_pred = torch.cat((occ_pred_occ, occ_pred_ori), dim=0)  # two branch occ predict
				else:
					occ_pred = occ_pred_occ
			else:
				occ_pred = None
			# score_ori, feat_ori, occ_pred = None, None, None
			return {"ori": score_ori, "occ": score_occ}, \
			       {"ori": feat_ori, "occ": feat_occ}, occ_pred, \
			       {"ori": attn_ori, "occ": attn_occ}, {"ori": ch_attn_ori, "occ": ch_attn_occ}

		# 推理
		else:
			print(f"推理x_ori:{x_ori}")
			
			features_ori, occ_pred_ori, _ = self.base(x_ori, cam_label=cam_label, view_label=view_label,mid_feature=mid_feature, 
											 occ_fix=False)  # [64, 129, 768]
			features, occ_pred, attn = self.base(x, cam_label=cam_label, view_label=view_label, mid_feature=mid_feature,
			                                     occ_fix=False)  # [64, 129, 768]
			# features, _ = self.channel_attn(features)
			# feat, _ = self.head_occ(features)
			print(f"推理features_ori：{features_ori.shape}") # 推理feat：torch.Size([256, 3840])  
			print(f"推理occ_pred：{occ_pred.shape}") # 推理occ_pred：torch.Size([256, 128, 2])
			feat, _ = self.my_head_occ(features,features_ori)

			print(f"推理feat：{feat.shape}") # 推理feat：torch.Size([256, 3840])  
			# print(f"推理occ_pred：{occ_pred.shape}") # 推理occ_pred：torch.Size([256, 128, 2])
			return feat, occ_pred, attn

	def load_param(self, trained_path):
		param_dict = torch.load(trained_path)
		for i in param_dict:
			if self.test_only and 'classifier' in i:
				continue
			if i.replace('module.', '') in self.state_dict():
				# print(i)
				self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
		print('Loading pretrained model from {}'.format(trained_path))
 
	def load_param_finetune(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			self.state_dict()[i].copy_(param_dict[i])
		print('Loading pretrained model for finetuning from {}'.format(model_path))

# 改进的共享Vit
class build_transformer_head(nn.Module):
	# two branch with shared transformer encoder
	# compatible with occ token
	def __init__(self, base, num_classes, cfg, rearrange):
		super(build_transformer_head, self).__init__()
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.occ_token = cfg.MODEL.OCC_AWARE
		self.in_planes = 768
		# self.base = base
		block = base.blocks[-1]
		layer_norm = base.norm
		self.b1_blk = copy.deepcopy(block)
		self.b1_norm = copy.deepcopy(layer_norm)
		self.b2_blk = copy.deepcopy(block)
		self.b2_norm = copy.deepcopy(layer_norm)
		self.num_classes = num_classes
		self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
		if self.ID_LOSS_TYPE == 'arcface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Arcface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'cosface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Cosface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'amsoftmax':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = AMSoftmax(self.in_planes, self.num_classes,
			                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'circle':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = CircleLoss(self.in_planes, self.num_classes,
			                             s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		else:
			self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier.apply(weights_init_classifier)
			self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_1.apply(weights_init_classifier)
			self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_2.apply(weights_init_classifier)
			self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_3.apply(weights_init_classifier)
			self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_4.apply(weights_init_classifier)

		self.bottleneck = nn.BatchNorm1d(self.in_planes)
		self.bottleneck.bias.requires_grad_(False)
		self.bottleneck.apply(weights_init_kaiming)
		self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_1.bias.requires_grad_(False)
		self.bottleneck_1.apply(weights_init_kaiming)
		self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_2.bias.requires_grad_(False)
		self.bottleneck_2.apply(weights_init_kaiming)
		self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_3.bias.requires_grad_(False)
		self.bottleneck_3.apply(weights_init_kaiming)
		self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_4.bias.requires_grad_(False)
		self.bottleneck_4.apply(weights_init_kaiming)

		self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
		print('using shuffle_groups size:{}'.format(self.shuffle_groups))
		self.shift_num = cfg.MODEL.SHIFT_NUM
		print('using shift_num size:{}'.format(self.shift_num))
		self.divide_length = cfg.MODEL.DEVIDE_LENGTH
		print('using divide_length size:{}'.format(self.divide_length))
		self.rearrange = rearrange

	def forward(self, features, occ_score=None, head_suppress=None):  # label is unused if self.cos_layer == 'no'
		'''
        if self.occ_token:
            patch_num = features.size(1) - 2
        else:
            patch_num = features.size(1) - 1'''
		print("开始调试模块--------")
		# print(f"输入的features：{features.shape}") # 输入的features：torch.Size([64, 129, 768])
		patch_num = features.size(1) - 1
		# print(f"patch_num:{patch_num}") # 128
		# norm feat
		# norm_feat = self.b_norm(features)
		# global branch
		# norm feat
		# norm_feat = self.b_norm(features)
		# global branch
		b1_feat, attn = self.b1_blk(features)

		
		b1_feat = self.b1_norm(b1_feat)
		# print(f"b1_feat:{b1_feat.shape}") # b1_feat:torch.Size([64, 129, 768])
		global_feat = b1_feat[:, 0]
		# print(f"global_feat:{global_feat.shape}") # global_feat:torch.Size([64, 768])
		# JPM branch
		feature_length = patch_num  # 128
		patch_length = feature_length // self.divide_length  # 32
		token = features[:, 0:1]
		# print(f"token:{token.shape}") # token:torch.Size([64, 1, 768])
		if self.rearrange:  # false
			x = shuffle_unit(features, self.shift_num, self.shuffle_groups,
			                 begin=features.size(1) - patch_num)  # num: 5, groups: 2
		else:
			x = features[:, -1 * patch_num:]
		print(f"build_transformer_head中的x的形状：{x.shape}")
		print("结束调试模块--------")
# 在此处前对x进行修改

		# lf_1
		local_feat_1_ = x[:, :patch_length]
		local_feat_1_, _ = self.b2_blk(torch.cat((token, local_feat_1_), dim=1))
		local_feat_1_ = self.b2_norm(local_feat_1_)
		local_feat_1 = local_feat_1_[:, 0]

		# lf_2
		local_feat_2_ = x[:, :patch_length]
		local_feat_2_, _ = self.b2_blk(torch.cat((token, local_feat_2_), dim=1))
		local_feat_2_ = self.b2_norm(local_feat_2_)
		local_feat_2 = local_feat_2_[:, 0]

		# lf_3
		local_feat_3_ = x[:, :patch_length]
		local_feat_3_, _ = self.b2_blk(torch.cat((token, local_feat_3_), dim=1))
		local_feat_3_ = self.b2_norm(local_feat_3_)
		local_feat_3 = local_feat_3_[:, 0]

		# lf_4
		local_feat_4_ = x[:, :patch_length]
		local_feat_4_, _ = self.b2_blk(torch.cat((token, local_feat_4_), dim=1))
		local_feat_4_ = self.b2_norm(local_feat_4_)
		local_feat_4 = local_feat_4_[:, 0]

		# bottleneck for classifier
		global_feat_bn = self.bottleneck(global_feat)

		local_feat_1_bn = self.bottleneck_1(local_feat_1)
		local_feat_2_bn = self.bottleneck_2(local_feat_2)
		local_feat_3_bn = self.bottleneck_3(local_feat_3)
		local_feat_4_bn = self.bottleneck_4(local_feat_4)

		if self.training:
			if head_suppress is not None:
				global_feat_noise = add_noise_2_suppress_head(global_feat_bn, head_suppress)
				cls_score = self.classifier(global_feat_noise)
				local_feat_1_noise = add_noise_2_suppress_head(local_feat_1_bn, head_suppress)
				cls_score_1 = self.classifier_1(local_feat_1_noise)
				local_feat_2_noise = add_noise_2_suppress_head(local_feat_2_bn, head_suppress)
				cls_score_2 = self.classifier_2(local_feat_2_noise)
				local_feat_3_noise = add_noise_2_suppress_head(local_feat_3_bn, head_suppress)
				cls_score_3 = self.classifier_3(local_feat_3_noise)
				local_feat_4_noise = add_noise_2_suppress_head(local_feat_4_bn, head_suppress)
				cls_score_4 = self.classifier_4(local_feat_4_noise)
			else:
				cls_score = self.classifier(global_feat_bn)
				cls_score_1 = self.classifier_1(local_feat_1_bn)
				cls_score_2 = self.classifier_2(local_feat_2_bn)
				cls_score_3 = self.classifier_3(local_feat_3_bn)
				cls_score_4 = self.classifier_4(local_feat_4_bn)
			return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
			       [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4], \
			       attn
		else:
			if occ_score is not None:
				non_occ_sum = occ_score[:, :, 0].reshape((occ_score.shape[0], 4, -1)).mean(dim=-1) / 2
				loc_weight = non_occ_sum.softmax(dim=-1).unsqueeze(-1)
				# print(non_occ_sum[0], loc_weight[0])
			if self.neck_feat == 'after':
				return torch.cat(
					[global_feat_bn, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4,
					 local_feat_4_bn / 4], dim=1)
			else:  # before bn layer for test
				if occ_score is not None:
					feats = torch.cat(
						[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
					'''
                    feats = torch.cat(
                        [global_feat,
                         local_feat_1 * loc_weight[:, 0],
                         local_feat_2 * loc_weight[:, 1],
                         local_feat_3 * loc_weight[:, 2],
                         local_feat_4 * loc_weight[:, 3]], dim=1)'''
					return feats
				else:
					return torch.cat(
						[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
						dim=1), attn

class my_build_transformer_head(nn.Module):
	# two branch with shared transformer encoder
	# compatible with occ token
	def __init__(self, base, num_classes, cfg, rearrange):
		super(my_build_transformer_head, self).__init__()
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.occ_token = cfg.MODEL.OCC_AWARE
		self.in_planes = 768
		# self.base = base
		block = base.blocks[-1]
		layer_norm = base.norm
		# self.b_norm = copy.deepcopy(layer_norm)
		'''
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        '''
		self.b1_blk = copy.deepcopy(block)
		self.b1_norm = copy.deepcopy(layer_norm)
		self.b2_blk = copy.deepcopy(block)
		self.b2_norm = copy.deepcopy(layer_norm)
		self.num_classes = num_classes
		self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
		if self.ID_LOSS_TYPE == 'arcface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Arcface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'cosface':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = Cosface(self.in_planes, self.num_classes,
			                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'amsoftmax':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = AMSoftmax(self.in_planes, self.num_classes,
			                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		elif self.ID_LOSS_TYPE == 'circle':
			print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
			                                         cfg.SOLVER.COSINE_MARGIN))
			self.classifier = CircleLoss(self.in_planes, self.num_classes,
			                             s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
		else:
			self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier.apply(weights_init_classifier)
			self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_1.apply(weights_init_classifier)
			self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_2.apply(weights_init_classifier)
			self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_3.apply(weights_init_classifier)
			self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
			self.classifier_4.apply(weights_init_classifier)

		self.bottleneck = nn.BatchNorm1d(self.in_planes)
		self.bottleneck.bias.requires_grad_(False)
		self.bottleneck.apply(weights_init_kaiming)
		self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_1.bias.requires_grad_(False)
		self.bottleneck_1.apply(weights_init_kaiming)
		self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_2.bias.requires_grad_(False)
		self.bottleneck_2.apply(weights_init_kaiming)
		self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_3.bias.requires_grad_(False)
		self.bottleneck_3.apply(weights_init_kaiming)
		self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
		self.bottleneck_4.bias.requires_grad_(False)
		self.bottleneck_4.apply(weights_init_kaiming)

		self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
		print('using shuffle_groups size:{}'.format(self.shuffle_groups))
		self.shift_num = cfg.MODEL.SHIFT_NUM
		print('using shift_num size:{}'.format(self.shift_num))
		self.divide_length = cfg.MODEL.DEVIDE_LENGTH
		print('using divide_length size:{}'.format(self.divide_length))
		self.rearrange = rearrange
	def apply_random_block_mask(self,tokens, mask_size=(4, 4)):
		"""
		对 ViT 编码后的特征嵌入应用随机块掩码。

		参数：
		- tokens (torch.Tensor): 形状为 [batch_size, num_patches, embed_dim] 的特征嵌入。
		- mask_size (tuple): 掩码块的大小，默认为 (4, 4)。

		返回：
		- masked_tokens (torch.Tensor): 应用掩码后的特征嵌入。
		"""
		_, num_patches, _ = tokens.shape

		# 计算 num_patches 的近似形状
		grid_size = int(np.ceil(np.sqrt(num_patches)))  # 向上取整
		max_x = grid_size - mask_size[0]
		max_y = grid_size - mask_size[1]

		# 随机选择掩码块的起始位置
		start_x = np.random.randint(0, max_x + 1)
		start_y = np.random.randint(0, max_y + 1)

		# 创建掩码矩阵
		mask = np.ones((grid_size, grid_size), dtype=np.float32)
		mask[start_x:start_x + mask_size[0], start_y:start_y + mask_size[1]] = 0
		mask = mask.flatten()  # 展平为一维数组

		# 如果 num_patches 不是完全平方数，调整掩码大小
		if len(mask) < num_patches:
			mask = np.pad(mask, (0, num_patches - len(mask)), 'constant', constant_values=1)
		elif len(mask) > num_patches:
			mask = mask[:num_patches]

		# 将掩码应用于每个批次的特征嵌入
		mask = torch.tensor(mask, device=tokens.device).unsqueeze(0).unsqueeze(-1)  # [1, 196, 1]
		masked_tokens = tokens * mask  # [batch_size, 196, 768]

		return masked_tokens
	def forward(self, features,ori_features, occ_score=None, head_suppress=None):  # label is unused if self.cos_layer == 'no'
		'''
        if self.occ_token:
            patch_num = features.size(1) - 2
        else:
            patch_num = features.size(1) - 1'''
		print("开始调试模块--------")
		# print(f"输入的features：{features.shape}") # 输入的features：torch.Size([64, 129, 768])
		patch_num = features.size(1) - 1
		# print(f"patch_num:{patch_num}") # 128
		# norm feat
		# norm_feat = self.b_norm(features)
		# global branch

		# b1_feat, attn = self.b1_blk(features)

		#######修改后的：#######
		# 拼接 features 和 ori_features
		ori_features = self.apply_random_block_mask(ori_features)
		merged_features = torch.cat((features, ori_features), dim=1)  # 形状为 [64, 129*2, 768]

		# 送入 self.b1_blk
		b1_feat, attn = self.b1_blk(merged_features)

		# 从第二个维度取出前 129 个元素
		b1_feat = b1_feat[:, :129, :]  # 形状为 [64, 129, 768]
		attn = attn[:, :, :129]  # 形状为 [64, num_heads=12, 129]
		#######以上是修改后的#######

		b1_feat = self.b1_norm(b1_feat)
		# print(f"b1_feat:{b1_feat.shape}") # b1_feat:torch.Size([64, 129, 768])
		global_feat = b1_feat[:, 0]
		# print(f"global_feat:{global_feat.shape}") # global_feat:torch.Size([64, 768])
		# JPM branch
		feature_length = patch_num  # 128
		patch_length = feature_length // self.divide_length  # 32
		token = features[:, 0:1]
		# print(f"token:{token.shape}") # token:torch.Size([64, 1, 768])
		if self.rearrange:  # false
			x = shuffle_unit(features, self.shift_num, self.shuffle_groups,
			                 begin=features.size(1) - patch_num)  # num: 5, groups: 2
		else:
			x = features[:, -1 * patch_num:]
		print(f"build_transformer_head中的x的形状：{x.shape}")
		print("结束调试模块--------")
# 在此处前对x进行修改

		# lf_1
		local_feat_1_ = x[:, :patch_length]
		local_feat_1_, _ = self.b2_blk(torch.cat((token, local_feat_1_), dim=1))
		local_feat_1_ = self.b2_norm(local_feat_1_)
		local_feat_1 = local_feat_1_[:, 0]

		# lf_2
		local_feat_2_ = x[:, :patch_length]
		local_feat_2_, _ = self.b2_blk(torch.cat((token, local_feat_2_), dim=1))
		local_feat_2_ = self.b2_norm(local_feat_2_)
		local_feat_2 = local_feat_2_[:, 0]

		# lf_3
		local_feat_3_ = x[:, :patch_length]
		local_feat_3_, _ = self.b2_blk(torch.cat((token, local_feat_3_), dim=1))
		local_feat_3_ = self.b2_norm(local_feat_3_)
		local_feat_3 = local_feat_3_[:, 0]

		# lf_4
		local_feat_4_ = x[:, :patch_length]
		local_feat_4_, _ = self.b2_blk(torch.cat((token, local_feat_4_), dim=1))
		local_feat_4_ = self.b2_norm(local_feat_4_)
		local_feat_4 = local_feat_4_[:, 0]

		# bottleneck for classifier
		global_feat_bn = self.bottleneck(global_feat)

		local_feat_1_bn = self.bottleneck_1(local_feat_1)
		local_feat_2_bn = self.bottleneck_2(local_feat_2)
		local_feat_3_bn = self.bottleneck_3(local_feat_3)
		local_feat_4_bn = self.bottleneck_4(local_feat_4)

		if self.training:
			if head_suppress is not None:
				global_feat_noise = add_noise_2_suppress_head(global_feat_bn, head_suppress)
				cls_score = self.classifier(global_feat_noise)
				local_feat_1_noise = add_noise_2_suppress_head(local_feat_1_bn, head_suppress)
				cls_score_1 = self.classifier_1(local_feat_1_noise)
				local_feat_2_noise = add_noise_2_suppress_head(local_feat_2_bn, head_suppress)
				cls_score_2 = self.classifier_2(local_feat_2_noise)
				local_feat_3_noise = add_noise_2_suppress_head(local_feat_3_bn, head_suppress)
				cls_score_3 = self.classifier_3(local_feat_3_noise)
				local_feat_4_noise = add_noise_2_suppress_head(local_feat_4_bn, head_suppress)
				cls_score_4 = self.classifier_4(local_feat_4_noise)
			else:
				cls_score = self.classifier(global_feat_bn)
				cls_score_1 = self.classifier_1(local_feat_1_bn)
				cls_score_2 = self.classifier_2(local_feat_2_bn)
				cls_score_3 = self.classifier_3(local_feat_3_bn)
				cls_score_4 = self.classifier_4(local_feat_4_bn)
			return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
			       [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4], \
			       attn
		else:
			if occ_score is not None:
				non_occ_sum = occ_score[:, :, 0].reshape((occ_score.shape[0], 4, -1)).mean(dim=-1) / 2
				loc_weight = non_occ_sum.softmax(dim=-1).unsqueeze(-1)
				# print(non_occ_sum[0], loc_weight[0])
			if self.neck_feat == 'after':
				return torch.cat(
					[global_feat_bn, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4,
					 local_feat_4_bn / 4], dim=1)
			else:  # before bn layer for test
				if occ_score is not None:
					feats = torch.cat(
						[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)
					'''
                    feats = torch.cat(
                        [global_feat,
                         local_feat_1 * loc_weight[:, 0],
                         local_feat_2 * loc_weight[:, 1],
                         local_feat_3 * loc_weight[:, 2],
                         local_feat_4 * loc_weight[:, 3]], dim=1)'''
					return feats
				else:
					return torch.cat(
						[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
						dim=1), attn
__factory_T_type = {
	'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
	'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
	'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
	'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num, test_only=False):
	if cfg.MODEL.NAME == 'transformer':
		if cfg.MODEL.ZZWEXP:
			model = build_transformer_exp(num_class, camera_num, view_num, cfg, __factory_T_type,
			                              rearrange=cfg.MODEL.RE_ARRANGE, test_only=test_only)
			print('===========building transformer with ZZW ablation study ===========')
		elif cfg.MODEL.JPM:
			model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
			                                rearrange=cfg.MODEL.RE_ARRANGE)
			print('===========building transformer with JPM module ===========')
		else:
			model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
			print('===========building transformer===========')
	else:
		model = Backbone(num_class, cfg)
		print('===========building ResNet===========')
	return model
