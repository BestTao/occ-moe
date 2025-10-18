import torch
import torch.nn as nn
import copy
from .occ_moe import TransReID,vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
	deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from omegaconf import OmegaConf
from functools import partial

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




class build_transformer_moe(nn.Module):  # ablation
	def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, patch_size=16, test_only=False,
	             embed_dim=768):
		super(build_transformer_moe, self).__init__()
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
			print("自动触发self.training")
			self.img_size = cfg.INPUT.SIZE_TRAIN
		else:
			print("没有触发self.training")
			self.img_size = cfg.INPUT.SIZE_TEST
		self.patch_num = int(self.img_size[0] * self.img_size[1] / (patch_size ** 2))
		# backbone
		if cfg.MODEL.SIE_CAMERA:
			camera_num = camera_num
		else:
			camera_num = 0

		if cfg.MODEL.SIE_VIEW:
			view_num = view_num
		else:
			view_num = 0
		self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](encoder=cfg.ENCODER,img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
		                                                local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
		                                                stride_size=cfg.MODEL.STRIDE_SIZE,
		                                                drop_path_rate=cfg.MODEL.DROP_PATH,
														drop_rate=cfg.MODEL.DROP_OUT,
														moe=cfg.MOE,
		                                                )
		# 不加载预训练的模型了
		if pretrain_choice == 'imagenet':
			self.base.load_param(model_path)
			# self.base.load_param_finetune(model_path)
			print('Loading pretrained ImageNet model......from {}'.format(model_path))
		
		# head
		if self.two_branched:
			self.head_ori = build_transformer_head(self.base, num_classes, cfg, rearrange,mode='ori')
		self.head_occ = build_transformer_head(self.base, num_classes, cfg, rearrange,mode='occ')
		# print(f"self.head_occ的结构:{self.head_occ}")

		
	def forward(self, x, x_ori=None, cam_label=None, view_label=None,
	            head_suppress=None):  # label is unused if self.cos_layer == 'no'
		


		if self.training:
			# feature extraction
			if self.two_branched:
				features, occ_pred_occ, _,metrics_occ,moe_data_occ = self.base(x, cam_label=cam_label, view_label=view_label,
				                                      is_occ=True)  # [64, 129, 768]
				
				score_occ, feat_occ, attn_occ = self.head_occ(features)

				features_ori, occ_pred_ori, _,metrics_ori,moe_data_ori = self.base(x_ori, cam_label=cam_label, view_label=view_label,
				                                        is_occ=False)  # [64, 129, 768]

				ch_attn_occ = None
				ch_attn_ori = None
				if self.inference:
					# 训练实际走的这里
					score_ori, feat_ori, attn_ori = self.head_ori(features_ori)

				# 训练阶段： 
				else:
					# 我认为的Holistic Head:
					
					score_ori, feat_ori, attn_ori = self.head_occ(features_ori)

			# 若不采用双分支：		
			else: 
				if x_ori is not None:  # for data aug
					flag = torch.rand((x.shape[0])).cuda()
					x_mask = (flag > 0.5).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
					x_input = x * x_mask + x_ori * ~x_mask
				else:
					x_input = x
				features, occ_pred_occ, attn_base = self.base(x_input, cam_label=cam_label, view_label=view_label,
				                                )  # [64, 129, 768]
				# features, ch_attn_occ = self.channel_attn(features)
				score_occ, feat_occ, attn_occ = self.head_occ(features, head_suppress=head_suppress)
				ch_attn_occ = None
				score_ori, feat_ori, attn_ori, ch_attn_ori = None, None, None, None
			# occ aware
			# 我认为的后续用于遮挡感知预测器
			if self.occ_aware and not self.occ_aug:
				if self.two_branched:
					print("检验双分支")
					occ_pred = torch.cat((occ_pred_occ, occ_pred_ori), dim=0)  # two branch occ predict
					print(f"occ_pred.shape:{occ_pred.shape}")  # occ_pred.shape:torch.Size([128, 128, 2])
					print(f"occ_pred_occ:{occ_pred_occ.shape}") # occ_pred_occ:torch.Size([64, 128, 2])
				else:
					print("走错了不是双分支")
					occ_pred = occ_pred_occ
			else:
				occ_pred = None
			# score_ori, feat_ori, occ_pred = None, None, None
			return {"ori": score_ori, "occ": score_occ}, \
			       {"ori": feat_ori, "occ": feat_occ}, occ_pred, \
			       {"ori": attn_ori, "occ": attn_occ}, {"ori": ch_attn_ori, "occ": ch_attn_occ},metrics_occ,metrics_ori,moe_data_occ,moe_data_ori

		# 推理
		else:
			# print(f"推理的x形状：{x.shape}")
			features, occ_pred, attn,_,moe_data = self.base(x, cam_label=cam_label, view_label=view_label, 
			                                    is_occ=True)  # [64, 129, 768]
			# features, _ = self.channel_attn(features) 
			feat, _ ,moe_data12= self.head_occ(features)
			if moe_data12 is not None and isinstance(moe_data12, dict):
				# 将第12层数据追加到 all_moe_data
				moe_data.append({
					'layer_index': 11,  # 层索引从0开始，因此第12层索引为11
					'expert_attentions': moe_data12.get('expert_attentions'),
					'expert_contributions': moe_data12.get('expert_contributions'),
					'dispatch_weights':moe_data12.get('dispatch_weights')
				})
			# print(f"推理feat：{feat.shape}") # 推理feat：torch.Size([256, 3840])  
			return feat, occ_pred, attn,moe_data

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


class build_transformer_branch(nn.Module):
	def __init__(self, base, depth=5):
		super(build_transformer_branch, self).__init__()
		self.blocks = copy.deepcopy(base.blocks[-1 * depth - 1:-1])
		self.norm = base.norm

	def forward(self, x):
		for blk in self.blocks:
			x, _ = blk(x)
		return x

# 改进的共享Vit
class  build_transformer_head(nn.Module):
	# two branch with shared transformer encoder
	# compatible with occ token
	def __init__(self, base, num_classes, cfg, rearrange,mode='occ'):
		super(build_transformer_head, self).__init__()
		self.cos_layer = cfg.MODEL.COS_LAYER
		self.neck = cfg.MODEL.NECK
		self.neck_feat = cfg.TEST.NECK_FEAT
		self.occ_token = cfg.MODEL.OCC_AWARE
		self.in_planes = 768
		# self.base = base
		# block2 = base.moe_blocks[-2]
		if mode == 'occ':
			block = base.moe_blocks[-2]
		else:
			block = base.moe_blocks[-1]  # base.moe_blocks[-1]
		block2 = base.moe_blocks[-1]
		# if rearrange:
		# 	block2 = base.moe_blocks[-1]
		# else:
		# 	block2 = block
		# block = nn.Sequential(*base.moe_blocks[-2:])
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

		self.b1_blk2 = copy.deepcopy(block2)
		self.b1_norm = copy.deepcopy(layer_norm)
		self.b2_blk = copy.deepcopy(block2)
		# self.b2_blk2 = copy.deepcopy(block2)
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

	def forward(self, features, head_suppress=None):  # label is unused if self.cos_layer == 'no'
		'''
        if self.occ_token:
            patch_num = features.size(1) - 2
        else:
            patch_num = features.size(1) - 1'''
		patch_num = features.size(1) - 1
		# norm_feat = self.b_norm(features)
		# global branch
		b1_feat, attn,_,moe_data12 = self.b1_blk(features)
		b1_feat = self.b1_norm(b1_feat)
		# print(f"b1_feat:{b1_feat.shape}") # b1_feat:torch.Size([64, 129, 768])
		global_feat = b1_feat[:, 0]
		# print(f"global_feat:{global_feat.shape}") # global_feat:torch.Size([64, 768])
		# JPM branch
		feature_length = patch_num  # 128
		patch_length = feature_length // self.divide_length  # 128/4=32
		token = features[:, 0:1]
		# print(f"token:{token.shape}") # token:torch.Size([64, 1, 768])
		if self.rearrange:  # false
			x = shuffle_unit(features, self.shift_num, self.shuffle_groups,
			                 begin=features.size(1) - patch_num)  # num: 5, groups: 2
		else:
			x = features[:, -1 * patch_num:]
		
		# lf_1
		local_feat_1_ = x[:, :patch_length]
		local_feat_1_, _,_,_ = self.b2_blk(torch.cat((token, local_feat_1_), dim=1))
		local_feat_1_ = self.b2_norm(local_feat_1_)
		local_feat_1 = local_feat_1_[:, 0]

		# lf_2   x[:, patch_length * 2:patch_length * 3]
		# local_feat_2_ = x[:, :patch_length]
		local_feat_2_ = x[:, patch_length:patch_length*2]
		local_feat_2_, _,_,_ = self.b2_blk(torch.cat((token, local_feat_2_), dim=1))
		local_feat_2_ = self.b2_norm(local_feat_2_)
		local_feat_2 = local_feat_2_[:, 0]

		# lf_3
		# local_feat_3_ = x[:, :patch_length]
		local_feat_3_ = x[:, patch_length*2:patch_length*3]
		local_feat_3_, _,_,_ = self.b2_blk(torch.cat((token, local_feat_3_), dim=1))
		local_feat_3_ = self.b2_norm(local_feat_3_)
		local_feat_3 = local_feat_3_[:, 0]

		# lf_4
		# local_feat_4_ = x[:, :patch_length]
		local_feat_4_ = x[:, patch_length*3:patch_length*4]
		local_feat_4_, _,_,_ = self.b2_blk(torch.cat((token, local_feat_4_), dim=1))
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
				print("head_suppress is not None")
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
				# print("head_suppress is None") # 走的这里！
				cls_score = self.classifier(global_feat_bn)
				cls_score_1 = self.classifier_1(local_feat_1_bn)
				cls_score_2 = self.classifier_2(local_feat_2_bn)
				cls_score_3 = self.classifier_3(local_feat_3_bn)
				cls_score_4 = self.classifier_4(local_feat_4_bn)
			return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
			       [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4], \
			       attn
		else:
			if self.neck_feat == 'after':
				return torch.cat(
					[global_feat_bn, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4,
					 local_feat_4_bn / 4], dim=1)
			else: 
				return torch.cat(
					[global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
					dim=1), attn,moe_data12



__factory_T_type = {
	'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
	'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
	'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
	'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def  make_model(cfg, num_class, camera_num, view_num, test_only=False):
	if cfg.MODEL.NAME == 'transformer':
		if cfg.MODEL.ZZWEXP:
			model = build_transformer_moe(num_class, camera_num, view_num, cfg, __factory_T_type,
			                              rearrange=cfg.MODEL.RE_ARRANGE, test_only=test_only)
			print('===========building transformer with ZZW ablation study ===========')

	return model
