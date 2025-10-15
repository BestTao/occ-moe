from yacs.config import CfgNode as CN

# ----------------------------------------------------------------------------- 
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------

# Config definition
_C = CN()

# ----------------------------------------------------------------------------- 
# MODEL
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.moeori_LOSS_WEIGHT =1.0
_C.MODEL.moeocc_LOSS_WEIGHT =1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# ZZW
_C.MODEL.ZZWEXP = False
_C.MODEL.ZZWTRY= False
_C.MODEL.TWO_BRANCHED = False

# Occlusion Augmentation
_C.MODEL.OCC_AUG = False
# for img block
_C.MODEL.OCC_RATIO = [0.2]
_C.MODEL.OCC_MARGIN = 0.3
_C.MODEL.OCC_ULRD = [0.25, 0.25, 0.25, 0.25]
_C.MODEL.OCC_ALIGN_BOUND = False
# for img_rect
_C.MODEL.OCC_ALIGN_BTM = False

# Occlusion Aware and Suppression
_C.MODEL.OCC_AWARE = False
_C.MODEL.EXTRA_OCC_BLOCKS = 3
_C.MODEL.OCC_LOSS_WEIGHT = 1.0
_C.MODEL.FIX_ALPHA = 0.1

# Occlusion Repairing
_C.MODEL.IFRC = False
_C.MODEL.IFRC_LOSS_WEIGHT = 0.01
_C.MODEL.IFRC_TARGET = 'feat'
_C.MODEL.IFRC_LOSS_TYPE = 'l2dist'
_C.MODEL.BRANCH_BLOCKS = 6
_C.MODEL.IFRC_HEAD_NUM = 6

# Head Enhancement
_C.MODEL.HEAD_ENHANCE = False
_C.MODEL.HEAD_DIV_LOSS_WEIGHT = 1.0

# Head Suppression
_C.MODEL.HEAD_SUP = False
_C.MODEL.SAMPLE_HEAD_SUP = False
_C.MODEL.OCC_TYPE = ''
_C.MODEL.OCC_TYPES = []
_C.MODEL.OCC_TYPES_RATIO=[]
_C.MODEL.USE_DECODER_FEAT = ''

# New entry for PATCH_ALIGN_OCC
_C.MODEL.PATCH_ALIGN_OCC = True  # Add this line

# Continue with other parameters below
_C.MODEL.qkv_bias= True
_C.MODEL.mlp_ratio=4
_C.MODEL.depth=12
_C.MODEL.embed_dim = 768


# ----------------------------------------------------------------------------- 
# ENCODER
# -----------------------------------------------------------------------------
_C.ENCODER = CN()
_C.ENCODER.num_layers = 12
_C.ENCODER.mlp_dim = 8
_C.ENCODER.num_heads = 12
_C.ENCODER.attention_dropout_rate = 0.1

# ----------------------------------------------------------------------------- 
# MOE
# -----------------------------------------------------------------------------
_C.MOE = CN()
# MoE层配置
_C.MOE.layers = [3, 6, 9]        # 插入MoE的层索引
_C.MOE.num_experts = 4           # 专家数量
_C.MOE.group_size = 4            # 专家分组大小

# 路由器配置
_C.MOE.router = CN()
_C.MOE.router.num_selected_experts = 1  # 选择专家数
_C.MOE.router.noise_std = 1e-3          # 路由噪声标准差
_C.MOE.router.importance_loss_weight = 0.02  # 重要性损失权重
_C.MOE.router.load_loss_weight = 0.02        # 负载平衡损失权重

# 分发器配置
_C.MOE.router.dispatcher = CN()
_C.MOE.router.dispatcher.name = "einsum"    # 分发算法类型
_C.MOE.router.dispatcher.capacity = 2       # 专家容量系数
_C.MOE.router.dispatcher.batch_priority = False  # 批次优先级
_C.MOE.router.dispatcher.bfloat16 = False        # 使用bfloat16

# ----------------------------------------------------------------------------- 
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.INPUT.SEG_CFG = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
# Value of padding size
_C.INPUT.PADDING = 10
# Occultation type
_C.INPUT.OCC_TYPE = 'instance_mask'
_C.INPUT.OCC_TYPES = []
_C.INPUT.AUG_TYPES = []
_C.INPUT.OCC_TYPES_RATIO = []

# ----------------------------------------------------------------------------- 
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
_C.MODEL.PRETEXT = 'feat'
# ----------------------------------------------------------------------------- 
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.OCC_PRED_FROZEN = 60

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.EVAL = True

# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
_C.TEST.NECK_FEAT = 'before'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.USE_FEAT = 'dec'

# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST

_C.OUTPUT_DIR= ""
