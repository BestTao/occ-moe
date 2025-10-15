# -*- coding=utf-8 -*-
from enum import Enum


class AugmentationType:
	OCCLUSION = 'occlusion'


class OcclusionType:
	INSTANCE_MASK = 'instance_mask'
	IMG_BLOCK = 'img_block'
	IMG_RECT = 'img_rect'
