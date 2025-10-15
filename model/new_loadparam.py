    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        
        # 键名转换规则（注意全部使用小写）
        key_mapping = [
            ('blocks.', 'moe_blocks.'),       # 主模块前缀
            ('.mlp.fc1.', '.mlp_block.fc1.'),  # 普通层MLP路径
            ('.mlp.fc2.', '.mlp_block.fc2.')

        ]


        model_dict = self.state_dict()
        updated_keys = []
        skipped_keys = []
        
        for raw_k, v in param_dict.items():
            # if any(skip_key in raw_k for skip_key in ['head', 'dist', 'moe_occ_blocks']):  # 显式跳过 # xz
            #     print(f'[Skip] 结构过滤: {raw_k}')# xz
            #     continue  # xz
            # if 'head' in raw_k or 'dist' in raw_k:
            #     continue
                
            # 执行键名转换 -------------------------------------------------
            k = raw_k
            for old, new in key_mapping:
                k = k.replace(old, new)
            # Step 2: 处理特殊参数形状
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
                
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)            
            # 特殊处理MoE层 -------------------------------------------------
            # if 'moe_blocks.' in k:
            #     # 提取层号 (e.g. "moe_blocks.3.norm1" -> 3)
            #     layer_idx = int(k.split('.')[1])
            #     # 如果该层是MoE层且参数属于原始MLP部分
            #     if layer_idx in self.moe_blocks and 'mlp_block.mlp.' in k:
            #         print(f'[Skip] MoE层参数: {raw_k} -> {k}')
            #         skipped_keys.append(k)
            #         continue
                    
            # 处理特殊参数形状 -----------------------------------------------
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
                
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            
            # 参数加载 -----------------------------------------------------
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k].copy_(v)
                    updated_keys.append(k)
                else:
                    print(f'[Mismatch] {k}: pretrained {v.shape} vs model {model_dict[k].shape}')
                    skipped_keys.append(k)
            else:
                print(f'[Missing] {k} (原始键名: {raw_k})')
                skipped_keys.append(k)
        
        # 打印统计信息 -----------------------------------------------------
        print(f'\n===== 参数加载统计 =====')
        print(f'成功加载: {len(updated_keys)}/{len(param_dict)}')
        print(f'跳过参数: {len(skipped_keys)}')
        print(f'匹配率: {len(updated_keys)/len(param_dict):.1%}')
        
        # 可选：打印前5个成功加载的键示例
        print('\n----- 成功加载示例 -----')
        for k in updated_keys[:5]:
            print(f'+ {k}')
        
        # 可选：打印前5个跳过的键示例
        if skipped_keys:
            print('\n----- 跳过键示例 -----')
            for k in skipped_keys[:5]:
                print(f'- {k}')


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb