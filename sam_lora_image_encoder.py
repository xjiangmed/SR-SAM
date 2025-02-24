from segment_anything import sam_model_registry
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
import copy
import numpy as np


def ema_update(model, rate):
    encoders = []
    encoders.append(model.sam.image_encoder)
    
    for encoder in encoders:
        for i, _ in enumerate(encoder.blocks.children()):
            avg_model_params = list(_.attn.qkv.ema_linear_a_q.parameters()) + list(_.attn.qkv.ema_linear_b_q.parameters()) + \
                list(_.attn.qkv.ema_linear_a_v.parameters()) + list(_.attn.qkv.ema_linear_b_v.parameters())
            model_params = list(_.attn.qkv.linear_a_q.parameters()) + list(_.attn.qkv.linear_b_q.parameters()) + \
                list(_.attn.qkv.linear_a_v.parameters()) + list(_.attn.qkv.linear_b_v.parameters())  
            for moving_avg_param, param in zip(avg_model_params, model_params):
                moving_avg_param.data = rate * moving_avg_param.data + (1-rate) * param.data.detach()   

class _LoRAtruncation_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            ema_linear_a_q = None,
            ema_linear_b_q = None,
            ema_linear_a_v = None,
            ema_linear_b_v = None,
            index = 8,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        if ema_linear_a_q is not None:
            self.ema_linear_a_q = ema_linear_a_q
            self.ema_linear_b_q = ema_linear_b_q
            self.ema_linear_a_v = ema_linear_a_v
            self.ema_linear_b_v = ema_linear_b_v
    
        self.index = index
        
    def calculate_change_rate(self, a, bb, r):
        change_rate = abs(bb) / abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices
    
    def update_base_weight(self, ema=False, logging=None):
        if ema:
            delta_W_q = self.ema_linear_b_q.weight @ self.ema_linear_a_q.weight 
        else:
            delta_W_q = self.linear_b_q.weight @ self.linear_a_q.weight 
        base_W_q = self.qkv.weight[:self.dim, :].clone()
        weight_u_q, weight_sigma_q, weight_vt_q = torch.linalg.svd(base_W_q, full_matrices=False)
        delta_sigma_q = torch.diag(torch.matmul(torch.matmul(weight_u_q.T, delta_W_q), weight_vt_q.T))
        top_index_q = self.calculate_change_rate(weight_sigma_q, delta_sigma_q, self.index)
        remain_index_q = torch.tensor([idx for idx in range(weight_u_q.shape[1]) if idx not in top_index_q]).cuda()
        logging.info(top_index_q)
        
        new_base_W_q = weight_u_q[:, remain_index_q] @ torch.diag(weight_sigma_q[remain_index_q]) @ weight_vt_q[remain_index_q, :] 
        self.qkv.weight[:self.dim, :] = new_base_W_q.clone()
        
        if ema:
            delta_W_v = self.ema_linear_b_v.weight @ self.ema_linear_a_v.weight
        else:
            delta_W_v = self.linear_b_v.weight @ self.linear_a_v.weight 
        base_W_v = self.qkv.weight[-self.dim:, :].clone()
        weight_u_v, weight_sigma_v, weight_vt_v = torch.linalg.svd(base_W_v, full_matrices=False)
        delta_sigma_v = torch.diag(torch.matmul(torch.matmul(weight_u_v.T, delta_W_v), weight_vt_v.T))
        top_index_v = self.calculate_change_rate(weight_sigma_v, delta_sigma_v, self.index)
        remain_index_v = torch.tensor([idx for idx in range(weight_u_v.shape[1]) if idx not in top_index_v ]).cuda()
        
        new_base_W_v = weight_u_v[:, remain_index_v] @ torch.diag(weight_sigma_v[remain_index_v]) @ weight_vt_v[remain_index_v, :]
        self.qkv.weight[-self.dim:, :] = new_base_W_v.clone()
        return top_index_q, top_index_v
        
    def forward(self, x, ema=False):
        qkv = self.qkv(x)
        
        if ema:
            new_q = self.ema_linear_b_q(self.ema_linear_a_q(x))
            new_v = self.ema_linear_b_v(self.ema_linear_a_v(x))
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))
            
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
 
class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None, ema_mode=True, Dash_index=8, truncation=True):
        super(LoRA_Sam, self).__init__()
        self.ema_mode = ema_mode
        self.truncation = truncation

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default

        # lets freeze first
        # freeze prompt encoder
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
            
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.reset_A_parameters(w_a_linear_q)
            self.reset_B_parameters(w_b_linear_q)
            self.reset_A_parameters(w_a_linear_v)
            self.reset_B_parameters(w_b_linear_v)
            if self.ema_mode:
                ema_w_a_linear_q = copy.deepcopy(w_a_linear_q)
                ema_w_b_linear_q = copy.deepcopy(w_b_linear_q)
                ema_w_a_linear_v = copy.deepcopy(w_a_linear_v)
                ema_w_b_linear_v = copy.deepcopy(w_b_linear_v)
            
            if self.ema_mode and self.truncation:
                self.index = Dash_index
                blk.attn.qkv = _LoRAtruncation_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    ema_w_a_linear_q,
                    ema_w_b_linear_q,
                    ema_w_a_linear_v,
                    ema_w_b_linear_v,
                    index=self.index,
                )
        self.sam = sam_model
    
    def reset_A_parameters(self, w_A) -> None:
        nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
    
    def reset_B_parameters(self, w_B) -> None:
        nn.init.zeros_(w_B.weight)
    
    def forward(self, batched_input, multimask_output, image_size, ema=False):
        return self.sam(batched_input, multimask_output, image_size, ema=ema)
