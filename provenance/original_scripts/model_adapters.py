import importlib.util
import sys
import numpy as np
import torch
import torch.nn.functional as F
from common import *

sys.path.insert(0,str(ROOT/'_deps'))
sys.path.insert(0,str(CORE))
sys.path.insert(0,str(CORE/'vendor/fairseq_v0122'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class NiiFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from fairseq.models.wav2vec import Wav2Vec2Model,Wav2Vec2Config
        cfg=Wav2Vec2Config(encoder_layers=12,encoder_embed_dim=768,quantize_targets=True,latent_dim=256,final_dim=256)
        self.model=Wav2Vec2Model(cfg)

    def forward(self,x):
        return self.model(x,mask=False,features_only=True)['x']


class NiiDetector(torch.nn.Module):
    def __init__(self):
        super().__init__();self.m_ssl=NiiFrontend();self.adap_pool1d=torch.nn.AdaptiveAvgPool1d(1);self.proj_fc=torch.nn.Linear(768,2)

    def forward(self,x):
        # Per-input normalization follows the author's inference example.
        x=F.layer_norm(x,(x.shape[-1],))
        emb=self.m_ssl(x).transpose(1,2)
        return self.proj_fc(self.adap_pool1d(emb).squeeze(-1))


def build(name,device):
    if name in ('aasist','ssl'):
        from shortform import build_model
        model,path=build_model(name,device)
    elif name=='rawnet2':
        import yaml
        cls=module('expanded_rawnet',ROOT/'sources/rawnet_model.py').RawNet
        config=yaml.safe_load((ROOT/'sources/rawnet_config.yaml').read_text())['model']
        model=cls(config,device)
        path=ROOT/'models/rawnet2.pth'
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True);model.to(device).eval()
    elif name=='antideepfake':
        from safetensors.torch import load_file
        path=ROOT/'models/antideepfake_small.safetensors'
        model=NiiDetector();model.load_state_dict(load_file(str(path),device='cpu'),strict=True);model.to(device).eval()
    else:raise ValueError(name)
    assert not any(m.training for m in model.modules())
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,path


def output(model,name,batch):
    pair=model(batch);pair=pair[1] if name=='aasist' else pair
    assert pair.ndim==2 and pair.shape[1]==2 and torch.isfinite(pair).all()
    if name in ('aasist','rawnet2'):score=-pair[:,1]
    elif name=='antideepfake':score=pair[:,0]-pair[:,1]
    else:score=pair[:,0]
    return score,pair


def sync(device):
    if device=='mps':torch.mps.synchronize()
