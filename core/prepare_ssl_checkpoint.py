"""Make the official pretraining config loadable by Fairseq 0.12.2.

Follows the organizer's fix_xlsr2.py: remove unsupported *task config* keys.
No model tensor is modified, no adaptation is performed.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf,open_dict

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor/fairseq_v0122'))
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    original=HERE/'checkpoints/xlsr2_300m.pt';target=HERE/'checkpoints/xlsr2_300m_runtime.pt'
    report=target.with_suffix('.config_fix.json')
    if target.exists() and report.exists():return
    cp=torch.load(original,map_location='cpu')
    cfg=OmegaConf.create(cp['cfg'])
    supported={f.name for f in dataclasses.fields(AudioPretrainingConfig)}|{'_name'}
    removed=[k for k in cfg.task.keys() if k not in supported]
    # Hash tensors before config-only editing; byte strings are not rewritten.
    tensors=hashlib.sha256()
    for key,tensor in sorted(cp['model'].items()):
        if torch.is_tensor(tensor):tensors.update(key.encode());tensors.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    before=tensors.hexdigest()
    with open_dict(cfg.task):
        for key in removed:del cfg.task[key]
    cp['cfg']=OmegaConf.to_container(cfg,resolve=True)
    torch.save(cp,target)
    check=torch.load(target,map_location='cpu');after=hashlib.sha256()
    for key,tensor in sorted(check['model'].items()):
        if torch.is_tensor(tensor):after.update(key.encode());after.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    assert before==after.hexdigest()
    report.write_text(json.dumps({'original_sha256':sha(original),'runtime_sha256':sha(target),'removed_task_config_keys':removed,
                                 'model_tensor_sha256':before,'all_model_tensors_identical':True,
                                 'organizer_reference':'vendor/radar/fix_xlsr2.py'},indent=2)+'\n')
    print('Task config compatibility fixed; all model tensors unchanged',flush=True)


if __name__=='__main__':main()
