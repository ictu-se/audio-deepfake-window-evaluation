"""Fetch pinned source and weights on explicit request; verify model fingerprints."""
import hashlib,io,json,tarfile,urllib.request,zipfile,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def fetch(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  partial=path.with_suffix('.partial')
  with urllib.request.urlopen(url,timeout=120) as source,partial.open('wb') as out:
   for data in iter(lambda:source.read(4*1024*1024),b''):out.write(data)
  partial.replace(path)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
 return h.hexdigest()
def main():
 study=ROOT/'study';core=ROOT/'core';models=study/'models';models.mkdir(exist_ok=True)
 info=json.loads((study/'sources/nii_small_info.json').read_text())
 v=json.loads((study/'sources/silero_head.json').read_text())['sha']
 fetch(f"https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake/resolve/{info['sha']}/model.safetensors",models/'antideepfake_small.safetensors')
 fetch(f'https://raw.githubusercontent.com/snakers4/silero-vad/{v}/src/silero_vad/data/silero_vad.jit',models/'silero_vad.jit')
 revision=json.loads((core/'vendor/source_revisions.json').read_text())
 aa=revision['aasist']['commit']
 fetch(f'https://raw.githubusercontent.com/clovaai/aasist/{aa}/models/weights/AASIST.pth',core/'vendor/aasist/models/weights/AASIST.pth')
 record=json.loads((study/'sources/rawnet2_download.json').read_text());archive=models/'rawnet2.zip';fetch(record['url'],archive)
 assert digest(archive)==record['archive_sha256']
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  (models/'rawnet2.pth').write_bytes(z.read(record['member']))
 fair=core/'vendor/fairseq_v0122'
 if not fair.exists():
  commit=revision['fairseq']['commit'];data=urllib.request.urlopen(f'https://codeload.github.com/facebookresearch/fairseq/tar.gz/{commit}',timeout=120).read()
  with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
   for m in tar:
    rel=Path(*Path(m.name).parts[1:]);target=fair/rel
    assert '..' not in rel.parts
    if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(m).read())
 fetch('https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt',core/'checkpoints/xlsr2_300m.pt')
 target=core/'checkpoints/Best_LA_model_for_DF.pth'
 if not target.exists():
  import gdown
  gdown.download(id='1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG',output=str(target),quiet=False)
 config=json.loads((study/'frozen_run.json').read_text())
 paths={'aasist':core/'vendor/aasist/models/weights/AASIST.pth','ssl':target,'rawnet2':models/'rawnet2.pth','antideepfake':models/'antideepfake_small.safetensors'}
 for name,path in paths.items():assert digest(path)==config['models'][name]['checkpoint_sha256'],name
 subprocess.run([sys.executable,'core/prepare_ssl_checkpoint.py'],cwd=ROOT,check=True)
 print('Pinned classifier checkpoints verified. Downloads are not redistributed in this repository.')
if __name__=='__main__':main()
