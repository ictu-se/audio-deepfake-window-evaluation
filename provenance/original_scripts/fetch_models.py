import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'sources'
    t = json.loads((source/'tssd_head.json').read_text())['sha']
    v = json.loads((source/'silero_head.json').read_text())['sha']
    info = json.loads((source/'nii_small_info.json').read_text())
    n = info['sha']
    tree = json.loads((source/'tssd_tree.json').read_text())['tree']
    res = next(x for x in tree if x['path'].startswith('pretrained/Res_TSSDNet_') and x['path'].endswith('.pth'))
    vad = next(x for x in json.loads((source/'silero_tree.json').read_text())['tree'] if x['path'].endswith('silero_vad.jit'))
    ni = next(x for x in info['siblings'] if x['rfilename']=='model.safetensors')
    tasks = [dict(name='res_tssd.pth',url='https://raw.githubusercontent.com/ghua-ac/end-to-end-synthetic-speech-detection/'+t+'/'+urllib.parse.quote(res['path']),size=res['size'],blob=res['sha']),
        dict(name='silero_vad.jit',url='https://raw.githubusercontent.com/snakers4/silero-vad/'+v+'/'+vad['path'],size=vad['size'],blob=vad['sha']),
        dict(name='antideepfake_small.safetensors',url='https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake/resolve/'+n+'/model.safetensors',size=ni['size'],sha256=ni['lfs']['sha256'])]
    def fetch(item):
        path = ROOT/'models'/item['name']
        if not path.exists():
            partial = path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(item['url'],timeout=60) as response, partial.open('wb') as target:
                while True:
                    chunk = response.read(4*1024*1024)
                    if not chunk:break
                    target.write(chunk)
            os.replace(partial,path)
        body = path.read_bytes()
        assert len(body)==item['size']
        digest = hashlib.sha256(body).hexdigest()
        if 'sha256' in item:assert digest==item['sha256']
        if 'blob' in item:assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==item['blob']
        return dict(**item,verified_sha256=digest,integrity_verified=True)
    records=list(concurrent.futures.ThreadPoolExecutor(3).map(fetch,tasks))
    (ROOT/'sources/model_integrity.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('name','size','verified_sha256','integrity_verified')} for r in records],indent=2))


if __name__=='__main__':main()
