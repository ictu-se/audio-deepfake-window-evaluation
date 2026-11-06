import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CORE = PROJECT / 'core'
SPEECH = PROJECT
CACHE = PROJECT / 'workspace/cache'
RAW = PROJECT / 'workspace/audio'
MODELS = ('aasist', 'ssl', 'rawnet2', 'antideepfake')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def load(path):return json.loads(Path(path).read_text())
