"""Download only frozen candidate WAV members from the author's pinned ZIP.

Each member uses two bounded requests, exact HTTP ranges, raw-deflate decoding,
and the upstream ZIP CRC32 and size. Never claim the whole ZIP SHA was checked.
"""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request
import zlib

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parents[1] / 'data/raw/itw_20261106/wav'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    plan = json.loads((HERE / 'external_plan.json').read_text())
    assert digest((HERE / 'candidate_manifest.csv').read_bytes()) == plan['candidate_manifest_sha256']
    with (HERE / 'candidate_manifest.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    repo = json.loads((HERE / 'source/hf_repository.json').read_text())
    size = next(r['size'] for r in repo['siblings'] if r['rfilename'].endswith('.zip'))
    url = plan['source']
    RAW.mkdir(parents=True, exist_ok=True)

    def fetch_range(first, last):
        assert 0 <= first <= last < size and last-first < 4*1024*1024
        for attempt in range(6):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=45) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {first}-{last}/{size}'
                    body = response.read(last-first+2)
                assert len(body) == last-first+1
                return body
            except Exception as error:
                if attempt == 5:
                    raise
                delay = int(error.headers.get('Retry-After', '5')) if hasattr(error, 'headers') else 5
                time.sleep(max(5, min(delay, 300)))

    def member(row):
        target = RAW / row['file']
        proof = target.with_suffix('.json')
        n, crc = int(row['file_size']), int(row['CRC'])
        if target.exists() and proof.exists():
            data = target.read_bytes()
            known = json.loads(proof.read_text())
            assert len(data) == n and zlib.crc32(data) == crc and digest(data) == known['sha256']
            return target
        offset, length = int(row['header_offset']), int(row['compress_size'])
        header = fetch_range(offset, offset+29)
        signature, version, flags, method, mt, md, c, cs, us, nl, xl = struct.unpack('<4s5H3I2H', header)
        assert signature == b'PK\x03\x04' and not flags & 1 and method == int(row['compress_type'])
        block = fetch_range(offset+30, offset+30+nl+xl+length-1)
        assert block[:nl].decode('utf-8') == row['filename']
        compressed = block[nl+xl:]
        assert len(compressed) == length
        data = zlib.decompress(compressed, -15) if method == 8 else compressed
        assert method in (0, 8) and len(data) == n and zlib.crc32(data) == crc
        temp = target.with_suffix('.incoming')
        temp.write_bytes(data)
        os.replace(temp, target)
        proof.write_text(json.dumps(dict(bytes=n, crc32=crc, sha256=digest(data),
            commit=plan['upstream_commit'], archive_member=row['filename'], range_header_offset=offset,
            integrity='Verified HTTP Content-Range, uncompressed size, ZIP CRC32; local SHA256'), indent=2) + '\n')
        return target

    started, done = time.time(), 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(member, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == len(rows):
                temp = HERE / 'download_status.tmp'
                result = dict(verified_candidates=done, expected_candidates=len(rows), seconds=time.time()-started,
                    complete=done == len(rows), integrity='Per-member ZIP CRC32 and local SHA256; whole ZIP not downloaded')
                temp.write_text(json.dumps(result, indent=2) + '\n')
                os.replace(temp, HERE / 'download_status.json')
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
