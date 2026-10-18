"""Fetch selected members of MIDV-2020 photo.tar by FTP byte ranges (no full download).
Selects even-numbered photos (00,02,...,98) of every document type = 500 photos."""
import subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor
idx, out = sys.argv[1], sys.argv[2]
URL = 'ftp://smartengines.com/midv-2020/dataset/photo.tar'
jobs = []
for line in open(idx):
    name, off, size, typ = line.rstrip('\n').split('\t')
    if typ != '0' or not name.startswith('images/') or not name.endswith('.jpg'): continue
    if int(os.path.basename(name)[:2]) % 2: continue
    dst = os.path.join(out, name)
    if os.path.exists(dst) and os.path.getsize(dst) == int(size): continue
    jobs.append((name, int(off), int(size), dst))
print(len(jobs), 'to fetch', sum(j[2] for j in jobs) / 1e9, 'GB')
def get(j):
    name, off, size, dst = j
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(4):
        subprocess.run(['curl', '-s', '--max-time', '300', '-r', f'{off}-{off+size-1}', URL, '-o', dst + '.part'])
        if os.path.exists(dst + '.part') and os.path.getsize(dst + '.part') == size:
            os.rename(dst + '.part', dst); return 1
    return 0
with ThreadPoolExecutor(8) as ex: print('ok', sum(ex.map(get, jobs)))
