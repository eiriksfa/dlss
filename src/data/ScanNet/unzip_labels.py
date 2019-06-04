from pathlib import Path
from zipfile import ZipFile

p = Path('F:/data/ScanNet/scans/')
n = '_2d-label-filt.zip'
for p2 in p.iterdir():
    p3 = p2 / (p2.stem + n)
    print(str(p2))
    with ZipFile(p3, 'r') as z:
        z.extractall(p2)
