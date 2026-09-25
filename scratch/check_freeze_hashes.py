import hashlib

files = [
    'backend/app/perception/frame_source.py',
    'backend/app/perception/detector.py',
    'backend/app/perception/bytetrack.py',
    'backend/app/perception/plate_association.py',
]

for f in files:
    h1 = hashlib.sha256(open(f, 'rb').read()).hexdigest().upper()
    h2 = hashlib.sha256(open('checkpoints/POINT_D_STAGE4_FIX1_FREEZE/' + f, 'rb').read()).hexdigest().upper()
    status = "MATCH" if h1 == h2 else "MISMATCH"
    print(f"{f}: {status} (Active: {h1}, Freeze: {h2})")
