import glob
import os

crops = sorted(glob.glob("outputs/anpr_debug/*TRK_106*"))
for c in crops:
    print(c)
