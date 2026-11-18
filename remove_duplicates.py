import hashlib, os
from collections import defaultdict

hashes = defaultdict(list)
for c in ["no", "yes"]:
    folder = f"Veri Seti/{c}"
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                hashes[hashlib.md5(fh.read()).hexdigest()].append(p)

silinen = 0
for v in hashes.values():
    for p in v[1:]:      # ilkini tut, kalanları sil
        os.remove(p)
        silinen += 1

print("Silinen dosya sayısı:", silinen)
for c in ["no", "yes"]:
    print(c, len(os.listdir(f"Veri Seti/{c}")))