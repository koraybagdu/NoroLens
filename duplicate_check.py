import hashlib, os
from collections import defaultdict

hashes = defaultdict(list)
for c in ["no", "yes"]:
    folder = f"Veri Seti/{c}"
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                hashes[hashlib.md5(fh.read()).hexdigest()].append(p)

dups = {h: v for h, v in hashes.items() if len(v) > 1}
print("Tekrar eden grup sayısı:", len(dups))
print("Fazla kopya sayısı:", sum(len(v) - 1 for v in dups.values()))

# Etiket çelişkisi: aynı görsel hem no hem yes klasöründe
conflicts = [v for v in dups.values()
             if len({p.split("/")[1] for p in v}) > 1]
print("no ve yes arasında çelişen grup:", len(conflicts))

# İlk 5 grubu göster
for v in list(dups.values())[:5]:
    print(v)