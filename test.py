import torch
from collections import OrderedDict
old = torch.load("logs/logs_20230514/model050000.pt",map_location='cpu')
new = OrderedDict()
for k in old:
    print(k)
    k_new = k
    if k_new == "label_emb.weight":
        k_new = "content_emb.weight"
    new[k_new] = old[k]
torch.save(new,"logs/logs_20230514/model050000_new.pt")