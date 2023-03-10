import torch
from vit_pytorch.recurrent_vit import MaxViT

v = MaxViT()

img = torch.randn(5, 20, 384, 640)

preds = v(img)
