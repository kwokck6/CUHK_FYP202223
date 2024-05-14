from datasets import ShopeeCL
import numpy as np
import torch

from contrastive_loss import ContrastiveLoss

torch.random.manual_seed(1028)
batch_size = 32
target = torch.randint(0, 2, (batch_size,))
print(target)

for i in range(5):
    img1_embed = torch.rand((batch_size, 128))
    img2_embed = torch.rand((batch_size, 128))
    loss_fn = ContrastiveLoss(margin=1.0)
    loss = loss_fn(img1_embed, img2_embed, target)
    print(loss)
