import torch
from vggt.models.vggt import VGGT  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 这里是不是应该设计成一个类，方便后续扩展

# [ ]: 这里是在运行时通过预训练模型计算 vggt 还是预先计算好存储在硬盘上?
    # 如果预先计算好存储起来, 那么我应该写一个 preprocess 相关的东西, 来完成这个事情

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)




