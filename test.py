import numpy as np
from os.path import join
from PIL import Image
from tqdm import tqdm
import torch

from vCycle import Cyclic
from utils import *



gpuID = 0
use_gpu = True
model_dir = 'test.pt'

device = torch.device("cuda:%i" %gpuID if use_gpu else "cpu")

model = Cyclic().to(device)
cp = torch.load(model_dir)
model.load_state_dict(cp)

imgDir = '/home/falmasri/Desktop/Datasets/AIM2020/RISRC(x2)/TestLR'
test_loader = AIM20(imgDir)


model.eval()
test_bar = tqdm(test_loader)
for im_lr, fname in test_bar:
    batch_size = im_lr.size(0)

    tile_dim = 800
    overlap_dim = 10

    imgslst = make_tiles(im_lr[0].permute(1,2,0).data.numpy(), tile_dim, overlap_dim)

    restoredSRlst = []
    for f in imgslst:
        f = torch.FloatTensor(f).unsqueeze(0).permute(0,3,1,2).to(device)
        f = ((f * 2) - 1)
        with torch.no_grad():
            output0, _ = model(f)
            sr = (output0[0] + 1) / 2
            restoredSRlst.append(sr.permute(1, 2, 0).data.cpu().numpy())

    img = merge_tiles(restoredSRlst, overlap_dim * 2, (im_lr.shape[2] * 2, im_lr.shape[3] * 2, im_lr.shape[1]))

    mainSR = np.clip(img * 255, 0, 255).round().astype(np.uint8)
    im = Image.fromarray(mainSR)
    im.save(join('imgs', fname[0]), "PNG", optimize=False, compress_level=0)



