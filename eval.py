from os.path import join
from tqdm import tqdm
import torch
import argparse
import sys

from vCycle import Cyclic
from utils import *


parser = argparse.ArgumentParser(description="PyTorch LapSRN Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--gpuID", default=0, type=int, help="GPU ID")
parser.add_argument("--model", default="model.pt", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--patch", default=800, type=int, help="image patch size")



opt = parser.parse_args()
model_dir = opt.model
tile_dim = opt.patch
overlap_dim = 10

device = torch.device("cuda:%i" %opt.gpuID if opt.cuda else "cpu")

model = Cyclic().to(device)
cp = torch.load(model_dir)
model.load_state_dict(cp)

img_dir = opt.dataset #'/home/falmasri/Desktop/Datasets/AIM2020/RISRC(x2)/TestLR'
test_loader = AIM20(img_dir)


model.eval()
test_bar = tqdm(test_loader)
for im_lr, fname in test_bar:
    batch_size = im_lr.size(0)


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

