import os
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json 
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, display
from PIL import Image, ImageDraw, ImageFont
FONT_PATH = "../../fonts/arial.ttf"


compound_coef = 2
force_input_size = 640  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
obj_list = ['Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ',
            'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

test_list = os.listdir("datasets/traffic_2020/traffic_public_test/images")

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             anchors_scales=[0.625, 1.4375, 0.28125],
                             anchors_ratios=[(1.0, 1.0), (1.0, 0.9782608695652174), (1.0, 0.95)])

model.load_state_dict(torch.load('logs/traffic_2020/' +
                                 'efficientdet-d2_26_59591.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

all_img_preds = []

img_path = [os.path.join(
    "datasets/traffic_2020/traffic_public_test/images", img_) for img_ in test_list]

for img in img_path:
    print(f'[INFO] Processing {img}...')
    ori_imgs, framed_imgs, framed_metas = preprocess(
        img, max_size=input_size)

    img_id = os.path.basename(img).split(".")[0]

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(
        0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    objs_json = display(img_id, out, ori_imgs, obj_list, imshow=False, imwrite=True)
    all_img_preds.extend(objs_json)

with open('submission.json', 'w') as f:
    json.dump(all_img_preds, f)