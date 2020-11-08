"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import glob
import torch
import pandas as pd
from torch.backends import cudnn
from matplotlib import colors


from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


############## REPLACEABLE PARAMETERS ################################

compound_coef = 6
force_input_size = 1024  # set None to use default size, 1024 is the size required by DHAKA AI Outputs
run_number = 1

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.4
iou_threshold = 0.4

weight_loc = '../drive/My Drive/logs/'+str(run_number)+'/dhaka_ai_v2/efficientdet-d'+str(compound_coef)+'.pth'
############## KEEP IT AS IT IS ######################################

list_image_id = []
list_class = []
list_score = []
list_xmin = []
list_ymin = []
list_xmax = []
list_ymax = []
list_width = 1024
list_height = 1024

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ["rickshaw","car","three wheelers (CNG)","bus","motorbike",
            "wheelbarrow","bicycle","auto rickshaw","truck",
            "pickup","minivan","human hauler","suv","van","minibus",
            "ambulance","taxi","army vehicle","scooter","policecar","garbagevan"]

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                            ratios=anchor_ratios, scales=anchor_scales)
                            
model.load_state_dict(torch.load(f'{weight_loc}', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()
    
for img_path in glob.iglob('test/*.jpg'):

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    def display(preds, imgs, imshow=True, imwrite=False):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            imgs[i] = imgs[i].copy()

            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                list_image_id.append(img_path.split('/')[-1])
                list_class.append(obj)
                list_score.append(score)
                list_xmin.append(x1)
                list_ymin.append(y1)
                list_xmax.append(x2)
                list_ymax.append(y2)
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])



    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=False)

    # print('running speed test...')
    # with torch.no_grad():
    #     print('test1: model inferring and postprocessing')
    #     print('inferring image for 10 times...')
    #     t1 = time.time()
    #     for _ in range(10):
    #         _, regression, classification, anchors = model(x)

    #         out = postprocess(x,
    #                         anchors, regression, classification,
    #                         regressBoxes, clipBoxes,
    #                         threshold, iou_threshold)
    #         out = invert_affine(framed_metas, out)

    #     t2 = time.time()
    #     tact_time = (t2 - t1) / 10
    #     print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

        # uncomment this if you want a extreme fps test
        # print('test2: model inferring only')
        # print('inferring images for batch_size 32 for 10 times...')
        # t1 = time.time()
        # x = torch.cat([x] * 32, 0)
        # for _ in range(10):
        #     _, regression, classification, anchors = model(x)
        #
        # t2 = time.time()
        # tact_time = (t2 - t1) / 10
        # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')

pd.DataFrame({
    'image_id':list_image_id,
    'class':list_class,
    'score':list_score,
    'xmin':list_xmin,
    'ymin':list_ymin,
    'xmax':list_xmax,
    'ymax':list_ymax,
    'width':list_width,
    'height':list_height
}).to_csv('../drive/My Drive/dhaka-ai_'+run_number+"_"+time.ctime()+'.csv',index=False)