# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained  model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov3.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import ipdb 

import numpy as np
import torch
from tqdm import tqdm
from numpy.linalg import eig, pinv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, NCOLS, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

from utils.datasets import LoadImagesAndLabels
from utils.plots import plot_images

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

CROP_IMAGE_X = [293, 803]
CROP_IMAGE_Y = [134, 637]

def crop_image(img):
    return img[CROP_IMAGE_Y[0]:CROP_IMAGE_Y[1],CROP_IMAGE_X[0]:CROP_IMAGE_X[1]]   

def binary_search(thresh, labels):

    x = []
    y = []
    dx = [-1, 0, 1]
    dy = [-1, 0, 1]
    threshold = 1

    for label in labels:
        [x1, y1, x2, y2, _, _] = label
        center = [int((x2+x1)/2), int((y2+y1)/2)]
        # print('label', label)
        i, j = 0, 0
        while i < len(dx) and j < len(dy):
            if dx[i] == 0 and dy[j] == 0:
                i += 1
                continue
            elif dy[j] != 0: 
                min = 0
                max = int((y2-y1)/2)
            elif dx[i] != 0: 
                min = 0
                max = int((x2-x1)/2)
            else:
                min = 0
                max = int(np.sqrt(((y2-y1)/2)**2 + ((x2-x1)/2)**2))

            found = False
            count = 0

            while not found:
                step = int((max+min)/2)
                _x = center[0] + step*dx[i]
                _y = center[1] + step*dy[j]
                pdx, ndx = _x + threshold*dx[i], _x - threshold*dx[i]
                pdy, ndy = _y + threshold*dy[j], _y - threshold*dy[j]

                if count<2:
                    if thresh[_y, _x] == 0 and thresh[pdy, pdx] == 0 and thresh[ndy, ndx] == 0:
                        min = step
                        count += 1
                        continue
                    if thresh[_y, _x] != 0 and thresh[pdy, pdx] != 0 and thresh[ndy, ndx] != 0:
                        max = step
                        count += 1
                        continue

                found = True
                # print('FOUND. count', count)
                x.append(_x)
                y.append(_y)
                count += 1

            i += 1
            if i == len(dx):
                i = 0
                j += 1

    # print('out of main loop')
    return x, y

def fit_ellipse(xx, yy):

    x = xx[:,np.newaxis]
    y = yy[:,np.newaxis]

    J = np.hstack((x*x, x*y, y*y, x, y))
    K = np.ones_like(x)

    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=pinv(JTJ)
    ABC= np.dot(InvJTJ, np.dot(JT,K))
    v=np.append(ABC,-1)

    Amat = np.array([[v[0],     v[1]/2.0, v[3]/2.0],
                    [v[1]/2.0, v[2],     v[4]/2.0],
                    [v[3]/2.0, v[4]/2.0, v[5]    ]])

    A2=Amat[0:2,0:2]
    A2Inv=pinv(A2)
    ofs=v[3:5]/2.0
    cc = -np.dot(A2Inv,ofs)

    Tofs=np.eye(3)
    Tofs[2,0:2]=cc
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))

    R2=R[0:2,0:2]
    s1=-R[2, 2]
    RS=R2/s1
    (el,ec)=eig(RS)

    recip=1.0/np.abs(el)
    axes=np.sqrt(recip)

    rads=np.arctan2(ec[1,0],ec[0,0])
    deg=np.degrees(rads)
    inve=pinv(ec)

    return [cc[0], cc[1], axes[0], axes[1], deg, inve]

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt = next(model.parameters()).device, True  # get model device, PyTorch model

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        # Data
        print("check shape of data")
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        print("skipping dataloader and sending a single image")

    ####################################################################
    '''
    REPLACED BELOW WITH IMAGE FROM ULTRASOUND IMAGE TOPIC
    '''
    ####################################################################
    # size is not being an issue when sending to model below 
    # im = torch.zeros(1, 3, 480, 512)
    import os
    import cv2
    import copy

    test_dir = '/home/tejasr/projects/tracir_segmentation/yolov3/dataset/test'
    num_test_img = 1
    images = os.listdir(test_dir+'/images')

    for i in range(num_test_img):
        # img_path = os.path.join(test_dir, 'images', images[i])
        # img = cv2.imread(img_path)
        img = cv2.imread('/home/tejasr/projects/tracir_segmentation/yolov3/dataset/test/images/pig_7_251.png')
        result = np.zeros_like(img)
        im = torch.Tensor(img).to(device)
        im = im.permute(2, 0, 1)
        im = torch.unsqueeze(im, dim=0)

        dt = [0.0, 0.0, 0.0]
        t1 = time_sync()
        if pt:
            im = im.to(device, non_blocking=True)
            # targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # NMS
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=0, multi_label=False, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Plot images
        # directly call below function 
        # fname = os.path.join(save_dir, 'val_batch{}_pred.jpg'.format(i))
        # fname = os.path.join(save_dir, images[i])
        fname = os.path.join(save_dir, 'pig_7_251.png')
        plot_images(im, output_to_target(out), paths=None, fname=fname, names=None, max_size=1920, max_subplots=16)
        print('i', i)
        print('image', images[i])

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = (((gray_image/255)**1.5)*255).astype(np.uint8)
        ret, thresh = cv2.threshold(thresh, 30, 150, cv2.THRESH_BINARY)
        labels = out[0].cpu().numpy().astype('int')
        x, y = binary_search(thresh, labels)
        ipdb.set_trace()
        for j in range(int(len(x)/8)):
            xx = np.array(x[j*8:(j+1)*8])
            yy = np.array(y[j*8:(j+1)*8])
            # ipdb.set_trace()
            [cx, cy, a, b, deg, _] = fit_ellipse(xx, yy)
            # result = cv2.ellipse(result, (int(cx), int(cy)), (int(a), int(b)), int(deg), 0, 360, color=(255, 0, 0), thickness=1)
            result = cv2.ellipse(result, (int(cx), int(cy)), (int(a), int(b)), int(deg), 0, 360, color=(255, 0, 0), thickness=-1)
        # fname2 = os.path.join(save_dir, 'ellipse_'+images[i])
        fname2 = os.path.join(save_dir, 'mask_'+'pig_7_251.png')
        print('ellipse')
        cv2.imwrite(fname2, result)

        # Plots
        if plots:
            # confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            callbacks.run('on_val_end')




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') # 0.001
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold') # 0.6
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov3.pt yolov3-spp.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov3.pt yolov3-spp.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
