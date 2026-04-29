# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:30:43 2026

@author: user
"""

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import copy
import math
import pkg_resources
import re
from pathlib import Path
import torch
import cv2
import numpy as np

from models.build import BuildNet
from utils.version_utils import digit_version
from utils.train_utils import file2dict
from utils.misc import to_2tuple
from utils.inference import init_model
from core.datasets.compose import Compose
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm

try:
    from pytorch_grad_cam import (
        EigenCAM, EigenGradCAM, GradCAM,
        GradCAMPlusPlus, LayerCAM, XGradCAM
    )
    from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        'Please run `pip install "grad-cam>=1.3.6"`'
    )

FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--target-layers', default=[], nargs='+', type=str)
    parser.add_argument('--preview-model', action='store_true')
    parser.add_argument('--method', default='gradcam')
    parser.add_argument('--target-category', default=[], nargs='+', type=int)
    parser.add_argument('--eigen-smooth', action='store_true')
    parser.add_argument('--aug-smooth', action='store_true')
    parser.add_argument('--save-path', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--vit-like', action='store_true')
    parser.add_argument('--num-extra-tokens', type=int)
    return parser.parse_args()


def apply_transforms(img_path, pipeline_cfg):
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        image_t, format_t = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for t in pipeline_cfg:
            (format_t if t['type'] in FORMAT_TRANSFORMS_SET else image_t).append(t)
        return image_t, format_t

    image_t, format_t = split_pipeline_cfg(pipeline_cfg)
    image_t = Compose(image_t)
    format_t = Compose(format_t)

    mid = image_t(data)
    src_img = copy.deepcopy(mid['img'])
    data = format_t(mid)
    return data, src_img


class MMActivationsAndGradients(ActivationsAndGradients):
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x, return_loss=False, softmax=False, post_process=False)


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    cam_cls = METHOD_MAP[method]
    cam = cam_cls(model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.activations_and_grads.release()
    cam.activations_and_grads = MMActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform
    )
    return cam


def show_cam_grad(grayscale_cam, src_img, method, img_path):
    grayscale_cam = grayscale_cam[0]
    src_img = np.float32(src_img)[:, :, ::-1] / 255.0
    vis_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)

    os.makedirs("outputs2", exist_ok=True)
    img_name = Path(img_path).stem
    save_path = Path("outputs2") / f"cam_{method}_{img_name}.png"
    cv2.imwrite(str(save_path), vis_img)
    print(f"[✓] Saved: {save_path}")


def get_default_traget_layers(model):
    norms = [m for m in model.backbone.modules()
             if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d))]
    return [norms[-1]]


def run_all_cams(model, data, src_img, target_layers, args):
    use_cuda = 'cuda' in args.device
    reshape_transform = None

    targets = None
    if args.target_category:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(c) for c in args.target_category]

    for method in METHOD_MAP.keys():
        print(f"\n[▶] Running {method.upper()} ...")
        cam = init_cam(method, model, target_layers, use_cuda, reshape_transform)
        grayscale_cam = cam(
            data['img'].unsqueeze(0),
            targets,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth
        )
        show_cam_grad(grayscale_cam, src_img, method, args.img)


def main():
    args = parse_args()

    if args.config.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        model_cfg = cfg.model_cfg
        val_pipeline = cfg.val_pipeline
        data_cfg = cfg.data_cfg
    else:
        model_cfg, _, val_pipeline, data_cfg, _, _ = file2dict(args.config)

    device = torch.device(args.device)
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    if args.preview_model:
        print(model)
        return

    data, src_img = apply_transforms(args.img, val_pipeline)
    target_layers = get_default_traget_layers(model)

    run_all_cams(model, data, src_img, target_layers, args)


if __name__ == '__main__':
    main()
