#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : common_utils.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""

"""
import yaml
import numpy as np
import matplotlib.pyplot as plt

class EvalConfig(dict):
    def __getattr__(self,val):
        rv = self[val]
        if isinstance(rv, dict):
            return EvalConfig(rv)
        return rv


def read_yaml(file_name):
    with open(file_name, 'r') as f:
        config = yaml.safe_load(f.read())
    return EvalConfig(config)


def crop_image(im, mask, margin_pixel=10, return_bbox=False, pad_to_square=True):
    h, w = im.shape[:2]
    bbox_y, bbox_x = np.where(mask > 0)
    ymin, ymax, xmin, xmax = max(0, bbox_y.min() - margin_pixel), min(bbox_y.max() + margin_pixel, h), \
                             max(0, bbox_x.min() - margin_pixel), min(bbox_x.max() + margin_pixel, w)
    if pad_to_square:
        xrange = xmax - xmin
        yrange = ymax - ymin
        if xrange < yrange:
            short_edge = 'x'
            minval, maxval = xmin, xmax
            maxlimit = w - 1
        else:
            short_edge = 'y'
            minval, maxval = ymin, ymax
            maxlimit = h - 1
        pad_val = abs(xrange - yrange)
        pad_side1, pad_side2 = pad_val // 2, pad_val - pad_val // 2
        minval -= pad_side1
        maxval += pad_side2
        if minval < 0:
            shift_delta = abs(minval)
            maxval += shift_delta
            minval = 0
            if maxval > maxlimit:
                maxval = maxlimit
        elif maxval > maxlimit:
            shift_delta = maxval - maxlimit
            maxval = maxlimit
            minval -= shift_delta
            if minval < 0:
                minval = 0
        if short_edge == 'x':
            xmin, xmax = minval, maxval
        else:
            ymin, ymax = minval, maxval

    if return_bbox:
        return im[ymin:ymax, xmin:xmax], ymin, ymax, xmin, xmax
    return im[ymin:ymax, xmin:xmax]


def plot_images(images, titles=None):
    fig, axes = plt.subplots(1, len(images), figsize=(30, 10), squeeze=False)
    axes = axes[0]
    for i, im in enumerate(images):
        axes[i].imshow(im)
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=12, wrap=True)
    plt.show()
    plt.close()


def overlay_mask_simple(rgb_im, mask: np.ndarray, colors=None, mask_alpha=.5):
    if rgb_im.max() > 2:
        rgb_im = rgb_im.astype(np.float32) / 255.
    if colors is None:
        colors = np.array([1, 0, 0])
    return (rgb_im * (1 - mask_alpha) + mask[..., np.newaxis] * colors * mask_alpha).copy()


def show_image_with_mask(im, mask):
    im_with_mask = overlay_mask_simple(im, mask)
    plot_images([im, im_with_mask])
