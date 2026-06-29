#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : perception.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""
Perception Interface
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from uncos import UncOS

from beepp.gemini_client import GeminiClient
from beepp.utils.common_utils import crop_image


def transform_pointcloud(transform_mat, input_pointcloud, set_invalid=False):
    """
    Transform a point cloud with tform mat
    Args:
        transform_mat:      4 x 4
        input_pointcloud:   N x 3

    Returns:
        transformed point cloud
    """
    original_pointcloud_hom = np.concatenate((input_pointcloud, np.ones((input_pointcloud.shape[0], 1))), axis=1)
    transformed_pointcloud = transform_mat.dot(original_pointcloud_hom.T)[:3].T
    if set_invalid:
        transformed_pointcloud[(input_pointcloud == 0).all(axis=1)] = 0
    return transformed_pointcloud


def point_cloud_from_depth_image_camera_frame(depth_image, camera_intrinsics):
    """
    Project depth image back to 3D to obtain partial point cloud.
    """
    height, width = depth_image.shape
    xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coord = np.concatenate((xmap.reshape(1, -1), ymap.reshape(1, -1), np.ones((1, height * width))))
    rays = np.linalg.inv(camera_intrinsics).dot(homogenous_coord)
    point_cloud = depth_image.reshape(1, height * width) * rays
    point_cloud = point_cloud.T.reshape(height, width, 3)
    return point_cloud


@dataclass
class RGBDObservation:
    def __init__(self, rgb_im, depth_im, intrinsic, extrinsic=None):
        self.rgb_im = rgb_im
        self.depth_im = depth_im
        self.intrinsic = intrinsic
        if extrinsic is None:
            extrinsic = np.eye(4)
        self.extrinsic = extrinsic

    @cached_property
    def pcd_cameraframe(self):
        """
        Convert the RGBD image to a point cloud in the world frame.
        Returns:
            np.ndarray: The point cloud in the world frame.
        """
        return point_cloud_from_depth_image_camera_frame(self.depth_im, self.intrinsic)

    @cached_property
    def pcd_worldframe(self):
        """
        Convert the RGBD image to a point cloud in the world frame.
        Returns:
            np.ndarray: The point cloud in the world frame.
        """
        im_h, im_w = self.rgb_im.shape[:2]
        return transform_pointcloud(self.extrinsic, self.pcd_cameraframe.reshape(-1, 3), set_invalid=True).reshape(im_h, im_w, 3)


class PerceptionInterface:
    def __init__(self):
        self.uncos = UncOS()
        self.gemini_client = GeminiClient()

    def get_object_mask(self, rgbd_observation: RGBDObservation, object_name, vis=False):
        fast_query_masks = self.uncos.grounded_sam_wrapper.process_image(rgbd_observation.rgb_im, text_prompt=object_name)
        if len(fast_query_masks) == 1:
            mask = fast_query_masks[0]()
            return mask
        pred_masks_boolarray, _ = self.uncos.segment_scene(rgbd_observation.rgb_im, rgbd_observation.pcd_worldframe, return_most_likely_only=True, pointcloud_frame='world')
        most_likely_mask_id = self.select_object(
            [crop_image(rgbd_observation.rgb_im, mask) for mask in pred_masks_boolarray],
            object_name,
            allow_none=False,
            vis=vis
        )
        return pred_masks_boolarray[most_likely_mask_id]

    def select_object(self, im_patches, object_name, allow_none=True, vis=False):
        selected_object_id = self.gemini_client.obtain_mostlikelyobject_auto(
            im_patches,
            object_name,
            allow_none=allow_none,
            vis=vis
        )
        return selected_object_id


def initialize_perception_interface():
    """
    Initialize the perception interface.
    Returns:
        PerceptionInterface: The initialized perception interface.
    """
    perception_interface = PerceptionInterface()
    return perception_interface
