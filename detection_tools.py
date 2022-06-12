# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

metadata_COCO = detectron2.data.MetadataCatalog.get('coco_2017_train')


def create_predictor(config_file, model_weights):
  cfg = detectron2.config.get_cfg()
  # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
  cfg.merge_from_file(config_file) #model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
  cfg.MODEL.WEIGHTS = model_weights #model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  cfg.MODEL.DEVICE = 'cpu'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set very low. Predictions will be filtered later on
  predictor = DefaultPredictor(cfg)
  return predictor


def process_output(predictor_output, classes, threshold=0.5):
  relevant_mask = (predictor_output['instances'].scores.numpy() > threshold) & np.isin(predictor_output['instances'].pred_classes, classes)
  processed_instances = detectron2.structures.Instances(
      predictor_output['instances']._image_size,
      scores=predictor_output['instances'].scores[relevant_mask],
      pred_classes=predictor_output['instances'].pred_classes[relevant_mask],
      pred_boxes=predictor_output['instances'].pred_boxes[relevant_mask]
      )
  return processed_instances


def get_predictions(img, predictor, classes, threshold=0.5):
    return process_output(predictor(img), classes=classes, threshold=threshold)

def visualize_predictions(img_bgr, pred_instances, metadata):

  v = Visualizer(img_bgr[:, :, ::-1], metadata, scale=1)
  out = v.draw_instance_predictions(pred_instances)
  out_img = out.get_image()[:, :, ::-1]

  return out_img

def resize_to(image, larger_size=1024):
    if larger_size is None:
        return image
    scale = larger_size/max(image.shape[:2])
    dim_out = round(scale*image.shape[1]), round(scale*image.shape[0])
    resized = cv2.resize(image, dim_out, interpolation = cv2.INTER_AREA)
    return resized

def predict_and_visualize(img_path, predictor, classes, threshold, metadata=metadata_COCO):
    img = cv2.imread(img_path)
    predictions = get_predictions(img, predictor, classes, threshold)
    visualization = visualize_predictions(img, predictions, metadata)
    return {'image': visualization, 'n_preds': len(predictions)}


# config_file = 'model/.../...'
# model_weights = 'model/.../model_final_f10217.pkl'
# OBJECT_CLASSES = [0]
# DETECTION_THRESHOLD = 0.5
# predictor = create_predictor(config_file, model_weights)
# img_path = ...
# predict_and_visualize(img_path, predictor, OBJECT_CLASSES, DETECTION_THRESHOLD, metadata=metadata_COCO)
