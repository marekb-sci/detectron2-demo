#!/bin/bash
mkdir -p models/mask_rcnn_R_50_FPN_3x
wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl -P models/mask_rcnn_R_50_FPN_3x
