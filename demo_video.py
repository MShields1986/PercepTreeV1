#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MOV to mp4
# ffmpeg -i input.mov -vcodec h264 -acodec mp2 output.mp4
# 4k to 1080
# ffmpeg -i sample.mp4 -vf scale=1920:1080 sample.mp4
# 60Hz to 30Hz
# ffmpeg -i input.mp4 -r 30 output.mp4

"""
Test trained network on a video
"""
from __future__ import absolute_import

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os, cv2
import torch

# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer


model_path = "./models/"
input_path = "./input/"
output_path = "./output/"

model_name = "X-101_RGB_60k.pth"
# model_name = 'R-50_RGB_60k.pth'

# input_file = 'forest_walk_1min.mp4'
# input_file = 'img_5342.mp4'
# input_file = 'img_5345.mp4'
# input_file = 'img_5346.mp4'
# input_file = 'img_5347.mp4'
# input_file = "img_5348.mp4"
# input_file = "img_5349.mp4"
# input_file = "img_0797.mp4"
input_file = "new_4k_30.mp4"


if __name__ == "__main__":
    torch.cuda.is_available()
    logger = setup_logger(name=__name__)

    # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
    )
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 * 4  # faster (default: 512)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster (default: 512)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True

    cfg.MODEL.WEIGHTS = model_path + model_name
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time

    # set detector
    predictor_synth = DefaultPredictor(cfg)

    # set metadata
    tree_metadata = MetadataCatalog.get("my_tree_dataset").set(
        thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"]
    )

    # Get one video frame
    vcap = cv2.VideoCapture(input_path + input_file)

    # get vcap property
    w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input File {input_file} - Frame Width = {w}")
    print(f"Input File {input_file} - Frame Height = {h}")
    print(f"Input File {input_file} - FPS = {fps}")
    print(f"Input File {input_file} - Frame Count = {n_frames}")

    # VIDEO recorder
    # Grab the stats from the input to use for the ouput video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path + input_file, fourcc, fps, (w, h), True)

    # Check if camera opened successfully
    if vcap.isOpened() == False:
        print("Error opening video stream or file")

    vid_vis = VideoVisualizer(metadata=tree_metadata)

    nframes = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        y = 000
        # h = 800
        x = 000
        # w = 800
        crop_frame = frame
        # crop_frame = frame[y:y+h, x:x+w]
        # cv2.imshow('frame', crop_frame)
        if cv2.waitKey(1) == ord("q"):
            break

        # 5 FPS
        # if nframes % fps/5 == 0:

        # All frames
        if nframes % 1 == 0:
            print(f"Processing Frame {nframes} of {n_frames}")
            outputs_pred = predictor_synth(crop_frame)
            # v_synth = Visualizer(crop_frame[:, :, ::-1],
            #                     metadata=tree_metadata,
            #                     scale=1,
            #                     instance_mode =  ColorMode.IMAGE     # remove color from image, better see instances
            #     )

            # print(f'kCP X - {outputs_pred["instances"].pred_keypoints[0][0][0]}')
            # print(f'kCP Y - {outputs_pred["instances"].pred_keypoints[0][0][1]}')
            # print(f'kCP score - {outputs_pred["instances"].pred_keypoints[0][0][2]}')

            out = vid_vis.draw_instance_predictions(
                crop_frame, outputs_pred["instances"].to("cpu")
            )

            vid_frame = out.get_image()
            video.write(vid_frame)
            # cv2.imshow("frame", vid_frame)

        nframes += 1

    video.release()
    vcap.release()
    cv2.destroyAllWindows()
