import logging
import math
import os
import pickle
import time

import cv2
import numpy as np
import tensorrt as trt
import torch
import _init_paths
from torchvision import transforms
from opts_pose import opts
from centernet_tensorrt_engine import CenterNetTensorRTEngine

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger()  # required by TensorRT 


def build_engine(onnx_file_path, engine_file_path, precision, max_batch_size, cache_file=None):
    """Builds a new TensorRT engine and saves it, if no engine presents"""

    if os.path.exists(engine_file_path):
        logger.info('{} TensorRT engine already exists. Skip building engine...'.format(precision))
        return

    logger.info('Building {} TensorRT engine from onnx file...'.format(precision))
    with trt.Builder(TRT_LOGGER) as b, b.create_network() as n, trt.OnnxParser(n, TRT_LOGGER) as p:
        b.max_workspace_size = 1 << 30  # 1GB
        b.max_batch_size = max_batch_size
        if precision == 'fp16':
            b.fp16_mode = True
        elif precision == 'int8':
            from ..calibrator import Calibrator
            b.int8_mode = True
            b.int8_calibrator = Calibrator(cache_file=cache_file)
        elif precision == 'fp32':
            pass
        else:
            logger.error('Engine precision not supported: {}'.format(precision))
            raise NotImplementedError
        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            p.parse(model.read())
        if p.num_errors:
            logger.error('Parsing onnx file found {} errors.'.format(p.num_errors))
        engine = b.build_cuda_engine(n)
        print(engine_file_path)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


def add_coco_bbox(image, bbox, conf=1):
    txt = '{}{:.1f}'.format('person', conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
    cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
                font, 0.5, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

def add_coco_hp(image, points, keypoints_prob):
    for j in range(5):
        if keypoints_prob[j] > 0.5:
            cv2.circle(image, (points[j, 0], points[j, 1]), 2, (255, 255, 0), -1)
    return image


if __name__ == '__main__':
    # 0. build trnsorrt engine/转成tensorrt模型
    # onnx_path = '../output/onnx_model/mobilev2_large.onnx'
    trt_path = '../output/onnx_model/mobilev2.trt'
    # build_engine(onnx_path, trt_path, 'fp32', 1)
    # print('build trnsorrt engine done')
    config = opts().init()

    # 1. load trnsorrt engine
    body_engine = CenterNetTensorRTEngine(weight_file=trt_path, config=config)
    print('load trnsorrt engine done')

    # 2. video for the tracking
    cap = cv2.VideoCapture('../readme/唐人街网剧.mp4')

    # 3. write the result image into video
    if config.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        write_cap = cv2.VideoWriter(config.output_video, fourcc, 50, (im_width, im_height))

    k = 1; start_time = time.time()
    while cap.grab():
        k += 1
        ret, image = cap.retrieve()                             # Capture frame-by-frame
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = body_engine.run(rgb_img)[1]

        print('fps is:{:.3f}'.format(k/(time.time() - start_time)))

        for i, bbox in enumerate(detections):
            if bbox[4] > 0.4:
                body_bbox = np.array(bbox[:4], dtype=np.int32)
                body_prob = bbox[4]
                add_coco_bbox(image, body_bbox, body_prob)
                body_pose = np.array(bbox[5:15], dtype=np.int32)
                keypoints = np.array(body_pose, dtype=np.int32).reshape(5, 2)
                keypoints_prob = bbox[15:]
                image = add_coco_hp(image, keypoints, keypoints_prob)
        # debug show
        # cv2.imshow('image result', image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # write into video
        if config.output_video:
            write_cap.write(image)
