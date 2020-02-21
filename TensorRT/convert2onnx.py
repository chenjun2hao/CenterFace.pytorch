
import logging
import os
import _init_paths

import cv2
import numpy as np
import onnxruntime as nxrun
import torch

from opts_pose import opts
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from utils.image import get_affine_transform
from detectors.detector_factory import detector_factory

logger = logging.getLogger(__name__)

class class_centernet(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

           
def main(opt):
    # init model
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    debug = 0            # return the detect result without show
    threshold = 0.05
    TASK = 'multi_pose'  # or 'multi_pose' for human pose estimation
    input_h, intput_w = 800, 800
    MODEL_PATH = '/home/deepblue/deepbluetwo/chenjun/4_face_detect/centerface/exp/multi_pose/mobilev2_10/model_15.pth'
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)

    out_onnx_path = "../output/onnx_model/mobilev2_aspaper.onnx"
    image = cv2.imread('../readme/test.png')
    torch_input, meta = detector.pre_process(image, scale=1)
    torch_input = torch_input.cuda()
    # pytorch output
    torch_output = detector.model(torch_input)
    torch.onnx.export(detector.model, torch_input, out_onnx_path, verbose=False)
    sess = nxrun.InferenceSession(out_onnx_path)

    print('save done')
    input_name = sess.get_inputs()[0].name
    output_onnx = sess.run(None, {input_name:  torch_input.cpu().data.numpy()})
    temp = 1

  
if __name__ == '__main__':
    opt = opts().init()
    main(opt)
