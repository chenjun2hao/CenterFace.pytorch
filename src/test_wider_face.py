import sys
import os
import scipy.io as sio
import cv2


path = os.path.dirname(__file__)
CENTERNET_PATH = os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts_pose import opts
from datasets.dataset_factory import get_dataset


def test_img(MODEL_PATH):
    debug = 1            # draw and show the result image
    TASK = 'multi_pose'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h).split(' '))

    detector = detector_factory[opt.task](opt)

    img = '../readme/000388.jpg'
    ret = detector.run(img)['results']


def test_wider_Face(model_path):
    Path = '/home/deepblue/deepbluetwo/data/WIDER_val/images/'
    wider_face_mat = sio.loadmat('/home/deepblue/deepbluetwo/data/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = '../output/widerface/'

    debug = 0            # return the detect result without show
    threshold = 0.05
    TASK = 'multi_pose'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img_path = os.path.join(Path, zip_name)
            dets = detector.run(img_path)['results']
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    MODEL_PATH = '../exp/multi_pose/mobilev2_10/model_best.pth'
    test_img(MODEL_PATH)
    # test_wider_Face(MODEL_PATH)