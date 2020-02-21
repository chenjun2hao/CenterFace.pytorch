import sys, cv2, time, os
from UI import Ui_TabWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QWidget, QProgressBar
from py_util import read_show
from center_main import main   
from opts2 import opts
import glob
from py_util import product_show
# from win32process import SuspendThread, ResumeThread

from opts2 import opts
from detectors.detector_factory import detector_factory
model_path = '/home/yangna/deepblue/2_MOT/CenterNet/exp/ctdet/dla/model_best.pth'
arch = 'dla_34'
task = 'ctdet'
opt = opts().init('--task {} --load_model {} --arch {}'.format(task, model_path, arch).split(' '))



class mywindow(QTabWidget,Ui_TabWidget): #这个窗口继承了用QtDesignner 绘制的窗口

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.thread = train_thred()
        self.thread.my_signal.connect(self.set_step)  # 3

        global imgnums
        path = r'/home/yangna/deepblue/2_MOT/CenterNet/data/pig/image/*.png'
        self.datas = glob.glob(path)
        imgnums = len(self.datas)

        self.save_nums = 0                  # 采集的图片数量

    def collect_image(self):
        '''自动化采集图片
            只能采用线程的方式进行摄像头的显示
        '''
        self.collect_image_thread = collect_image_thread()
        self.collect_image_thread.signal.connect(self.set_label)
        self.collect_image_thread.start()

    def collect_save_image(self):
        folder = f'./data/{self.line51.text()}/image'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.label53.pixmap().save(f'{folder}/{self.save_nums}.jpg')
        # cv2.imwrite(f'{folder}/{self.save_nums}.jpg', img)
        self.save_nums += 1
        self.label52.setText('已采集图片： ' + str(self.save_nums))

    def set_label(self, image):
        '''显示采集了多少张图片'''
        # self.label52.setText(text)
        self.label53.setPixmap(QPixmap.fromImage(image))


    def choose_train(self):
        global train_json
        train_json, file_type = QFileDialog.getOpenFileName(self,
                                            '选择训练数据集',
                                            "",
                                            'All Files (*)')
        self.label11.setText(train_json)

    def choose_val(self):
        global val_json
        val_json, file_type = QFileDialog.getOpenFileName(self,
                                            '选择验证数据集',
                                            "",
                                            'All Files (*)')
        self.label12.setText(val_json)
        
    def count_func(self):
        self.thread.start()

    def set_step(self, num):
        self.bar.setValue(num)

    def load_model(self):
        opt.debug = min(opt.debug, 0)       # 检测结果以cv2的格式返回
        self.detector = detector_factory[opt.task](opt)

    def load_picture(self):
        '''
            验证流程中的选择图片
        '''
        global imgname
        if self.pushbutton_22.text() == '选择图片':
            imgname, file_type = QFileDialog.getOpenFileName(self,
                                            '选择图片',
                                            "",
                                            'All Files (*)')
            read_show(imgname, self.label_21, 
                      choose_id=self.combobox21.currentIndex() + 1)   # 显示图片
    
    def test(self):
        '''验证流程中的测试过程'''
        read_show(imgname, self.label_21, self.detector,
                  choose_id=self.combobox21.currentIndex() + 1)


    def product_start(self):
        '''流水线开始'''
        if not hasattr(self, 'detector'):       # 没有载入模型
            opt.debug = min(opt.debug, 0)
            self.detector = detector_factory[opt.task](opt)

        if not hasattr(self, 'product_thread'):         # 声明进程
            # video_path = 'rtsp://admin:Shenlan2018@171.211.125.44:1554/h264/ch1/main/av_stream'
            video_path = 0
            self.product_thread = product_thread(self.detector, video_path, self.combobox41)
            self.product_thread.mysignal.connect(self.product_cess)
        self.product_thread.start()

    def product_stop(self):
        '''流水线暂停'''
        self.product_thread.stop()
        self.product_thread.quit()
        self.product_thread.wait()

    def exit(self):
        sys.exit()

    def product_cess(self, image):
        self.label41.setPixmap(QPixmap.fromImage(image))


class collect_image_thread(QThread):
    '''
        数据采集页：
        读取视频流;保存到指定文件夹;实时显示保存的图片数量
        在线程中读取视频流，在推到UI进程
    '''
    signal = pyqtSignal(QImage)
    
    def __init__(self):
        super(collect_image_thread, self).__init__()
        # self.cap = cv2.VideoCapture('rtsp://admin:Shenlan2018@171.211.125.44:1554/h264/ch1/main/av_stream')
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    img = cv2.resize(frame, (1000,600))
                    h, w, c = img.shape
                    byteperlin = c * w
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                    image = QImage(img.data, w, h, byteperlin, QImage.Format_RGB888)
                    self.signal.emit(image)
            except:
                self.signal.emit('something wrong with the input video source')


class product_thread(QThread):
    '''
        将这里做成一个API接口的样子, 模型，
        模型一直加载在线程中，视频流可以释放、重启
    '''
    mysignal = pyqtSignal(QImage)

    def __init__(self, detector, video_path, combobox):
        super(product_thread, self).__init__()
        self.flag = 1                   # 实现开始暂停
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.detector = detector
        self.combobox = combobox
        self.index = 0
        
    def run(self):
        '''4帧处理一次'''
        self.flag = 1
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)

        while self.cap.isOpened() and self.flag:
            if self.index > 1000000000:
                self.index = 0
            self.index += 1

            try:
                # ret, frame = self.cap.read()
                ret = self.cap.grab()
                if ret and self.index % 4 == 0:
                    tret, frame = self.cap.retrieve()
                    image = product_show(frame, self.detector, 
                                        choose_id=self.combobox.currentIndex() + 1)
                    self.mysignal.emit(image)
            except:
                print('something wrong with the product_thread')

    def stop(self):
        self.flag = 0
        self.cap.release()          # 释放摄像头


class train_thred(QThread):
    my_signal = pyqtSignal(int)  # 1

    def __init__(self):
        super(train_thred, self).__init__()
        self.max_iter = 50         # 共训练50个epoch

    def run(self):
        opt = opts(train_json, val_json).parse()            # 这是串行的
        center_train = main(opt)
        for i in range(self.max_iter):
            self.my_signal.emit(i)  # 2
            center_train.train(i)
        center_train.logger.close() # 关闭日志文件


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())