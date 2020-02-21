import cv2
from PyQt5.QtGui import QPixmap, QImage

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def read_show(imgname, label, detector=None, confidence=0.5, choose_id=2):
    try:
        img = cv2.imread(imgname)
        if detector:
            # img = detector.run(img)['plot_img']
            bbox = detector.run(img)['results'][choose_id]
            bbox = bbox[bbox[:, 4] > confidence, :]
            bbox = bbox[:, :4]
            img = draw_bboxes(img, bbox)
        img = cv2.resize(img, (1000,600))
        h, w, c = img.shape
        byteperlin = c * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, w, h, byteperlin, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))
    except:
        label.setText('something wrong with the model')

def product_show(img, detector, confidence=0.5, choose_id=1):
    try:
        if detector:
            # img = detector.run(img)['plot_img']
            bbox = detector.run(img)['results'][choose_id]
            bbox = bbox[bbox[:, 4] > confidence, :]
            bbox = bbox[:, :4]
            img = draw_bboxes(img, bbox)
        img = cv2.resize(img, (1000,600))
        h, w, c = img.shape
        byteperlin = c * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, w, h, byteperlin, QImage.Format_RGB888)
        return image
    except:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        image = QImage(img.data, w, h, byteperlin, QImage.Format_RGB888)
        return image


def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id%len(COLORS_10)]
        # label = '{} {}'.format("defect", id)
        label = 'defect'
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = cv2.getTextSize(label, font, 0.5 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
        cv2.rectangle(img,(x1, y1-t_size[1]-2),(x1+t_size[0],y1-2), color,-1)
        cv2.putText(img,label,(x1,y1-4), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img