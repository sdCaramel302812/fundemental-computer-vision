from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets, uic
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as opt
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time



class MyUI(QtWidgets.QMainWindow):
    key_point = np.zeros((0, 1, 2),dtype=np.float32)
    def __init__(self):
        super(MyUI, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        self.btn1_1.clicked.connect(self.func1_1_btn)
        self.btn2_1.clicked.connect(self.func2_1_btn)
        self.btn3_1.clicked.connect(self.func3_1_btn)
        self.btn3_2.clicked.connect(self.func3_2_btn)
        self.btn4_1.clicked.connect(self.func4_1_btn)

        
    def func1_1_btn(self):
        print('1.1')
        imgL = cv2.imread('imL.png',0)
        imgR = cv2.imread('imR.png',0)

        stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=9)
        disparity = stereo.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()
        pass

    def func2_1_btn(self):
        print('2.1')
        video = cv2.VideoCapture('bgSub.mp4')
        ret, bg = video.read()
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        bs = cv2.bgsegm.createBackgroundSubtractorMOG()
        while(True):
            ret, frame = video.read()
            cv2.imshow('video',frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bs.apply(gray_frame) 
            cv2.imshow('frontground', fg)
            if cv2.waitKey(10) == ord('q'):
                break

        video.release()
        pass

    def func3_1_btn(self):
        print('3.1')
        video = cv2.VideoCapture('featureTracking.mp4')
        ret, frame = video.read()

        param = cv2.SimpleBlobDetector_Params()
        param.filterByCircularity = True
        param.minCircularity = 0.87
        param.filterByConvexity = True
        param.minConvexity = 0.1
        param.maxConvexity = 1

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detec = cv2.SimpleBlobDetector(param)
        else : 
            detec = cv2.SimpleBlobDetector_create(param)

        tmp_result = detec.detect(frame)
        for i in range(len(tmp_result)):
            x, y = tmp_result[i].pt[0], tmp_result[i].pt[1]
            self.key_point = np.append(self.key_point, [[[np.float32(x), np.float32(y)]]], axis = 0) 
        for i in range(0, len(self.key_point)):
            x = np.int(self.key_point[i][0][0])
            y = np.int(self.key_point[i][0][1])
            frame = cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)
        cv2.imshow('optical flow', frame)

        video.release()
        pass

    def func3_2_btn(self):
        print('3.2')

        lk_params = dict( winSize  = (21, 21), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))   

        video = cv2.VideoCapture('featureTracking.mp4')
        ret, prev_frame = video.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        line_img = np.zeros_like(prev_frame)
        while(True):
            ret, frame = video.read()
            if ret == False:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, self.key_point, None, **lk_params)

            for i in range(len(self.key_point)):
                a, b = p1[i][0][0], p1[i][0][1]
                c, d = self.key_point[i][0][0], self.key_point[i][0][1]
                line_img = cv2.line(line_img, (a, b), (c, d), (0, 0, 255), 2)
                x = np.int(a)
                y = np.int(b)
                frame = cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)
            self.key_point = p1

            final_img = cv2.add(frame, line_img)
            cv2.imshow('optical flow', final_img)
            prev_gray = gray_frame
            if cv2.waitKey(10) == ord('q'):
                break

        video.release()
        pass

    def func4_1_btn(self):
        print('4.1')
        intrinsic = np.array([[2225.49585482,    0,           1025.5459589],
                              [     0,       2225.18414074,   1038.58518846],
                              [     0,           0,               1       ]])
        distortion = np.array([[-0.12874225,  0.09057782, -0.00099125,  0.00000278,  0.0022925]])
        rvec1 = np.array([[-0.97157425,  -0.01827487,  0.23602862],
                           [ 0.07148055, -0.97312723,  0.2188925],
                           [ 0.22568565,  0.22954177,  0.94677165]])

        tvec1 = np.array([6.81253889, 3.37330384, 16.71572319])
        rvec2 = np.array([[-0.8884799,  -0.14530922, -0.435303],
                           [ 0.07148066, -0.98078915,  0.18150248],
                           [-0.45331444,  0.13014556,  0.88179825]])
        tvec2 = np.array([[3.3925504,  4.36149229, 22.15957429]])
        rvec3 = np.array([[-0.52390938,  0.22312793,  0.82202974],
                           [ 0.00530458, -0.96420621,  0.26510046],
                           [ 0.85175749,  0.14324914,  0.50397308]])
        tvec3 = np.array([[2.68774801,  4.70990021, 12.98147662]])
        rvec4 = np.array([[-0.63108673,  0.53013053,  0.566296],
                           [ 0.13263301, -0.64553994,  0.75212145],
                           [ 0.76429823,  0.54976341,  0.33707888]])
        tvec4 = np.array([[1.22781875,  3.48023006, 10.9840538]])
        rvec5 = np.array([[-0.87676843, -0.23020567,  0.42223508],
                           [ 0.19708207, -0.97286949, -0.12117596],
                           [ 0.43867502, -0.02302829,  0.89835067]])
        tvec5 = np.array([[4.43641198,  0.67177428, 16.24069227]])

        img1 = cv2.imread('1.bmp',0)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.imread('2.bmp',0)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        img3 = cv2.imread('3.bmp',0)
        img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
        img4 = cv2.imread('4.bmp',0)
        img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
        img5 = cv2.imread('5.bmp',0)
        img5 = cv2.cvtColor(img5, cv2.COLOR_GRAY2BGR)
        point =  np.float32([[1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0], [3, 3, -4]]).reshape(-1,3)
        # img1
        position, _ = cv2.projectPoints(point, rvec1, tvec1, intrinsic, distortion)
        print(position)
        x1, y1 = np.int(position[0][0][0]), np.int(position[0][0][1])
        x2, y2 = np.int(position[1][0][0]), np.int(position[1][0][1])
        x3, y3 = np.int(position[2][0][0]), np.int(position[2][0][1])
        x4, y4 = np.int(position[3][0][0]), np.int(position[3][0][1])
        x5, y5 = np.int(position[4][0][0]), np.int(position[4][0][1])
        img1 = cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x1, y1), (x4, y4), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x1, y1), (x5, y5), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x2, y2), (x3, y3), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x2, y2), (x5, y5), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x3, y3), (x4, y4), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x3, y3), (x5, y5), (0, 0, 255), 4)
        img1 = cv2.line(img1, (x4, y4), (x5, y5), (0, 0, 255), 4)
        img1 = cv2.resize(img1, (400, 400), interpolation=cv2.INTER_CUBIC)
        # img2
        position, _ = cv2.projectPoints(point, rvec2, tvec2, intrinsic, distortion)
        print(position)
        x1, y1 = np.int(position[0][0][0]), np.int(position[0][0][1])
        x2, y2 = np.int(position[1][0][0]), np.int(position[1][0][1])
        x3, y3 = np.int(position[2][0][0]), np.int(position[2][0][1])
        x4, y4 = np.int(position[3][0][0]), np.int(position[3][0][1])
        x5, y5 = np.int(position[4][0][0]), np.int(position[4][0][1])
        img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x1, y1), (x4, y4), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x1, y1), (x5, y5), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x2, y2), (x3, y3), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x2, y2), (x5, y5), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x3, y3), (x4, y4), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x3, y3), (x5, y5), (0, 0, 255), 4)
        img2 = cv2.line(img2, (x4, y4), (x5, y5), (0, 0, 255), 4)
        img2 = cv2.resize(img2, (400, 400), interpolation=cv2.INTER_CUBIC)
        # img3
        position, _ = cv2.projectPoints(point, rvec3, tvec3, intrinsic, distortion)
        print(position)
        x1, y1 = np.int(position[0][0][0]), np.int(position[0][0][1])
        x2, y2 = np.int(position[1][0][0]), np.int(position[1][0][1])
        x3, y3 = np.int(position[2][0][0]), np.int(position[2][0][1])
        x4, y4 = np.int(position[3][0][0]), np.int(position[3][0][1])
        x5, y5 = np.int(position[4][0][0]), np.int(position[4][0][1])
        img3 = cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x1, y1), (x4, y4), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x1, y1), (x5, y5), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x2, y2), (x3, y3), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x2, y2), (x5, y5), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x3, y3), (x4, y4), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x3, y3), (x5, y5), (0, 0, 255), 4)
        img3 = cv2.line(img3, (x4, y4), (x5, y5), (0, 0, 255), 4)
        img3 = cv2.resize(img3, (400, 400), interpolation=cv2.INTER_CUBIC)
        # img4
        position, _ = cv2.projectPoints(point, rvec4, tvec4, intrinsic, distortion)
        print(position)
        x1, y1 = np.int(position[0][0][0]), np.int(position[0][0][1])
        x2, y2 = np.int(position[1][0][0]), np.int(position[1][0][1])
        x3, y3 = np.int(position[2][0][0]), np.int(position[2][0][1])
        x4, y4 = np.int(position[3][0][0]), np.int(position[3][0][1])
        x5, y5 = np.int(position[4][0][0]), np.int(position[4][0][1])
        img4 = cv2.line(img4, (x1, y1), (x2, y2), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x1, y1), (x4, y4), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x1, y1), (x5, y5), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x2, y2), (x3, y3), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x2, y2), (x5, y5), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x3, y3), (x4, y4), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x3, y3), (x5, y5), (0, 0, 255), 4)
        img4 = cv2.line(img4, (x4, y4), (x5, y5), (0, 0, 255), 4)
        img4 = cv2.resize(img4, (400, 400), interpolation=cv2.INTER_CUBIC)
        # img5
        position, _ = cv2.projectPoints(point, rvec5, tvec5, intrinsic, distortion)
        print(position)
        x1, y1 = np.int(position[0][0][0]), np.int(position[0][0][1])
        x2, y2 = np.int(position[1][0][0]), np.int(position[1][0][1])
        x3, y3 = np.int(position[2][0][0]), np.int(position[2][0][1])
        x4, y4 = np.int(position[3][0][0]), np.int(position[3][0][1])
        x5, y5 = np.int(position[4][0][0]), np.int(position[4][0][1])
        img5 = cv2.line(img5, (x1, y1), (x2, y2), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x1, y1), (x4, y4), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x1, y1), (x5, y5), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x2, y2), (x3, y3), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x2, y2), (x5, y5), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x3, y3), (x4, y4), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x3, y3), (x5, y5), (0, 0, 255), 4)
        img5 = cv2.line(img5, (x4, y4), (x5, y5), (0, 0, 255), 4)
        img5 = cv2.resize(img5, (400, 400), interpolation=cv2.INTER_CUBIC)

        iter = 0
        while True:
            iter += 1
            if iter == 1:
                cv2.imshow('augument reality', img1)
            if iter == 2:
                cv2.imshow('augument reality', img2)
            if iter == 3:
                cv2.imshow('augument reality', img3)
            if iter == 4:
                cv2.imshow('augument reality', img4)
            if iter == 5:
                cv2.imshow('augument reality', img5)
            if cv2.waitKey(500) == ord('q'):
                break

        pass



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    

    widget = MyUI()



    app.exec_()



