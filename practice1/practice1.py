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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.full_con1 = nn.Linear(16 * 5 * 5, 120)
        self.full_con2 = nn.Linear(120, 84)
        self.full_con3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.full_con1(x))
        x = func.relu(self.full_con2(x))
        x = self.full_con3(x)
        return x

    def reset(self):
        #nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.conv1.weight.data.normal_(0.0, 0.02)
        if self.conv1.bias is not None:
            self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.normal_(0.0, 0.02)
        if self.conv2.bias is not None:
            self.conv2.bias.data.fill_(0)
        self.full_con1.weight.data.normal_(0.0, 0.1)
        if self.full_con1.bias is not None:
            self.full_con1.bias.data.fill_(0)
        self.full_con2.weight.data.normal_(0.0, 0.1)
        if self.full_con2.bias is not None:
            self.full_con2.bias.data.fill_(0)
        self.full_con3.weight.data.normal_(0.0, 0.1)
        if self.full_con3.bias is not None:
            self.full_con3.bias.data.fill_(0)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
transform2 = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 9999, shuffle = False, num_workers = 2)
indexset = torchvision.datasets.CIFAR10(root='./data', train = False, download = False, transform = transform2)
indexloader = torch.utils.data.DataLoader(indexset, batch_size = 9999, shuffle = False, num_workers = 2)
showset = torchvision.datasets.CIFAR10(root='./data', train = False, download = False, transform = transform2)
showloader = torch.utils.data.DataLoader(showset, batch_size = 10, shuffle = True, num_workers = 2)

testiter = iter(testloader)
test_img_n, _ = testiter.next()
indexiter = iter(indexloader)
test_img, _ = indexiter.next()

net = Network()
CEloss = nn.CrossEntropyLoss()
optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)  

def load_image(file_name):
    img = cv2.imread(file_name)

    height, width, channel = img.shape
    bytes_per_line = 3 * width
    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

    return (img, q_img)

def get_q_image(img):
    height, width, channel = img.shape
    bytes_per_line = 3 * width
    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
    return q_img

def func4():
    original_img = cv2.imread('./Contour.png')
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (11, 11), 0)
    gray_img = cv2.Canny(gray_img, 30, 150)
    (cnts, _) = cv2.findContours(gray_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = original_img.copy()
    cv2.drawContours(contours_img, cnts, -1, (0, 0, 255), 2)

    return (get_q_image(original_img), get_q_image(contours_img))

def func5_1():
    showiter = iter(showloader)
    show_images, show_labels = showiter.next()

    #img = np.ndarray(())
    #cv2.cvtColor(np.float32(images[0]), img, cv2.COLOR_BGR2RGB)
    #img = np.asarray(images[0])
    #print(type(images[0]))
    #print(type(img))
    #widget.set_image(img)
    #plt.imshow(torchvision.utils.make_grid(images))
    grid_img = torchvision.utils.make_grid(show_images, nrow = 10)
    #tensor_image = grid_img.view(grid_img.shape[1], grid_img.shape[2], grid_img.shape[0])
    to_pil = torchvision.transforms.ToPILImage()
    tensor_image = to_pil(grid_img)
    plt.imshow(tensor_image)
    plt.xticks([16, 50, 84, 118, 152, 186, 220, 254, 288, 322],
                 [classes[show_labels[0]], classes[show_labels[1]], classes[show_labels[2]], classes[show_labels[3]], classes[show_labels[4]],
                  classes[show_labels[5]], classes[show_labels[6]], classes[show_labels[7]], classes[show_labels[8]], classes[show_labels[9]]])
    plt.show()

def func5_2():
    print('hyperparameters : \nbatch size = 32\nlearning rate = 0.001\noptimizer : SGD\n')

def func5_3():
    loss_change = []
    net.reset()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()       
        outputs = net(inputs)                
        loss = CEloss(outputs, labels)  
        loss.backward()  
        optimizer.step()   

        loss_change.append(loss.item())
    
    plt.plot(loss_change)
    plt.show()
    

def func5_4_run():
    loss_change = []
    train_acc = []
    for epoch in range(0, 100):
        correct = 0
        count = 0
        iner_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()       
            outputs = net(inputs)                
            loss = CEloss(outputs, labels)  
            loss.backward()  
            optimizer.step()  

            pred = outputs.argmax(dim=1) 
            correct += torch.eq(pred, labels).sum().float().item()

            iner_loss += loss.item()
        loss_change.append(iner_loss * 32 / len(trainloader.dataset))
        train_acc.append(correct / len(trainloader.dataset))
    
    plt.plot(loss_change)
    plt.plot(train_acc)
    plt.legend(('loss', 'accuracy'), loc='upper right')
    plt.xlabel('epoches')
    plt.show()
    torch.save(net.state_dict(), 'model.pkl')

def func5_4_show():
    func4()

def fun5_5(index):
    net.load_state_dict(torch.load('model.pkl'))

    outputs = func.softmax(net(Variable(test_img_n[index].reshape(1, 3, 32, 32))))

    outputs = outputs.detach()
    arr = [float(outputs[0][0]), float(outputs[0][1]), float(outputs[0][2]), float(outputs[0][3]), float(outputs[0][4]),
            float(outputs[0][5]), float(outputs[0][6]), float(outputs[0][7]), float(outputs[0][8]), float(outputs[0][9])]

    grid_img = torchvision.utils.make_grid(test_img[index].reshape(1, 3, 32, 32))
    to_pil = torchvision.transforms.ToPILImage()
    tensor_image = to_pil(grid_img)
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_image)
    plt.subplot(1, 2, 2)
    plt.bar(range(10), arr, tick_label = classes)
    plt.show() 


class MyUI(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(768, 768)
        
        self.button1_1 = QtWidgets.QPushButton(self)
        self.button1_1.setGeometry(QRect(10, 10, 100, 30))
        self.button1_1.setText("1.1")
        self.button1_2 = QtWidgets.QPushButton(self)
        self.button1_2.setGeometry(QRect(10, 60, 100, 30))
        self.button1_2.setText("1.2")
        self.button1_3 = QtWidgets.QPushButton(self)
        self.button1_3.setGeometry(QRect(10, 110, 100, 30))
        self.button1_3.setText("1.3")
        self.button1_4 = QtWidgets.QPushButton(self)
        self.button1_4.setGeometry(QRect(10, 160, 100, 30))
        self.button1_4.setText("1.4")
        self.button2 = QtWidgets.QPushButton(self)
        self.button2.setGeometry(QRect(10, 210, 100, 30))
        self.button2.setText("2")
        self.button3_1 = QtWidgets.QPushButton(self)
        self.button3_1.setGeometry(QRect(10, 260, 100, 30))
        self.button3_1.setText("3.1")
        self.button3_2 = QtWidgets.QPushButton(self)
        self.button3_2.setGeometry(QRect(10, 310, 100, 30))
        self.button3_2.setText("3.2")
        self.button4 = QtWidgets.QPushButton(self)
        self.button4.setGeometry(QRect(10, 360, 100, 30))
        self.button4.setText("4")
        self.button5_1 = QtWidgets.QPushButton(self)
        self.button5_1.setGeometry(QRect(10, 410, 100, 30))
        self.button5_1.setText("5.1")
        self.button5_2 = QtWidgets.QPushButton(self)
        self.button5_2.setGeometry(QRect(10, 460, 100, 30))
        self.button5_2.setText("5.2")
        self.button5_3 = QtWidgets.QPushButton(self)
        self.button5_3.setGeometry(QRect(10, 510, 100, 30))
        self.button5_3.setText("5.3")
        self.button5_4 = QtWidgets.QPushButton(self)
        self.button5_4.setGeometry(QRect(10, 560, 100, 30))
        self.button5_4.setText("5.4 run")
        self.show5_4 = QtWidgets.QPushButton(self)
        self.show5_4.setGeometry(QRect(10, 610, 100, 30))
        self.show5_4.setText("5.4 show")
        self.button5_5 = QtWidgets.QPushButton(self)
        self.button5_5.setGeometry(QRect(10, 660, 100, 30))
        self.button5_5.setText("5.5")
        
        self.alart_label = QtWidgets.QLabel(self)
        self.alart_label.setGeometry(QRect(120, 560, 200, 30))
        self.alart_label.setText('training button, don\'t touch')

        self.hint_label = QtWidgets.QLabel(self)
        self.hint_label.setGeometry(QRect(120, 710, 200, 30))
        self.hint_label.setText('5.5 input (0 ~ 9998)')
        
        self.label1 = QtWidgets.QLabel(self)
        self.label1.setGeometry(QRect(200, 25, 500, 360))
        self.label1.setScaledContents(True)
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setGeometry(QRect(200, 375, 500, 360))
        self.label2.setScaledContents(True)

        self.input_line = QtWidgets.QLineEdit(self)
        self.input_line.setGeometry(QRect(10, 710, 100, 30))

        self.button4.clicked.connect(self.button4_callback)
        self.button5_1.clicked.connect(self.button5_1_callback)
        self.button5_2.clicked.connect(self.button5_2_callback)
        self.button5_3.clicked.connect(self.button5_3_callback)
        self.button5_4.clicked.connect(self.button5_4_callback)
        self.button5_5.clicked.connect(self.button5_5_callback)
        self.show5_4.clicked.connect(self.show5_4_callback)

    def button4_callback(self):
        img1, img2 = func4()
        self.label1.setPixmap(QPixmap.fromImage(img1))
        self.label2.setPixmap(QPixmap.fromImage(img2))

    def button5_1_callback(self):
        func5_1()
    
    def button5_2_callback(self):
        func5_2()

    def button5_3_callback(self):
        func5_3()

    def button5_4_callback(self):
        func5_4_run()

    def button5_5_callback(self):
        try:
            if int(self.input_line.text()) < 9999 and int(self.input_line.text()) >= 0:
                fun5_5(int(self.input_line.text()))
        except:
            pass
        self.input_line.clear()

    def show5_4_callback(self):
        pass
        img = cv2.imread('./Figure_1.png')
        cv2.imshow('loss and accuracy', img)
        #try:
        #    self.img, q_img = load_image(self.input_line.text())
        #    self.label.setPixmap(QPixmap.fromImage(q_img))
        #except:
        #    print('there is no file named : ', self.input_line.text())
        #self.input_line.clear()
    
    def set_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        #self.label.setPixmap(QPixmap.fromImage(QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()))




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)


    '''
    for epoch in range(2):
 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()       
            outputs = net(inputs)                
            loss = CEloss(outputs, labels)  
            loss.backward()  
            optimizer.step()   
    
            #running_loss += loss.data[0]
       


    dataiter = iter(testloader)
    images, labels = dataiter.next() 

    print('GroundTruth : ', labels)

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1) 
    print('Predicted   : ', predicted)

    '''

    

    widget = MyUI()
    widget.show()


    app.exec_()