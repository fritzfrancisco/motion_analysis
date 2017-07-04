import datetime
import numpy as np
import time
import cv2
import os.path
import math
import h5py
import matplotlib as mpl
mpl.use("Qt4Agg")
import matplotlib.pyplot as plt
from PyQt4.QtCore import QObject, pyqtSignal, pyqtSlot
# from datetime import datetime

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')# set codec
# vout = cv2.VideoWriter("output.mp4", fourcc, 30.0, (1920, 1080), True)

class analyser(QObject):

    def __init__(self,videofile,i,j,start,stop,f):
        QObject.__init__(self)
        self.video = videofile
        self.j = j
        self.wait = start
        self.timelimit = stop
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 99, detectShadows=False)
        self.kernel = np.ones((5,5))
        self.frame_count = 0
        self.grp = f.create_group("/tank_%s/"%(i+1))
        self.f = f
        print(self.grp.name)

    def plot(self,data):
        plt.style.use('ggplot')
        plt.figure(1)
        for i,element in enumerate(data):
            plt.subplot(311+i)
            plt.plot(data[i][0],data[i][1],color='cornflowerblue')
            plt.title('tank_%s'%(1+i))
        plt.show()

    def write_data(self):
        dset_x = self.grp.create_dataset("x",shape=(0,),dtype=np.float64,maxshape=(None,))
        dset_y = self.grp.create_dataset("y",shape=(0,),dtype=np.float64,maxshape=(None,))
        dset_framenr = self.grp.create_dataset("framenr",shape=(0,),dtype=np.int64,maxshape=(None,))
        dset_filename = self.grp.create_dataset("filename",shape=(0,),dtype=h5py.special_dtype(vlen=str),maxshape=(None,))

        first,snd = zip(*self.point)
        for i in range(0,len(np.asarray(first)),1):
            self.filename_list.append(str((os.path.splitext(os.path.basename(self.video))[0])))

        dset_list = [dset_x,dset_y,dset_framenr,dset_filename]
        data_list = [np.asarray(first),np.asarray(snd),np.asarray(self.framenr),np.asarray(self.filename_list)]

        for element in zip(dset_list,data_list):
            newsize = element[0].resize(element[1].shape)
            element[0][0:element[1].shape[0]] = element[1]

    def cumsum(self,x):
    ### cumulative sum according to unique objects of list x
        cumsum = []
        for element in np.unique(x):
            cumsum.append(sum(e == element for e in x)) if (e == e+1 for e in x) else cumsum.append(0)
        return(cumsum)

    def euclidean(self,x1,x2,y1,y2):
        return np.sqrt(pow(x2-x1,2)+pow(y2-y1,2))

    def start(self):
        self.cap = cv2.VideoCapture(self.video)
        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.point = []
        self.framenr = []
        self.pre = []
        self.filename_list = []

        while(1):
            ret, frame = self.cap.read()
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            if ret:
                if self.wait > 0:
                    self.wait = self.wait-(1/fps)
                    frame = frame[int(self.j[0][1]*frame.shape[0]):int(self.j[1][1]*frame.shape[0]),int(self.j[0][0]*frame.shape[1]):int(self.j[1][0]*frame.shape[1])]
                    cv2.imshow('frame',frame)
                    cv2.waitKey(10)
                    self.frame_count += 1
                    print("wait :",round(self.wait,1),end='\r')

                elif ((self.frame_count/fps) >= self.timelimit and self.timelimit != 0):
                    self.write_data()
                    # print(self.dset_x[()],self.dset_y[()],self.dset_framenr[()],self.dset_filename[()])
                    print("timelimit reached!")
                    print(datetime.now() - startTime)
                    break

                else:
                    startTime = datetime.now()
                    self.frame_count = self.frame_count + 1
                    frame = frame[int(self.j[0][1]*frame.shape[0]):int(self.j[1][1]*frame.shape[0]),int(self.j[0][0]*frame.shape[1]):int(self.j[1][0]*frame.shape[1])]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,5)
                    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,20)
                    blur = cv2.GaussianBlur(gray, (15,15),3)
                    fgmask = self.fgbg.apply(blur)
                    erode = cv2.erode(fgmask, self.kernel)
                    dilate = cv2.erode(erode, self.kernel)
                    fg = cv2.bitwise_and(gray,gray,mask = fgmask)
                    _, contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    self.current = []

                    for c in contours:
                        if cv2.contourArea(c) < 220:
                            (x, y, w, h) = cv2.boundingRect(c)
                            fish_x = float(x+w/2) / float(gray.shape[1])
                            fish_y = float(y+h/2) / float(gray.shape[0])
                            self.current.append((fish_x,fish_y))

                            cv2.drawContours(frame, c, -1, (0,0,255), 2)
                            cv2.drawContours(frame, c, -1, (0,0,255), -1)
                        else:
                            cv2.drawContours(frame, c, -1, (0,255,255), 2)
                            cv2.drawContours(frame, c, -1, (0,255,255), -1)

                    for i in self.current:
                        for previous_element in self.pre:
                            if i != previous_element:
                                dist_euclidean = 0
                                previous_element = (int(previous_element[0]*gray.shape[1]),int(previous_element[1]*gray.shape[0]))
    ### Calculating euclidean distance between points:
                                dist_euclidean = self.euclidean(previous_element[0],int(i[0]*gray.shape[1]),previous_element[1],int(i[1]*gray.shape[0]))
                                if dist_euclidean < 5:
                                    self.framenr.append(self.frame_count)
                                    self.point.append((i[0],i[1]))
    ### find neighbors in fixed radius
                    # for j in self.current:
                    #     self.neighbors = []
                    #     for o in self.point:
                    #         if j != o:
                    #             R = 5
                    #             dx = abs(o[0]-j[0])*gray.shape[1]
                    #             dy = abs(o[1]-j[1])*gray.shape[0]
                    #             if dx < R and dy < R:
                    #                 self.neighbors.append((o[0],o[1]))
                    #                 for i, element in enumerate(self.current):
                    #                     if len(self.neighbors) <= 1:
                    #                         self.current.remove(j)
                    #                         print("here")
                    self.pre = self.current

                    for i in self.point:
                        cv2.circle(frame,(int(i[0]*gray.shape[1]),int(i[1]*gray.shape[0])),1,(255,255,255),1,lineType=cv2.LINE_AA)

                    if self.timelimit != 0:
                        print(round((self.frame_count/(fps*self.timelimit)*100),1),"%",end='\r')
                    else:
                        print(round((self.frame_count/self.total_frame_count*100),1),"%",end='\r')

                    # print(int(self.total_frame_count),self.frame_count,self.timelimit)
                    cv2.imshow('frame',frame)
                    cv2.waitKey(1)
            else:
                self.write_data()
                # print(self.dset_x[()],self.dset_y[()],self.dset_framenr[()],self.dset_filename[()])
                print("timelimit reached!")
                break

class reader(QObject):

    def __init__(self,f,videofile,start,stop):
        QObject.__init__(self)
        start = start
        stop = stop
        video = videofile
        cap = cv2.VideoCapture(video)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def plot(self,data):
        plt.style.use('ggplot')
        plt.figure(1)
        for i,element in enumerate(data):
            plt.subplot(311+i)
            plt.plot(data[i][0],data[i][1],color='cornflowerblue')
            plt.title('tank_%s'%(1+i))
        plt.show()

    def cumulative_plot(self,data):
        plt.style.use('ggplot')
        plt.figure(1)
        for i,element in enumerate(data):
            values, base = np.histogram(data[i][1], bins=2800)
            cumulative = np.cumsum(values)
            plt.subplot(311+i)
            plt.plot(cumulative,base[:-1], c='blue')
            plt.title('tank_%s'%(1+i))
        plt.show()
