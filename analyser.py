import datetime
import numpy as np
import time
import cv2
import os.path
import math
import h5py
import matplotlib as mpl
import random
mpl.use("Qt4Agg")
import matplotlib.pyplot as plt
from PyQt4.QtCore import QObject, pyqtSignal, pyqtSlot
from datetime import datetime

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
        self.col_dict = {}
        print(self.grp.name)

    def plot(self,data):
        plt.style.use('ggplot')
        plt.figure(1)
        for i,element in enumerate(data):
            plt.subplot(311+i)
            plt.plot(data[i][0],data[i][1],color='cornflowerblue')
            plt.title('tank_%s'%(1+i))
        plt.show()

    def createColDict(self,identities):
        for e in np.unique(identities):
            if e not in self.col_dict:
                self.col_dict[e] = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    def write_data(self):
        dset_x = self.grp.create_dataset("x",shape=(0,),dtype=np.float64,maxshape=(None,))
        dset_y = self.grp.create_dataset("y",shape=(0,),dtype=np.float64,maxshape=(None,))
        dset_framenr = self.grp.create_dataset("framenr",shape=(0,),dtype=np.int64,maxshape=(None,))
        dset_filename = self.grp.create_dataset("filename",shape=(0,),dtype=h5py.special_dtype(vlen=str),maxshape=(None,))

        first,snd,trd,frth = zip(*self.point)
        for i in range(0,len(np.asarray(first)),1):
            self.filename_list.append(str((os.path.splitext(os.path.basename(self.video))[0])))

        dset_list = [dset_x,dset_y,dset_framenr,dset_filename]
        data_list = [np.asarray(first),np.asarray(snd),np.asarray(self.framenr),np.asarray(self.filename_list)]

        for element in zip(dset_list,data_list):
            newsize = element[0].resize(element[1].shape)
            element[0][0:element[1].shape[0]] = element[1]

    def cumsum(self,x): ### cumulative sum according to unique objects of list x
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
        mem_x = []
        mem_y = []
        mem_i = []
        mem_frame = []
        max_ind = 0

        ret, frame = self.cap.read() ### record video:
        fourcc = cv2.VideoWriter_fourcc(*'X264')# set codec
        vid_img = frame[int(self.j[0][1]*frame.shape[0]):int(self.j[1][1]*frame.shape[0]),int(self.j[0][0]*frame.shape[1]):int(self.j[1][0]*frame.shape[1])]
        vout = cv2.VideoWriter("output.mkv", fourcc, 30,(vid_img.shape[1],vid_img.shape[0]), True)

        while(1):
            ret, frame = self.cap.read()
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            if ret and vout.isOpened():
                frame = frame[int(self.j[0][1]*frame.shape[0]):int(self.j[1][1]*frame.shape[0]),int(self.j[0][0]*frame.shape[1]):int(self.j[1][0]*frame.shape[1])]
                if self.wait > 0:
                    self.wait = self.wait-(1/fps)
                    cv2.imshow('frame',frame)
                    cv2.waitKey(10)
                    self.frame_count += 1
                    print("wait :",round(self.wait,1),end='\r')

                elif ((self.frame_count/fps) >= self.timelimit and self.timelimit != 0):
                    self.write_data()
                    snippet_id = [x[2] for x in self.point]
                    print("timelimit reached!")
                    print(datetime.now() - startTime,"\nTotal snippets detected:",max(snippet_id))
                    vout.release()
                    break

                else:
                    ingroup = []
                    self.current = []
                    startTime = datetime.now()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,5)
                    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,20)
                    blur = cv2.GaussianBlur(gray, (15,15),3)
                    fgmask = self.fgbg.apply(blur)
                    erode = cv2.erode(fgmask, self.kernel)
                    dilate = cv2.erode(erode, self.kernel)
                    fg = cv2.bitwise_and(gray,gray,mask = fgmask)
                    _, contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = np.asarray(contours)
                    ingroup = contours[([cv2.contourArea(c)<220 for c in contours])] ### Limit maximum detected contour size

                    # if len(ingroup)>0:
                    #     if max_ind < max([int(cv2.contourArea(c)) for c in ingroup]):
                    #         max_ind = max([int(cv2.contourArea(c)) for c in ingroup])
                    #     print(max_ind)

                    for c in ingroup:
                        # if cv2.contourArea(c) >= max_ind: ### identify largest contour in frame:
                        #     (x, y, w, h) = cv2.boundingRect(c)
                        #     fish_x = float(x+w/2) / float(gray.shape[1])
                        #     fish_y = float(y+h/2) / float(gray.shape[0])
                        #     cv2.drawContours(frame, c, -1, (0,0,255), 2)
                        #     cv2.drawContours(frame, c, -1, (0,0,255), -1)

                        # else:
                        (x, y, w, h) = cv2.boundingRect(c)
                        fish_x = float(x+w/2) / float(gray.shape[1])
                        fish_y = float(y+h/2) / float(gray.shape[0])
                        self.current.append([fish_x,fish_y,np.nan])

                    if mem_x != []: ### Limit memory to last 10 frames:
                        flist = [10 > (self.frame_count-d) for d in mem_frame]
                        mem_x = mem_x[flist]
                        mem_y = mem_y[flist]
                        mem_i = mem_i[flist]
                        mem_frame = mem_frame[flist]

                    for element in self.current:
                        if len(self.pre) > 0 and self.frame_count != 0:

                            for previous_element in self.pre:
                                dist_euclidean = self.euclidean(int(previous_element[0]*gray.shape[1]),int(element[0]*gray.shape[1]),int(previous_element[1]*gray.shape[0]),int(element[1]*gray.shape[0])) ### Calculating euclidean distance between points:
                                if dist_euclidean < 5: ### Check distance to last object of all lists and append if bellow distance:
                                    element[2] = previous_element[2]
                                    self.point.append([element[0],element[1],element[2],self.frame_count])

                                else:
                                    reassigned = 0
                                    for i in range(len(mem_i)):
                                        dist_euclidean = self.euclidean(int(mem_x[i]*gray.shape[1]),int(element[0]*gray.shape[1]),int(mem_y[i]*gray.shape[0]),int(element[1]*gray.shape[0]))
                                        if dist_euclidean < 15 and reassigned == 0:
                                            element[2] = int(mem_i[i])
                                            self.point.append([element[0],element[1],element[2],self.frame_count])
                                            reassigned = 1
                                            break

                                    if reassigned == 0:
                                        list_id = [x[2] for x in self.point]
                                        element[2] = max(list_id) + 1

                        elif len(self.point) == 0:
                            element[2] = 0
                            self.point.append([element[0],element[1],element[2],self.frame_count])

                        else:
                            list_id = [x[2] for x in self.point]
                            element[2] = max(list_id) + 1

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

                    list_id = [x[2] for x in self.current]
                    for previous_element in self.pre:
                        if previous_element[2] not in list_id:
                            ### Implementing memory function:
                            mem_x = np.append(mem_x,previous_element[0])
                            mem_y = np.append(mem_y,previous_element[1])
                            mem_i = np.append(mem_i,previous_element[2])
                            mem_frame = np.append(mem_frame,self.frame_count)

                    if len(self.point) != 0:
                        self.createColDict([x[2] for x in self.point])

                    for i in self.point:
                        # if np.isfinite(i[2]) and (i[3] > self.frame_count - 40): ### trailing track for last 40 frames:
                        cv2.circle(frame,(int(i[0]*gray.shape[1]),int(i[1]*gray.shape[0])),1,self.col_dict[i[2]],1,lineType=cv2.LINE_AA)
                        # cv2.circle(frame,(int(i[0]*gray.shape[1]),int(i[1]*gray.shape[0])),1,(255,255,255),1,lineType=cv2.LINE_AA)
                    #
                    # if self.timelimit != 0:
                    #     print(round((self.frame_count/(fps*self.timelimit)*100),1),"%",end='\r')
                    # else:
                    #     print(round((self.frame_count/self.total_frame_count*100),1),"%",end='\r')

                    self.pre = self.current
                    self.frame_count = self.frame_count + 1
                    vout.write(frame)
                    cv2.imshow('frame',frame)
                    cv2.waitKey(1)
            else:
                self.write_data()
                snippet_id = [x[2] for x in self.point]
                print("timelimit reached!")
                print(datetime.now() - startTime,"\nTotal snippets detected:",max(snippet_id))
                vout.release()
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
