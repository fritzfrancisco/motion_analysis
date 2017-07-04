import numpy as np
import cv2
from collections import Counter
# import argparse
import analyser as an
import h5py
import inspect, os
import sys
# import glob

#ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
# ap.add_argument("-o", "--outfile", help="storage file")
#args = vars(ap.parse_args())
#print(args)

data = []
dist = []
crop = []
### start and stop time (s)
start = 0
stop = 0
xyreturn=None
crp_lst = []
p1 = (0,0)
p2 = (1,1)
switch = 0

# videofile = 'helder_test.MP4'
videofile = '/home/cronk/Documents/Projects/Jordan Lab/HIWI/1_2014-09-06_08-29-03.mp4'
cap = cv2.VideoCapture(videofile)

def divide_frame(event,x,y,flags,param):
    global xyreturn, switch, crp_lst

    if event == cv2.EVENT_LBUTTONDOWN:
        switch = 1
        crp_lst = [(x,y)]

    elif event == cv2.EVENT_LBUTTONUP:
        switch = 0
        crp_lst.append((x,y))

    if event == cv2.EVENT_MOUSEMOVE and switch == 1:
        crp_lst.append((x,y))
#
while xyreturn == None:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    cv2.namedWindow("Crop")
    cv2.setMouseCallback("Crop",divide_frame)
    key = cv2.waitKey(30) & 0xFF

    if len(crp_lst) >= 1:
        cv2.rectangle(frame,min(crp_lst),crp_lst[-1],(0,0,255),2)

    else:
        p1 = (0,0)
        p2 = (frame.shape[1],frame.shape[0])
        print("No ROI selected!",end='\r')

    if key == ord("c") or key == ord("C"):
        p1 = min(crp_lst)[0]/float(width),min(crp_lst)[1]/float(height)
        p2 = crp_lst[-1][0]/float(width),crp_lst[-1][1]/float(height)
        crop.append([p1,p2])
        print(len(crop)," ROIs selected")

    if key == 27:# escape key
    ### remove duplicated ROIs
        for g in crop:
            if crop.count(g) > 1:
                crop.remove(g)
        cv2.destroyAllWindows()
        break

    cv2.imshow("Crop",frame)

while(1):
    # file_list = sorted(glob.glob(str(os.path.dirname(videofile) + '/' + '*.{}'.format('mp4'))))
    # print(file_list)
    filename = (os.path.splitext(os.path.basename(videofile))[0])
    f = h5py.File('test_'+ filename +'.h5' ,'a')
    print("File: ",filename)

    for i,j in enumerate(crop):
    ### input: capture,iteration object,wait (in seconds),timelimit (in seconds),output file
        analyser = an.analyser(videofile,i,j,start,stop,f)
        analyser.start()

    for i,name in enumerate(f):
        data.append((range(0,len(analyser.cumsum(f["tank_%s/framenr"%(i+1)].value))),analyser.cumsum(f["tank_%s/framenr"%(i+1)].value)))
    reader = an.reader(f,videofile,start,stop)
    # reader.plot(data)
    # reader.cumulative_plot(data)
    f.close()
    break
