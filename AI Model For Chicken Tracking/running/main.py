import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
 
cap=cv2.VideoCapture('CCTV.mp4')
 
 
def POINTS(event,x,y,flags,param):
   if event == cv2.EVENT_MOUSEMOVE :  
       colorsBGR = [x, y]
       print(colorsBGR)
       
 
cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
 
tracker = Tracker()
 
area_1=[(80,1),(80,499),(940,499),(937,2)]
area1=set()
 
while True:
   ret,frame=cap.read()
   frame=cv2.resize(frame,(1020,500))
   cv2.polylines(frame,[np.array(area_1,np.int32)],True,(0,255,0),3)
   results=model(frame)
   list=[]
#    print(results.pandas().xyxy[0].name)
#    print(results.pandas().xyxy[0])

   for index,row in results.pandas().xyxy[0].iterrows():
    #    print(index,*row)
       x1=int(row['xmin'])
       y1=int(row['ymin'])
       x2=int(row['xmax'])
       y2=int(row['ymax'])
       b=str(row['name'])
       if 'bird' in b:
           list.append([x1,y1,x2,y2,index])
   boxes_ids=tracker.update(list)
   for box_id in boxes_ids:
       x,y,w,h,id,obid=box_id
       print(obid)
       cv2.rectangle(frame,(x,y),(w,h),(255,0,255),2)
       cv2.putText(frame,str(id)+"(CID:"+str(obid)+")",(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
       result=cv2.pointPolygonTest(np.array(area_1,np.int32),(int(w),int(h)),False)
       if result > 0:
           area1.add(id)                
   p=len(area1)
   cv2.putText(frame,str(p),(20,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
   
     
   cv2.imshow('FRAME',frame)
   if cv2.waitKey(1)&0xFF==27:
       break
cap.release()
cv2.destroyAllWindows()