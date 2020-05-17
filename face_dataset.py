import cv2
import sys
 
from PIL import Image
 
def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    
    cap = cv2.VideoCapture(camera_idx)                
    
   
    classfier = cv2.CascadeClassifier("/home/pi/project/face_recg/face.xml")

 
    face_id = input('\n enter user id end press <return> ==>  ')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    
  
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)             
 
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:                                            
            for faceRect in faceRects:  
                x, y, w, h = faceRect                        
               
                #img_name = "User." + str(face_id) + '.' + str(num) + ".jpg"               
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite("/home/pi/project/dataset/User." + str(face_id) + '.' + str(num) + ".jpg", image)                                
                                
                num += 1                
                if num > (catch_pic_num):  
                    break
                
                
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        
        if num > (catch_pic_num): break                
                       
        
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("Intercepting face", 0, 100, '/home/pi/project/dataset')

