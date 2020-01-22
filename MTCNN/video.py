import sys
import cv2
from mtcnn.mtcnn import MTCNN

def main (args):
    cap = cv2.VideoCapture(args[1])
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter('result.avi', codec, fps, size)
    while True: 
        #Capture frame-by-frame
        __, frame = cap.read()    
        #Use MTCNN to detect faces
        detector = MTCNN()
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                bounding_box = person['box']
                keypoints = person['keypoints']
        
                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0,155,255),
                              2)    
                cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
        #display resulting frame
        cv2.imshow('frame',frame)
        output.write(frame)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    #When everything's done, release capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(sys.argv)