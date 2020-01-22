import sys
import cv2
from mtcnn.mtcnn import MTCNN

def main (args):
	detector = MTCNN()
	image = cv2.imread(args[1])
	result = detector.detect_faces(image)

	for person in result:
	    bounding_box = person['box']
	    keypoints = person['keypoints']
	    
	    cv2.rectangle(image,
	                  (bounding_box[0], bounding_box[1]),
	                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
	                  (0,155,255),
	                  2)
	    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
	    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
	    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
	    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
	    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

	cv2.imwrite("result.jpg", image)
	cv2.imshow("image",image)
	cv2.waitKey(0)

if __name__=="__main__":
    main(sys.argv)