import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image=[]
while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), (0, 255, 0), 2)
		crop_img = frame[y:y + h, x:x + w]
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	image.append(crop_img)
cap.release()
cv2.destroyAllWindows()
cv2.imshow("Face",image[0])
cv2.waitKey(0)
cv2.imwrite("tf_files/face.png",image[0])
cv2.destroyAllWindows()