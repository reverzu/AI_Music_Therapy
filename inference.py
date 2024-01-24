import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


# Loading Model
model  = load_model("model.h5")

# Loading Labels
label = np.load("labels.npy")


# Holistics pipeline generates seperate models for pose, face and hand components
holistic = mp.solutions.holistic
# Hands Data
hands = mp.solutions.hands
# Holistics pipeline calling
holis = holistic.Holistic()
# Drawing holistics on window
drawing = mp.solutions.drawing_utils

# Starting video camera capture
cap = cv2.VideoCapture(0)


# Reading video camera frames
while True:
	# Frame List
	lst = []

	# Frames capture
	_, frm = cap.read()

	# Frames flipping to mirror
	frm = cv2.flip(frm, 1)

	# Holistics Result
	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))


	# Reading Facial Landmarks
	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)
		# Left Hand
		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		# Right Hand
		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		# Reshaping the Numpy Array
		lst = np.array(lst).reshape(1,-1)
		# Predicting facial emotional data
		pred = label[np.argmax(model.predict(lst))]

		# Printing Results
		print(pred)
		# Putting predicted data on window
		cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

	# Applying FaceMesh
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	# Left hand Skeleton
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	# Right hand Skeleton
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	# Showing Camera Frames Window
	cv2.imshow("window", frm)

	# Loop out from the Process
	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

