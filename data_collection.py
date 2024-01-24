import mediapipe as mp 
import numpy as np 
import cv2 

# Starting video camera capture
cap = cv2.VideoCapture(0)
# Naming the Emotion data
name = input("Enter the name of the data : ")


# Holistics pipeline generates seperate models for pose, face and hand components
holistic = mp.solutions.holistic
# Hands Data
hands = mp.solutions.hands
# Holistics pipeline calling
holis = holistic.Holistic()
# Drawing holistics on window
drawing = mp.solutions.drawing_utils

# Emotions List
X = []
# Daat Size
data_size = 0

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

		# Appending list and incresing the size
		X.append(lst)
		data_size = data_size+1

	# Applying FaceMesh
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	# Left hand Skeleton
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	# Right hand Skeleton
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	# Putting predicted data on window
	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
	# Showing Camera Frames Window
	cv2.imshow("window", frm)
	# Loop out from the Process
	if cv2.waitKey(1) == 27 or data_size>999:
		cv2.destroyAllWindows()
		cap.release()
		break

# Saving emotions as a Numpy Array
np.save(f"{name}.npy", np.array(X))
# Printing the Array Shape
print(np.array(X).shape)
