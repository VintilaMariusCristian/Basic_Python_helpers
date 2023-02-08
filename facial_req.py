#! /usr/bin/python

# import the necessary packages
import PySimpleGUI as sg
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import ctypes
import time
import requests

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#### trimitere mail , configurare functie si serviciu api
def send_message(name):
    return requests.post(
        "https://api.mailgun.net/v3/sandboxc5b43439f39040a7b6528f627b99f9d1.mailgun.org/messages",
        auth=("api", "fc0a956fc427e9a978309173ec9ea6eb-1553bd45-6669c5d9"),
        files = [("attachment", ("image.jpg", open("image.jpg", "rb").read()))],
        data={"from": 'Licenta Andreea Elena Circeag <andreeaelena1898@gmail.com>',
            "to": ["andreeaelenac18@gmail.com"],
            "subject": "You have a visitor",
            "html": "<html>" + name + " is at your door.  </html>"})


#########################################################################3

from datetime import datetime, timedelta
#########################################
#banner 1
layout = [
		[sg.Text("ACCEPTAM O SINGURA PERSOANA IN CAMERA ",size=(80, 2))],
			  [sg.Text(size=(60,1), key='-OUTPUT-')],
			  [sg.Button('Ok')]
		]

# Create the window
window = sg.Window('Reguli de conduita', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Ok':
        break
    # Output a message to the window
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] )

# Finish up by removing from the screen
window.close()



#########################################

time.sleep(3.0)

#########################################
# banner 2

layout = [
		[sg.Text("INCEPE PROCESUL DE RECUNOASTERE FACIALA",size=(80, 2))],
		[sg.Text("                                       ",size=(80, 2))],
		[sg.Text("VA ROG DEPLASATI CAPUL IN STANGA SI DREAPTA , APOI TINETI CAPUL NEMISCAT ",size=(80, 2))],
			  [sg.Text(size=(60,1), key='-OUTPUT-')],
			  [sg.Button('Ok')]
		]

# Create the window
window = sg.Window('Reguli de conduita', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Ok':
        break
    # Output a message to the window
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] )

# Finish up by removing from the screen
window.close()





#########################################

#Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"

room_limit = "overcrowded"
overcrowded = 0

#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
#use this xml file
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("incarcam fisierele encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# initialize the video stream and allow the camera sensor to warm up

print(" incepem verificarea faciala ")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# memoram timpul de incepere al executiei
start_time = time.time()
durata = 5
# loop over frames from the video file stream
while (int(time.time()-start_time) < durata):
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face

			## verificam daca sunt cate persoane trebuie in camera


			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1



			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
			
			#If someone in your dataset is identified, print their name on the screen

			if currentname != name:
				currentname = name
				print(currentname)




		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15

			#time.sleep(3)
		#if len(counts) == 1 :
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,.8, (0, 255, 255), 2)



	# se fiseaza timp de 10 secunde imagine cu utilizatorul , perioada in care acesta executa instrucitunile afisate pe banner



	##### rahat facut de mine

	cv2.imshow("Recunoastere Faciala", frame)
	key = cv2.waitKey(1) & 0xFF


	# numara cadrele pe scunda (FPS)
	fps.update()



# opreste numaratoarea fps urilor si afiseaza in consola informatia
fps.stop()


print("FPS : {:.2f}".format(fps.fps()))
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



img_name = "image.jpg"
cv2.imwrite(img_name, frame)
print('Taking a picture.')


time.sleep(2)
###################3## face counter
image = cv2.imread('image.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayImage)

if len(faces) == 0:
	print ("No faces found")

else:
	print( "Number of faces detected: " + str(faces.shape[0]))

	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

	cv2.rectangle(image, ((0, image.shape[0] - 25)), (270, image.shape[0]), (255, 255, 255), -1)
	cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0, image.shape[0] - 10),
				cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)


if ( str(faces.shape[0]) != '1' ):
	overcrowded = 1





currentname1 = "unknown"
#BANNER 3

if(overcrowded != 1):


	if currentname == currentname1:
		layout = [
			[sg.Text("UTILIZATORUL NU ESTE INREGISTRAT IN BAZA DE DATE , VA ROG CONTACTATI BIROUL DE INREGISTRARI", size=(80, 2))],
			[sg.Text("ADMINISTRATIA VA FI  INSTIINTATA DE PREZENTA DUMNEAVOASTRA", size=(80, 2))],
			[sg.Text(currentname, size=(80, 2))],
			[sg.Text(size=(60, 1), key='-OUTPUT-')],
			[sg.Button('Ok')]
		]

	else:


		layout = [
			[sg.Text("BINE AI VENIT ", size=(80, 2))],
			[sg.Text("ADMINISTRATIA VA FI  INSTIINTATA DE PREZENTA DUMNEAVOASTRA", size=(80, 2))],
			[sg.Text(currentname, size=(80, 2))],
			[sg.Text(size=(60, 1), key='-OUTPUT-')],
			[sg.Button('Ok')]
		]

else:

	layout = [
			[sg.Text("Regulament incalcat ", size=(80, 2))],
			[sg.Text("In camera este permisa o singura persoana", size=(80, 2))],
			[sg.Text(currentname, size=(80, 2))],
			[sg.Text(size=(60, 1), key='-OUTPUT-')],
			[sg.Button('Ok')]
		]



# Create the window
window = sg.Window('Reguli de conduita', layout)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Ok':
        break
    # Output a message to the window
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] )



cv2.imshow('Image with faces', image)


# Finish up by removing from the screen
window.close()




request = send_message(currentname)
print ('Status Code: '+format(request.status_code))

