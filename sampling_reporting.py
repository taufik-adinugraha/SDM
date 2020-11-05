# import the necessary packages
from imutils.video import VideoStream
import detection
import numpy as np
import argparse
import imutils
import cv2
import time
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--time_shot", type=int, default=5, help="path to (optional) output video file")
ap.add_argument("--save", action='store_true')
args = vars(ap.parse_args())

if args["save"]:
	os.system('rm -rf images_out')
	os.system('mkdir images_out')

# load the COCO class labels our YOLO model was trained on
labelsPath = 'models/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = 'models/yolov3.weights'
configPath = 'models/yolov3.cfg'

# load YOLO object detector trained on COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
step = 0
percent, Npeople = [], []
# initialize the set of indexes that violate the minimum social distance
violate = set()
t1 = time.time()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=700)
	(h, w) = frame.shape[:2]

	t2 = time.time()
	# AI is active for every 5 seconds
	if t2-t1 > args["time_shot"]:
		t1 = t2

		objects = ['person']
		results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in objects if LABELS.index(i) is not None])

		# initialize the set of indexes that violate the minimum social distance
		violate = set()

		# ensure there are at least two people detections
		if len(results) >= 2:
			# extract all centroids from the results
			centroids = np.array([r[3] for r in results])
			# get the widths of bounding boxes
			dXs = [r[2][2]-r[2][0] for r in results]

			for i in range(len(results)):
				c1 = centroids[i]
				for j in range(i + 1, len(results)):
					c2 = centroids[j]
					Dx, Dy = np.sqrt((c2[0]-c1[0])**2), np.sqrt((c2[1]-c1[1])**2)
					thresX = (dXs[i] + dXs[j]) * 0.7
					thresY = (dXs[i] + dXs[j]) * 0.25
					# check to see if the distance between any pairs is less than the threshold
					if Dx<thresX and Dy<thresY:
						# update our violation set with the indexes of the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
		for (i, (classID, prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			dX, dY = endX-startX, endY-startY
			(cX, cY) = centroid[0], centroid[1]-dY//2
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then update the color
			if i in violate:
				color = (0, 0, 255)

			# draw a bounding box around the person
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			# cv2.circle(frame, (cX, cY), 4, (255,255,255), -1)
		
		# save snapshot
		if args["save"]:
			cv2.imwrite(f'images_out/{round(time.clock()*1e6)}.jpg', frame)

		# calculate the average of social distancing violations per minute
		if len(results) != 0:
			percent.append(len(violate) / len(results) * 100)
			Npeople.append(len(results))
		step += 1
		# report every 1 minute (averaging 12 data points)
		if step == 60//args["time_shot"]:
			step = 0
			if percent:
				out = (round(np.mean(percent),2), round(np.mean(Npeople),1))
			else:
				out = (0,0)
			percent = []
			print(f'Jumlah Pelanggaran: {out[0]} % \t\t Jumlah Orang: {out[1]}')
			



