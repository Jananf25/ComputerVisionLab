import cv2
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video")

args = parser.parse_args()

MODE = "MPI"

if MODE == "COCO":
    protoFile = "./coco/pose_deploy_linevec.prototxt"
    weightsFile = "./coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI":
    protoFile = "./mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

inWidth = 368
inHeight = 368
threshold = 0.1

input_source = args.video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

if not hasFrame:
    raise RuntimeError("Could not read video file.")

save_name = os.path.splitext(os.path.basename(input_source))[0]
print(save_name)

vid_writer = cv2.VideoWriter(
    f"{save_name}_openpose.avi",
    cv2.VideoWriter_fourcc('M','J','P','G'),
    10,
    (frame.shape[1], frame.shape[0])
)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")

elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

frame_count = 0
max_frames = 30

while True:
    t = time.time()
    hasFrame, frame = cap.read()

    if not hasFrame:
        break

    frame_count += 1
    if frame_count > max_frames:
        break

    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 255,
        (inWidth, inHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False
    )

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(
        frame,
        "time taken = {:.2f} sec".format(time.time() - t),
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        .8,
        (255, 50, 0),
        2,
        lineType=cv2.LINE_AA
    )

    cv2.imshow('Output-Skeleton', frame)
    vid_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()

print("Finished OpenPose video.")