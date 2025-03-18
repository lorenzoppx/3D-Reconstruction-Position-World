# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils
import time
import json

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

# Carrega a imagem a ser substituida
img_subs = cv2.imread('hello.jpg')
img_subs = cv2.cvtColor(img_subs, cv2.COLOR_BGR2RGB)
# Get the limits of the image that will be inserted in the original one
[l,c,ch] = np.shape(img_subs)
#print(l,' ',c,' ',' ',ch)

# Source points are the corners of the image that will be warped
pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])

#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values
aruco_id = 0
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(aruco_id)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

count = 0

videos = ["camera-00.mp4","camera-01.mp4","camera-02.mp4","camera-03.mp4"]
calibrations = ["0.json","1.json","2.json","3.json"]


# Processa o vídeo para recuperar o centro dos arucos com id == 0
have_aruco = 0
for video,calibration in zip(videos,calibrations):
    coord_video = np.array([])
    cap = cv2.VideoCapture("videos/" + video)
    coords = np.array([])
    while True:
        #captura um frame do video
        ret, frame = cap.read()
        if ret:
            h,  w = frame.shape[:2]
            print('Image size: ',h,' ',w)
            frame_out = np.copy(frame)
            # Detect the markers in the image
            markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)
            # Draw aruco markers
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners,markerIds)

            if len(markerCorners) != 0:
                have_aruco = 1
                for mark,id in zip(markerCorners,markerIds):
                    # Testa se o aruco é o correto e está presente no frame
                    if id[0]==0:
                        # Define the source and destiny point for calculating the homography
                        # Destiny points are the corners of the marker
                        pts_dst = np.array(mark[0])
                        #print(np.shape(pts_dst))
                        frame_0 = []
                        frame_1 = []
                        for i in range(0,np.shape(pts_dst)[0]):
                            frame_0.append(pts_dst[i][0])
                            frame_1.append(pts_dst[i][1])
                            frame = cv2.circle(frame,(int(pts_dst[i][0]),int(pts_dst[i][1])),4,(150,150,0),3)
                        frame_0_mean = int(np.mean(frame_0))
                        frame_1_mean = int(np.mean(frame_1))
                        #print(frame_0_mean,frame_1_mean)

                        # Calculate Homography
                        h, status = cv2.findHomography(pts_src, pts_dst)

                        # Warp source image to destination based on homography
                        warped_image = cv2.warpPerspective(img_subs, h, (frame.shape[1],frame.shape[0]))
                        warp_out = np.copy(warped_image)
                    
                        # Prepare a mask representing region to copy from the warped image into the original frame.
                        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)

                        cv2.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv2.LINE_AA)

                        # Erode the mask to not copy the boundary effects from the warping
                        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                        mask = cv2.erode(mask, element, iterations=3)

                        # Copy the mask into 3 channels.
                        warped_image = warped_image.astype(float)
                        mask3 = np.zeros_like(warped_image)

                        for i in range(0, 3):

                            mask3[:,:,i] = mask/255

                        # Copy the masked warped image into the original frame in the mask region.

                        warped_image_masked = cv2.multiply(warped_image, mask3)
                        frame_masked = cv2.multiply(frame.astype(float), 1-mask3)

                        frame = cv2.add(warped_image_masked, frame_masked)
                        #frame *= 255
                        frame = np.uint8(frame)
                        frame = cv2.circle(frame,(frame_0_mean,frame_1_mean),10,(0,0,255),3)
                        
                        if len(coords)==0:
                            coords = np.array([frame_0_mean,frame_1_mean])
                        else:
                            coords = np.vstack((coords,np.array([frame_0_mean,frame_1_mean])))
                        
                        print("Exit")


            else:
                have_aruco = 0
                # Quando nao tem aruco reconhecido empilha [np.nan,np.nan]
                if len(coords)==0:
                    coords = np.array([np.nan,np.nan])
                else:
                    coords = np.vstack((coords,np.array([np.nan,np.nan])))
                    
            time.sleep(0.03)
            cv2.imshow('ImageFrame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            np.save(video.replace(".mp4",""),coords)
            break

cap.release()
cv2.destroyAllWindows()

# Processa as coordenadas para recuperar a informação 3D
coord0 = np.load(f"camera-00.npy")
coord1 = np.load(f"camera-01.npy")
coord2 = np.load(f"camera-02.npy")
coord3 = np.load(f"camera-03.npy")
print(coord0)

# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis

K0, R0, T0, res0, dis0 = camera_parameters('calibracao/0.json')
K1, R1, T1, res1, dis1 = camera_parameters('calibracao/1.json')
K2, R2, T2, res2, dis2 = camera_parameters('calibracao/2.json')
K3, R3, T3, res3, dis3 = camera_parameters('calibracao/3.json')
intr = [[K0, R0.T, -R0.T@T0, res0, dis0],
        [K1, R1.T, -R1.T@T1, res1, dis1],
        [K2, R2.T, -R2.T@T2, res2, dis2],
        [K3, R3.T, -R3.T@T3, res3, dis3]]

x = []
y = []
z = []
count = 0
for i in range(0,len(coord1)):
    print("Count:",count)
    count+=1

    coord = np.array([[coord0[i]],[coord1[i]],[coord2[i]],[coord3[i]]], dtype=np.float64)
    cam = []
    for i in range(0,4):
        if np.any(np.isnan(coord[i])):
            continue
        else:
            cam.append(i)
    print("cam",cam)
    quant_cam = len(cam)
    if quant_cam>=2:
        flag = 0
        offset = 1
        B = np.array([])
        for j in cam:
            if flag==0:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                pi0 = np.hstack((np.identity(3),np.zeros((3,1))))
                G = np.vstack((np.hstack((R,T)),np.array([0.0,0.0,0.0,1.0])))
                P = K@pi0@G
                minus_m_til = np.array([float(-coord[j][0][0]),float(-coord[j][0][1]),-1]).reshape(-1,1)
                B = np.hstack((P,minus_m_til,np.zeros((3,len(cam)-1))))
                b = np.zeros((3,1))
                flag = 1
            else:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                pi0 = np.hstack((np.identity(3),np.zeros((3,1))))
                G = np.vstack((np.hstack((R,T)),np.array([0.0,0.0,0.0,1.0])))
                P = K@pi0@G
                minus_m_til = np.array([float(-coord[j][0][0]),float(-coord[j][0][1]),-1]).reshape(-1,1)
                Bi = np.hstack((P,np.zeros((3,offset)),minus_m_til,np.zeros((3,len(cam)-1-offset))))
                B = np.vstack((B,Bi))
                b = np.vstack((b,np.zeros((3,1))))
                offset+=1
        print(coord)
        print(np.isnan(coord))
        #lambda_ = np.linalg.pinv(B)@b
        U,S,V = np.linalg.svd(B)
        X = V[-1,:4]
        if X[3]==0:
            X[3]==1e-10
        lambda_ = X / X[3]
        print("x:",lambda_[0])
        x.append(lambda_[0])
        print("y:",lambda_[1])
        y.append(lambda_[1])
        print("z:",lambda_[2])
        z.append(lambda_[2])
        print(np.shape(lambda_))
        print(lambda_)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")
ax.set_title("Trajetória do Objeto no Espaço 3D")

# Plot the points
ax.scatter(np.array(x).T, np.array(y).T, np.array(z).T, c='r', marker='o')

plt.show()

