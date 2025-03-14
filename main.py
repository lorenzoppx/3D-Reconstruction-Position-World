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

# Função para calcular W_i a partir dos parâmetros da câmera e coordenadas da imagem
def compute_Wi(K, R, m_i):
    m_i_h = np.array([m_i[0], m_i[1], 1]).reshape(3,1)
    W_i = np.linalg.inv(K @ R) @ m_i_h
    return W_i

# Função para construir o sistema linear A * X = B
def build_system(K_list, R_list, T_list, points):
    A = []
    B = []
    for K, R, T, m in zip(K_list, R_list, T_list, points):
        Wi = compute_Wi(K, R, m).flatten()

        # Adiciona linhas à matriz A
        A.append(np.hstack([-np.eye(3), Wi.reshape(-1, 1)]))
        B.append(np.linalg.inv(R) @ T)

    A = np.vstack(A)
    B = np.vstack(B)
    return A, B

# Função para resolver o sistema usando pseudo-inversa
def solve_system(A, B):
    return np.linalg.pinv(A) @ B

# Função para exibir os pontos reconstruídos no espaço 3D
def plot_3D(coord_videos):
    if coord_videos.size == 0:
        print("Nenhum ponto reconstruído para exibir.")
        return

    W_x, W_y, W_z = coord_videos[:, 0], coord_videos[:, 1], coord_videos[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(W_x, W_y, W_z, c='red', marker='o', label="Pontos reconstruídos")
    ax.plot(W_x, W_y, W_z, c='blue', linestyle='dashed')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajetória do Objeto no Espaço 3D")
    ax.legend()
    plt.show()

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
coord_videos = np.array([])
K_list = []
R_List = []
T_List = []
have_aruco = 0
for video,calibration in zip(videos,calibrations):
    coord_video = np.array([])
    K0, R0, T0, res0, dis0 = camera_parameters('calibracao/' + calibration)
    K_list.append(K0)
    R_List.append(R0)
    T_List.append(T0)
    cap = cv2.VideoCapture("videos/" + video)
    while True:
        #captura um frame do video
        ret, frame = cap.read()
        if ret:
            frame_out = np.copy(frame)
            # Detect the markers in the image
            markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)
            # Draw aruco markers
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners,markerIds)
            
            coords = np.array([])

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
                        
                        if len(coord_video)==0:
                            coord_video = np.array(coords)
                        else: 
                            coord_video = np.vstack((coord_video,coords))
                        
            else:
                have_aruco = 0
                # Quando nao tem aruco reconhecido empilha [np.nan,np.nan]
                if len(coords)==0:
                    coords = np.array([np.nan,np.nan])
                else:
                    coords = np.vstack((coords,np.array([np.nan,np.nan])))   
                if len(coord_video)==0:
                    coord_video = np.array(coords)
                else: 
                    coord_video = np.vstack((coord_video,coords))
            time.sleep(0.03)
            cv2.imshow('ImageFrame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            np.save(video.replace(".mp4",""),coords)
            print(len(coord_video))
            if(len(coord_video)==197): #temporario
                if len(coord_videos)==0:
                    coord_videos = coord_video
                else: 
                    coord_videos = np.column_stack((coord_videos, coord_video))
                print(coord_videos)
            break
        


    #print(np.max(frame))

cap.release()
cv2.destroyAllWindows()


coord_trajeto = np.array([])
for linha in coord_videos:
    pts = []
    for i in range(0, linha.shape[0], 2):
        if(linha[i] != np.nan and linha[i+1] != np.nan):
            pts.append([linha[i], linha[i+1]])
    if len(pts)==0:
        break
    A, B = build_system(K_list, R_List, T_List, pts)
    X = solve_system(A, B)  # Coordenada 3D estimada

    # Adiciona as coordenadas reconstruídas ao array de trajetórias
    new_coords = np.array([[X[0,0], X[1,0], X[2,0]]])
    if len(coord_trajeto)==0:
        coord_trajeto = new_coords
    else:
        coord_trajeto = np.vstack((coord_trajeto, new_coords))
    print("Coordenada 3D reconstruída:", X[:3].flatten())
plot_3D(coord_trajeto) 
