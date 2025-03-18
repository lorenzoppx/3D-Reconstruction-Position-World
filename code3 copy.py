import json
import numpy as np
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

def calculate_wi(K,R,ui,vi):
    W = np.linalg.inv(K@R)
    hom_coord = np.array([[ui],[vi],[1]])
    W = W@hom_coord
    #print(np.shape(W))
    #print(W)
    return W

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

    coord = np.array([[coord0[i]],[coord1[i]],[coord2[i]],[coord3[i]]])
    cam = []
    for i in range(0,4):
        if np.any(np.isnan(coord[i])):
            continue
        else:
            cam.append(i)
    print("cam",cam)
    quant_cam = len(cam)

    """
    if quant_cam == 2:
        flag = 0
        W = np.array([])
        for j in cam:
            if flag==0:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                W = np.array(calculate_wi(K,R,coord[j][0][0],coord[j][0][1]))
                b = np.array(np.linalg.inv(R)@T)
                flag = 1
            else:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                W = np.hstack((W,calculate_wi(K,R,coord[j][0][0],coord[j][0][1])))
                b = np.hstack((b,np.linalg.inv(R)@T))
        I = np.identity(3)
        O = np.zeros((3,1))
        A1 = np.hstack([-I,W[:,0].reshape(-1, 1),O])
        A2 = np.hstack([-I,O,W[:,1].reshape(-1, 1)])
        A = np.vstack([A1,A2])
        print(coord)
        print(np.isnan(coord))
        lambda_ = np.linalg.pinv(A)@b.reshape(-1, 1)
        
        print("x:",lambda_[0,0]/lambda_[-1,0])
        x.append(lambda_[0,0]/lambda_[-1,0])
        print("y:",lambda_[1,0]/lambda_[-1,0])
        y.append(lambda_[1,0]/lambda_[-1,0])
        print("z:",lambda_[2,0]/lambda_[-1,0])
        z.append(lambda_[2,0]/lambda_[-1,0])
        print(np.shape(lambda_))
        print(lambda_/lambda_[-1,0])
    """
    if quant_cam == 3:
        flag = 0
        W = np.array([])
        for j in cam:
            if flag==0:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                if np.any(np.isnan(coord[j])):
                    raise nan_error
                W = np.array(calculate_wi(K,R,coord[j][0][0],coord[j][0][1]))
                b = np.array(np.linalg.inv(R)@T)
                flag = 1
            else:
                K = intr[j][0]
                R = intr[j][1]
                T = intr[j][2]
                W = np.hstack((W,calculate_wi(K,R,coord[j][0][0],coord[j][0][1])))
                b = np.hstack((b,np.linalg.inv(R)@T))
        I = np.identity(3)
        O = np.zeros((3,1))
        A1 = np.hstack([-I,W[:,0].reshape(-1, 1),O,O])
        A2 = np.hstack([-I,O,W[:,1].reshape(-1, 1),O])
        A3 = np.hstack([-I,O,O,W[:,2].reshape(-1, 1)])
        A = np.vstack([A1,A2,A3])
        print(coord)
        print(np.isnan(coord))
        #lambda_ = np.linalg.pinv(A)@b.reshape(-1, 1)
        lambda_, residuals, rank, s = np.linalg.lstsq(A, b.reshape(-1, 1), rcond=None)
        #lambda_ = np.linalg.pinv(A, rcond=1e-10) @ b.reshape(-1, 1)
        rank_A = np.linalg.matrix_rank(A)
        #if rank_A != min(A.shape):
        #    raise Error
        print("x:",lambda_[0,0])
        x.append(lambda_[0,0])
        print("y:",lambda_[1,0])
        y.append(lambda_[1,0])
        print("z:",lambda_[2,0])
        z.append(lambda_[2,0])
        print(np.shape(lambda_))
        print(lambda_)
    
    """
    #error
    if quant_cam == 4:
        W0 = calculate_wi(K0,R0,coord0[i][0],coord0[i][1])
        W1 = calculate_wi(K1,R1,coord1[i][0],coord1[i][1])
        W2 = calculate_wi(K2,R2,coord2[i][0],coord2[i][1])
        W3 = calculate_wi(K3,R3,coord3[i][0],coord3[i][1])
        I = np.identity(3)
        O = np.zeros((3,1))
        b = np.vstack((np.linalg.inv(R0)@T0,np.linalg.inv(R1)@T1,np.linalg.inv(R2)@T2,np.linalg.inv(R3)@T3))
        print("b:",b)
        A1 = np.hstack([-I,W0,O,O,O])
        A2 = np.hstack([-I,O,W1,O,O])
        A3 = np.hstack([-I,O,O,W2,O])
        A4 = np.hstack([-I,O,O,O,W3])
        A = np.vstack([A1,A2,A3,A4])
        print(coord)
        print(np.isnan(coord))
        lambda_ = np.linalg.pinv(A)@b
        
        print("x:",lambda_[0,0])
        x.append(lambda_[0,0])
        print("y:",lambda_[1,0])
        y.append(lambda_[1,0])
        print("z:",lambda_[2,0])
        z.append(lambda_[2,0])
        print(np.shape(lambda_))
        print(lambda_)
    """
import matplotlib.pyplot as plt

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

# Plot the points
ax.scatter(np.array(x).T, np.array(y).T, np.array(z).T, c='r', marker='o')

plt.show()
