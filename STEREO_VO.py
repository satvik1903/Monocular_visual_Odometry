#for stereo vo
import numpy as np
import cv2 as cv

class Stereo_VO():

    def __init__(self):
        self.K,self.b=self.calibration()
        self.orb=cv.ORB_create(5000)
        self.matcher=cv.BFMatcher(cv.NORM_HAMMING)
        self.prev_points3d=None
        self.prev_des=None
        self.cur_pose=np.eye(4)
        self.trajectory=[]
        
        
        
        

    def load_images(self,i):
        path_l=f"Dataset_kitti/dataset_odometry_gray/sequences/00/image_0/{i:06d}.png"
        path_r=f"Dataset_kitti/dataset_odometry_gray/sequences/00/image_1/{i:06d}.png"

        img_l=cv.imread(path_l,0)
        img_r=cv.imread(path_r,0)
        #passed the loading check 
        if img_l is None or img_r is None:
            raise FileNotFoundError(f"Image not found")
        return img_l,img_r

    def calibration(self):
        path=f"Dataset_kitti/dataset_odometry_calib/sequences/00/calib.txt"
        with open(path,'r') as f:
            P0=f.readline().split()
            P1=f.readline().split()
        P0drop=P0[1:]
        p0array=np.asarray(P0drop,dtype='float')
        K_matrix=p0array.reshape(3,4)
        K=np.delete(K_matrix,3,axis=1)
        self.fx=K[0,0]
        self.fy=K[1,1]
        self.cx=K[0,2]
        self.cy=K[1,2]

        P1drop=P1[1:]
        p1array=np.asarray(P1drop,dtype='float')
        p1shape=p1array.reshape(3,4)
        tp1=p1shape[:,-1]
        b=-tp1[0]/self.fx
        return K,b

    def stereo_detection_matching(self,img_l,img_r,):
        
        kpl,dsl=self.orb.detectAndCompute(img_l,None)
        kpr,dsr=self.orb.detectAndCompute(img_r,None)

        matches=self.matcher.knnMatch(dsl,dsr,k=2)
        object_pt=[]
        ds=[]
        for pair in matches:
            if len(pair)==2:
                m,n=pair
                if m.distance <0.75*n.distance:
                    left_kp=kpl[m.queryIdx] #m.queryIdx is an attribue #that is index of keypoint 
                    u,v=left_kp.pt # .pt is the atttribute of the keypoint object that hold 2D image coordinates as floating tuples
                    right_kp=kpr[m.trainIdx]
                    xr,yr=right_kp.pt
                    if abs(v-yr)>2: 
                        continue #Continue skips the match that did not survive 
                    d=u-xr
                    if d<=0:
                        continue
                    Z=self.fx*self.b/d
                    X=Z*(u-self.cx)/self.fx
                    Y=Z*(v-self.cy)/self.fy
                    object_pt.append([X,Y,Z])
                    ds.append(dsl[m.queryIdx])
        return np.array(object_pt), np.array(ds)

    def temporal_detection_matching(self,img_l):
        kp_cur,ds_cur=self.orb.detectAndCompute(img_l,None)
        matcherl=self.matcher.knnMatch(self.prev_des,ds_cur,k=2)
        object_pts=[] #this are the 3d points of the object.
        image_pts=[] #this are the next in line pixel that the object falls on.
        for m,n in matcherl:
            if m.distance<0.75*n.distance:
                object_pts.append(self.prev_points3d[m.queryIdx])
                image_pts.append(kp_cur[m.trainIdx].pt)

        return np.array(object_pts, dtype=np.float32), np.array(image_pts, dtype=np.float32)

    def motion_estimation(self,obj_pts,img_pts):
        success,Rvec,tvec,inliers=cv.solvePnPRansac(obj_pts,img_pts,self.K,None)
        if not success or inliers is None or len(inliers) < 30:
            return

# keep only the inlier correspondences
        inlier_obj = obj_pts[inliers.flatten()]
        inlier_img = img_pts[inliers.flatten()]

        # re-solve on the clean subset, using the RANSAC pose as starting guess
        success, Rvec, tvec = cv.solvePnP(
            inlier_obj, inlier_img, self.K, None,
            Rvec, tvec, useExtrinsicGuess=True
        )
        T=np.eye(4)
        R, _ = cv.Rodrigues(Rvec)#convert the vector to a 3*3 rotation matrix
        T[:3, :3] = R      # the rotation matrix from Rodrigues (NOT inverted)
        T[:3, 3] = tvec.flatten()   # tvec is (3,1); flatten to (3,) to fit the column slot
        T_cam = np.linalg.inv(T) #this is to get camera-world from world-camera
        self.cur_pose = self.cur_pose @ T_cam
        self.trajectory.append(self.cur_pose[:3, 3])

    #The function that i will un everything 
    def process_function(self,i):
        img_l,img_r=self.load_images(i)
        point3D,ds=self.stereo_detection_matching(img_l,img_r)
        if self.prev_des is None:
            self.prev_points3d = point3D
            self.prev_des=ds
            return
        obj_pts,img_pts=self.temporal_detection_matching(img_l)
        self.motion_estimation(obj_pts, img_pts)

        self.prev_points3d, self.prev_des = point3D, ds #for the next iteration.

    def load_ground_truth(self):
            gt_path="Dataset_kitti/dataset_odometry_poses/poses/00.txt"
            gt=[]
            with open(gt_path,"r") as f:
                for line in f:
                    if line.strip():
                        row=np.fromstring(line,sep=" ")
                        mat=row.reshape(3,4)
                        traje=mat[:,3]
                        gt.append(traje)
            return np.array(gt)

def align(est,gt):
        mean_est=est.mean(axis=0)
        mean_gt=gt.mean(axis=0)
        est_c=est-mean_est
        gt_c=gt-mean_gt
        H=est_c.T @gt_c
        U,S,Vt=np.linalg.svd(H) #U=orientation of estimated data Vt=Orientation of ground truth S=how strongly those direction correspond
        R=Vt.T@U.T
        if np.linalg.det(R) < 0: #If R is anywere negative that means there is a reflection which we do not want
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mean_gt - R @ mean_est
        return (R @ est.T).T + t

def compute_ate(est, gt):
    est_aligned = align(est, gt)
    errors = np.linalg.norm(est_aligned - gt, axis=1)   # per-frame distance
    return np.sqrt(np.mean(errors**2))                   # RMS

def compute_rpe(est, gt):
    # relative (frame-to-frame) motions
    est_rel = est[1:] - est[:-1]     # each row: motion from frame i to i+1, estimated
    gt_rel  = gt[1:]  - gt[:-1]      # same, ground truth
    diff    = est_rel - gt_rel       # how wrong each relative step is
    errors  = np.linalg.norm(diff, axis=1)   # distance error per step
    rpe = np.sqrt(np.mean(errors**2))        # RMS
    return rpe

svo = Stereo_VO()
N = 4541                      # full sequence
for i in range(N):
    svo.process_function(i)

est = np.array(svo.trajectory)          # (N-1, 3)
gt  = svo.load_ground_truth()[:len(est)]  # slice GT to match

ate = compute_ate(est, gt)          # your mono ATE fn
print("ATE RMSE:", ate)

rpe = compute_rpe(est, gt)
print("ATE RMSE:", ate)
print("RPE RMSE:", rpe)

import matplotlib.pyplot as plt
plt.plot(gt[:,0],  gt[:,2],  label="Ground truth")
plt.plot(est[:,0], est[:,2], label="Stereo VO")
plt.legend(); plt.axis("equal"); plt.xlabel("X (m)"); plt.ylabel("Z (m)")
plt.title("Stereo VO vs Ground Truth — KITTI 00")
plt.show()