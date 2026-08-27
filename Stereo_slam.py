#Building on top of stereo VO the backend SLAM

import numpy as np
import cv2 as cv
import gtsam
import matplotlib.pyplot as plt
class SLAM():
    def __init__(self):
        self.K,self.b=self.calibration()
        self.orb=cv.ORB_create(5000)
        self.matcher=cv.BFMatcher(cv.NORM_HAMMING)
        self.prev_points3d=None
        self.prev_des=None
        self.cur_pose=np.eye(4)
        self.trajectory=[]
        self.keyframes=[]
        
        
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
        kp_list=[]
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
                    kp_list.append(left_kp.pt)
        return np.array(object_pt), np.array(ds), np.array(kp_list, dtype=np.float32)


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


    #THE BACKEND OF SLAM
    def keyframe_detection(self,i,ds,point3D,kp):
        if i%10==0:
            self.keyframes.append({
                'frame_id':i,
                'descriptors': ds,
                'point3d': point3D,
                'keypoints': kp,
                'pose': self.cur_pose.copy() #with .copy() are able to create a snapshot of the current pose 
            })
            return True
        return False


    def compute_relative_motion(self):#We will compute the relative(optimizer) to fix the absolute 
        edges=[]
        for i in range(len(self.keyframes)-1):
            pose_A=self.keyframes[i]['pose']
            pose_B=self.keyframes[i+1]['pose']
            T_rel=np.linalg.inv(pose_A)@ pose_B
            edges.append((i,i+1,T_rel))
        return edges
    

    def build_pose_graph(self):
        graph=gtsam.NonlinearFactorGraph()
        initial=gtsam.Values()
        prior_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
        odom_noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
        #the inital estimate for each node
        for i,kf in enumerate(self.keyframes) :
            initial.insert(i,gtsam.Pose3(kf['pose']))
            # Converts the 4* 4 transformation matrix kf['pose'] into a
            #  GTSAM 3D pose object (gtsam.Pose3) and assigns it to key i in the initial value dictionary.
            #anchoring the graph with a prior node zero.
        graph.add(gtsam.PriorFactorPose3(0,gtsam.Pose3(self.keyframes[0]['pose']),prior_noise))
        edges = self.compute_relative_motion()
        for (i, j, T_rel) in edges:
            graph.add(gtsam.BetweenFactorPose3(
            i, j, gtsam.Pose3(T_rel), odom_noise))
        return graph, initial


    def build_pose_graph_with_loops(self, loops):
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
        odom_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
        loop_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05]*6))
        for i, kf in enumerate(self.keyframes):
            initial.insert(i, gtsam.Pose3(kf['pose']))
        graph.add(gtsam.PriorFactorPose3(
            0, gtsam.Pose3(self.keyframes[0]['pose']), prior_noise))
        edges = self.compute_relative_motion()
        for (a, b, T_rel) in edges:
            graph.add(gtsam.BetweenFactorPose3(a, b, gtsam.Pose3(T_rel), odom_noise))
        # loop closure edges — identity approximation (same place → identity transform)
        for (current, past) in loops:
            T_loop = self.compute_loop_transform(current, past)
            if T_loop is None:
                continue
            graph.add(gtsam.BetweenFactorPose3(current, past, gtsam.Pose3(T_loop), loop_noise))
        return graph, initial


    def optimize(self, graph, initial):
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        return result


    def detect_loop(self, current_idx,THRESHOLD=250):
        cur_des = self.keyframes[current_idx]['descriptors']
        best_j = None
        best_count = 0
        for j in range(current_idx - 20):
            past_des = self.keyframes[j]['descriptors']
            matches = self.matcher.knnMatch(cur_des, past_des, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) > best_count:      # track the strongest candidate
                best_count = len(good)
                best_j = j
        if best_count > THRESHOLD:
            return best_j
        return None

    def compute_loop_transform(self, current_idx, past_idx):
        past_kf = self.keyframes[past_idx]
        cur_kf  = self.keyframes[current_idx]

        # match PAST descriptors against CURRENT descriptors
        matches = self.matcher.knnMatch(past_kf['descriptors'], cur_kf['descriptors'], k=2)

        obj_pts = []   # 3D points from the PAST keyframe
        img_pts = []   # 2D pixels in the CURRENT keyframe
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    obj_pts.append(past_kf['point3d'][m.queryIdx])   # 3D from past
                    img_pts.append(cur_kf['keypoints'][m.trainIdx])  # 2D in current

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        if len(obj_pts) < 30:          # not enough matches to trust PnP
            return None

        success, rvec, tvec, inliers = cv.solvePnPRansac(obj_pts, img_pts, self.K, None)
        if not success:
            return None

        T = np.eye(4)
        R, _ = cv.Rodrigues(rvec)
        T[:3, :3] = R
        T[:3, 3]  = tvec.flatten()
        return T        
            

    
    #The function that i will un everything 
    def process_function(self,i):
        img_l,img_r=self.load_images(i)
        point3D,ds,kp=self.stereo_detection_matching(img_l,img_r)
        if self.prev_des is None:
            self.prev_points3d = point3D
            self.prev_des=ds
            return
        obj_pts,img_pts=self.temporal_detection_matching(img_l)
        self.motion_estimation(obj_pts, img_pts)

        #keyframe after motion estimation so that we get the cur pose 
        self.keyframe_detection(i,ds,point3D,kp)

        self.prev_points3d, self.prev_des = point3D, ds #for the next iteration.

    def load_ground_truth(self):
        gt_path = "Dataset_kitti/dataset_odometry_poses/poses/00.txt"
        gt = []
        with open(gt_path, "r") as f:
            for line in f:
                if line.strip():
                    row = np.fromstring(line, sep=" ")
                    mat = row.reshape(3, 4)
                    gt.append(mat[:, 3])
        return np.array(gt)

def align(est, gt):
    mean_est = est.mean(axis=0)
    mean_gt = gt.mean(axis=0)
    est_c = est - mean_est
    gt_c = gt - mean_gt
    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mean_gt - R @ mean_est
    return (R @ est.T).T + t

def compute_ate(est, gt):
    est_aligned = align(est, gt)
    errors = np.linalg.norm(est_aligned - gt, axis=1)
    return np.sqrt(np.mean(errors**2))






vo = SLAM()
for i in range(4541):
    vo.process_function(i)

loops = []
for idx in range(len(vo.keyframes)):
    j = vo.detect_loop(idx)
    if j is not None:
        loops.append((idx, j))
print(f"{len(loops)} loops detected:", loops[:5])


graph, initial = vo.build_pose_graph_with_loops(loops)   # ← you'll extend build_pose_graph
result = vo.optimize(graph, initial)


opt_pos = np.array([[result.atPose3(i).x(),
                     result.atPose3(i).y(),
                     result.atPose3(i).z()]
                    for i in range(len(vo.keyframes))])


vo_pos = np.array([kf['pose'][:3,3] for kf in vo.keyframes])


gt_full = vo.load_ground_truth()
gt_kf = np.array([gt_full[kf['frame_id']] for kf in vo.keyframes])


ate_before = compute_ate(vo_pos, gt_kf)
ate_after  = compute_ate(opt_pos, gt_kf)
print(f"ATE before loop closure: {ate_before:.3f} m")
print(f"ATE after  loop closure: {ate_after:.3f} m")


plt.figure(figsize=(10,8))
plt.plot(gt_kf[:,0],  gt_kf[:,2],  'k-',  label="Ground truth", linewidth=2)
plt.plot(vo_pos[:,0], vo_pos[:,2], 'r--', label=f"Before (VO, ATE {ate_before:.1f}m)")
plt.plot(opt_pos[:,0], opt_pos[:,2],'g-',  label=f"After (SLAM, ATE {ate_after:.1f}m)")
plt.legend(); plt.axis("equal"); plt.xlabel("X (m)"); plt.ylabel("Z (m)")
plt.title("Stereo SLAM: Loop Closure Correction — KITTI 00")
plt.savefig("stereo_slam_kitti00.png", dpi=150, bbox_inches="tight")
plt.show()