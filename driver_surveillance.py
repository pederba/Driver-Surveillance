import cv2 
import numpy as np
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

frames = []
EAR_calibration_data_eyes_open = []
EAR_calibration_data_eyes_closed = []
threshold_EAR = 0
time_elapsed = 0

capture = cv2.VideoCapture(0)

while capture.isOpened():
    success, frame = capture.read()
    start = time.time()
    

    if not success:
        print("Capture failed")
        break

    frame.flags.writeable = False

    results = face_mesh.process(frame)

    frame.flags.writeable = True

    frame_height, frame_width, frame_channels = frame.shape

    # The camera matrix
    focal_length = 1 * frame_width
    cam_matrix = np.array([ [focal_length, 0, frame_height / 2],
    [0, focal_length, frame_width / 2],
    [0, 0, 1]])
    # The distorsion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Eye Gaze (Iris Tracking)
    
    # Left eye indices list
    #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    # Right eye indices list
    #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
    #LEFT_IRIS = [473, 474, 475, 476, 477]
    #RIGHT_IRIS = [468, 469, 470, 471, 472]

    point_right_eye_right = []
    point_right_eye_bottom = [] 
    point_right_eye_left = [] 
    point_right_eye_top = [] 

    point_left_eye_right = [] 
    point_left_eye_bottom = []
    point_left_eye_left= []
    point_left_eye_top = []

    point_right_eye_iris_center = []
    point_left_eye_iris_center = []

    right_eye_2d = []
    right_eye_3d = []
    left_eye_2d = []
    left_eye_3d = []
    face_2d = []
    face_3d = []

    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index, landmark in enumerate(face_landmarks.landmark):

                if index == 33:
                    point_right_eye_right = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 145:
                    point_right_eye_bottom = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 133:
                    point_right_eye_left = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 159:
                    point_right_eye_top = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 362:
                    point_left_eye_right = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 374:
                    point_left_eye_bottom = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 263:
                    point_left_eye_left = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 386:
                    point_left_eye_top = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 468:
                    point_right_eye_iris_center = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(255, 255, 0), thickness=-1)                    

                if index == 469:
                    point_469 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 255, 0), thickness=-1)

                if index == 470:
                    point_470 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 255, 0), thickness=-1)

                if index == 471:
                    point_471 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 255, 0), thickness=-1)

                if index == 472:
                    point_472 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 255, 0), thickness=-1)

                if index == 473:
                    point_left_eye_iris_center = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 255, 255), thickness=-1)

                if index == 474:
                    point_474 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(255, 0, 0), thickness=-1)

                if index == 475:
                    point_475 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(255, 0, 0), thickness=-1)

                if index == 476:
                    point_476 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(255, 0, 0), thickness=-1)

                if index == 477:
                    point_477 = (landmark.x * frame_width, landmark.y * frame_height)
                    #cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(255, 0, 0), thickness=-1)

                if index == 33 or index == 263 or index == 1 or index == 61 or index == 291 or index == 199:

                    if index == 1:
                        nose_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        nose_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    face_2d.append((x, y))
                    face_3d.append((x, y, landmark.z * 3000))

                #LEFT_IRIS = [473, 474, 475, 476, 477]
                if index == 473 or index == 362 or index == 374 or index == 263 or index == 386: # iris points
                #if index == 473 or index == 474 or index == 475 or index == 476 or index == 477: # eye border
                    if index == 473:
                        left_pupil_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        left_pupil_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    left_eye_2d.append((x, y))
                    left_eye_3d.append((x, y, landmark.z * 3000))

                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if index == 468 or index == 33 or index == 145 or index == 133 or index == 159: # iris points
                # if index == 468 or index == 469 or index == 470 or index == 471 or index == 472: # eye border
                    if index == 468:
                        right_pupil_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        right_pupil_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    right_eye_2d.append((x, y))
                    right_eye_3d.append((x, y, landmark.z * 3000))
                    

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            left_eye_2d = np.array(left_eye_2d, dtype=np.float64)
            left_eye_3d = np.array(left_eye_3d, dtype=np.float64)
            right_eye_2d = np.array(right_eye_2d, dtype=np.float64)
            right_eye_3d = np.array(right_eye_3d, dtype=np.float64)

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            success_left_eye, rotation_vector_left_eye, translation_vector_left_eye = cv2.solvePnP(left_eye_3d, left_eye_2d, cam_matrix, dist_matrix)
            success_right_eye, rotation_vector_right_eye, translation_vector_right_eye = cv2.solvePnP(right_eye_3d, right_eye_2d, cam_matrix, dist_matrix)
            
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
            rotation_matrix_left_eye, jacobian_left_eye = cv2.Rodrigues(rotation_vector_left_eye)
            rotation_matrix_right_eye, jacobian_right_eye = cv2.Rodrigues(rotation_vector_right_eye)

            euler_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
            euler_angles_left_eye, mtxR_left_eye, mtxQ_left_eye, Qx_left_eye, Qy_left_eye, Qz_left_eye = cv2.RQDecomp3x3(rotation_matrix_left_eye)
            euler_angles_right_eye, mtxR_right_eye, mtxQ_right_eye, Qx_right_eye, Qy_right_eye, Qz_right_eye = cv2.RQDecomp3x3(rotation_matrix_right_eye)

            pitch = euler_angles[0] * 180 / np.pi
            yaw = -euler_angles[1] * 1800
            roll = 180 + (np.arctan2(point_right_eye_right[1] - point_left_eye_left[1], point_right_eye_right[0] - point_left_eye_left[0]) * 180 / np.pi)

            if roll > 180:
                roll = roll - 360

            pitch_left_eye = euler_angles_left_eye[0] * 1800
            yaw_left_eye = euler_angles_left_eye[1] * 1800
            pitch_right_eye = euler_angles_right_eye[0] * 1800
            yaw_right_eye = euler_angles_right_eye[1] * 1800

            print("Pitch: ", pitch)
            # print("Yaw: ", yaw)
            # print("Roll: ", roll)
            # print("Pitch left eye: ", pitch_left_eye)
            # print("Yaw left eye: ", yaw_left_eye)
            # print("Pitch right eye: ", pitch_right_eye)
            # print("Yaw right eye: ", yaw_right_eye)

            # if abs(pitch + pitch_left_eye + pitch_right_eye) + abs(yaw + yaw_left_eye + yaw_right_eye) > 30:
            #     cv2.putText(frame, "DISTRACTED", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                

            # 4.4. - Draw the positions on the frame
            l_eye_width = point_left_eye_left[0] - point_left_eye_right[0]
            l_eye_height = point_left_eye_bottom[1] - point_left_eye_top[1]
            l_eye_center = [(point_left_eye_left[0] + point_left_eye_right[0])/2 ,(point_left_eye_bottom[1] + point_left_eye_top[1])/2]

            #cv2.circle(frame, (int(l_eye_center[0]), int(l_eye_center[1])), radius=int(horizontal_threshold * l_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            cv2.circle(frame, (int(point_left_eye_iris_center[0]), int(point_left_eye_iris_center[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(frame, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("Left eye: x = " + str(np.round(point_left_eye_iris_center[0],0)) + " , y = " + str(np.round(point_left_eye_iris_center[1],0)))
            #cv2.putText(frame, "Left eye:  x = " + str(np.round(point_left_eye_iris_center[0],0)) + " , y = " + str(np.round(point_left_eye_iris_center[1],0)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

            r_eye_width = point_right_eye_left[0] - point_right_eye_right[0]
            r_eye_height = point_right_eye_bottom[1] - point_right_eye_top[1]
            r_eye_center = [(point_right_eye_left[0] + point_right_eye_right[0])/2 ,(point_right_eye_bottom[1] + point_right_eye_top[1])/2]

            #cv2.circle(frame, (int(r_eye_center[0]), int(r_eye_center[1])), radius=int(horizontal_threshold * r_eye_width), color=(255, 0, 0), thickness=-1) #center of eye and its radius 
            
            cv2.circle(frame, (int(point_right_eye_iris_center[0]), int(point_right_eye_iris_center[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
            cv2.circle(frame, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye
            #print("right eye: x = " + str(np.round(point_right_eye_iris_center[0],0)) + " , y = " + str(np.round(point_right_eye_iris_center[1],0)))
            #cv2.putText(frame, "Right eye: x = " + str(np.round(point_right_eye_iris_center[0],0)) + " , y = " + str(np.round(point_right_eye_iris_center[1],0)), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

            # 4.5 - Calculate the EAR
            right_eye_EAR = r_eye_height / r_eye_width 
            left_eye_EAR = l_eye_height / l_eye_width
            EAR = (right_eye_EAR + left_eye_EAR) / 2
            
            #if time_elapsed < 3:
            #    cv2.putText(frame, "Calibrating EAR baseline", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #    cv2.putText(frame, "Please keep your eyes open", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #    EAR_calibration_data_eyes_open.append(EAR)
            #    baseline_eyes_open = np.mean(EAR_calibration_data_eyes_open)
            #elif time_elapsed > 3 and time_elapsed < 6:
            #    cv2.putText(frame, "Calibrating EAR baseline", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #    cv2.putText(frame, "Please keep your eyes closed", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #    EAR_calibration_data_eyes_closed.append(EAR)
            #    baseline_eyes_closed = np.mean(EAR_calibration_data_eyes_closed)

            #    threshold_EAR = baseline_eyes_open - (baseline_eyes_open - baseline_eyes_closed) * 0.8

            #else:
            #    if EAR < threshold_EAR: # eyes are 80% closed
            #        frames.append(1)
            #    else:
            #        frames.append(0)

            # speed reduction (comment out for full speed)

            time.sleep(1/25) # [s]

        end = time.time()
        total_time = end-start
        time_elapsed += total_time

        if total_time>0:
            fps = 1 / total_time
        else:
            fps=0
            
    #    if len(frames) > 10 * fps:
    #        PERCLOS = frames.count(1) / len(frames)
    #        print("PERCLOS: ", PERCLOS)
#
    #        if PERCLOS > 0.8: # 80% of the time eyes are closed
    #            print("Drowsy")
    #            cv2.putText(frame, "Drowsy", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    #        frames.pop(0)

        cv2.putText(frame, f'FPS : {int(fps)}', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, f'TIME : {round(time_elapsed, 2)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI', frame)             

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly source and eventual log file
capture.release()
