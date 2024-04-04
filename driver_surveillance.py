import cv2 
import numpy as np
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

capture = cv2.VideoCapture(0)
frames = []

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

                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if index == 468 or index == 33 or index == 145 or index == 133 or index == 159: # iris points
                # if index == 468 or index == 469 or index == 470 or index == 471 or index == 472: # eye border
                    if index == 468:
                        right_pupil_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        right_pupil_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)

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
            
            #print("EAR: ", EAR)

            if EAR < 0.2:
                frames.append(1)
                #print("Blinking")
            else:
                frames.append(0)

            # speed reduction (comment out for full speed)

            time.sleep(1/25) # [s]

        end = time.time()
        totalTime = end-start

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0

        PERCLOS = frames.count(1) / len(frames)

        #print("PERCLOS: ", PERCLOS)
            
        if len(frames) > 10 * fps:
            if PERCLOS > 0.8:
                cv2.putText(frame, "Drowsy", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            #while len(frames) > 10 * fps:
            frames.pop(0)

        print(len(frames)/fps)
        #print("FPS:", fps)

        cv2.putText(frame, f'FPS : {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI', frame)             

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly source and eventual log file
capture.release()
