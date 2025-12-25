import cv2
import numpy as np
import mediapipe as mp

class MediaPipeProcessor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Persistent solution objects (lazy init could be better but let's init here for simplicity)
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame, mode):
        """
        Process a generic opencv frame (BGR) based on the selected mode.
        Returns the annotated frame (RGB).
        """
        # Convert BGR to RGB
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = None
        
        if mode == "Face Detection":
            results = self.face_detection.process(frame_rgb)
        elif mode == "Face Mesh":
            results = self.face_mesh.process(frame_rgb)
        elif mode == "Hands":
            results = self.hands.process(frame_rgb)
        elif mode == "Pose":
            results = self.pose.process(frame_rgb)
            
        # Draw annotations
        frame.flags.writeable = True
        # Note: frame is BGR here, but drawing utils usually work on it fine, 
        # or we can draw on RGB. Let's draw on RGB to return valid RGB for Qt.
        annotated_image = frame_rgb.copy() 
        
        if results:
            if mode == "Face Detection" and results.detections:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(annotated_image, detection)
                    
            elif mode == "Face Mesh" and results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        
            elif mode == "Hands" and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                        
            elif mode == "Pose" and results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
        return annotated_image

    def close(self):
        self.face_detection.close()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
