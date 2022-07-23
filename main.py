import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def process_image(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        nparr = np.fromstring(image, np.uint8)
        image = cv2.imdecode(nparr, flags=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image)

    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        print("face_landmarks:", face_landmarks)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return annotated_image


def st_ui():
    uploaded_file = st.file_uploader(label="Upload image")
    st.image(process_image(uploaded_file.getvalue()))


if __name__ == "__main__":
    st_ui()
