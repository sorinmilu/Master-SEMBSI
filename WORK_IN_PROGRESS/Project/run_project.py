import cv2
import mediapipe as mp
import numpy as np
import math
import os
import argparse

def get_face_landmarks(image_path):
    """Detects face landmarks and mesh in an image using MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return image, None, None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            return image, results.multi_face_landmarks[0].landmark, results.multi_face_landmarks[0]
        else:
            print("No face detected in the image.")
            return image, None, None

def get_oval_face_mask(image, landmarks):
    """Creates a mask for the face oval."""
    if landmarks is None:
        return None
    height, width, _ = image.shape
    points = np.array([(int(lm.x * width), int(lm.y * height)) for lm in landmarks])
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, (255), thickness=cv2.FILLED)
    return mask

def extract_oval_face(image, mask):
    """Extracts the portion of the face inside the mask."""
    if mask is None:
        return None
    oval_face = cv2.bitwise_and(image, image, mask=mask)
    return oval_face

def draw_oval_face_with_landmarks_mesh(image, landmarks, face_mesh_results):
    """Draws the extracted oval face with landmarks and mesh."""
    if image is None or landmarks is None or face_mesh_results is None:
        return None
    output_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Draw landmarks
    for landmark in landmarks:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(output_image, (x, y), 2, (0, 255, 0), -1)

    # Draw mesh
    mp_drawing.draw_landmarks(
        image=output_image,
        landmark_list=face_mesh_results,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, color=(255, 255, 0)))

    return output_image

def normalize_landmarks(landmarks, image_shape):
    """Normalizes face landmarks."""
    if landmarks is None:
        return None
    height, width = image_shape
    oval_indices = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
    oval_points = np.array([(landmarks[i[0]].x, landmarks[i[0]].y) for i in oval_indices])
    center_x = np.mean(oval_points[:, 0])
    center_y = np.mean(oval_points[:, 1])
    distances = np.sqrt((oval_points[:, 0] - center_x)**2 + (oval_points[:, 1] - center_y)**2)
    radius = np.max(distances)
    normalized_landmarks = [( (lm.x - center_x) / radius, (lm.y - center_y) / radius ) for lm in landmarks]
    return normalized_landmarks

def get_deformed_landmarks(landmarks, image_shape):
    """Transforms the original face oval into a circle and deforms other landmarks accordingly."""
    if landmarks is None:
        return None
    height, width = image_shape
    oval_indices = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
    oval_points_orig = np.array([(landmarks[i[0]].x, landmarks[i[0]].y) for i in oval_indices])

    # 1. Calculate the center of the original oval
    center_x_orig = np.mean(oval_points_orig[:, 0])
    center_y_orig = np.mean(oval_points_orig[:, 1])

    # 2. Calculate the average radius of the original oval
    distances = np.sqrt((oval_points_orig[:, 0] - center_x_orig)**2 + (oval_points_orig[:, 1] - center_y_orig)**2)
    avg_radius_orig = np.mean(distances)

    # 3. Place the deformed oval landmarks on a perfect circle
    deformed_landmarks = [None] * len(landmarks)
    for i, index_pair in enumerate(oval_indices):
        angle = 2 * np.pi * i / len(oval_indices)  # Distribute evenly around the circle
        deformed_x = 0.5 * np.cos(angle) + 0.5
        deformed_y = 0.5 * np.sin(angle) + 0.5
        deformed_landmarks[index_pair[0]] = (deformed_x, deformed_y)

    # 4. Transform the internal landmarks based on the scaling and translation
    #    from the original oval's center and radius to the unit circle space.
    for i, lm in enumerate(landmarks):
        if deformed_landmarks[i] is None: # If it's not an oval landmark
            norm_x = (lm.x - center_x_orig) / avg_radius_orig
            norm_y = (lm.y - center_y_orig) / avg_radius_orig
            deformed_x = norm_x * 0.5 + 0.5
            deformed_y = norm_y * 0.5 + 0.5
            deformed_landmarks[i] = (deformed_x, deformed_y)

    return deformed_landmarks

def calculate_parameters(deformed_landmarks):
    """Calculates facial parameters based on deformed landmarks."""
    if deformed_landmarks is None:
        return None

    def get_landmark(index):
        return np.array([deformed_landmarks[index][0], deformed_landmarks[index][1]])

    parameter_config = [
        {"nose_height": {"first_point": 1, "second_point": 168}},
        {"nose_width": {"first_point": 48, "second_point": 294}},
        {"mouth_width": {"first_point": 76, "second_point": 291}},
        {"interocular_distance_small": {"first_point": 133, "second_point": 362}},
        {"left_eye_width": {"first_point": 362, "second_point": 263}},
        {"right_eye_width": {"first_point": 130, "second_point": 133}},
        {"eye_to_eye": {"first_point": 130, "second_point": 263}},
        {"bottom_lip_width": {"first_point": 14, "second_point": 17}},
        {"top_lip_width": {"first_point": 0, "second_point": 13}},
    ]

    parameters = []
    for param_dict in parameter_config:
        key = list(param_dict.keys())[0]
        indices = param_dict[key]
        p1 = get_landmark(indices["first_point"])
        p2 = get_landmark(indices["second_point"])
        value = np.linalg.norm(p1 - p2)
        param_dict[key]["value"] = value
        parameters.append(param_dict)

    # Calculate face width and height (using min/max of oval landmarks in deformed space)
    oval_indices = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
    oval_x = [deformed_landmarks[i[0]][0] for i in oval_indices]
    oval_y = [deformed_landmarks[i[0]][1] for i in oval_indices]
    face_width = max(oval_x) - min(oval_x)
    face_height = max(oval_y) - min(oval_y)

    face_dimensions = {"face_width": face_width, "face_height": face_height}

    # Calculate relative sizes
    for param_dict in parameters:
        key = list(param_dict.keys())[0]
        if "height" in key:
            param_dict[key]["relative_size"] = param_dict[key]["value"] / face_dimensions["face_height"]
        elif "width" in key or "distance" in key:
            param_dict[key]["relative_size"] = param_dict[key]["value"] / face_dimensions["face_width"]
        else:
            param_dict[key]["relative_size"] = None  # Default value for parameters without relative size

    return parameters, face_dimensions

def draw_parameters(image_shape, deformed_landmarks, parameters):
    """Draws deformed landmarks and parameter lines on a black rectangle."""
    if deformed_landmarks is None or parameters is None:
        return None
    height, width = image_shape
    canvas_width = 500
    canvas_height = 500
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    def scale_landmark(landmark):
        return int(landmark[0] * canvas_width), int(landmark[1] * canvas_height)

    def get_deformed_landmark(index):
        return np.array(deformed_landmarks[index])

    # Draw deformed landmarks
    for lm in deformed_landmarks:
        cv2.circle(canvas, scale_landmark(lm), 3, (0, 255, 0), -1)

    def draw_line(start_index, end_index, color=(255, 0, 0), thickness=2):
        start_point = scale_landmark(get_deformed_landmark(start_index))
        end_point = scale_landmark(get_deformed_landmark(end_index))
        cv2.line(canvas, start_point, end_point, color, thickness)

    # Draw parameter lines
    for param_dict in parameters:
        key = list(param_dict.keys())[0]
        indices = param_dict[key]
        draw_line(indices["first_point"], indices["second_point"])

    return canvas

def get_parameter_list(parameters, face_dimensions):
    """Returns a list of all calculated parameters."""
    if parameters is None:
        return None
    parameter_list = face_dimensions.copy()
    for param_dict in parameters:
        key = list(param_dict.keys())[0]
        parameter_list[key + "_value"] = param_dict[key]["value"]
        parameter_list[key + "_relative"] = param_dict[key]["relative_size"]
    return parameter_list

parser = argparse.ArgumentParser(prog='run_project.py',
                                 description="Scriptul extrage si afiseaza punctele de interes ale fetei folosind MediaPipe, normalizeaza punctele si extrage o serie de masuratori",
                                 usage='run_project.py -f <input_file>')

parser.add_argument('-f', "--input_file", help="Imaginea de intrare")
parser.add_argument('-os', "--output_size", help="Dimensiunea imaginii de iesire")
parser.add_argument('-op', "--output_prefix", help="Prefixul imaginilor generate")

args = parser.parse_args()

if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file '{args.input_file}' does not exist.")


original_image, landmarks, face_mesh_results = get_face_landmarks(args.input_file)

if landmarks:
    face_mask = get_oval_face_mask(original_image, landmarks)
    oval_face_extracted = extract_oval_face(original_image, face_mask)
    oval_face_with_lm_mesh = draw_oval_face_with_landmarks_mesh(oval_face_extracted, landmarks, face_mesh_results)
    if oval_face_with_lm_mesh is not None:
        cv2.imwrite(f"{args.output_prefix}_oval_face_lm_mesh.jpg", oval_face_with_lm_mesh)
        print(f"\nSaved oval face with landmarks and mesh to {args.output_prefix}_oval_face_lm_mesh.jpg")

    normalized_landmarks = normalize_landmarks(landmarks, original_image.shape[:2])
    deformed_landmarks = get_deformed_landmarks(landmarks, original_image.shape[:2])
    parameters, face_dimensions = calculate_parameters(deformed_landmarks)

    print("\nCalculated Parameters:")
    print(face_dimensions)
    for param in parameters:
        key = list(param.keys())[0]
        relative_size = param[key].get("relative_size", "N/A")  # Use "N/A" if relative_size is not present
        if relative_size is not None:
            print(f"Parametru: {key}, Dimensiune: {relative_size:.4f}")

    parameter_image = draw_parameters(original_image.shape[:2], deformed_landmarks, parameters)
    if parameter_image is not None:
        cv2.imwrite(f"{args.output_prefix}_parameters.jpg", parameter_image)
        print(f"\nSaved parameter visualization to {args.output_prefix}_parameters.jpg")


    parameter_list = get_parameter_list(parameters, face_dimensions)
    print("\nParameter List:")
    print(parameter_list)