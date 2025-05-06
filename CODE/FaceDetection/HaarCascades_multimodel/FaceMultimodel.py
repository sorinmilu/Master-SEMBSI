import cv2
import numpy as np

haarcascade_face = None
haarcascade_eye = None
haarcascade_nose = None
haarcascade_mouth = None

color_face = (0,255,0)
color_eye = (3,186,252)
color_nose = (255,0,0)
color_mouth = (0,0,255)

font = cv2.FONT_HERSHEY_SIMPLEX

def contains(r1, r2):
   return r1.x1 < r2.x1 < r2.x2 < r1.x2 and r1.y1 < r2.y1 < r2.y2 < r1.y2

def is_inside(bx,by,bw,bh,sx,sy,sw,sh):
   return bx < sx < (sx + sw) < (bx + bw) and by < sy < (sy + sh) < (by + bh)

def init_cascades(args):
    global haarcascade_face
    global haarcascade_eye
    global haarcascade_nose
    global haarcascade_mouth

    haarcascade_face = cv2.CascadeClassifier(args.haarcascade_face)
    haarcascade_eye = cv2.CascadeClassifier(args.haarcascade_eye)
    haarcascade_nose = cv2.CascadeClassifier(args.haarcascade_nose)
    haarcascade_mouth = cv2.CascadeClassifier(args.haarcascade_mouth)


def detect_frame(gray, args):
    report={}
    report['labels_face'] = hits_face(gray, args)
    report['labels_eye'] = hits_eye(gray, args)
    report['labels_nose'] = hits_nose(gray, args)
    report['labels_mouth'] = hits_mouth(gray, args)
    return report


def hits_face(gray, args):
    labels = []
    hits_face = haarcascade_face.detectMultiScale(
                                    gray,
                                    scaleFactor=args.haarScaleFactor_face,
                                    minNeighbors=args.haarMinNeighbours_face,
                                    minSize=(args.haarMinSize_face, args.haarMinSize_face)
                                )
    if len(hits_face) > 0:
        for (x,y,w,h) in hits_face:
            labels.append({'x': int(x),'y': int(y),'w': int(w), 'h': int(h)})

    return labels


def hits_eye(gray, args):
   labels = []
   hits_eye = haarcascade_eye.detectMultiScale(
           gray,
           scaleFactor=args.haarScaleFactor_eye,
           minNeighbors=args.haarMinNeighbours_eye,
           minSize=(args.haarMinSize_eye, args.haarMinSize_eye)
       )

   if len(hits_eye) > 0:
      for (x,y,w,h) in hits_eye:
         labels.append({'x': int(x),'y': int(y),'w': int(w), 'h': int(h)})

   return labels

def hits_nose(gray, args):
    labels = []
    hits_nose = haarcascade_nose.detectMultiScale(
                gray,
                scaleFactor=args.haarScaleFactor_nose,
                minNeighbors=args.haarMinNeighbours_nose,
                minSize=(args.haarMinSize_nose, args.haarMinSize_nose)
            )

    if len(hits_nose) > 0:
        for (x,y,w,h) in hits_nose:
            labels.append({'x': int(x),'y': int(y),'w': int(w), 'h': int(h)})

    return labels


def hits_mouth(gray, args):
    labels = []
    hits_mouth = haarcascade_mouth.detectMultiScale(
        gray,
        scaleFactor=args.haarScaleFactor_mouth,
        minNeighbors=args.haarMinNeighbours_mouth,
        minSize=(args.haarMinSize_mouth, args.haarMinSize_mouth)
    )

    if len(hits_mouth) > 0:
        for (x,y,w,h) in hits_mouth:
            labels.append({'x': int(x),'y': int(y),'w': int(w), 'h': int(h)})

    return labels

def draw_all_frames(img, report, args):
#     print(report)
    if 'labels_face' in report:
        for face in report['labels_face']:
            cv2.rectangle(img, (face['x'], face['y']), (face['x'] + face['w'], face['y'] + face['h']), color_face, 2)

    if 'labels_eye' in report:
        for eye in report['labels_eye']:
            cv2.rectangle(img, (eye['x'], eye['y']), (eye['x'] + eye['w'], eye['y'] + eye['h']), color_eye, 2)

    if 'labels_nose' in report:
        for nose in report['labels_nose']:
            cv2.rectangle(img, (nose['x'], nose['y']), (nose['x'] + nose['w'], nose['y'] + nose['h']), color_nose, 2)

    if 'labels_mouth' in report:
        for mouth in report['labels_mouth']:
            cv2.rectangle(img, (mouth['x'], mouth['y']), (mouth['x'] + mouth['w'], mouth['y'] + mouth['h']), color_mouth, 2)

    return img

def draw_face_content(img, face_content, args):
    for face in face_content:
        cv2.rectangle(img, (face['facebox']['x'], face['facebox']['y']), (face['facebox']['x'] + face['facebox']['w'], face['facebox']['y'] + face['facebox']['h']), color_face, 2)
        cv2.putText(img, face['type'], (face['facebox']['x']-12, face['facebox']['y']-19), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, face['type'], (face['facebox']['x']-13, face['facebox']['y']-20), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        for eye in face['eyes']:
            cv2.rectangle(img, (eye['x'], eye['y']), (eye['x'] + eye['w'], eye['y'] + eye['h']), color_eye, 2)

        for nose in face['noses']:
            cv2.rectangle(img, (nose['x'], nose['y']), (nose['x'] + nose['w'], nose['y'] + nose['h']), color_nose, 2)

        for mouth in face['mouths']:
            cv2.rectangle(img, (mouth['x'], mouth['y']), (mouth['x'] + mouth['w'], mouth['y'] + mouth['h']), color_mouth, 2)

    return img


def extract_image(img, face, prc):
    (imgy, imgx,depth) = img.shape

    neww = face['facebox']['w'] + face['facebox']['w']*prc
    newh = face['facebox']['h'] + face['facebox']['h']*prc
    newx = face['facebox']['x'] - (neww - face['facebox']['w'])/2
    newy = face['facebox']['y'] - (newh - face['facebox']['h'])/2

    newx2 = newx + neww
    newy2 = newy + newh

    if (newx < 0):
        newx = 0
    if (newy < 0):
        newy = 0
    if (newx2 > imgx):
        newx2 = imgx

    if (newy2 > imgy):
        newy2 = imgy

    roi = img[int(newy):int(newy2),int(newx):int(newx2),:]
    return roi

def extract_faces(img, face_content, args, types):
    for face in face_content:
        if face['type'] in types:
            face['image'] = extract_image(img, face, args.enlargepurcent)

    return face_content


def check_face_types(report,args):
    faces = []
    if 'labels_face' in report:
        for face in report['labels_face']:
            face_content = {'facebox': face, 'eyes': [], 'noses' : [], 'mouths': []}
            eyes_count = 0
            noses_count = 0
            mouths_count = 0

            if 'labels_eye' in report:
                for eye in report['labels_eye']:
                    if (is_inside(face['x'],face['y'],face['w'],face['h'], eye['x'],eye['y'],eye['w'],eye['h'])):
                        eyes_count = eyes_count + 1
                        face_content['eyes'].append(eye)

            if 'labels_nose' in report:
                for nose in report['labels_nose']:
                    if (is_inside(face['x'],face['y'],face['w'],face['h'], nose['x'],nose['y'],nose['w'],nose['h'])):
                        noses_count = noses_count + 1
                        face_content['noses'].append(nose)

            if 'labels_mouth' in report:
                 for mouth in report['labels_mouth']:
                    if (is_inside(face['x'],face['y'],face['w'],face['h'], mouth['x'],mouth['y'],mouth['w'],mouth['h'])):
                        mouths_count = mouths_count + 1
                        face_content['mouths'].append(mouth)

            if (eyes_count == 2 and noses_count == 1 and mouths_count == 1):
                face_content['type'] = 'perfect'
            elif (eyes_count > 0 and noses_count > 0 and mouths_count > 0):
                face_content['type'] = 'complete'
            elif (eyes_count > 0 and noses_count > 0 and mouths_count == 0):
                face_content['type'] = 'eyenose'
            elif (eyes_count > 0 and mouths_count > 0 and noses_count == 0):
                face_content['type'] = 'eyemouth'
            elif (eyes_count == 0 and mouths_count > 0 and noses_count > 0):
                face_content['type'] = 'nosesmouth'
            elif (eyes_count > 0 and mouths_count == 0 and noses_count == 0):
                face_content['type'] = 'eyes'
            elif (eyes_count == 0 and mouths_count > 0 and noses_count == 0):
                face_content['type'] = 'mouths'
            elif (eyes_count == 0 and mouths_count == 0 and noses_count > 0):
                face_content['type'] = 'noses'
            else:
                face_content['type'] = 'empty'

            faces.append(face_content)
    return faces


def extract_lab_data(img, args):
    kplist = []
    #resize
    img = cv2.resize(img, (60,60),interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(img)
    return l_channel.flatten()

def extract_cany_data(img,args):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (60,60),interpolation = cv2.INTER_AREA)
    edges = cv2.Canny(img,100,200)
    return edges.flatten()

def extract_sift_data(img, args):
    kplist = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img,None)
    mcp = list(keypoints)
    mcp.sort(key=lambda x: x.size, reverse=True)
    for kp in mcp:
        kplist.append({
                           'class_id' : kp.class_id,
                           'octave': kp.octave,
                           'angle' : kp.angle,
                           'size' : kp.size,
                        })

    return { 'features': 'SIFT',
              'keypoints' : kplist,
              'descriptors' : descriptors
            }


# Function to calculate Euclidean distance between two points
def calculate_distance(p1, p2):
    # Accessing x, y, and z from NormalizedLandmark objects
    p1_x, p1_y = p1.x, p1.y
    p2_x, p2_y = p2.x, p2.y
    
    # Compute Euclidean distance (ignoring z for now)
    return np.linalg.norm(np.array([p1_x, p1_y]) - np.array([p2_x, p2_y]))

def calculate_area(landmarks, indices, image_width, image_height):
    points = [(landmarks[i].x * image_width, landmarks[i].y * image_height) for i in indices]
    points = np.array(points)

    # Use polygon area calculation (Shoelace theorem)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Function to calculate face metrics, including areas and distances
def calculate_face_metrics(landmarks, width, height):
    # Landmark indices for different regions
    left_eye_indices = [33, 133, 160, 144, 163, 33]  # Example left eye indices
    right_eye_indices = [362, 263, 249, 466, 467, 362]  # Example right eye indices
    mouth_indices = [61, 185, 40, 39, 37, 0, 61]  # Example mouth indices
    upper_lip = [13]  # Upper lip landmark (just below the nose)
    nose_tip = [1]  # Tip of the nose (for distance calculation)

    # Facial metrics
    interocular_distance = calculate_distance(landmarks[33], landmarks[133])
    nose_length = calculate_distance(landmarks[1], landmarks[152])
    nose_width = calculate_distance(landmarks[49], landmarks[279])
    mouth_width = calculate_distance(landmarks[61], landmarks[291])
    face_width = calculate_distance(landmarks[178], landmarks[454])
    face_height = calculate_distance(landmarks[10], landmarks[152])

    # Area of the eyes
    left_eye_area = calculate_area(landmarks, left_eye_indices, width, height)
    right_eye_area = calculate_area(landmarks, right_eye_indices, width, height)

    # Area of the mouth
    mouth_area = calculate_area(landmarks, mouth_indices, width, height)

    # Distance between upper lip and nose
    upper_lip_nose_distance = calculate_distance(landmarks[13], landmarks[1])

    # Calculate angle between landmarks (example: jawline angle)
    def angle_between_points(p1, p2, p3):
        # Convert landmarks to coordinates
        p1_coords = np.array([p1.x, p1.y])
        p2_coords = np.array([p2.x, p2.y])
        p3_coords = np.array([p3.x, p3.y])
        
        vector_1 = p1_coords - p2_coords
        vector_2 = p3_coords - p2_coords
        
        dot_product = np.dot(vector_1, vector_2)
        magnitude_1 = np.linalg.norm(vector_1)
        magnitude_2 = np.linalg.norm(vector_2)
        
        cos_angle = dot_product / (magnitude_1 * magnitude_2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Jawline angle
    jawline_angle = angle_between_points(landmarks[178], landmarks[152], landmarks[454])

    return {
        "interocular_distance": interocular_distance,
        "nose_length": nose_length,
        "nose_width": nose_width,
        "mouth_width": mouth_width,
        "face_width": face_width,
        "face_height": face_height,
        "left_eye_area": left_eye_area,
        "right_eye_area": right_eye_area,
        "mouth_area": mouth_area,
        "upper_lip_nose_distance": upper_lip_nose_distance,
        "jawline_angle": jawline_angle
    }