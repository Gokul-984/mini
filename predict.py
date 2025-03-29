import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os

# ---- Configuration ----
MODEL_INPUT_SIZE = (400, 400)  # Models expect 400x400 images.
SKELETON_SIZE = 400            # White canvas size used for skeleton processing.
# SKELETON_POS is no longer needed since we won't overlay the skeleton.
scale_margin = 0.7             # Margin factor for scaling the hand in the canvas.

# ---- MediaPipe Setup ----
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# ---- Load Coarse Model and Mapping ----
coarse_model = tf.keras.models.load_model("coarse_model.h5")
with open("coarse_class_map.pkl", "rb") as f:
    coarse_class_map = pickle.load(f)
inv_coarse_map = {v: k for k, v in coarse_class_map.items()}

# ---- Load Fine Models and Their Label Maps ----
group_list = ["group_0", "group_1", "group_2", "group_3", "group_4", "group_5"]
group_models = {}
group_label_maps = {}
for group in group_list:
    model_path = f"{group}_model.h5"
    map_path = f"{group}_map.pkl"
    if os.path.exists(model_path) and os.path.exists(map_path):
        group_models[group] = tf.keras.models.load_model(model_path)
        with open(map_path, "rb") as f:
            group_label_maps[group] = pickle.load(f)
    else:
        group_models[group] = None
        group_label_maps[group] = {}
        print(f"Warning: Model or map for {group} not found.")

# ---- Preprocessing Function ----
def preprocess_image(img, target_size=MODEL_INPUT_SIZE):
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

# ---- Helper: Transform and center landmarks onto a white canvas ----
def get_centered_skeleton(hand_landmarks, frame_width, frame_height, canvas_size=SKELETON_SIZE):
    pts = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        pts.append((x, y))
    
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width == 0 or bbox_height == 0:
        return None

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    scale_factor = (canvas_size * scale_margin) / max(bbox_width, bbox_height)
    canvas_cx = canvas_size / 2
    canvas_cy = canvas_size / 2

    transformed = []
    for (x, y) in pts:
        x_new = int((x - cx) * scale_factor + canvas_cx)
        y_new = int((y - cy) * scale_factor + canvas_cy)
        transformed.append((x_new, y_new))
    
    return transformed

# ---- Helper: Draw skeleton on a white canvas given transformed points ----
def draw_skeleton_on_canvas(canvas, points):
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), thickness=2)
    for pt in points:
        cv2.circle(canvas, pt, radius=3, color=(0, 255, 0), thickness=-1)

# ---- Real-Time Pipeline (Skeleton processing hidden) ----
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect.
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Process entire frame for hand detection.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)

    predicted_letter = "No hand"

    # Create a white canvas for background skeleton processing.
    skeleton_canvas = np.ones((SKELETON_SIZE, SKELETON_SIZE, 3), dtype=np.uint8) * 255

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        transformed_points = get_centered_skeleton(hand_landmarks, frame_w, frame_h, canvas_size=SKELETON_SIZE)
        if transformed_points is not None:
            draw_skeleton_on_canvas(skeleton_canvas, transformed_points)

            # Prediction pipeline using the skeleton canvas (hidden from the main frame)
            cnn_input = preprocess_image(skeleton_canvas, target_size=MODEL_INPUT_SIZE)
            coarse_pred = coarse_model.predict(cnn_input)
            coarse_idx = np.argmax(coarse_pred)
            coarse_group = inv_coarse_map.get(coarse_idx, None)
            
            if coarse_group is not None and group_models.get(coarse_group) is not None:
                sub_model = group_models[coarse_group]
                sub_map = group_label_maps.get(coarse_group, {})
                inv_sub_map = {v: k for k, v in sub_map.items()}
                fine_pred = sub_model.predict(cnn_input)
                fine_idx = np.argmax(fine_pred)
                predicted_letter = inv_sub_map.get(fine_idx, "Unknown")
            else:
                predicted_letter = "No sub-model"
    else:
        predicted_letter = "No hand"

    # Overlay the predicted letter on the main frame.
    cv2.putText(frame, predicted_letter, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # (Optional) If you want to debug, you could display the skeleton canvas in a separate window:
    # cv2.imshow("Skeleton Canvas", skeleton_canvas)

    # Display only the main feed.
    cv2.imshow("Real-Time Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
