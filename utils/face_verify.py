import cv2
import numpy as np
from PIL import Image

def verify_face_ghost_match(pil_img):
    """
    Dynamically locates faces using OpenCV Haar Cascades and compares 
    the largest detected face (Main Portrait) against smaller faces (Ghost Hologram).
    """
    try:
        # Convert PIL to OpenCV BGR
        img_np = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Load OpenCV default face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Multi-scale face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        # If fewer than 2 faces are found, fallback to adaptive region detection
        if len(faces) < 2:
            h, w, _ = img_np.shape
            main_crop = gray[int(h*0.15):int(h*0.75), int(w*0.55):int(w*0.95)]
            ghost_crop = gray[int(h*0.65):int(h*0.98), int(w*0.05):int(w*0.45)]
        else:
            # Sort faces by bounding box area (largest first)
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            
            x1, y1, w1, h1 = faces[0]  # Main Portrait
            x2, y2, w2, h2 = faces[1]  # Ghost Hologram
            
            main_crop = gray[y1:y1+h1, x1:x1+w1]
            ghost_crop = gray[y2:y2+h2, x2:x2+w2]

        if main_crop.size == 0 or ghost_crop.size == 0:
            return False, 1.0

        # Resize ghost face to match main face dimensions
        ghost_resized = cv2.resize(ghost_crop, (main_crop.shape[1], main_crop.shape[0]))
        
        # 1. Histogram Similarity Analysis
        hist_ghost = cv2.calcHist([ghost_resized], [0], None, [256], [0, 256])
        hist_main = cv2.calcHist([main_crop], [0], None, [256], [0, 256])
        cv2.normalize(hist_ghost, hist_ghost, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_main, hist_main, 0, 1, cv2.NORM_MINMAX)
        
        hist_sim = float(cv2.compareHist(hist_ghost, hist_main, cv2.HISTCMP_CORREL))
        
        # 2. Template Feature Matching
        res = cv2.matchTemplate(main_crop, ghost_resized, cv2.TM_CCOEFF_NORMED)
        template_score = float(np.max(res))
        
        combined_score = (hist_sim * 0.4) + (template_score * 0.6)
        
        # Forgery threshold: scores below 0.35 indicate mismatched faces
        is_mismatch = combined_score < 0.35
        return is_mismatch, max(0.0, combined_score)
        
    except Exception as e:
        return False, 1.0