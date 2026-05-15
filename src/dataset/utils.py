import numpy as np
import cv2

def crop_square_containing_face_patch(image, face_box):
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = face_box

    # 人脸中心和尺寸
    face_cx = (x_min + x_max) // 2
    face_cy = (y_min + y_max) // 2

    face_w = x_max - x_min
    face_h = y_max - y_min
    new_w = new_h = max(face_w, face_h)
    
    new_x_min = int(round(face_cx - new_w / 2))
    new_x_max = int(round(face_cx + new_w / 2))
    new_y_min = int(round(face_cy - new_h / 2))
    new_y_max = int(round(face_cy + new_h / 2))

    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(w, new_x_max)
    new_y_max = min(h, new_y_max)

    face_patch = image[new_y_min:new_y_max, new_x_min:new_x_max]

    return face_patch

def check_oob(bbox, size):
    left, top, right, bot = bbox
    return left < 0 or top < 0 or right > size[1] - 1 or bot > size[0] - 1

def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    if check_oob((left, top, right, bot), size):
        return bbox
    else:
        return np.array([left, top, right, bot])
    
def get_bbox_param(bbox, ref_bbox):
    left, top, right, bot = bbox
    center = np.array([(bot + top) * 0.5, (left + right) * 0.5])
    length = max(right - left, bot - top)

    ref_left, ref_top, ref_right, ref_bot = ref_bbox
    ref_center = np.array([(ref_bot + ref_top) * 0.5, (ref_left + ref_right) * 0.5])
    ref_length = max(ref_right - ref_left, ref_bot - ref_top)  # 最好保证bbox是正方形

    return np.asarray(
        ((center - ref_center) / ref_length).tolist() + [length / ref_length]
    )

def draw_box(box, mask, value=255):
    x1, y1, x2, y2 = box
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), value, -1)
    return mask

def get_mask(face_box, left_eye_box, right_eye_box, mouth_box, side):
    face_mask = np.zeros((side, side), dtype=np.uint8)
    local_mask = np.zeros((side, side), dtype=np.uint8)

    draw_box(face_box, face_mask)
    draw_box(left_eye_box, local_mask)
    draw_box(right_eye_box, local_mask)
    draw_box(mouth_box, local_mask)
    return face_mask, local_mask

