from flask import Flask, request, send_file, jsonify
from PIL import Image, ImageEnhance
from rembg import remove
#from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import io
import os
from flask_cors import CORS
from waitress import serve
from datetime import datetime

app = Flask(__name__)
CORS(app)

# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").cuda()
# predictor = SamPredictor(sam)

def center_align_image(final_img_rgb):
    # RGBA 변환해서 알파 채널 추출
    final_img_rgba = final_img_rgb.convert("RGBA")
    alpha = np.array(final_img_rgba.split()[-1])  # 알파 채널만 추출

    coords = np.argwhere(alpha > 0)
    
    if coords.size == 0:
        print("비어있는 이미지입니다. 원본 반환")
        return final_img_rgb

    # 실제 이미지 좌표 범위 계산
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    crop_x1, crop_y1 = top_left[1], top_left[0]
    crop_x2, crop_y2 = bottom_right[1], bottom_right[0]

    real_w = crop_x2 - crop_x1 + 1
    real_h = crop_y2 - crop_y1 + 1

    # 중심 위치 계산
    final_x = (500 - real_w) // 2
    final_y = (500 - real_h) // 2

    # 유효 영역만 crop
    cropped_img = final_img_rgb.crop((crop_x1, crop_y1, crop_x2 + 1, crop_y2 + 1))

    # 새로운 500x500 흰 배경에 중앙 배치
    result_img = Image.new("RGB", (500, 500), (255, 255, 255))
    result_img.paste(cropped_img, (final_x, final_y))

    return result_img

def re_center_image(img_rgb):
    """
    RGB 이미지 바이트를 입력으로 받아, 피사체를 500x500 흰색 배경 중앙에 다시 정렬합니다.

    Args:
        image_rgb_bytes: 가운데 정렬이 필요한 RGB 이미지의 바이트 데이터 (io.BytesIO 객체).

    Returns:
        io.BytesIO: 가운데 정렬된 RGB 이미지의 바이트 데이터.
    """
    img_gray = img_rgb.convert("L") # 흑백 이미지로 변환
    np_gray = np.array(img_gray)

    _, binary_mask = cv2.threshold(np_gray, 254, 255, cv2.THRESH_BINARY_INV) # 거의 흰색이 아닌 픽셀들을 마스크로 만듦

    coords = cv2.findNonZero(binary_mask)
    if coords is None:
        # 피사체를 찾지 못한 경우 (예: 완전 흰색 이미지) 원본 반환
        output_buffer = io.BytesIO()
        img_rgb.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return output_buffer

    x, y, w, h = cv2.boundingRect(coords) # 피사체의 바운딩 박스 계산

    # 피사체 영역만 크롭
    cropped = img_rgb.crop((x, y, x + w, y + h))

    # 새로운 500x500 흰색 배경 이미지 생성
    final_img_re_centered = Image.new("RGB", (500, 500), (255, 255, 255))

    # 크롭된 피사체를 500x500 이미지의 중앙에 배치
    img_w, img_h = cropped.size
    pos_x = (500 - img_w) // 2
    pos_y = (500 - img_h) // 2

    final_img_re_centered.paste(cropped, (pos_x, pos_y))

    # 결과를 바이트 스트림으로 변환하여 반환
    output_buffer = io.BytesIO()
    final_img_re_centered.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return output_buffer

def process_image_bytes(image_bytes):
    removed_bg = remove(image_bytes)
    img = Image.open(io.BytesIO(removed_bg)).convert("RGBA")
    np_img = np.array(img)

    alpha_channel = np_img[:, :, 3]
    _, sure_foreground_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(sure_foreground_mask)
    x, y, w, h = cv2.boundingRect(coords)

    cropped = img.crop((x, y, x + w, y + h))

    width_range = (72, 428)
    height_range = (28, 470)

    img_w, img_h = cropped.size

    if img_w >= img_h:
        new_w = width_range[1] - width_range[0]
        new_h = int((new_w / img_w) * img_h)
    else:
        new_h = height_range[1] - height_range[0]
        new_w = int((new_h / img_h) * img_w)
        # if new_w < width_range[0] or new_w > width_range[1]:
        if new_w > (width_range[1] - width_range[0]):
            new_h = width_range[1] - width_range[0]
            new_w = int((new_h / img_h) * img_w)

    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    final_img = Image.new("RGBA", (500, 500), (255, 255, 255, 0))
    pos_x = (500 - new_w) // 2
    pos_y = (500 - new_h) // 2
    final_img.paste(resized, (pos_x, pos_y), resized)

    # 흰 배경에 합성 후 RGB 변환
    white_bg = Image.new("RGB", final_img.size, (255, 255, 255))
    final_img_rgb = Image.alpha_composite(white_bg.convert("RGBA"), final_img).convert("RGB")

    return re_center_image(final_img_rgb)

    # output_buffer = io.BytesIO()
    # final_img_rgb.save(output_buffer, format="PNG")
    # output_buffer.seek(0)
    # return output_buffer


# def process_image_bytes(image_bytes):
#     removed_bg = remove(image_bytes)
#     img = Image.open(io.BytesIO(removed_bg)).convert("RGBA")

#     np_img = np.array(img)
#     alpha_channel = np_img[:, :, 3]
#     coords = cv2.findNonZero(alpha_channel)
#     x, y, w, h = cv2.boundingRect(coords)

#     cropped = img.crop((x, y, x + w, y + h))

#     width_range = (71, 433)
#     height_range = (29, 480)

#     img_w, img_h = cropped.size

#     if img_w >= img_h:
#         new_w = width_range[1] - width_range[0]
#         new_h = int((new_w / img_w) * img_h)
#     else:
#         new_h = height_range[1] - height_range[0]
#         new_w = int((new_h / img_h) * img_w)
#         if new_w < width_range[0] or new_w > width_range[1]:
#             new_h = width_range[1] - width_range[0]
#             new_w = int((new_h / img_h) * img_w)

#     resized = cropped.resize((new_w, new_h), Image.LANCZOS)

#     final_img = Image.new("RGBA", (500, 500), (255, 255, 255, 0))
#     pos_x = (500 - new_w) // 2
#     pos_y = (500 - new_h) // 2
#     final_img.paste(resized, (pos_x, pos_y), resized)

#     output_buffer = io.BytesIO()
#     final_img.save(output_buffer, format="PNG")
#     output_buffer.seek(0)

#     return output_buffer

@app.route('/upload', methods=['POST'])
def upload_image():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"요청 들어옴 - [{now}]")
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    processed_image = process_image_bytes(file.read())
    print("다운로드 진행!")
    return send_file(processed_image, mimetype='image/png', as_attachment=True, download_name='processed.png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9091))
    serve(app, host='0.0.0.0', port=port)