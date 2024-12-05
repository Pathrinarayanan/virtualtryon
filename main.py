from flask import Flask, render_template, Response, request, jsonify, send_file, make_response
import cv2
import dlib
from imutils import face_utils
from rembg import remove
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def load_image_from_url_opencv(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)  # Preserve alpha channel if present
        if image is None:
            raise Exception("Failed to decode image.")
        return image
    except Exception as e:
        raise Exception(f"Error loading image from URL: {e}")


def overlay_image(frame, overlay, x_offset, y_offset):
    # Get the dimensions of the frame and overlay image
    frame_height, frame_width = frame.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]
    
    # Ensure the overlay doesn't exceed the frame's boundaries
    if x_offset + overlay_width > frame_width or y_offset + overlay_height > frame_height:
        raise ValueError("Overlay region exceeds frame boundaries.")

    # Create an alpha mask for the overlay image
    alpha = overlay[:, :, 3] / 255.0  # Assuming RGBA overlay
    for c in range(0, 3):
        frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width, c] = (
            alpha * overlay[:, :, c] + (1 - alpha) * frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width, c]
        )

    return frame



def generate_frames(ear_url=None, necklace_url=None, scale_factor=12):
    cap = cv2.VideoCapture(0)
    earring_image = load_image_from_url_opencv(ear_url) if ear_url else None

    # Process necklace image
    necklace_image = None
    if necklace_url:
        try:
            raw_necklace_image = load_image_from_url_opencv(necklace_url)
            pil_image = Image.fromarray(cv2.cvtColor(raw_necklace_image, cv2.COLOR_BGR2RGB))
            # Convert PIL to bytes for rembg
            input_bytes = BytesIO()
            pil_image.save(input_bytes, format="PNG")
            input_bytes = input_bytes.getvalue()
            # Remove background
            output_bytes = remove(input_bytes)
            # Convert back to PIL and then to OpenCV
            bg_removed = Image.open(BytesIO(output_bytes)).convert("RGBA")
            necklace_image = np.array(bg_removed)  # Convert PIL to NumPy array
            necklace_image = cv2.cvtColor(necklace_image, cv2.COLOR_RGBA2BGRA)  # Ensure OpenCV format
        except Exception as e:
            print(f"Error processing necklace image: {e}")

    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            if necklace_image is not None:
                # Define neck points and calculate bounding box
                neck_points = shape[6:11]
                (x, y, w, h) = calculate_boundbox(neck_points)

                # Apply scaling factor to the necklace size while preserving aspect ratio
                aspect_ratio = necklace_image.shape[1] / necklace_image.shape[0]
                new_height = int(h * scale_factor)
                new_width = int(new_height * aspect_ratio)

                # Resize the necklace with the new dimensions
                resized_necklace = cv2.resize(necklace_image, (new_width, new_height))

                # Ensure that the overlay doesn't exceed frame boundaries
                if x + new_width > frame.shape[1]:
                    new_width = frame.shape[1] - x  # Adjust width to fit within the frame
                    resized_necklace = cv2.resize(resized_necklace, (new_width, new_height))

                if y + new_height + 20 > frame.shape[0]:
                    new_height = frame.shape[0] - (y + 20)  # Adjust height to fit within the frame
                    resized_necklace = cv2.resize(resized_necklace, (new_width, new_height))

                # Apply the overlay with the resized necklace
                frame = overlay_image(frame, resized_necklace, x-100, y + 20)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()





def calculate_boundbox(points):
    x = min(points[:, 0])
    y = min(points[:, 1])
    w = max(points[:, 0]) - x
    h = max(points[:, 1]) - y
    return (x, y, w, h)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    ear_url = request.args.get('ear_url')
    necklace_url = request.args.get('necklace_url')
    return Response(generate_frames(ear_url, necklace_url), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/remove_bg', methods=['GET'])
def remove_bg():
    image_url = request.args.get('image_url')
    if not image_url:
        return jsonify({"error": "image_url parameter is required"}), 400
    try:
        image = load_image_from_url_opencv(image_url)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
        bg_removed = remove(pil_image)
        output_bytes = BytesIO()
        bg_removed.save(output_bytes, format="PNG")
        output_bytes.seek(0)
        return send_file(output_bytes, mimetype='image/png', as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
