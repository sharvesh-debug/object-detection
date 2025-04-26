import cv2
import base64
import requests
import json
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import threading
import socket

# Ollama server address (your Windows PC)
OLLAMA_SERVER = "http://192.168.73.146:11434"

# The API endpoint for generating responses
API_URL = f"{OLLAMA_SERVER}/api/generate"

# Configuration
MODEL_NAME = "llava"  # A commonly available vision model
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
DETECTION_INTERVAL = 5  # Seconds between detections
CONNECTION_TIMEOUT = 3  # Timeout for server connection checks

# Set to True to enable fallback mode from the start
FORCE_FALLBACK = False

# OpenCV DNN settings for local detection
OPENCV_DNN_CONFIG = {
    'model_path': os.path.expanduser('~/mobilenet_ssd/frozen_inference_graph.pb'),
    'config_path': os.path.expanduser('~/mobilenet_ssd/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'),
    'class_names_path': os.path.expanduser('~/mobilenet_ssd/coco_class_names.txt'),
    'confidence_threshold': 0.5
}

# COCO class names (common objects)
COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
    'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
    'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color map for bounding boxes (for consistent colors per class)
COLOR_MAP = {}

def setup_local_detection():
    """Set up the local object detection with OpenCV DNN"""
    global net, COCO_CLASSES
    try:
        # Try to load the pre-trained model if it exists
        if os.path.exists(OPENCV_DNN_CONFIG['model_path']) and os.path.exists(OPENCV_DNN_CONFIG['config_path']):
            print("Loading pre-trained object detection model...")
            net = cv2.dnn.readNetFromTensorflow(
                OPENCV_DNN_CONFIG['model_path'],
                OPENCV_DNN_CONFIG['config_path']
            )
            
            # Load class names if custom file exists
            if os.path.exists(OPENCV_DNN_CONFIG['class_names_path']):
                with open(OPENCV_DNN_CONFIG['class_names_path'], 'r') as f:
                    COCO_CLASSES = [line.strip() for line in f.readlines()]
                    
            print("Local detection model loaded successfully")
            return True
        else:
            print("Pre-trained model files not found. Using basic detection.")
            return False
    except Exception as e:
        print(f"Error setting up local detection: {e}")
        return False

def download_model_files():
    """Download required model files for local detection if they don't exist"""
    # Create directory for model files
    model_dir = os.path.expanduser('~/mobilenet_ssd')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define model files URLs and paths
    model_files = {
        'frozen_inference_graph.pb': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pb',
        'ssd_mobilenet_v2_coco_2018_03_29.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt'
    }
    
    # Try to download model files
    try:
        for filename, url in model_files.items():
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {filename}")
                else:
                    print(f"Failed to download {filename}: {response.status_code}")
        
        # Create class names file
        class_names_path = os.path.join(model_dir, 'coco_class_names.txt')
        if not os.path.exists(class_names_path):
            with open(class_names_path, 'w') as f:
                for class_name in COCO_CLASSES:
                    f.write(f"{class_name}\n")
            print("Created class names file")
        
        return True
    except Exception as e:
        print(f"Error downloading model files: {e}")
        return False

def check_server_connection():
    """Check if the Ollama server is reachable"""
    try:
        # Parse the server URL to get host and port
        import urllib.parse
        parsed_url = urllib.parse.urlparse(OLLAMA_SERVER)
        host = parsed_url.hostname
        port = parsed_url.port
        
        # If port is not specified in the URL, use default HTTP port
        if port is None:
            if parsed_url.scheme == 'https':
                port = 443
            else:
                port = 80
        
        # Try to connect to the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(CONNECTION_TIMEOUT)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"Server at {host}:{port} is reachable.")
            return True
        else:
            print(f"Server at {host}:{port} is not reachable (error code {result}).")
            return False
    except Exception as e:
        print(f"Error checking server connection: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False
    cap.release()
    return True

def encode_image_to_base64(image):
    """Convert a CV2 image to base64 encoded string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def query_ollama_with_image(base64_image):
    """Send an image to Ollama for object detection"""
    prompt = """
    Analyze this image and identify the objects present.
    Provide a response in the following exact JSON format:
    {"objects": ["object1", "object2", ...]}
    Only include the objects you're confident about.
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')
            
            # Try to parse the JSON response
            try:
                # Find JSON content in the response text
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx]
                    data = json.loads(json_text)
                    objects = data.get('objects', [])
                    
                    # Convert to a format compatible with our detection system
                    detection_results = []
                    for obj in objects:
                        # Since we don't have coordinates from Ollama, use placeholder values
                        detection_results.append({
                            'class_name': obj,
                            'confidence': 0.9,  # Placeholder confidence
                            'box': [10, 10, 100, 100]  # Placeholder box coordinates
                        })
                    
                    return detection_results
                else:
                    # Parse as text if can't find JSON
                    words = response_text.lower().split()
                    objects = []
                    for word in words:
                        for class_name in COCO_CLASSES:
                            if class_name != 'background' and class_name.lower() in word:
                                objects.append({
                                    'class_name': class_name,
                                    'confidence': 0.7,  # Placeholder confidence
                                    'box': [10, 10, 100, 100]  # Placeholder box coordinates
                                })
                    return objects
            except json.JSONDecodeError:
                # Fallback to simple parsing if JSON parsing fails
                words = response_text.lower().split()
                objects = []
                for word in words:
                    for class_name in COCO_CLASSES:
                        if class_name != 'background' and class_name.lower() in word:
                            objects.append({
                                'class_name': class_name,
                                'confidence': 0.7,  # Placeholder confidence
                                'box': [10, 10, 100, 100]  # Placeholder box coordinates
                            })
                return objects
                
        return []
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return []

def get_color_for_class(class_name):
    """Get a consistent color for a class name"""
    global COLOR_MAP
    
    if class_name not in COLOR_MAP:
        # Generate a random color (but avoid too dark colors)
        color = tuple(map(int, np.random.randint(80, 255, size=3)))
        COLOR_MAP[class_name] = color
    
    return COLOR_MAP[class_name]

def local_object_detection(frame):
    """Perform object detection using OpenCV DNN with bounding boxes"""
    global net
    
    # Check if we have the DNN model
    if 'net' not in globals():
        # Try basic method
        return basic_object_detection(frame)
    
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            frame, 
            size=(300, 300), 
            swapRB=True, 
            crop=False
        )
        
        # Set the input for the neural network
        net.setInput(blob)
        
        # Run forward pass to get output
        output = net.forward()
        
        # Process detections
        detected_objects = []
        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            
            # Filter by confidence
            if confidence > OPENCV_DNN_CONFIG['confidence_threshold']:
                # Get class ID
                class_id = int(output[0, 0, i, 1])
                
                # Get class name
                if 0 <= class_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_id]
                    
                    if class_name != 'background':
                        # Get bounding box coordinates
                        box = output[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (start_x, start_y, end_x, end_y) = box.astype("int")
                        
                        # Add to detected objects
                        detected_objects.append({
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'box': [start_x, start_y, end_x, end_y]
                        })
        
        return detected_objects
            
    except Exception as e:
        print(f"Error in local object detection: {e}")
        return basic_object_detection(frame)

def basic_object_detection(frame):
    """Perform basic local object detection using OpenCV"""
    # Convert frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use a simple edge detection as fallback
    edges = cv2.Canny(gray, 50, 150)
    
    # Count significant edges as a simple metric
    edge_count = np.count_nonzero(edges)
    
    # Create a single generic detection
    if edge_count > 10000:
        return [{'class_name': 'object', 'confidence': 0.6, 'box': [50, 50, 200, 200]}]
    elif edge_count > 5000:
        return [{'class_name': 'object', 'confidence': 0.5, 'box': [100, 100, 200, 200]}]
    else:
        return []

class AsyncDetector:
    """Handles asynchronous detection to keep UI responsive"""
    def __init__(self, use_fallback=False):
        self.latest_result = []
        self.processing = False
        self.lock = threading.Lock()
        self.use_fallback = use_fallback
        self.server_check_time = 0
        self.server_available = not use_fallback
        
    def start_detection(self, frame):
        """Start a new detection if not already processing"""
        if not self.processing:
            self.processing = True
            thread = threading.Thread(target=self._process_frame, args=(frame.copy(),))
            thread.daemon = True
            thread.start()
    
    def check_server_periodically(self):
        """Periodically check if server becomes available"""
        current_time = time.time()
        # Check every 30 seconds
        if not self.server_available and (current_time - self.server_check_time > 30):
            self.server_available = check_server_connection()
            self.server_check_time = current_time
            if self.server_available:
                print("Server is now available. Switching to Ollama detection.")
                self.use_fallback = False
            
    def _process_frame(self, frame):
        """Process the frame in a separate thread"""
        try:
            # Check if server is available periodically
            if not self.server_available:
                self.check_server_periodically()
            
            # Use fallback or regular detection based on settings
            if self.use_fallback or not self.server_available:
                print("Using local fallback detection...")
                result = local_object_detection(frame)
            else:
                # Encode image to base64
                base64_image = encode_image_to_base64(frame)
                
                # Query Ollama
                print("Sending image to Ollama for analysis...")
                result = query_ollama_with_image(base64_image)
                
                # Check if we got no results
                if not result:
                    print("No results from Ollama, switching to fallback mode")
                    self.server_available = False
                    self.server_check_time = time.time()
                    self.use_fallback = True
                    # Try fallback for this frame
                    result = local_object_detection(frame)
            
            # Update result
            with self.lock:
                self.latest_result = result
                
            print(f"Detected {len(result)} objects")
        except Exception as e:
            print(f"Error in detection thread: {e}")
            # On error, default to fallback
            try:
                result = local_object_detection(frame)
                with self.lock:
                    self.latest_result = result
            except:
                with self.lock:
                    self.latest_result = []
        finally:
            self.processing = False
            
    def get_latest_result(self):
        """Get the latest detection result"""
        with self.lock:
            return self.latest_result.copy()

def draw_detection_boxes(frame, detections):
    """Draw bounding boxes and labels for detected objects"""
    result_frame = frame.copy()
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        box = detection['box']
        
        # Get color for this class
        color = get_color_for_class(class_name)
        
        # Draw bounding box
        cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate label position
        label_y = box[1] - 10 if box[1] > 20 else box[1] + 30
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_frame, 
                     (box[0], label_y - label_size[1] - 5), 
                     (box[0] + label_size[0], label_y + 5), 
                     color, 
                     -1)
        
        # Draw label text
        cv2.putText(result_frame, 
                   label, 
                   (box[0], label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 255), 
                   2)
    
    return result_frame

def display_results(frame, detections):
    """Display the results on the frame with bounding boxes and info panel"""
    try:
        # First draw detection boxes
        result_frame = draw_detection_boxes(frame, detections)
        
        # Add info panel at the top
        # Convert to PIL Image for better text handling
        img_pil = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
        
        # Try to use a system font if available
        try:
            # Try common fonts that might be available on Raspberry Pi
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]
            
            font = None
            for path in font_paths:
                if os.path.exists(path):
                    font = ImageFont.truetype(path, 18)
                    break
                    
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Create a simple overlay without alpha blending
        box_height = 40
        # Create a copy of the image
        img_with_box = img_pil.copy()
        draw = ImageDraw.Draw(img_with_box)
        # Draw a semi-transparent black rectangle
        draw.rectangle([(0, 0), (img_pil.width, box_height)], fill=(0, 0, 0))
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # Write text
        draw.text((10, 10), f"Object Detection | {timestamp} | Objects: {len(detections)}", 
                 font=font, fill=(255, 255, 255))
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(img_with_box), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to a simple text overlay if PIL fails
        print(f"Error in display_results: {e}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Object Detection | {time.strftime('%H:%M:%S')} | Objects: {len(detections)}", 
                   (10, 25), font, 0.5, (255, 255, 255), 1)
        return frame

def check_available_models():
    """Check available models in Ollama"""
    if not check_server_connection():
        print("Cannot check models: server is not reachable")
        return []
        
    try:
        response = requests.get(f"{OLLAMA_SERVER}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            print(f"Available models: {models}")
            return models
        return []
    except requests.exceptions.RequestException as e:
        print(f"Could not check available models: {e}")
        return []

def main():
    """Main function to capture images and detect objects"""
    print("Starting Raspberry Pi Webcam Object Detection...")
    
    # Check if camera is available
    if not check_camera():
        print("Cannot continue without a camera. Exiting.")
        return
    
    # Try to set up local detection
    try:
        has_model = setup_local_detection()
        if not has_model:
            print("Attempting to download object detection model...")
            download_model_files()
            has_model = setup_local_detection()
    except Exception as e:
        print(f"Error setting up local detection: {e}")
    
    # Check if server is reachable first
    server_available = check_server_connection()
    
    # Determine if we should use fallback mode
    use_fallback = FORCE_FALLBACK or not server_available
    
    if use_fallback:
        print("Using local detection fallback mode.")
    else:
        # Check available models if server is available
        models = check_available_models()
        
        # If our preferred model isn't available but others are, use the first available
        global MODEL_NAME
        if models and MODEL_NAME not in models:
            if any(model for model in models if "llava" in model.lower()):
                # Prefer llava models if available
                MODEL_NAME = next(model for model in models if "llava" in model.lower())
            else:
                # Otherwise use the first available model
                MODEL_NAME = models[0]
            print(f"Using model: {MODEL_NAME}")
        
        if not models:
            print("Warning: No models available. Please run 'ollama pull llava' on your Windows machine.")
            print("Switching to fallback mode for now.")
            use_fallback = True
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    # Initialize detector with appropriate mode
    detector = AsyncDetector(use_fallback=use_fallback)
    last_detection_time = 0
    
    print("Press 'q' to quit, 's' to save a screenshot, 'f' to toggle fallback mode")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process every DETECTION_INTERVAL seconds
            current_time = time.time()
            if current_time - last_detection_time > DETECTION_INTERVAL and not detector.processing:
                detector.start_detection(frame)
                last_detection_time = current_time
            
            # Get latest detection result
            detections = detector.get_latest_result()
            
            # Display results on frame
            display_frame = display_results(frame, detections)
            
            # Show the frame
            cv2.imshow('Raspberry Pi Object Detection', display_frame)
            
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, display_frame)  # Save the annotated frame
                print(f"Screenshot saved as {filename}")
            elif key == ord('f'):
                # Toggle fallback mode
                detector.use_fallback = not detector.use_fallback
                print(f"Fallback mode {'enabled' if detector.use_fallback else 'disabled'}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
