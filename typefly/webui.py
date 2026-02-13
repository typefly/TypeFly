import io, time, json, os, queue, sys
from flask import Flask, Response, render_template, request, jsonify
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typefly.robot_info import RobotInfo
from typefly.utils import print_t, CURRENT_PROJ_DIR
from typefly.llm_controller import LLMController, _USER_LOG_QUEUE

# Import Janah CV
try:
    from typefly.janah_cv_integrated import janah_cv_integrated
    from typefly.reference_manager import reference_manager
    FACE_RECOGNITION_ENABLED = True
    print_t("[ط¬ظ†ط§ط­ | Janah] âœ… Face Recognition enabled - ظ…ط§ ط¯ط§ظ… ط§ظ„ط¬ظ†ط§ط­ ظ…ظ…ط¯ظˆط¯ط§ظ‹ ظپط§ظ„ط£ظ…ظ„ ظ‚ط±ظٹط¨")
except Exception as e:
    FACE_RECOGNITION_ENABLED = False
    print_t(f"[ط¬ظ†ط§ط­ | Janah] âڑ ï¸ڈ Face Recognition disabled: {e}")

class TypeFly:
    def __init__(self, robot_info: RobotInfo):
        self.llm_controller = LLMController(robot_info)
        self.running = True
        self.app = Flask(__name__, 
                        template_folder=os.path.join(CURRENT_PROJ_DIR, 'assets'))
        
        # Janah SAR specific
        self.missing_person_info = None
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds between alerts
        
        self.setup_routes()

    def setup_routes(self):
        """Sets up the Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Handle chat messages and stream responses using SSE."""
            data = request.get_json()
            user_message = data.get('message', '')
            
            if not user_message:
                return jsonify({'type': 'text', 'content': '[WARNING] Empty command!'})
            
            if user_message == "exit":
                self.running = False
                return jsonify({'type': 'text', 'content': 'Shutting down...'})
            
            # Send instruction to LLM Controller
            self.llm_controller.put_instruction(user_message)
            
            def generate():
                # Send initial acknowledgment
                yield f"data: {json.dumps({'type': 'text', 'content': 'Okay! Working on it...'})}\n\n"
                
                # Stream messages from the queue as they arrive
                while True:
                    try:
                        msg = _USER_LOG_QUEUE.get(timeout=3.0)
                        if msg == '#end':
                            print_t("[UI] End of plan")
                            return "data: [DONE]\n\n"
                    
                    except queue.Empty:
                        continue

                    print_t(f"[UI] New message: {msg}")
                    msg_str = str(msg)
                    
                    # Check if message contains an image (base64 encoded)
                    if '<img src="data:image/' in msg_str:
                        response_data = json.dumps({'type': 'image', 'content': msg_str}, ensure_ascii=False)
                        yield f"data: {response_data}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': msg_str})}\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        # â•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گ
        # JANAH SAR: Upload Missing Person Photo
        # â•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گâ•گ
        @self.app.route('/upload-reference', methods=['POST'])
        def upload_reference():
            """Upload reference photo of missing person."""
            if not FACE_RECOGNITION_ENABLED:
                return jsonify({
                    'success': False,
                    'message': 'Face Recognition not available'
                })
            
            try:
                # Get uploaded file
                if 'photo' not in request.files:
                    return jsonify({
                        'success': False,
                        'message': 'No photo uploaded'
                    })
                
                file = request.files['photo']
                
                # Get person info
                name = request.form.get('name', 'Unknown')
                age = request.form.get('age', '0')
                clothing_color = request.form.get('clothing_color', 'unknown')
                description = request.form.get('description', '')
                
                # Save uploaded photo
                upload_dir = Path('data/references')
                upload_dir.mkdir(parents=True, exist_ok=True)
                photo_path = upload_dir / 'current_missing_person.jpg'
                
                # Read and save image
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                cv2.imwrite(str(photo_path), img)
                
                # Setup person info
                person_info = {
                    'name': name,
                    'age': int(age),
                    'clothing_color': clothing_color,
                    'description': description
                }
                
                # Train face recognition
                success = janah_cv_integrated.setup_reference(str(photo_path), person_info)
                
                if success:
                    self.missing_person_info = person_info
                    print_t(f"[ط¬ظ†ط§ط­ | Janah] âœ… Reference setup for: {name} - ط§ظ„ط¨ط­ط« ط¨ط¯ط£ ط§ظ„ط¢ظ†")
                    
                    return jsonify({
                        'success': True,
                        'message': f'âœ… طھظ… ط¥ط¹ط¯ط§ط¯ ط§ظ„ط¨ط­ط« ط¹ظ†: {name}\nâœ… Reference setup for: {name}',
                        'person_info': person_info
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'â‌Œ Failed to process photo - no face detected'
                    })
                    
            except Exception as e:
                print_t(f"[ط¬ظ†ط§ط­ | Janah] â‌Œ Upload error: {e}")
                return jsonify({
                    'success': False,
                    'message': f'Error: {str(e)}'
                })
        
        @self.app.route('/get-reference-status', methods=['GET'])
        def get_reference_status():
            """Get current reference photo status."""
            if FACE_RECOGNITION_ENABLED and janah_cv_integrated.is_face_trained:
                return jsonify({
                    'setup': True,
                    'person_info': self.missing_person_info
                })
            else:
                return jsonify({
                    'setup': False
                })
        
        @self.app.route('/robot-pov/')
        def video_feed_pov():
            """Stream robot POV video feed with face recognition."""
            return Response(
                self.generate_mjpeg_stream('pov'),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'running',
                'robot': self.running,
                'face_recognition': FACE_RECOGNITION_ENABLED,
                'reference_setup': FACE_RECOGNITION_ENABLED and janah_cv_integrated.is_face_trained
            })

    def process_frame_with_face_recognition(self, frame_pil):
        """
        Process frame with face recognition
        Returns: annotated frame, alert message (if target found)
        """
        if not FACE_RECOGNITION_ENABLED or not janah_cv_integrated.is_face_trained:
            return frame_pil, None
        
        try:
            # Convert PIL to OpenCV
            frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            h, w = frame_cv.shape[:2]
            
            # Simple detection: assume whole frame contains a person
            yolo_detections = [{
                'class': 'person',
                'confidence': 1.0,
                'bbox': {'x': 0.5, 'y': 0.5, 'width': 0.8, 'height': 0.8}
            }]
            
            # Face recognition
            enriched = janah_cv_integrated.process_frame(frame_cv, yolo_detections)
            
            alert_message = None
            
            # Draw results
            for det in enriched:
                if det.get('class') != 'person':
                    continue
                
                bbox = det['bbox']
                x1 = int((bbox['x'] - bbox['width']/2) * w)
                y1 = int((bbox['y'] - bbox['height']/2) * h)
                x2 = int((bbox['x'] + bbox['width']/2) * w)
                y2 = int((bbox['y'] + bbox['height']/2) * h)
                
                is_target = det.get('is_target', False)
                match_score = det.get('face_match_score', 0)
                
                # Color and label
                if is_target:
                    color = (0, 255, 0)  # Green
                    label = f"TARGET FOUND! {match_score}%"
                    
                    # Create alert (with cooldown)
                    current_time = time.time()
                    if current_time - self.last_detection_time > self.detection_cooldown:
                        alert_message = {
                            'type': 'target_found',
                            'name': self.missing_person_info['name'],
                            'match_score': match_score,
                            'info': self.missing_person_info
                        }
                        self.last_detection_time = current_time
                        print_t(f"[ط¬ظ†ط§ط­ | Janah] ًںژ¯ TARGET FOUND! {self.missing_person_info['name']} ({match_score}%)")
                else:
                    color = (128, 128, 128)  # Gray
                    label = f"Person ({match_score}%)"
                
                # Draw bounding box
                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), color, 3)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame_cv, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame_cv, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to PIL
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(frame_rgb)
            
            return annotated_pil, alert_message
            
        except Exception as e:
            print_t(f"[ط¬ظ†ط§ط­ | Janah] âڑ ï¸ڈ Frame processing error: {e}")
            return frame_pil, None

    def generate_mjpeg_stream(self, source: str):
        """Generate MJPEG stream for video feeds with face recognition."""
        while self.running:
            if source == 'pov':
                frame = self.llm_controller.fetch_robot_pov()
            else:
                frame = None
            
            if frame is None:
                time.sleep(1.0 / 30.0)
                continue
            
            # Process with face recognition
            frame, alert = self.process_frame_with_face_recognition(frame)
            
            # Send alert to chat if target found
            if alert:
                alert_msg = f"""
ًںڑ¨ طھظ†ط¨ظٹظ‡! | ALERT!
â”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پâ”پ
ًںژ¯ طھظ… ط§ظ„ط¹ط«ظˆط± ط¹ظ„ظ‰: {alert['name']}
ًںژ¯ Target Found: {alert['name']}

ًں“ٹ ظ†ط³ط¨ط© ط§ظ„ظ…ط·ط§ط¨ظ‚ط© | Match: {alert['match_score']}%
ًں‘¤ ط§ظ„ط¹ظ…ط± | Age: {alert['info']['age']}
ًں‘• ط§ظ„ظ„ظˆظ† | Color: {alert['info']['clothing_color']}

ًں•ٹï¸ڈ "ظ„ظ† ظٹط±طھط§ط­ ط§ظ„ط¬ظ†ط§ط­ ط­طھظ‰ طھطھط­ظ‚ظ‚ ظ„ط­ط¸ط© ط§ظ„ط¹ظˆط¯ط©"
"""
                _USER_LOG_QUEUE.put(alert_msg)
            
            # Convert to JPEG
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def run(self):
        """Start the TypeFly system with Flask server."""
        # Start the LLM controller
        self.llm_controller.start_controller()

        # Start the Flask server
        print_t("[ط¬ظ†ط§ط­ | Janah] Starting Flask server on http://0.0.0.0:50000")
        print_t(f"[ط¬ظ†ط§ط­ | Janah] Face Recognition: {'âœ… Enabled' if FACE_RECOGNITION_ENABLED else 'â‌Œ Disabled'}")
        print_t("[ط¬ظ†ط§ط­ | Janah] ًں•ٹï¸ڈ ظ…ط§ ط¯ط§ظ… ط§ظ„ط¬ظ†ط§ط­ ظ…ظ…ط¯ظˆط¯ط§ظ‹ ظپط§ظ„ط£ظ…ظ„ ظ‚ط±ظٹط¨")
        
        self.app.run(host='127.0.0.1', port=50000, debug=False, threaded=True)
        
        # When Flask stops, stop the LLM controller
        self.llm_controller.stop_controller()

def main():
    with open(os.path.join(CURRENT_PROJ_DIR, 'config/robot_info.json'), 'r') as f:
        typefly = TypeFly(RobotInfo.from_dict(json.load(f)))
        typefly.run()

if __name__ == '__main__':
    main()


