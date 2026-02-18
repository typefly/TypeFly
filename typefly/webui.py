# -*- coding: utf-8 -*-
import io, time, json, os, queue, sys
from flask import Flask, Response, render_template, request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64

sys.path.insert(0, str(Path(__file__).parent.parent))

from typefly.robot_info import RobotInfo
from typefly.utils import print_t, CURRENT_PROJ_DIR
from typefly.llm_controller import LLMController, _USER_LOG_QUEUE

try:
    from typefly.janah_cv_integrated import janah_cv_integrated
    from typefly.reference_manager import reference_manager
    FACE_RECOGNITION_ENABLED = True
    print_t("[Janah] ✅ Face Recognition enabled")
except Exception as e:
    FACE_RECOGNITION_ENABLED = False
    print_t(f"[Janah] ⚠️ Face Recognition disabled: {e}")


class TypeFly:
    def __init__(self, robot_info: RobotInfo):
        self.llm_controller = LLMController(robot_info)
        self.running = True
        self.app = Flask(__name__,
                        template_folder=os.path.join(CURRENT_PROJ_DIR, 'assets'))
        self.app.config['JSON_AS_ASCII'] = False
        self.app.config['JSON_SORT_KEYS'] = False

        self.missing_person_info = None
        self.last_detection_time = 0
        self.detection_cooldown = 15.0

        self.setup_routes()

    def setup_routes(self):

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/chat', methods=['POST'])
        def chat():
            data = request.get_json()
            user_message = data.get('message', '')

            if not user_message:
                return Response(
                    json.dumps({'type': 'text', 'content': '[WARNING] Empty command!'}, ensure_ascii=False),
                    mimetype='application/json'
                )
            if user_message == "exit":
                self.running = False
                return Response(
                    json.dumps({'type': 'text', 'content': 'Shutting down...'}, ensure_ascii=False),
                    mimetype='application/json'
                )

            self.llm_controller.put_instruction(user_message)

            def generate():
                ack = json.dumps({'type': 'text', 'content': 'Okay! Working on it...'}, ensure_ascii=False)
                yield f"data: {ack}\n\n"

                while True:
                    try:
                        msg = _USER_LOG_QUEUE.get(timeout=3.0)
                        if msg == '#end':
                            print_t("[UI] End of plan")
                            yield "data: [DONE]\n\n"
                            return
                    except queue.Empty:
                        continue

                    print_t(f"[UI] New message: {msg}")
                    msg_str = str(msg)

                    if 'data:image/' in msg_str and 'base64,' in msg_str:
                        import re
                        match = re.search(r'src="(data:image/[^"]+)"', msg_str)
                        img_src = match.group(1) if match else msg_str
                        yield f"data: {json.dumps({'type': 'image', 'content': img_src}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': msg_str}, ensure_ascii=False)}\n\n"

            return Response(
                generate(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Content-Type': 'text/event-stream; charset=utf-8'
                }
            )

        @self.app.route('/upload-reference', methods=['POST'])
        def upload_reference():
            if not FACE_RECOGNITION_ENABLED:
                return Response(
                    json.dumps({'success': False, 'message': 'Face Recognition not available'}, ensure_ascii=False),
                    mimetype='application/json'
                )
            try:
                if 'photo' not in request.files:
                    return Response(
                        json.dumps({'success': False, 'message': 'No photo uploaded'}, ensure_ascii=False),
                        mimetype='application/json'
                    )

                file = request.files['photo']
                name = request.form.get('name', 'Unknown')
                age = request.form.get('age', '0')
                clothing_color = request.form.get('clothing_color', 'unknown')
                description = request.form.get('description', '')

                upload_dir = Path('data/references')
                upload_dir.mkdir(parents=True, exist_ok=True)
                photo_path = upload_dir / 'current_missing_person.jpg'

                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                cv2.imwrite(str(photo_path), img)

                person_info = {
                    'name': name,
                    'age': int(age),
                    'clothing_color': clothing_color,
                    'description': description
                }

                success = janah_cv_integrated.setup_reference(str(photo_path), person_info)

                if success:
                    self.missing_person_info = person_info
                    # ✅ FIX #10: أرسل المعلومات للـ LLMController
                    self.llm_controller.set_missing_child_info(person_info)
                    print_t(f"[Janah] ✅ Reference + LLM updated: {name}, {age}y, {clothing_color}")

                    return Response(
                        json.dumps({
                            'success': True,
                            'message': f'✅ تم إعداد البحث عن: {name}\n✅ Reference setup for: {name}',
                            'person_info': person_info
                        }, ensure_ascii=False),
                        mimetype='application/json'
                    )
                else:
                    return Response(
                        json.dumps({'success': False, 'message': '❌ Failed to process photo - no face detected'}, ensure_ascii=False),
                        mimetype='application/json'
                    )

            except Exception as e:
                print_t(f"[Janah] ❌ Upload error: {e}")
                return Response(
                    json.dumps({'success': False, 'message': f'Error: {str(e)}'}, ensure_ascii=False),
                    mimetype='application/json'
                )

        @self.app.route('/get-reference-status', methods=['GET'])
        def get_reference_status():
            if FACE_RECOGNITION_ENABLED and janah_cv_integrated.is_face_trained:
                return Response(
                    json.dumps({'setup': True, 'person_info': self.missing_person_info}, ensure_ascii=False),
                    mimetype='application/json'
                )
            return Response(json.dumps({'setup': False}, ensure_ascii=False), mimetype='application/json')

        @self.app.route('/robot-pov/')
        def video_feed_pov():
            return Response(
                self.generate_mjpeg_stream('pov'),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/health')
        def health():
            return Response(
                json.dumps({
                    'status': 'running',
                    'robot': self.running,
                    'face_recognition': FACE_RECOGNITION_ENABLED,
                    'reference_setup': FACE_RECOGNITION_ENABLED and janah_cv_integrated.is_face_trained,
                    'llm_child_info': bool(self.llm_controller.missing_child_info.get('name'))
                }, ensure_ascii=False),
                mimetype='application/json'
            )

    def process_frame_with_face_recognition(self, frame_pil: Image.Image):
        """
        ✅ FIX #2: استخدام YOLO detections الحقيقية من robot.obs
        """
        if not FACE_RECOGNITION_ENABLED or not janah_cv_integrated.is_face_trained:
            return frame_pil, None

        try:
            frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            h, w = frame_cv.shape[:2]

            # ✅ FIX #2: الـ detections الحقيقية من YOLO
            real_yolo = self.llm_controller.robot.obs.image_process_result.get("yolo", [])

            if real_yolo:
                yolo_detections = [
                    {
                        'class': obj.name,
                        'confidence': 1.0,
                        'bbox': {'x': obj.x, 'y': obj.y, 'width': obj.w, 'height': obj.h}
                    }
                    for obj in real_yolo if obj.name == 'person'
                ]
            else:
                # Fallback إذا YOLO مو شغال
                yolo_detections = [{
                    'class': 'person',
                    'confidence': 0.5,
                    'bbox': {'x': 0.5, 'y': 0.5, 'width': 0.9, 'height': 0.9}
                }]

            enriched = janah_cv_integrated.process_frame(frame_cv, yolo_detections)
            alert_message = None

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
                color = (0, 255, 0) if is_target else (128, 128, 128)
                label = f"TARGET! {match_score}%" if is_target else f"Person ({match_score}%)"

                if is_target:
                    current_time = time.time()
                    if current_time - self.last_detection_time > self.detection_cooldown:
                        alert_message = {
                            'type': 'target_found',
                            'name': self.missing_person_info['name'],
                            'match_score': match_score,
                            'info': self.missing_person_info
                        }
                        self.last_detection_time = current_time
                        print_t(f"[Janah] 🎯 TARGET FOUND! {self.missing_person_info['name']} ({match_score}%)")

                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), color, 3)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame_cv, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                cv2.putText(frame_cv, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            return Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)), alert_message

        except Exception as e:
            print_t(f"[Janah] ⚠️ Frame processing error: {e}")
            return frame_pil, None

    def generate_mjpeg_stream(self, source: str):
        while self.running:
            frame = self.llm_controller.fetch_robot_pov() if source == 'pov' else None
            if frame is None:
                time.sleep(1.0/30.0)
                continue

            frame, alert = self.process_frame_with_face_recognition(frame)

            if alert:
                alert_msg = (
                    f"🚨 تنبيه! | ALERT!\n{'─'*28}\n"
                    f"🎯 تم العثور على: {alert['name']}\n"
                    f"🎯 Target Found: {alert['name']}\n\n"
                    f"📊 نسبة المطابقة | Match: {alert['match_score']}%\n"
                    f"👤 العمر | Age: {alert['info']['age']}\n"
                    f"👕 اللون | Color: {alert['info']['clothing_color']}\n\n"
                    f"🕊️ \"لن يرتاح الجناح حتى تتحقق لحظة العودة\""
                )
                _USER_LOG_QUEUE.put(alert_msg)

            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0/30.0)

    def run(self):
        self.llm_controller.start_controller()
        print_t("[Janah] Starting Flask server on http://127.0.0.1:50000")
        print_t(f"[Janah] Face Recognition: {'✅' if FACE_RECOGNITION_ENABLED else '❌'}")
        print_t("[Janah] 🕊️ ما دام الجناح ممدوداً فالأمل قريب")
        self.app.run(host='127.0.0.1', port=50000, debug=False, threaded=True)
        self.llm_controller.stop_controller()


def main():
    with open(os.path.join(CURRENT_PROJ_DIR, 'config/robot_info.json'), 'r') as f:
        typefly = TypeFly(RobotInfo.from_dict(json.load(f)))
        typefly.run()


if __name__ == '__main__':
    main()