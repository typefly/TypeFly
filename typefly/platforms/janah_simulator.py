"""
Janah SAR Simulator
====================
يحاكي حركات الدرون بالكامل مع:
- كاميرا اللابتوب كـ "كاميرا الدرون"
- تتبع كل حركة (أمام/خلف/يمين/يسار/ارتفاع)
- فحص حدود آمنة (لا يصطدم)
- لوج كامل لكل أمر
- نفس الـ interface بالضبط للـ TelloWrapper

لما تريدين التبديل للدرون الحقيقي:
  robot_info.json → "robot_type": "tello"
  بس!
"""

import time
import math
import threading
from typing import Any, Optional
from PIL import Image
import cv2
import numpy as np
from overrides import overrides

# ===== إعدادات الحدود الآمنة =====
SAFE_BOUNDS = {
    'x_min': -500,   # أقصى يسار (سم)
    'x_max':  500,   # أقصى يمين (سم)
    'y_min': -500,   # أقصى خلف (سم)
    'y_max':  500,   # أقصى أمام (سم)
    'z_min':    0,   # الأرض (سم)
    'z_max':  300,   # أقصى ارتفاع (سم)
}

MOVEMENT_MIN = 20    # أقل حركة (سم) - نفس Tello
MOVEMENT_MAX = 300   # أكبر حركة (سم) - نفس Tello
EXECUTION_DELAY = 0.5  # ثواني انتظار بعد كل أمر

# ===== ألوان اللوج =====
class Colors:
    MOVE    = "\033[94m"   # أزرق
    ROTATE  = "\033[93m"   # أصفر
    LIFT    = "\033[96m"   # سماوي
    OK      = "\033[92m"   # أخضر
    WARN    = "\033[91m"   # أحمر
    RESET   = "\033[0m"
    BOLD    = "\033[1m"

class DronePosition:
    """تتبع موضع الدرون في الفضاء ثلاثي الأبعاد"""
    
    def __init__(self):
        self.x = 0.0      # يمين/يسار (سم)
        self.y = 0.0      # أمام/خلف (سم)
        self.z = 100.0    # ارتفاع (سم) - يبدأ من 1 متر
        self.heading = 0.0  # اتجاه (0 = شمال)
        self.is_flying = False
        self.history = []  # سجل كل الحركات
        self.total_distance = 0.0
        
    def record(self, command: str, dx=0, dy=0, dz=0, dh=0):
        """سجل حركة جديدة"""
        entry = {
            'time': time.strftime('%H:%M:%S'),
            'command': command,
            'dx': dx, 'dy': dy, 'dz': dz, 'dh': dh,
            'pos_after': (self.x, self.y, self.z, self.heading)
        }
        self.history.append(entry)
        self.total_distance += math.sqrt(dx**2 + dy**2 + dz**2)
        
    def check_bounds(self, new_x, new_y, new_z) -> tuple[bool, str]:
        """فحص إذا الحركة آمنة"""
        if new_x < SAFE_BOUNDS['x_min'] or new_x > SAFE_BOUNDS['x_max']:
            return False, f"خطر: حد X ({new_x:.0f}cm) خارج النطاق [{SAFE_BOUNDS['x_min']}, {SAFE_BOUNDS['x_max']}]"
        if new_y < SAFE_BOUNDS['y_min'] or new_y > SAFE_BOUNDS['y_max']:
            return False, f"خطر: حد Y ({new_y:.0f}cm) خارج النطاق [{SAFE_BOUNDS['y_min']}, {SAFE_BOUNDS['y_max']}]"
        if new_z < SAFE_BOUNDS['z_min'] or new_z > SAFE_BOUNDS['z_max']:
            return False, f"خطر: ارتفاع ({new_z:.0f}cm) خارج النطاق [{SAFE_BOUNDS['z_min']}, {SAFE_BOUNDS['z_max']}]"
        return True, "OK"

    def status(self) -> str:
        """حالة الدرون الحالية"""
        return (f"📍 الموضع: X={self.x:.0f} Y={self.y:.0f} Z={self.z:.0f}cm | "
                f"🧭 الاتجاه: {self.heading:.0f}° | "
                f"📏 إجمالي المسافة: {self.total_distance:.0f}cm")

    def print_map(self):
        """رسم خريطة نصية بسيطة للمسار"""
        print("\n" + "="*50)
        print("🗺️  مسار الدرون (نظرة من الأعلى):")
        print("="*50)
        
        # مقياس: كل وحدة = 50 سم
        scale = 50
        grid_size = 20
        center = grid_size // 2
        
        grid = [['·'] * grid_size for _ in range(grid_size)]
        
        # ارسم المسار
        for i, entry in enumerate(self.history):
            px = int(entry['pos_after'][0] / scale) + center
            py = int(entry['pos_after'][1] / scale) + center
            if 0 <= px < grid_size and 0 <= py < grid_size:
                grid[grid_size - 1 - py][px] = '○' if i < len(self.history)-1 else '◉'
        
        # نقطة البداية
        grid[grid_size - 1 - center][center] = '★'
        
        for row in grid:
            print('  ' + ' '.join(row))
        
        print(f"\n  ★ = نقطة البداية | ○ = مسار | ◉ = الموضع الحالي")
        print("="*50 + "\n")


class JanahSimulator:
    """
    محاكي درون Tello الكامل
    يستخدم كاميرا اللابتوب بدلاً من الدرون الحقيقي
    """
    
    def __init__(self, camera_index: int = 0):
        self.pos = DronePosition()
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[Image.Image] = None
        self.running = False
        self._lock = threading.Lock()
        self._camera_thread: Optional[threading.Thread] = None
        self.battery = 85  # بطارية افتراضية
        
        print(f"\n{Colors.BOLD}{'='*55}{Colors.RESET}")
        print(f"{Colors.OK}{Colors.BOLD}  🚁 Janah SAR Simulator جاهز{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*55}{Colors.RESET}")
        print(f"  📷 الكاميرا: #{camera_index} (كاميرا اللابتوب)")
        print(f"  🔋 البطارية: {self.battery}%")
        print(f"  📍 الموضع الابتدائي: X=0, Y=0, Z=0 (أرض)")
        print(f"  🛡️  الحدود الآمنة: ±5 متر أفقياً, 0-3 متر ارتفاعاً")
        print(f"{Colors.BOLD}{'='*55}{Colors.RESET}\n")

    # ==========================================
    # تحكم الطيران
    # ==========================================
    
    def connect(self):
        """اتصال بالدرون (simulation)"""
        print(f"{Colors.OK}✅ [SIM] اتصال ناجح بالدرون (simulation mode){Colors.RESET}")
        return True

    def takeoff(self):
        """إقلاع"""
        self.pos.z = 100  # يرتفع 1 متر
        self.pos.is_flying = True
        self.pos.record("takeoff", dz=100)
        print(f"{Colors.OK}🚀 [SIM] إقلاع! الارتفاع: {self.pos.z}cm{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def land(self):
        """هبوط"""
        old_z = self.pos.z
        self.pos.z = 0
        self.pos.is_flying = False
        self.pos.record("land", dz=-old_z)
        print(f"{Colors.OK}🛬 [SIM] هبوط آمن! الموضع النهائي: X={self.pos.x:.0f}, Y={self.pos.y:.0f}{Colors.RESET}")
        self._print_flight_summary()
        time.sleep(EXECUTION_DELAY)

    def move_forward(self, cm: int):
        """تحرك للأمام"""
        cm = self._cap_dist(cm)
        rad = math.radians(self.pos.heading)
        new_x = self.pos.x + cm * math.sin(rad)
        new_y = self.pos.y + cm * math.cos(rad)
        
        safe, msg = self.pos.check_bounds(new_x, new_y, self.pos.z)
        if not safe:
            print(f"{Colors.WARN}⚠️  [SIM] حركة مرفوضة: {msg}{Colors.RESET}")
            return
            
        self.pos.x = new_x
        self.pos.y = new_y
        self.pos.record("forward", dx=new_x-self.pos.x, dy=new_y-self.pos.y)
        print(f"{Colors.MOVE}➡️  [SIM] أمام {cm}cm | {self.pos.status()}{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def move_back(self, cm: int):
        """تحرك للخلف"""
        cm = self._cap_dist(cm)
        rad = math.radians(self.pos.heading)
        new_x = self.pos.x - cm * math.sin(rad)
        new_y = self.pos.y - cm * math.cos(rad)
        
        safe, msg = self.pos.check_bounds(new_x, new_y, self.pos.z)
        if not safe:
            print(f"{Colors.WARN}⚠️  [SIM] حركة مرفوضة: {msg}{Colors.RESET}")
            return
            
        self.pos.x = new_x
        self.pos.y = new_y
        self.pos.record("back", dx=new_x-self.pos.x, dy=new_y-self.pos.y)
        print(f"{Colors.MOVE}⬅️  [SIM] خلف {cm}cm | {self.pos.status()}{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def move_left(self, cm: int):
        """تحرك يسار"""
        cm = self._cap_dist(cm)
        rad = math.radians(self.pos.heading - 90)
        new_x = self.pos.x + cm * math.sin(rad)
        new_y = self.pos.y + cm * math.cos(rad)
        
        safe, msg = self.pos.check_bounds(new_x, new_y, self.pos.z)
        if not safe:
            print(f"{Colors.WARN}⚠️  [SIM] حركة مرفوضة: {msg}{Colors.RESET}")
            return
            
        self.pos.x = new_x
        self.pos.y = new_y
        self.pos.record("left")
        print(f"{Colors.MOVE}↖️  [SIM] يسار {cm}cm | {self.pos.status()}{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def move_right(self, cm: int):
        """تحرك يمين"""
        cm = self._cap_dist(cm)
        rad = math.radians(self.pos.heading + 90)
        new_x = self.pos.x + cm * math.sin(rad)
        new_y = self.pos.y + cm * math.cos(rad)
        
        safe, msg = self.pos.check_bounds(new_x, new_y, self.pos.z)
        if not safe:
            print(f"{Colors.WARN}⚠️  [SIM] حركة مرفوضة: {msg}{Colors.RESET}")
            return
            
        self.pos.x = new_x
        self.pos.y = new_y
        self.pos.record("right")
        print(f"{Colors.MOVE}↗️  [SIM] يمين {cm}cm | {self.pos.status()}{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def move_up(self, cm: int):
        """ارتفع"""
        cm = self._cap_dist(cm)
        new_z = self.pos.z + cm
        safe, msg = self.pos.check_bounds(self.pos.x, self.pos.y, new_z)
        if not safe:
            print(f"{Colors.WARN}⚠️  [SIM] ارتفاع مرفوض: {msg}{Colors.RESET}")
            return
        self.pos.z = new_z
        self.pos.record("up", dz=cm)
        print(f"{Colors.LIFT}⬆️  [SIM] ارتفاع {cm}cm | ارتفاع الآن: {self.pos.z}cm{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def move_down(self, cm: int):
        """انزل"""
        cm = self._cap_dist(cm)
        new_z = max(30, self.pos.z - cm)  # لا تنزل أقل من 30سم
        self.pos.z = new_z
        self.pos.record("down", dz=-cm)
        print(f"{Colors.LIFT}⬇️  [SIM] نزول {cm}cm | ارتفاع الآن: {self.pos.z}cm{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def rotate_clockwise(self, deg: int):
        """دوران يمين"""
        self.pos.heading = (self.pos.heading + deg) % 360
        self.pos.record("rotate_cw", dh=deg)
        print(f"{Colors.ROTATE}🔄 [SIM] دوران يمين {deg}° | الاتجاه: {self.pos.heading:.0f}°{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def rotate_counter_clockwise(self, deg: int):
        """دوران يسار"""
        self.pos.heading = (self.pos.heading - deg) % 360
        self.pos.record("rotate_ccw", dh=-deg)
        print(f"{Colors.ROTATE}🔃 [SIM] دوران يسار {deg}° | الاتجاه: {self.pos.heading:.0f}°{Colors.RESET}")
        time.sleep(EXECUTION_DELAY)

    def send_control_command(self, cmd: str):
        """keep-alive command"""
        pass  # في السيمولاتور لا نحتاج شيء

    def query_battery(self) -> int:
        """استعلام البطارية"""
        # محاكاة نزول البطارية تدريجياً
        self.battery = max(10, self.battery - 0.1)
        return int(self.battery)

    # ==========================================
    # الكاميرا
    # ==========================================

    def streamon(self):
        """تشغيل الكاميرا"""
        self.running = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        print(f"{Colors.OK}📷 [SIM] كاميرا اللابتوب شغّالة{Colors.RESET}")

    def streamoff(self):
        """إيقاف الكاميرا"""
        self.running = False
        if self.cap:
            self.cap.release()
        print(f"📷 [SIM] الكاميرا أُوقفت")

    def _camera_loop(self):
        """حلقة قراءة الكاميرا"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"{Colors.WARN}⚠️  [SIM] لم تُفتح الكاميرا {self.camera_index} - جارٍ استخدام صورة بديلة{Colors.RESET}")
            self._use_dummy_frame()
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # أضف HUD على الصورة
                frame = self._add_hud(frame)
                with self._lock:
                    self.current_frame = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
            time.sleep(1/30)
        
        self.cap.release()

    def _add_hud(self, frame: np.ndarray) -> np.ndarray:
        """أضف معلومات الطيران على الصورة"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # شريط علوي شفاف
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # معلومات الموضع
        info = f"SIM | X:{self.pos.x:.0f} Y:{self.pos.y:.0f} Z:{self.pos.z:.0f}cm | {self.pos.heading:.0f}deg | BAT:{int(self.battery)}%"
        cv2.putText(frame, info, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # نقطة المنتصف (crosshair)
        cx, cy = w//2, h//2
        cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 255, 0), 1)
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), 1)
        
        # مؤشر الاتجاه
        cv2.putText(frame, f"Janah SAR - Simulation Mode",
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame

    def _use_dummy_frame(self):
        """إنشاء صورة وهمية إذا الكاميرا مو متاحة"""
        while self.running:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SIMULATION MODE - No Camera",
                       (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Pos: X={self.pos.x:.0f} Y={self.pos.y:.0f} Z={self.pos.z:.0f}",
                       (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            with self._lock:
                self.current_frame = Image.fromarray(frame)
            time.sleep(1/10)

    def get_frame_read(self):
        """إرجاع frame reader متوافق مع djitellopy"""
        return SimFrameReader(self)

    # ==========================================
    # Helpers
    # ==========================================

    def _cap_dist(self, dist: int) -> int:
        """تحديد الحركة بين الحد الأدنى والأقصى"""
        return max(MOVEMENT_MIN, min(MOVEMENT_MAX, abs(dist)))

    def _print_flight_summary(self):
        """ملخص رحلة الدرون"""
        print(f"\n{Colors.BOLD}{'='*55}{Colors.RESET}")
        print(f"{Colors.BOLD}  📊 ملخص رحلة الدرون{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*55}{Colors.RESET}")
        print(f"  📏 إجمالي المسافة: {self.pos.total_distance:.0f}cm ({self.pos.total_distance/100:.1f}m)")
        print(f"  🔋 البطارية المتبقية: {int(self.battery)}%")
        print(f"  📋 عدد الأوامر: {len(self.pos.history)}")
        print(f"  📍 الموضع النهائي: X={self.pos.x:.0f}, Y={self.pos.y:.0f}")
        print(f"{Colors.BOLD}{'='*55}{Colors.RESET}\n")
        self.pos.print_map()


class SimFrameReader:
    """متوافق مع djitellopy BackgroundFrameRead"""
    def __init__(self, sim: JanahSimulator):
        self._sim = sim
        
    @property
    def frame(self):
        with self._sim._lock:
            if self._sim.current_frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return cv2.cvtColor(np.array(self._sim.current_frame), cv2.COLOR_RGB2BGR)


# ==========================================
# TelloWrapper محسّن يدعم السيمولاتور
# ==========================================

class JanahTelloWrapper:
    """
    TelloWrapper محسّن لـ Janah SAR
    يدعم وضعين:
    - simulation: يستخدم JanahSimulator + كاميرا اللابتوب
    - real: يستخدم djitellopy مع الدرون الحقيقي
    
    للتبديل: غيّري robot_info.json
    """
    
    @staticmethod
    def create(mode: str = "simulation", camera_index: int = 0):
        """
        إنشاء wrapper المناسب
        
        Args:
            mode: "simulation" أو "real"  
            camera_index: رقم الكاميرا (0 = كاميرا اللابتوب)
        """
        if mode == "simulation":
            print(f"{Colors.OK}🎮 [Janah] وضع السيمولاتور - كاميرا اللابتوب #{camera_index}{Colors.RESET}")
            return JanahSimulator(camera_index)
        else:
            print(f"{Colors.OK}🚁 [Janah] وضع الدرون الحقيقي - Tello{Colors.RESET}")
            try:
                from djitellopy import Tello
                return Tello()
            except ImportError:
                print(f"{Colors.WARN}⚠️ djitellopy غير مثبت، استخدام السيمولاتور بدلاً{Colors.RESET}")
                return JanahSimulator(camera_index)


# ==========================================
# اختبار مستقل
# ==========================================

def test_simulator():
    """اختبر السيمولاتور بدون TypeFly"""
    print("\n🧪 اختبار السيمولاتور...")
    
    sim = JanahSimulator(camera_index=0)
    sim.connect()
    
    print("\n--- اختبار حركات أساسية ---")
    sim.takeoff()
    
    print("\n[1] أمام 100cm")
    sim.move_forward(100)
    
    print("\n[2] دوران يمين 90°")
    sim.rotate_clockwise(90)
    
    print("\n[3] أمام 50cm")
    sim.move_forward(50)
    
    print("\n[4] ارتفاع 50cm")
    sim.move_up(50)
    
    print("\n[5] اختبار حد الأمان (500cm = يرفض)")
    sim.move_forward(600)  # يجب أن يُرفض
    
    print("\n[6] هبوط")
    sim.land()
    
    print(f"\n{Colors.OK}✅ اختبار السيمولاتور اكتمل!{Colors.RESET}")


if __name__ == "__main__":
    test_simulator()