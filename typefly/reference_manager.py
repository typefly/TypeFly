"""
Reference Photo Manager for Janah SAR
يدير الصورة المرجعية للشخص المفقود
"""

import os
import json
from pathlib import Path

class ReferenceManager:
    def __init__(self):
        self.config_dir = Path("data/references")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "current_reference.json"
        
        self.current_reference = self._load_current()
    
    def _load_current(self):
        """تحميل الصورة المرجعية الحالية"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def set_reference(self, image_path, person_info=None):
        """
        تعيين صورة مرجعية جديدة
        
        Args:
            image_path: مسار الصورة
            person_info: معلومات عن الشخص (اسم، عمر، إلخ)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # حفظ نسخة من الصورة
        from shutil import copy2
        saved_path = self.config_dir / "current_reference.jpg"
        copy2(image_path, saved_path)
        
        # حفظ المعلومات
        self.current_reference = {
            'image_path': str(saved_path),
            'original_path': image_path,
            'person_info': person_info or {},
            'timestamp': str(Path(image_path).stat().st_mtime)
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_reference, f, ensure_ascii=False, indent=2)
        
        return saved_path
    
    def get_reference(self):
        """الحصول على الصورة المرجعية الحالية"""
        if self.current_reference:
            return self.current_reference['image_path']
        return None
    
    def clear_reference(self):
        """مسح الصورة المرجعية"""
        self.current_reference = None
        if self.config_file.exists():
            self.config_file.unlink()

# Instance عام
reference_manager = ReferenceManager()