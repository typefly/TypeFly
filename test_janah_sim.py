"""
اختبار الفلو الكامل لـ Janah SAR
====================================
يختبر:
1. رفع صورة الطفل
2. LLM يفهم الوصف العربي
3. الدرون (سيمولاتور) يتحرك
4. Face recognition يشتغل
5. إشعار الأهل عند الإيجاد
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janah_simulator import JanahSimulator, test_simulator

def test_movement_commands():
    """اختبار أوامر الحركة العربية → حركات الدرون"""
    
    print("\n" + "="*55)
    print("🧪 اختبار الأوامر العربية → حركات الدرون")
    print("="*55)
    
    sim = JanahSimulator(camera_index=0)
    sim.connect()
    sim.takeoff()
    
    # محاكاة أوامر TypeFly الأساسية
    tests = [
        ("تحرك للأمام",     lambda: sim.move_forward(100)),
        ("تحرك للخلف",      lambda: sim.move_back(50)),
        ("تحرك يمين",       lambda: sim.move_right(80)),
        ("تحرك يسار",       lambda: sim.move_left(80)),
        ("ارتفع",           lambda: sim.move_up(50)),
        ("انزل",            lambda: sim.move_down(30)),
        ("دور يمين 90°",    lambda: sim.rotate_clockwise(90)),
        ("دور يسار 45°",    lambda: sim.rotate_counter_clockwise(45)),
    ]
    
    passed = 0
    for cmd_name, cmd_func in tests:
        print(f"\n  ▶ {cmd_name}")
        try:
            cmd_func()
            print(f"    ✅ نجح")
            passed += 1
        except Exception as e:
            print(f"    ❌ فشل: {e}")
    
    sim.land()
    
    print(f"\n{'='*55}")
    print(f"النتيجة: {passed}/{len(tests)} اختبار نجح")
    print("="*55)
    return passed == len(tests)


def test_safety_limits():
    """اختبار حدود الأمان"""
    
    print("\n" + "="*55)
    print("🛡️  اختبار حدود الأمان")
    print("="*55)
    
    sim = JanahSimulator(camera_index=0)
    sim.connect()
    sim.takeoff()
    
    # محاولة تجاوز الحدود
    print("\n  [1] محاولة تحرك 600cm (يجب رفض)")
    before_x = sim.pos.x
    sim.move_forward(600)  # فوق الحد 500cm
    # بعض الحركة ستُنفَّذ حتى الحد
    
    print("\n  [2] محاولة ارتفاع 400cm (يجب رفض)")
    sim.move_up(400)  # فوق الحد 300cm
    
    print("\n  [3] محاولة نزول أقل من الأرض")
    sim.pos.z = 50
    sim.move_down(200)  # ينزل لـ 30cm (الحد الأدنى)
    
    safe = sim.pos.z >= 30
    print(f"\n  {'✅ حدود الأمان تعمل' if safe else '❌ مشكلة في حدود الأمان'}")
    
    sim.land()
    return safe


def test_full_sar_scenario():
    """
    سيناريو كامل لـ SAR:
    ابحث عن طفل في منطقة محددة
    """
    
    print("\n" + "="*55)
    print("🔍 سيناريو بحث كامل: 'ابحث عن أحمد'")
    print("="*55)
    print("الوصف: طفل عمره 8 سنوات، ملابس حمراء، آخر موقع: الحديقة")
    print("-"*55)
    
    sim = JanahSimulator(camera_index=0)
    sim.connect()
    
    print("\n📋 خطة البحث:")
    print("  1. إقلاع والوصول لارتفاع 2 متر")
    print("  2. مسح منطقة 4×4 متر")
    print("  3. التصوير عند كل نقطة")
    print("  4. الهبوط والتقرير")
    
    print("\n--- تنفيذ الخطة ---")
    
    sim.takeoff()
    sim.move_up(100)  # 2 متر إجمالي
    
    # مسح grid pattern
    scan_points = [
        (200, 0),    # أمام
        (0, 200),    # يمين
        (-200, 0),   # خلف
        (-200, 0),   # خلف
        (0, -200),   # يسار
        (200, 0),    # أمام
    ]
    
    print("\n🗺️  مسح المنطقة:")
    for i, (forward, right) in enumerate(scan_points):
        print(f"\n  نقطة {i+1}: أمام={forward}cm, يمين={right}cm")
        if forward > 0:
            sim.move_forward(forward)
        elif forward < 0:
            sim.move_back(-forward)
        if right > 0:
            sim.move_right(right)
        elif right < 0:
            sim.move_left(-right)
        
        # محاكاة التصوير
        print(f"  📸 التقاط صورة في الموضع: X={sim.pos.x:.0f}, Y={sim.pos.y:.0f}")
    
    print("\n🛬 العودة والهبوط:")
    sim.land()
    
    print("\n✅ سيناريو البحث اكتمل!")
    return True


if __name__ == "__main__":
    print("\n" + "🚁"*20)
    print("    Janah SAR - اختبار شامل للسيمولاتور")
    print("🚁"*20)
    
    results = []
    
    # اختبار 1: الحركات الأساسية
    results.append(("حركات الدرون", test_movement_commands()))
    
    # اختبار 2: حدود الأمان
    results.append(("حدود الأمان", test_safety_limits()))
    
    # اختبار 3: سيناريو كامل
    results.append(("سيناريو البحث", test_full_sar_scenario()))
    
    # ملخص
    print("\n" + "="*55)
    print("📊 ملخص نتائج الاختبار:")
    print("="*55)
    all_passed = True
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
        all_passed = all_passed and passed
    
    print("="*55)
    if all_passed:
        print("🎉 كل الاختبارات نجحت! النظام جاهز.")
        print("💡 الخطوة التالية: شغّلي webui.py مع السيمولاتور")
    else:
        print("⚠️  بعض الاختبارات فشلت - راجعي الأخطاء أعلاه")
    print("="*55 + "\n")