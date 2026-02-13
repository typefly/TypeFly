"""
Enhanced WebUI with Face Recognition
"""

import gradio as gr
from typefly.janah_cv_integrated import janah_cv_integrated
import cv2

def setup_missing_person(image, name, age, clothing_color):
    """
    إعداد معلومات الشخص المفقود
    """
    if image is None:
        return "❌ Please upload an image"
    
    # حفظ الصورة مؤقتاً
    temp_path = "data/references/temp_upload.jpg"
    cv2.imwrite(temp_path, image)
    
    # إعداد المعلومات
    person_info = {
        'name': name,
        'age': age,
        'clothing_color': clothing_color
    }
    
    # تدريب النظام
    success = janah_cv_integrated.setup_reference(temp_path, person_info)
    
    if success:
        return f"✅ Reference set for: {name}, Age: {age}, Clothing: {clothing_color}"
    else:
        return "❌ Failed to process image"

def create_ui():
    """إنشاء واجهة Gradio محسّنة"""
    
    with gr.Blocks(title="Janah SAR System") as demo:
        gr.Markdown("# 🚁 Janah - Search and Rescue System")
        
        with gr.Tab("Setup Reference"):
            gr.Markdown("## Upload Missing Person Photo")
            
            with gr.Row():
                ref_image = gr.Image(label="Missing Person Photo")
                
                with gr.Column():
                    person_name = gr.Textbox(label="Name (الاسم)", placeholder="أحمد")
                    person_age = gr.Number(label="Age (العمر)", value=25)
                    clothing_color = gr.Dropdown(
                        choices=["red", "blue", "green", "white", "black", "yellow"],
                        label="Clothing Color (لون الملابس)"
                    )
                    
                    setup_btn = gr.Button("Setup Reference", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False)
            
            setup_btn.click(
                setup_missing_person,
                inputs=[ref_image, person_name, person_age, clothing_color],
                outputs=status_text
            )
        
        with gr.Tab("Live Feed"):
            gr.Markdown("## Drone Camera Feed")
            # ... باقي الكود للعرض المباشر
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=50000)