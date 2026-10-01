import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr

# -----------------------------------------------------------------------------
# Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Detecção de Marcas - Simulação Marketing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Inicialização de Modelos com Cache de Recursos
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    # Modelo YOLOv8 Nano leve para execução em CPU
    yolo_model = YOLO("yolov8n.pt")
    # Leitor OCR leve (Português/Inglês)
    ocr_reader = easyocr.Reader(['pt', 'en'], gpu=False)
    return yolo_model, ocr_reader

yolo_model, ocr_reader = load_models()

# -----------------------------------------------------------------------------
# Funções de Processamento de Imagem
# -----------------------------------------------------------------------------
def preprocess_for_ocr(crop_img):
    """Aplica filtros básicos de visão clássica para otimizar a leitura OCR."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    # Binarização adaptativa para aumentar contraste do texto
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def process_pipeline(image_bytes, conf_threshold=0.25):
    """Executa detecção, recorte, OCR e anotação mantendo baixo footprint de memória."""
    pil_img = Image.open(image_bytes).convert("RGB")
    
    # Redimensionamento defensivo para economia de memória
    pil_img.thumbnail((800, 800))
    img_np = np.array(pil_img)
    
    # Detecção com YOLO
    results = yolo_model.predict(img_np, conf=conf_threshold, verbose=False)[0]
    
    annotated_img = img_np.copy()
    detected_brands = []

    # Se a YOLO não detectar regiões específicas, processa a imagem inteira via OCR
    if len(results.boxes) == 0:
        ocr_res = ocr_reader.readtext(img_np)
        for bbox, text, prob in ocr_res:
            if prob > 0.3 and len(text.strip()) > 1:
                detected_brands.append(text.strip().upper())
        return annotated_img, list(set(detected_brands))

    # Processamento por região de interesse (Bounding Boxes)
    h_img, w_img, _ = img_np.shape
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Slices de segurança
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Otimização OCR no crop
        prep_crop = preprocess_for_ocr(crop)
        ocr_res = ocr_reader.readtext(prep_crop)
        
        brand_text = ""
        if ocr_res:
            # Seleciona o texto de maior confiança
            brand_text = sorted(ocr_res, key=lambda x: x[2], reverse=True)[0][1].strip().upper()
        
        if not brand_text:
            # Fallback para o nome da classe genérica do YOLO se não houver texto legível
            cls_id = int(box.cls[0])
            brand_text = f"LOGO ({yolo_model.names[cls_id].upper()})"

        detected_brands.append(brand_text)

        # Desenho da Bounding Box e Rótulo
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img,
            brand_text,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated_img, list(set(detected_brands))

# -----------------------------------------------------------------------------
# Interface de Usuário (Streamlit)
# -----------------------------------------------------------------------------
st.title("Simulador de Identificação de Marcas")
st.caption("Focado em eficiência computacional e execução local via CPU.")

uploaded_file = st.file_uploader("Envie uma imagem da campanha ou produto:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])
    
    with st.spinner("Analisando marcas e textos..."):
        processed_image, brands = process_pipeline(uploaded_file)
        
    with col1:
        st.subheader("Imagem Analisada")
        st.image(processed_image, use_container_width=True)
        
    with col2:
        st.subheader("Marcas Identificadas")
        if brands:
            for brand in brands:
                st.success(f"**{brand}**")
        else:
            st.warning("Nenhuma marca ou texto legível foi identificado.")