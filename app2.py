import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from io import BytesIO

# --- THEME DETECTION ---
theme_base = st.get_option("theme.base") or "dark"
is_dark = theme_base == "dark"

# --- DYNAMIC CSS FOR LIGHT/DARK ---
main_bg = "linear-gradient(135deg, #232526 0%, #414345 100%)" if is_dark else "linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%)"
main_fg = "#f3f3f3" if is_dark else "#232526"
sub_fg = "#b0b0b0" if is_dark else "#555"
box_bg = "#232526" if is_dark else "#fff"
box_fg = "#4dd0e1" if is_dark else "#1976d2"
border_col = "#444" if is_dark else "#e0e7ef"

st.markdown(
    f"""
    <style>
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
        background: {main_bg};
        color: {main_fg};
    }}
    .main-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-align: center;
        margin-bottom: 0.2em;
        color: {main_fg};
        text-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }}
    .subtitle {{
        font-size: 1.1rem;
        text-align: center;
        color: {sub_fg};
        margin-bottom: 1.5em;
    }}
    .kv-metrics-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 18px 32px;
        margin: 18px 0 10px 0;
        font-size: 1.08rem;
        font-weight: 500;
        justify-content: flex-start;
    }}
    .kv-metric {{
        background: {box_bg};
        border-radius: 8px;
        padding: 8px 18px;
        color: {box_fg};
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        border: 1.2px solid {border_col};
    }}
    .kv-metric-label {{
        color: {sub_fg};
        font-size: 0.98rem;
        font-weight: 500;
        margin-right: 8px;
    }}
    .img-card {{
        background: {box_bg};
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
        padding: 10px;
        margin-bottom: 12px;
    }}
    .img-card img {{
        border-radius: 10px;
        width: 100%;
        height: auto;
        display: block;
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    }}
    .stButton > button {{
        background: linear-gradient(90deg, #4dd0e1, #1976d2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 10px 24px;
        border: none;
        transition: background 0.3s;
    }}
    .stButton > button:hover {{
        background: linear-gradient(90deg, #1976d2, #4dd0e1);
        cursor: pointer;
    }}
    section[data-testid="stSidebar"] {{
        background: {box_bg};
        color: {main_fg};
        border-radius: 0 18px 18px 0;
        padding: 24px 18px 18px 18px;
        box-shadow: 4px 0 24px rgba(0,0,0,0.10);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- TITLE & SUBTITLE ---
st.markdown('<div class="main-title">FaceIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time & Uploaded Image Age & Gender Detection</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙ Settings")
threshold = st.sidebar.slider("Gender confidence threshold", 0.5, 0.99, 0.7, 0.01)
input_source = st.sidebar.radio("Choose input source:", ("Upload an Image", "Use Webcam"))
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use a clear, frontal face for best results.")
st.sidebar.markdown("### Quick Actions")
st.sidebar.write("- Adjust gender confidence threshold above.")
st.sidebar.write("- Download annotated image and CSV after analysis.")

# --- FACE ANALYSIS FUNCTIONS ---
def safe_float(val):
    try:
        return float(val)
    except Exception:
        return 0.0

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def filter_overlapping_detections(detections, iou_threshold=0.3):
    detections = sorted(detections, key=lambda d: d.get('face_confidence', 0), reverse=True)
    filtered = []
    for det in detections:
        box = (det['region']['x'], det['region']['y'], det['region']['w'], det['region']['h'])
        if all(iou(box, (f['region']['x'], f['region']['y'], f['region']['w'], f['region']['h'])) < iou_threshold for f in filtered):
            filtered.append(det)
    return filtered

def analyze_faces(image, threshold=0.7):
    image_np = np.array(image)
    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    detections = DeepFace.analyze(
        image_np, 
        actions=['age', 'gender'], 
        enforce_detection=False
    )

    if isinstance(detections, dict):
        detections = [detections]

    detections = filter_overlapping_detections(detections)

    MIN_FACE_SIZE = 40
    MIN_FACE_CONFIDENCE = 0.7
    detections = [
        det for det in detections
        if det['region']['w'] > MIN_FACE_SIZE
        and det['region']['h'] > MIN_FACE_SIZE
        and safe_float(det.get('face_confidence', 0)) > MIN_FACE_CONFIDENCE
    ]

    # Draw annotations
    annotated_image = Image.fromarray(image_np).convert("RGBA")
    draw = ImageDraw.Draw(annotated_image)
    for det in detections:
        x, y, w, h = det['region']['x'], det['region']['y'], det['region']['w'], det['region']['h']
        draw.rectangle([(x, y), (x + w, y + h)], outline="#4dd0e1", width=5)

    # Prepare data
    table_data = []
    male_count = 0
    female_count = 0
    ages = []

    for idx, det in enumerate(detections, 1):
        age = int(round(det['age']))
        ages.append(age)
        face_conf = safe_float(det.get('face_confidence', 0))
        gender = det['gender']

        if isinstance(gender, dict):
            male_score = safe_float(gender.get('Man', gender.get('Male', 0))) / 100.0
            female_score = safe_float(gender.get('Woman', gender.get('Female', 0))) / 100.0
            max_score = max(male_score, female_score)

            if max_score < threshold:
                gender_label = "Uncertain"
                confidence = f"{max_score*100:.1f}%"
            else:
                if male_score > female_score:
                    gender_label = "Male"
                    male_count += 1
                else:
                    gender_label = "Female"
                    female_count += 1
                confidence = f"{max_score*100:.1f}%"
        else:
            gender_label = "Male" if gender.lower() in ["man", "male"] else "Female"
            if gender_label == "Male":
                male_count += 1
            else:
                female_count += 1
            confidence = ""

        table_data.append({
            "Face #": idx,
            "Age": age,
            "Gender": gender_label,
            "Gender Confidence": confidence,
            "Face Confidence": f"{face_conf*100:.1f}%"
        })

    total_faces = len(table_data)
    avg_age = round(np.mean(ages), 1) if ages else 0
    min_age = min(ages) if ages else 0
    max_age = max(ages) if ages else 0
    median_age = int(np.median(ages)) if ages else 0
    male_percent = round((male_count / total_faces) * 100, 2) if total_faces else 0
    female_percent = round((female_count / total_faces) * 100, 2) if total_faces else 0
    unknown_count = total_faces - male_count - female_count
    unknown_percent = round((unknown_count / total_faces) * 100, 2) if total_faces else 0

    stats = {
        "total_faces": total_faces,
        "male_percent": male_percent,
        "female_percent": female_percent,
        "unknown_percent": unknown_percent,
        "avg_age": avg_age,
        "min_age": min_age,
        "max_age": max_age,
        "median_age": median_age,
        "male_count": male_count,
        "female_count": female_count,
        "unknown_count": unknown_count
    }

    return annotated_image, table_data, detections, stats

# --- MAIN PROCESSING & UI ---
def process_image_and_render(pil_image: Image.Image):
    annotated_image, table_data, detections, stats = analyze_faces(pil_image, threshold)
    table_df = pd.DataFrame(table_data)
    faces_detected = stats["total_faces"]

    # --- IMAGE DISPLAY ---
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.markdown("<strong>Input Image</strong>", unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with img_col2:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.markdown("<strong>Annotated Result</strong>", unsafe_allow_html=True)
        st.image(annotated_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TABLE ---
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Face Analysis Results")
    st.dataframe(table_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- METRICS (MULTIPLE FACES) ---
    if faces_detected > 1:
        st.markdown('<div class="kv-metrics-row">', unsafe_allow_html=True)
        for label, value in [
            ("Average Age", stats["avg_age"]),
            ("Median Age", stats["median_age"]),
            ("Youngest", stats["min_age"]),
            ("Oldest", stats["max_age"]),
            ("Male %", f"{stats['male_percent']}%"),
            ("Female %", f"{stats['female_percent']}%"),
            ("Unknown %", f"{stats['unknown_percent']}%"),
        ]:
            st.markdown(
                f"""<div class="kv-metric"><span class="kv-metric-label">{label}:</span> {value}</div>""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    elif faces_detected == 1 and not table_df.empty:
        row = table_df.iloc[0]
        st.markdown('<div class="kv-metrics-row">', unsafe_allow_html=True)
        for label, value in [
            ("Age", row["Age"]),
            ("Gender", row["Gender"]),
            ("Gender Confidence", row["Gender Confidence"]),
            ("Face Confidence", row["Face Confidence"]),
        ]:
            st.markdown(
                f"""<div class="kv-metric"><span class="kv-metric-label">{label}:</span> {value}</div>""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- SUMMARY & DOWNLOADS ---
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.markdown("### Summary & Downloads")
    st.markdown(f"- **Faces:** {faces_detected}")
    if faces_detected > 1:
        st.markdown(f"- **Male:** {stats['male_count']} ({stats['male_percent']}%)")
        st.markdown(f"- **Female:** {stats['female_count']} ({stats['female_percent']}%)")
        if stats['unknown_count']:
            st.markdown(f"- **Unknown Gender:** {stats['unknown_count']} ({stats['unknown_percent']}%)")
    buf_img = BytesIO()
    annotated_image.convert("RGB").save(buf_img, format="PNG")
    st.download_button("📥 Download Annotated Image", buf_img.getvalue(), file_name="annotated.png", mime="image/png")
    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Report", data=csv_bytes, file_name="results.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 Raw detection output (debug)"):
        st.json(detections)

# --- INPUT UI ---
st.markdown('<div class="img-card">', unsafe_allow_html=True)
st.markdown("## Input")
st.write("Choose an image file or take a photo with your webcam. Results will animate into the dashboard.")
if input_source == "Upload an Image":
    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image_obj = Image.open(uploaded_file).convert("RGB")
            process_image_and_render(image_obj)
        except Exception:
            st.error("Could not open image. Please try another file.")
else:
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image_obj = Image.open(camera_image).convert("RGB")
        process_image_and_render(image_obj)
st.markdown("</div>", unsafe_allow_html=True)