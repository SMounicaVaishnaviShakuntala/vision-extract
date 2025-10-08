import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import requests
import os
import io
import base64
import time
import uuid

# --- Initialize Gallery in Session State ---
if "gallery" not in st.session_state:
    st.session_state["gallery"] = []

# --- Configuration and File Paths ---
IMAGE_SIZE = 256
MODEL_PATH = 'deeplabv3_best_lr_5e-05.pth'
HF_MODEL_URL = "https://huggingface.co/Sighakolli-2Mounica/Vision_Extract/resolve/main/deeplabv3_best_lr_5e-05.pth"
DEVICE = torch.device("cpu")

# --- Custom Style ---
def set_custom_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Full-page animated Ocean Breeze gradient background */
        html, body, .stApp {
            height: 100%;
            margin: 0;
            background: linear-gradient(270deg, #cce7ff, #d0f4de, #ffd6d6);
            background-size: 600% 600%;
            animation: GradientShift 30s ease infinite;
        }

        /* Smooth slow motion animation */
        @keyframes GradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        h1 { text-align: center; font-weight: 700; color: #2c3e50; margin-bottom: 0.3em; }
        .app-description { text-align: center; font-size: 20px; line-height: 1.6; color: #34495e; margin-bottom: 1em; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        iframe { margin-bottom: 0px !important; }
        .image-frame { border-radius: 16px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .stButton button {
            background: #0d47a1; color: white !important; border-radius: 12px;
            padding: 12px 25px; font-size: 16px; font-weight: bold;
            border: none; transition: all 0.3s ease-in-out;
        }
        .stButton button:hover { transform: scale(1.05); background: #1565c0; }
        section[data-testid="stFileUploader"] {
            border: 2px dashed #3498db; border-radius: 12px; background-color: rgba(236,247,255,0.7);
        }
        section[data-testid="stFileUploader"]:hover {
            border-color: #2575fc; background-color: rgba(200,230,255,0.9);
        }
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1, .app-description { text-align: center !important; }
        h2, h3, p { text-align: left !important; }
        .stImage { width: 60% !important; }
        .juxtapose-credit { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Helper Functions ---
def get_image_download_link_button(img, filename, key: str = None, label: str = "📥 Download Masked Image"):
    if key is None:
        key = f"download_{uuid.uuid4().hex}"

    if isinstance(img, Image.Image):
        pil_img = img
    else:
        arr = np.array(img)
        if arr.dtype != np.uint8:
            try:
                arr = (arr * 255).astype(np.uint8)
            except Exception:
                arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr, mode="RGBA")
        else:
            pil_img = Image.fromarray(arr)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    buffered.seek(0)
    return st.download_button(
        label=label,
        data=buffered.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png",
        key=key
    )

resize_transform = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

@st.cache_resource
def download_model_file():
    try:
        response = requests.get(HF_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False

@st.cache_resource
def load_segmentation_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    except Exception:
        pass
    model.to(DEVICE)
    model.eval()
    return model

def run_segmentation_pipeline(image_pil, model, threshold=0.5):
    image_tensor = T.ToTensor()(resize_transform(image_pil))
    image_tensor = normalize(image_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)['out']
    probabilities = torch.sigmoid(output)
    predicted_mask = (probabilities > threshold).float().squeeze(0).cpu().numpy().squeeze()

    un_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
    original_image_np = un_normalize(image_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
    original_image_np = np.clip(original_image_np, 0, 1)

    masked_result = np.zeros_like(original_image_np)
    mask_bool = predicted_mask.astype(bool)
    masked_result[mask_bool] = original_image_np[mask_bool]

    orig_uint8 = (original_image_np * 255).astype(np.uint8)
    alpha = (predicted_mask * 255).astype(np.uint8)
    rgba = np.dstack([orig_uint8, alpha])
    transparent_pil = Image.fromarray(rgba, mode="RGBA")

    return original_image_np, masked_result, transparent_pil

# --- Custom Comparison Demo ---
def custom_image_comparison(img1: Image.Image, img2: Image.Image):
    def pil_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    img1_b64 = pil_to_base64(img1)
    img2_b64 = pil_to_base64(img2)

    html_code = f"""
    <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
    <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
    <div id="juxtapose-wrapper" style="max-width:350px; margin:auto;"></div>
    <script>
      new juxtapose.JXSlider('#juxtapose-wrapper',
        [
          {{ src: "data:image/png;base64,{img1_b64}", label: "Original" }},
          {{ src: "data:image/png;base64,{img2_b64}", label: "Masked" }}
        ],
        {{ animate: true, showLabels: true, showCredits: false, startingPosition: "50%" }}
      );
    </script>
    """
    st.components.v1.html(html_code, height=330)

# --- Results Display ---
def show_uploaded_results(original, masked, transparent, filename):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4 style='text-align:center;'>Original Image</h4>", unsafe_allow_html=True)
        st.image(original, use_container_width=True)

    with col2:
        st.markdown("<h4 style='text-align:center;'>Masked Image</h4>", unsafe_allow_html=True)
        st.image(masked, use_container_width=True)
        c_a, c_b, c_c = st.columns([1, 2, 1])
        with c_b:
            get_image_download_link_button((masked * 255).astype(np.uint8), f"{filename}_masked_download",
                                           key=f"dl_masked_{uuid.uuid4().hex}", label="📥 Download Masked Image")

    with col3:
        st.markdown("<h4 style='text-align:center;'>Transparent BG</h4>", unsafe_allow_html=True)
        st.image(transparent, use_container_width=True)
        c_d, c_e, c_f = st.columns([1, 2, 1])
        with c_e:
            get_image_download_link_button(transparent, f"{filename}_transparent",
                                           key=f"dl_trans_{uuid.uuid4().hex}", label="📥 Download Transparent BG Image")

# --- UI ---
set_custom_style()

if not os.path.exists(MODEL_PATH):
    if not download_model_file():
        st.stop()

segmentation_model = load_segmentation_model()

st.title("VisionExtract — AI-Powered Subject Isolation")
import streamlit as st
import time

# --- Description with Fade-In Animation ---
description_text = """
<span style="color:#2575fc; font-weight:600;">Vision Extract</span> is an 
<span style="color:#27ae60; font-weight:600;">AI-powered tool</span> that instantly 
separates objects from their backgrounds.  
Simply upload an image, and the system generates a clean cut-out mask you can use for 
<b>editing</b>, <b>presentations</b>, or <b>creative projects</b>.  
It’s <span style="color:#e67e22; font-weight:600;">fast</span>, 
<span style="color:#e67e22; font-weight:600;">accurate</span>, and built to make 
<b>image extraction effortless</b>.
"""

placeholder = st.empty()

# Split into words for fade-in effect
words = description_text.split(" ")
display_text = ""

for word in words:
    display_text += word + " "
    placeholder.markdown(f"<p class='app-description'>{display_text}</p>", unsafe_allow_html=True)
    time.sleep(0.07)  # Adjust speed here

# --- Demo ---
try:
    orig_demo = Image.open("sample_original_image.png")
    mask_demo = Image.open("sample_masked_resized.png")
    st.markdown("### See it in Action 👇")
    custom_image_comparison(orig_demo, mask_demo)
except Exception:
    st.info("⚠️ Demo images not found. Please add sample images.")

# --- Upload Section ---
from scipy.ndimage import binary_dilation, binary_erosion

def refine_mask(original_pil, model, threshold=0.5, expansion=0):
    """Reapply segmentation mask with user-tuned threshold and expansion."""
    image_tensor = T.ToTensor()(resize_transform(original_pil))
    image_tensor = normalize(image_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)['out']
    probabilities = torch.sigmoid(output)
    predicted_mask = (probabilities > threshold).float().squeeze(0).cpu().numpy().squeeze()

    # Expand or contract the mask using morphological ops
    if expansion > 0:
        predicted_mask = binary_dilation(predicted_mask, iterations=expansion)
    elif expansion < 0:
        predicted_mask = binary_erosion(predicted_mask, iterations=abs(expansion))

    un_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
    original_image_np = un_normalize(image_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
    original_image_np = np.clip(original_image_np, 0, 1)

    masked_result = np.zeros_like(original_image_np)
    mask_bool = predicted_mask.astype(bool)
    masked_result[mask_bool] = original_image_np[mask_bool]

    orig_uint8 = (original_image_np * 255).astype(np.uint8)
    alpha = (predicted_mask * 255).astype(np.uint8)
    rgba = np.dstack([orig_uint8, alpha])
    transparent_pil = Image.fromarray(rgba, mode="RGBA")

    return original_image_np, masked_result, transparent_pil

if segmentation_model is not None:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "✨ Upload an image to see instant subject isolation",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    st.markdown("---")

    if uploaded_file is not None:
        progress = st.progress(0)
        status_text = st.empty()

        status_text.text("📤 Uploading image...")
        time.sleep(0.3)
        progress.progress(20)

        original_image_pil = Image.open(uploaded_file).convert("RGB")

        status_text.text("🤖 Running AI model...")
        time.sleep(0.3)
        progress.progress(50)

        original_image_np, masked_image_np, transparent_pil = run_segmentation_pipeline(
            original_image_pil, segmentation_model
        )

        status_text.text("🎨 Generating mask...")
        time.sleep(0.3)
        progress.progress(80)

        status_text.text("✅ Finalizing results...")
        time.sleep(0.2)
        progress.progress(100)

        progress.empty()
        status_text.empty()

        st.subheader("Segmentation Result")

        # --- Mask Controls ---
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            threshold = st.slider(
                "🧠 Mask Threshold",
                0.1, 0.9, 0.5, 0.05,
                help="Higher values make the mask stricter, lower values include more area."
            )
        with col_t2:
            expansion = st.slider(
                "📏 Mask Expansion",
                -10, 10, 0, 1,
                help="Expand (positive) or contract (negative) the masked area in pixels."
            )

        # --- Apply refinement ---
        original_image_np, masked_image_np, transparent_pil = refine_mask(
            original_image_pil, segmentation_model, threshold, expansion
        )

        # --- Display updated result ---
        show_uploaded_results(original_image_np, masked_image_np, transparent_pil, uploaded_file.name)

        # --- Feedback Section ---
        import streamlit as st
        from streamlit_js_eval import streamlit_js_eval

        st.markdown("<h4 style='text-align:center; margin-top:1.5em;'>📝 Rate this Result</h4>", unsafe_allow_html=True)

        if "feedback" not in st.session_state:
            st.session_state["feedback"] = {}

        # Unique key per uploaded image using filename only (keeps feedback per image)
        image_key = uploaded_file.name.replace(" ", "_")

        slider_id = f"emojiSlider_{image_key}"  # unique slider id for HTML

        if image_key not in st.session_state["feedback"]:
            # Emoji slider HTML + JS
            st.components.v1.html(f"""
            <div style="display:flex; justify-content:center; margin-top:10px;">
                <input type="range" min="1" max="5" value="3" step="1" id="{slider_id}" style="width:300px;">
            </div>
            <div style="display:flex; justify-content:space-between; width:300px; margin:0 auto; font-size:24px; margin-top:5px;">
                <span>😡</span><span>😕</span><span>😐</span><span>🙂</span><span>😍</span>
            </div>
            <script>
                const slider = document.getElementById('{slider_id}');
                slider.addEventListener('mouseup', () => {{
                    const value = slider.value;
                    // Send feedback to Streamlit
                    const streamlitEvent = new CustomEvent("feedback_selected", {{detail: value}});
                    window.dispatchEvent(streamlitEvent);

                    // Lock the slider
                    slider.disabled = true;

                    // Toast message
                    const toast = document.createElement('div');
                    toast.innerHTML = '✅ Feedback recorded!';
                    toast.style.position = 'fixed';
                    toast.style.top = '20px';
                    toast.style.right = '20px';
                    toast.style.background = '#27ae60';
                    toast.style.color = 'white';
                    toast.style.padding = '10px 20px';
                    toast.style.borderRadius = '10px';
                    toast.style.zIndex = 9999;
                    document.body.appendChild(toast);
                    setTimeout(()=>{{ toast.remove(); }}, 3000);
                }});
            </script>
            """, height=120, scrolling=False)

            # Capture slider value in Python
            value = streamlit_js_eval(
                js_expressions=f"window.addEventListener('feedback_selected', e => e.detail);",
                key=f"feedback_{image_key}"
            )

            if value is not None:
                st.session_state["feedback"][image_key] = int(value)

        else:
            saved_rating = st.session_state["feedback"][image_key]
            st.markdown(f"<p style='text-align:center; font-size:20px;'>👍 You rated this result: {['😡','😕','😐','🙂','😍'][saved_rating-1]}</p>", unsafe_allow_html=True)

        # --- Add to Gallery ---
        uploaded_name = uploaded_file.name
        if not any(entry["filename"] == uploaded_name for entry in st.session_state["gallery"]):
            st.session_state["gallery"].insert(0, {
                "filename": uploaded_name,
                "original": (original_image_np * 255).astype(np.uint8),
                "masked": (masked_image_np * 255).astype(np.uint8),
                "transparent": transparent_pil
            })

#----Gallery Section----
import zipfile
import io
import uuid
from PIL import Image
import base64
import numpy as np
import streamlit as st

# --- Mini Gallery Toggle ---
if "show_gallery" not in st.session_state:
    st.session_state["show_gallery"] = False

if st.session_state.get("gallery") and len(st.session_state["gallery"]) > 0:
    col_show, col_spacer, col_close = st.columns([0.2, 0.6, 0.2])

    with col_show:
        if not st.session_state["show_gallery"]:
            if st.button("🖼️ Show Gallery", key="show_gallery_btn"):
                st.session_state["show_gallery"] = True

    with col_close:
        if st.session_state["show_gallery"]:
            if st.button("❌ Close Gallery", key="close_gallery_btn"):
                st.session_state["show_gallery"] = False
                st.rerun()

# --- Display gallery ---
if st.session_state["show_gallery"]:
    gallery = st.session_state.get("gallery", [])
    if gallery:
        st.markdown("""
        <h3 style="text-align:center; color:#2c3e50; margin-bottom:1em; font-weight:600;">
        Recent Uploads
        </h3>
        """, unsafe_allow_html=True)

        # --- Function to display scrollable row with modal + label ---
        def display_scrollable_row(images, label_prefix):
            html_content = """
            <div style="display:flex; overflow-x:auto; gap:10px; padding:5px;">
            """
            for idx, img_entry in enumerate(images):
                pil_img = Image.fromarray(img_entry) if isinstance(img_entry, np.ndarray) else img_entry
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format="PNG")
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
                label = f"{label_prefix} {idx+1}"

                # Image with click-to-enlarge
                html_content += f"""
                <div style="flex:0 0 auto; text-align:center; position:relative;">
                    <img src="data:image/png;base64,{img_b64}" width="250" 
                        style="border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.15);
                            transition: transform 0.3s; cursor:pointer;"
                        onclick="showModal(this.src, '{label}');"
                        onmouseover="this.style.transform='scale(1.05)';"
                        onmouseout="this.style.transform='scale(1)';"
                    >
                    <div style="text-align:center; font-weight:600; margin-top:5px;">{label}</div>
                </div>
                """
            html_content += "</div>"

            # Modal with download button
            html_content += """
            <div id="modal" style="display:none; position:fixed; z-index:1000; 
                left:0; top:0; width:100%; height:100%; background-color:rgba(0,0,0,0.8); 
                justify-content:center; align-items:center; flex-direction:column;">

                <!-- Close button -->
                <span id="closeModal" style="position:absolute; top:10px; right:55px; 
                    color:white; font-size:40px; font-weight:bold; cursor:pointer;">&times;</span>

                <!-- Download button -->
                <a id="downloadBtn" href="#" download style="position:absolute; top:15px; right:15px; color:white; font-size:30px; text-decoration:none;">
                    <i class="fa fa-download"></i>
                </a>

                <div id="modalLabel" style="position:absolute; top:10px; left:50%; 
                    transform:translateX(-50%); color:white; font-size:20px; font-weight:600;"></div>
                    
                <img id="modalImg" style="max-width:90%; max-height:90%; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
            </div>
            <script>
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modalImg');
            const closeModal = document.getElementById('closeModal');
            const modalLabel = document.getElementById('modalLabel');
            const downloadBtn = document.getElementById('downloadBtn');

            function showModal(src, label){
                modal.style.display='flex';
                modalImg.src = src;
                modalLabel.textContent = label;

                // Set download link dynamically
                downloadBtn.href = src;
                downloadBtn.download = label + ".png";
            }

            closeModal.onclick = function(){ modal.style.display='none'; }
            modal.onclick = function(event){ 
                if(event.target.id=='modal'){ modal.style.display='none'; } 
            }
            </script>
            """
            st.components.v1.html(html_content, height=350)

        # --- Display all rows with labels ---
        display_scrollable_row([entry["original"] for entry in gallery], "Original")
        display_scrollable_row([entry["masked"] for entry in gallery], "Masked")
        display_scrollable_row([entry["transparent"] for entry in gallery], "Transparent")

        # --- ZIP creation functions ---
        def create_original_zip():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                for i, entry in enumerate(gallery):
                    pil_img = Image.fromarray(entry["original"])
                    img_bytes = io.BytesIO()
                    pil_img.save(img_bytes, format="PNG")
                    zipf.writestr(f"original_{i+1}.png", img_bytes.getvalue())
            zip_buffer.seek(0)
            return zip_buffer.getvalue()

        def create_masked_zip():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                for i, entry in enumerate(gallery):
                    pil_img = Image.fromarray(entry["masked"])
                    img_bytes = io.BytesIO()
                    pil_img.save(img_bytes, format="PNG")
                    zipf.writestr(f"masked_{i+1}.png", img_bytes.getvalue())
            zip_buffer.seek(0)
            return zip_buffer.getvalue()

        def create_transparent_zip():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                for i, entry in enumerate(gallery):
                    pil_img = entry["transparent"]
                    img_bytes = io.BytesIO()
                    pil_img.save(img_bytes, format="PNG")
                    zipf.writestr(f"transparent_{i+1}.png", img_bytes.getvalue())
            zip_buffer.seek(0)
            return zip_buffer.getvalue()

        # --- Footer Download All ZIP buttons ---
        col_original_btn, col_masked_btn, col_trans_btn = st.columns(3)

        with col_original_btn:
            st.download_button(
                label="📥 Download All Original Images as ZIP",
                data=create_original_zip(),
                file_name="original_images.zip",
                mime="application/zip",
                key="original_zip_dl",
            )

        with col_masked_btn:
            st.download_button(
                label="📥 Download All Masked Images as ZIP",
                data=create_masked_zip(),
                file_name="masked_images.zip",
                mime="application/zip",
                key="masked_zip_dl",
            )

        with col_trans_btn:
            st.download_button(
                label="📥 Download All Transparent BG Images as ZIP",
                data=create_transparent_zip(),
                file_name="transparent_images.zip",
                mime="application/zip",
                key="transparent_zip_dl",
            )



