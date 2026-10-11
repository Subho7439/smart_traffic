import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ============================
# Load YOLO model
# ============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano model for faster inference

model = load_model()

st.title("🚦 AI-Powered Smart Traffic Signal System")

# ============================
# Sidebar settings
# ============================
with st.sidebar:
    st.header("Settings")
    num_roads = st.number_input("Number of roads", min_value=2, max_value=8, value=4, step=1)
    cycle_time = st.slider("Total cycle time (seconds)", min_value=30, max_value=300, value=120, step=5)
    confidence = st.slider("Detection confidence threshold", 0.1, 0.9, 0.35, step=0.05)
    min_green = st.slider("Minimum green time per road (s)", min_value=3, max_value=30, value=5, step=1)

# ============================
# Upload road images
# ============================
uploaded_files = st.file_uploader(
    f"Upload {num_roads} road images (one per road)",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

counts = []

if uploaded_files:
    if len(uploaded_files) != num_roads:
        st.error(f"⚠️ Please upload exactly {num_roads} images.")
    else:
        st.success("✅ All road images uploaded successfully!")

        # ============================
        # Vehicle detection loop
        # ============================
        for i, file in enumerate(uploaded_files, start=1):
            image = Image.open(file)
            st.image(image, caption=f"Road {i}", use_container_width=True)

            # Run YOLO detection
            results = model.predict(image, conf=confidence, verbose=False)

            # Count vehicles (car=2, motorcycle=3, bus=5, truck=7 in COCO dataset)
            car_count = 0
            for r in results:
                for c in r.boxes.cls:
                    if int(c) in [2, 3, 5, 7]:
                        car_count += 1
            counts.append(car_count)

        # ============================
        # Show car counts
        # ============================
        st.subheader("🚗 Vehicle Count per Road")
        for i, c in enumerate(counts, start=1):
            st.write(f"**Road {i}: {c} vehicles detected**")

        # ============================
        # Traffic Light Simulation
        # ============================
        st.subheader("🚦 Traffic Light Sequence Simulation")

        total_cars = sum(counts)

        green_times = []
        for c in counts:
            if c == 0:
                # Road with no cars → always RED
                green_times.append(0)
            else:
                # Proportional allocation with min green
                t = int((c / total_cars) * cycle_time) if total_cars > 0 else 0
                green_times.append(max(min_green, t))

        # Order roads: highest car count first, ignore empty ones
        order = sorted(
            [i for i in range(len(counts)) if counts[i] > 0],
            key=lambda x: -counts[x]
        )

        if not order:
            st.warning("⚠️ All roads are empty — all signals stay RED.")
        else:
            for idx in order:
                st.markdown(
                    f"### Road {idx+1} → 🟢 GREEN for {green_times[idx]}s | Cars: {counts[idx]}"
                )
                st.write(f"Other roads remain 🔴 RED during this time.")

            # Mark vacant roads explicitly
            for i in range(len(counts)):
                if counts[i] == 0:
                    st.markdown(f"### Road {i+1} → 🔴 RED (No cars present)")