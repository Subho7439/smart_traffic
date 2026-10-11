import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
from io import StringIO

# ============================
# Load and analyze traffic dataset
# ============================
def analyze_traffic_data(csv_content):
    # Parse the CSV content
    df = pd.read_csv(StringIO(csv_content))
    
    # Filter only green signals (as they represent movement time)
    green_signals = df[df['Signal_Status'] == 'Green']
    
    # Calculate average green time
    avg_green_time = green_signals['Duration_Seconds'].mean()
    
    # Since we don't have vehicle counts, we need to estimate
    # a reasonable vehicles per second rate based on typical traffic flow
    # Typical values: 1 vehicle every 2-3 seconds during normal flow
    # This would be ~0.3-0.5 vehicles per second
    
    # For this simulation, we'll use a conservative estimate
    # and calculate time per vehicle based on average green duration
    # assuming a moderate traffic volume
    
    # Estimate vehicles per green cycle (assuming moderate traffic)
    # This is a simplification - in a real application, you'd want actual counts
    estimated_vehicles_per_green = avg_green_time / 2.5  # Assuming 2.5s per vehicle
    
    # Calculate time per vehicle
    time_per_vehicle = avg_green_time / estimated_vehicles_per_green
    
    st.sidebar.write(f"📊 Data Analysis Results:")
    st.sidebar.write(f"Average Green Time: {avg_green_time:.1f}s")
    st.sidebar.write(f"Estimated Time per Vehicle: {time_per_vehicle:.1f}s")
    
    return time_per_vehicle

# Sample dataset (in a real app, you'd load from file)
dataset_content = """Timestamp,Intersection_ID,Signal_Status,Duration_Seconds,Day_of_Week
2025-06-06 10:01:05,INT_025,Red,64,Friday
2025-04-30 04:05:41,INT_041,Red,106,Wednesday
2025-05-01 04:15:53,INT_023,Yellow,8,Thursday
2025-05-28 10:08:18,INT_006,Red,97,Wednesday
2025-05-19 22:28:13,INT_007,Green,83,Monday
... [rest of your dataset] ..."""

# Calculate time per vehicle from dataset
time_per_vehicle = analyze_traffic_data(dataset_content)

# ============================
# Load YOLO model
# ============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🚦 Traffic Signal Simulation")

# ============================
# Sidebar controls
# ============================
num_roads = st.sidebar.number_input("Number of roads", min_value=2, max_value=10, value=4, step=1)
cycle_time = st.sidebar.slider("Total cycle time (seconds)", 30, 300, 120, 5)
min_green = st.sidebar.slider("Minimum green per road (s)", 3, 30, 5, 1)

# Use the calculated time per vehicle or allow manual override
calculated_time_per_vehicle = st.sidebar.slider(
    "Time per vehicle (seconds)", 
    min_value=1.0, 
    max_value=10.0, 
    value=float(time_per_vehicle), 
    step=0.5,
    help="Calculated from traffic dataset. Adjust if needed."
)

# ============================
# Initialize session state
# ============================
if "roads" not in st.session_state or st.session_state.get("last_num_roads") != num_roads:
    st.session_state.roads = list(range(1, num_roads + 1))
    st.session_state.last_num_roads = num_roads
    st.session_state.step_ready = False
    st.session_state.detected = {}

roads = st.session_state.roads
st.write(f"### Roads left this round: {roads}")

# ============================
# Upload images for active roads
# ============================
files = st.file_uploader(
    f"Upload {len(roads)} images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=f"step_{len(roads)}"
)

if files and len(files) == len(roads):
    counts = {}
    for rid, f in zip(roads, files):
        img = Image.open(f).convert("RGB")
        results = model.predict(img, conf=0.35, verbose=False)

        # Count vehicles (car=2, motorcycle=3, bus=5, truck=7)
        cnt = 0
        for r in results:
            for c in r.boxes.cls:
                if int(c) in [2, 3, 5, 7]:
                    cnt += 1

        counts[rid] = cnt
        st.image(img, caption=f"Road {rid}: {cnt} vehicles", use_container_width=True)

    st.session_state.detected = counts
    st.session_state.step_ready = True

# ============================
# If detection ready
# ============================
if st.session_state.step_ready:
    counts = st.session_state.detected
    st.write("### Detected vehicle counts:", counts)

    if sum(counts.values()) == 0:
        st.warning("⚠️ All roads empty → No green signal this step.")
        next_road = None
    else:
        # Pick busiest road
        next_road = max(counts, key=counts.get)

        # Green time logic using data-driven calculation
        raw_time = counts[next_road] * calculated_time_per_vehicle

        # Apply bounds
        if len(counts) == 1:
            green_time = min_green
        else:
            green_time = min(max(raw_time, min_green), 60)
        
        st.success(
            f"✅ Road {next_road} gets GREEN for {green_time}s "
            f"(Cars: {counts[next_road]}, {calculated_time_per_vehicle}s per vehicle)"
        )

    if st.button("➡️ Next Step"):
        if next_road is not None and next_road in st.session_state.roads:
            st.session_state.roads.remove(next_road)
        # Reset if last road finished
        if len(st.session_state.roads) == 0:
            st.session_state.roads = list(range(1, num_roads + 1))
        st.session_state.step_ready = False
        st.session_state.detected = {}
        st.rerun()



   



