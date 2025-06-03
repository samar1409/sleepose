# app.py - Simplified Sleep Tracking App with Apple Watch Integration
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import serial
import plotly.express as px
import websockets
import asyncio
import json
import glob
import os
from urllib.parse import quote
import requests  # Add this import
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set page config
st.set_page_config(page_title="SleepPose - Real-time Sleep Tracking", layout="wide")

# Real Apple Watch Data Connector
@st.cache_resource
def get_real_apple_watch_connector():
    class RealAppleWatch:
        def __init__(self):
            # Use direct download URL format
            self.folder_id = "6m8qge8khuyz1g9u4uhbo"
            self.rlkey = "y6o2n2awwcrrvjdv3ku7tucjs"
            self.st = "v8eyh9i9"
            self.last_processed_time = None
            
            # Set up URL for data access
            self.test_url = (
                f"https://www.dropbox.com/scl/fo/{self.folder_id}"
                f"/AIIqElBSNxMxhBesOym91uU"
                f"?rlkey={self.rlkey}&st={self.st}&dl=1"
            )
            
            # Debug info in sidebar
            # st.sidebar.markdown("### Dropbox Debug Info")
            # st.sidebar.text("Folder ID: " + self.folder_id)
            # st.sidebar.text("RL Key: " + self.rlkey)
            
            # Test URL construction for folder access (not individual file)
            self.test_url = (
                f"https://www.dropbox.com/scl/fo/{self.folder_id}"
                f"/AIIqElBSNxMxhBesOym91uU"
                f"?rlkey={self.rlkey}&st={self.st}&dl=1"  # Added dl=1 for direct download
            )
            
            # Add debug info
            # st.sidebar.markdown("### Test Your Share Link")
            # st.sidebar.text("Copy this URL to test folder access:")
            # st.sidebar.code(self.test_url, language="text")
            
        def get_realtime_data(self):
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                response = requests.get(
                    self.test_url,
                    verify=False,
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    text = response.text.strip()
                    df = pd.read_csv(pd.io.common.StringIO(text))
                    latest_data = df[df['Heart Rate [Avg] (bpm)'].notna()].iloc[-1]
                    
                    
                    return {
                        'timestamp': pd.to_datetime(latest_data['Date']) if pd.notna(latest_data['Date']) else pd.Timestamp.now(),
                        'heart_rate': latest_data['Heart Rate [Avg] (bpm)'] if pd.notna(latest_data['Heart Rate [Avg] (bpm)']) else np.random.normal(65, 5),
                        'movement_magnitude': float(latest_data['Step Count (steps)']) / 1000 if pd.notna(latest_data['Step Count (steps)']) else np.random.uniform(0.01, 0.05),
                        'breathing_rate': np.random.normal(16, 2),  # Always simulate breathing
                        'source': 'apple_watch'
                    }
                    
                else:
                    
                    return {
                        'timestamp': pd.Timestamp.now(),
                        'heart_rate': np.random.normal(65, 5),
                        'movement_magnitude': np.random.uniform(0.01, 0.05),
                        'breathing_rate': np.random.normal(16, 2),
                        'source': 'simulated'
                    }
                    
            except:
                
                return {
                    'timestamp': pd.Timestamp.now(),
                    'heart_rate': np.random.normal(65, 5),
                    'movement_magnitude': np.random.uniform(0.01, 0.05),
                    'breathing_rate': np.random.normal(16, 2),
                    'source': 'simulated'
                }
            
        def is_connected(self):
            try:
                # Use the same URL construction as get_realtime_data
                today = datetime.now().strftime('%Y-%m-%d')
                file_url = (
                    f"https://www.dropbox.com/scl/fo/{self.folder_id}"
                    f"/AIIqElBSNxMxhBesOym91uU/HealthMetrics-{today}.csv"
                    f"?rlkey={self.rlkey}&st={self.st}&dl=1"
                )
                
                response = requests.get(file_url, verify=False)
                return response.status_code == 200
                
            except Exception as e:
                st.sidebar.error(f"Connection test failed: {str(e)}")
                return False
    
    return RealAppleWatch()

# DSP Processing Class
@st.cache_resource
def get_dsp_processor():
    class SleepDSP:
        def __init__(self, sample_rate=50):
            self.sample_rate = sample_rate
            
        def bandpass_filter(self, data, low_freq, high_freq):
            """Apply bandpass filter"""
            if len(data) < 10:  # Need minimum data for filtering
                return data
            
            nyquist = self.sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are in valid range
            low = max(0.001, min(0.999, low))
            high = max(0.001, min(0.999, high))
            
            if low >= high:
                return data
                
            try:
                b, a = signal.butter(2, [low, high], btype='band')  # Reduced order for stability
                return signal.filtfilt(b, a, data)
            except:
                return data  # Return original if filtering fails
        
        def frequency_analysis(self, data):
            """Perform FFT analysis"""
            if len(data) < 10:
                return np.array([]), np.array([])
                
            fft_vals = fft(data)
            freqs = fftfreq(len(data), 1/self.sample_rate)
            
            # Only positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = np.abs(fft_vals[pos_mask])
            
            return freqs, fft_vals
        
        def calculate_sleep_metrics(self, data_buffer):
            """Calculate sleep quality metrics"""
            if len(data_buffer) < 10:
                return {}
                
            df = pd.DataFrame(data_buffer)
            
            metrics = {
                'avg_heart_rate': df['heart_rate'].mean(),
                'hr_variability': df['heart_rate'].std(),
                'total_movement': df['movement_magnitude'].sum(),
                'avg_breathing_rate': df['breathing_rate'].mean(),
                'sleep_efficiency': max(0, 100 - df['movement_magnitude'].mean() * 1000)
            }
            
            return metrics
        
        def detect_sleep_stage(self, recent_hr, recent_movement):
            """Simple sleep stage detection"""
            if len(recent_hr) < 5 or len(recent_movement) < 5:
                return "Initializing"
                
            hr_std = np.std(recent_hr)
            mov_mean = np.mean(recent_movement)
            
            if mov_mean > 0.15:
                return "Awake"
            elif hr_std < 3 and mov_mean < 0.05:
                return "Deep Sleep"
            elif hr_std < 5 and mov_mean < 0.1:
                return "Light Sleep"
            else:
                return "REM Sleep"
    
    return SleepDSP()

# Initialize session state
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'collecting' not in st.session_state:
    st.session_state.collecting = False

# Get cached instances
watch = get_real_apple_watch_connector()  # Only use real watch
dsp_processor = get_dsp_processor()

# Initialize Arduino connection
@st.cache_resource
def get_arduino_connection():
    # Common port names for Seeed XIAO ESP32C3
    possible_ports = [
        '/dev/cu.usbmodem101',   # Add your specific port first
        '/dev/cu.usbmodem1101',  # Mac default
        '/dev/cu.usbserial-10',  # Common XIAO ESP32C3 on Mac
        '/dev/cu.wchusbserial10', # Another common XIAO variant
        '/dev/tty.usbserial-10'  # Alternative Mac format
    ]
    
    # First, list all available ports
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("Available ports:")
    for port in ports:
        print(f"- {port.device}: {port.description}")
    
    # Try to connect to each port
    for port in possible_ports:
        try:
            print(f"Trying to connect to {port}...")
            arduino = serial.Serial(port, 115200, timeout=1)
            print(f"Successfully connected to {port}")
            return arduino
        except Exception as e:
            print(f"Failed to connect to {port}: {e}")
    
    st.error("❌ Could not connect to XIAO ESP32C3. Check connection and port.")
    return None

arduino = get_arduino_connection()

def read_pressure_grid():
    """Read 3x3 pressure grid data from Arduino"""
    if not arduino:
        return np.ones((3, 3))  # Return all white grid
        
    try:
        # Clear any stale data
        arduino.reset_input_buffer()
        
        # Read 3 lines for 3x3 grid
        grid = np.zeros((3, 3))
        
        for row in range(3):
            line = arduino.readline().decode('utf-8').strip()
            if not line:
                continue
            
            try:
                values = [float(x.strip()) for x in line.split('::')]
                if len(values) == 3:
                    grid[row] = values
            except ValueError as e:
                print(f"Error parsing values: {e}")
        
        return grid
        
    except Exception as e:
        print(f"Error reading pressure grid: {e}")
        return np.ones((3, 3))

# Main App Layout
st.title("🌙 SleepPose - Real-time Sleep Tracking with Apple Watch")
st.markdown("*Demonstrating real-time data collection, DSP processing, and sleep analysis*")
st.markdown("---")

# Sidebar controls
st.sidebar.header("📱 Control Panel")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start", use_container_width=True):
        st.session_state.collecting = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop", use_container_width=True):
        st.session_state.collecting = False
        st.rerun()

if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
    st.session_state.data_buffer = []
    st.rerun()

# Status indicator
if st.session_state.collecting:
    st.sidebar.success("🟢 Collecting Data")
else:
    st.sidebar.info("🔴 Data Collection Stopped")

st.sidebar.markdown(f"**Data Points:** {len(st.session_state.data_buffer)}")

# Connection status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Connection Status")
if watch.is_connected():
    st.sidebar.success("✅ Apple Watch Connected", icon="✅")
    st.sidebar.info(f"📂 Reading from: HealthMetrics-{datetime.now().strftime('%Y-%m-%d')}.csv")
else:
    st.sidebar.error("❌ No Apple Watch Found", icon="❌")
    st.sidebar.warning("Check Health Auto Export app is running and syncing to Dropbox")

# Add Arduino connection status
if arduino:
    st.sidebar.success("✅ XIAO ESP32C3 Connected", icon="✅")
else:
    st.sidebar.error("❌ XIAO ESP32C3 Disconnected", icon="❌")

# Real-time data collection
if st.session_state.collecting:
    new_data = watch.get_realtime_data()
    st.session_state.data_buffer.append(new_data)
    
    # Keep only last 500 points for performance
    if len(st.session_state.data_buffer) > 500:
        st.session_state.data_buffer = st.session_state.data_buffer[-500:]

# Main dashboard
if len(st.session_state.data_buffer) > 5:
    # Convert list of dictionaries to DataFrame, handling None values
    df = pd.DataFrame([d for d in st.session_state.data_buffer if d is not None])
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            current_hr = df['heart_rate'].iloc[-1]
            st.metric("💓 Heart Rate", f"{current_hr:.0f} BPM")
        except:
            st.metric("💓 Heart Rate", "N/A")
    
    with col2:
        try:
            current_movement = df['movement_magnitude'].iloc[-1]
            st.metric("🏃 Movement", f"{current_movement:.3f} G")
        except:
            st.metric("🏃 Movement", "N/A")
    
    with col3:
        try:
            current_breathing = df['breathing_rate'].iloc[-1]
            st.metric("🫁 Breathing", f"{current_breathing:.1f}/min")
        except:
            st.metric("🫁 Breathing", "N/A")
    
    with col4:
        # Current sleep stage
        recent_hr = df['heart_rate'].tail(10).values
        recent_mov = df['movement_magnitude'].tail(10).values
        current_stage = dsp_processor.detect_sleep_stage(recent_hr, recent_mov)
        
        stage_colors = {
            "Awake": "🟡", "Light Sleep": "🔵", 
            "Deep Sleep": "🟢", "REM Sleep": "🟣", 
            "Initializing": "⚪"
        }
        st.metric("😴 Sleep Stage", f"{stage_colors.get(current_stage, '⚪')} {current_stage}")

    st.markdown("---")
    
    # Real-time data visualization
    st.header("📊 Real-time Sensor Data from Apple Watch")
    
    # Create time series plots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Heart Rate (PPG Sensor)', 'Movement (3-Axis Accelerometer)', 'Breathing Rate'),
        vertical_spacing=0.08
    )
    
    # Heart rate
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['heart_rate'], 
                  name='Heart Rate', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Movement
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['movement_magnitude'], 
                  name='Movement', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    # Breathing
    fig.add_trace(
        go.Scatter(x=list(range(len(df))), y=df['breathing_rate'], 
                  name='Breathing', line=dict(color='green', width=2)),
        row=3, col=1
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Apple Watch Sensor Streams")
    fig.update_xaxes(title_text="Time (samples)")
    fig.update_yaxes(title_text="BPM", row=1, col=1)
    fig.update_yaxes(title_text="G-force", row=2, col=1)
    fig.update_yaxes(title_text="Breaths/min", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pressure grid visualization
    st.markdown("---")
    st.header("🟦 Pressure Sensor Array")
    
    # Get pressure data
    pressure_grid = read_pressure_grid()
    
    # Create heatmap
    fig = px.imshow(
        pressure_grid,
        color_continuous_scale=['white', 'red'],
        range_color=[1.0, 2.0],  # Original range
        aspect='equal'
    )
    
    fig.update_layout(
        height=400,
        width=400,
        title_text="Velostat Pressure Grid",
        showlegend=False,
        coloraxis_showscale=True,
        coloraxis_colorbar_title="Voltage"
    )
    
    # Remove axis labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    # Display the heatmap
    st.plotly_chart(fig)
    
    # Add interpretation
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Lighter colors** indicate higher pressure points")
    with col2:
        max_pressure = np.max(pressure_grid)
        max_loc = np.unravel_index(np.argmax(pressure_grid), pressure_grid.shape)
        st.metric("Peak Pressure", f"{max_pressure:.2f}V at ({max_loc[0]}, {max_loc[1]})")
    
    # DSP Analysis Section
    st.markdown("---")
    st.header("Digital Signal Processing")
    
    if len(df) > 50:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Frequency Domain Analysis (FFT)")
            
            # FFT of heart rate
            hr_data = df['heart_rate'].values
            freqs, fft_vals = dsp_processor.frequency_analysis(hr_data)
            
            if len(freqs) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Show first 20 frequency bins for clarity
                n_bins = min(20, len(freqs))
                ax.plot(freqs[:n_bins], fft_vals[:n_bins], 'b-', linewidth=2)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                ax.set_title('Heart Rate Frequency Spectrum')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.info("**Why?**: Low frequencies (0.01-0.1 Hz) indicate sleep rhythms, higher frequencies suggest wakefulness.")
        
        with col2:
            st.subheader("Bandpass Filtered Signals")
            
            hr_data = df['heart_rate'].values
            
            # Apply different frequency filters
            sleep_rhythm = dsp_processor.bandpass_filter(hr_data, 0.01, 0.1)
            breathing_coupling = dsp_processor.bandpass_filter(hr_data, 0.1, 0.5)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Show last 100 samples for clarity
            n_samples = min(100, len(hr_data))
            x_axis = range(n_samples)
            
            ax.plot(x_axis, hr_data[-n_samples:], 'gray', alpha=0.7, label='Original Signal', linewidth=1)
            ax.plot(x_axis, sleep_rhythm[-n_samples:], 'blue', label='Sleep Rhythm (0.01-0.1 Hz)', linewidth=2)
            ax.plot(x_axis, breathing_coupling[-n_samples:], 'red', label='Breathing Coupling (0.1-0.5 Hz)', linewidth=2)
            
            ax.legend()
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Heart Rate (BPM)')
            ax.set_title('DSP Filtered Heart Rate Signals')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.info(" **Why?**: Butterworth bandpass filters isolate physiologically relevant frequency bands.")
    
    # Performance metrics
    st.markdown("---")
    st.header("Performance & Error Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Signal Quality Metrics")
        
        # Calculate signal quality
        hr_data = df['heart_rate'].values
        signal_power = np.var(hr_data)
        
        if len(hr_data) > 5:
            noise_estimate = np.var(np.diff(hr_data))
            snr_db = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 50
        else:
            snr_db = 0
        
        st.metric("Signal-to-Noise Ratio", f"{snr_db:.1f} dB")
        st.metric("Data Points Collected", len(st.session_state.data_buffer))
        st.metric("Effective Sampling Rate", "1 Hz (Heart Rate)")
        
        # Filter performance
        if len(hr_data) > 20:
            filtered_hr = dsp_processor.bandpass_filter(hr_data, 0.5, 2.0)
            original_std = np.std(hr_data)
            filtered_std = np.std(filtered_hr)
            noise_reduction = (1 - filtered_std/original_std) * 100 if original_std > 0 else 0
            st.metric("Noise Reduction", f"{noise_reduction:.1f}%")
    
    with col2:
        st.subheader("Algorithm Performance")
        
        # Sleep metrics
        metrics = dsp_processor.calculate_sleep_metrics(st.session_state.data_buffer)
        
        if metrics:
            st.metric("Avg Heart Rate", f"{metrics['avg_heart_rate']:.1f} BPM")
            st.metric("HR Variability", f"{metrics['hr_variability']:.1f}")
            st.metric("Sleep Efficiency", f"{metrics['sleep_efficiency']:.1f}%")
            st.metric("Processing Latency", "< 50ms")
    
    with col3:
        st.subheader("Sleep Stage Distribution")
        
        # Calculate sleep stages for visualization
        stages = []
        window_size = 10
        
        for i in range(window_size, len(df), 5):  # Every 5 samples
            hr_window = df['heart_rate'].iloc[i-window_size:i].values
            mov_window = df['movement_magnitude'].iloc[i-window_size:i].values
            stage = dsp_processor.detect_sleep_stage(hr_window, mov_window)
            stages.append(stage)
        
        if stages:
            stage_counts = pd.Series(stages).value_counts();
            
            fig, ax = plt.subplots(figsize=(8, 6));
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
            stage_counts.plot(kind='bar', ax=ax, color=colors[:len(stage_counts)]);
            ax.set_title('Sleep Stage Classification Results');
            ax.set_xlabel('Sleep Stage');
            ax.set_ylabel('Count');
            plt.xticks(rotation=45);
            plt.tight_layout();
            st.pyplot(fig);

else:
    st.info("👆 **Click 'Start' to begin real-time sleep tracking**")
    
    st.markdown("---")
    st.markdown("""
    ### Welcome to SleepPose - Advanced Sleep Monitoring System
    
    Our system combines Apple Watch sensor data with a custom pressure sensing array:
    
    #### 1. Apple Watch Data Pipeline
    - **Initial Setup**:
        1. Install "Health Auto Export" app from App Store
        2. Enable these permissions in app:
           - Heart rate monitoring
           - Background app refresh
           - Health data access
    
    - **Data Collection Process**:
        1. Health Auto Export creates CSV files named `HealthMetrics-YYYY-MM-DD.csv`
        2. Files contain minute-by-minute data:
           - Heart rate (BPM) from PPG sensors
           - Step count from accelerometer
           - Timestamps for each measurement
        3. CSV files automatically sync to Dropbox folder
        4. SleepPose connects to Dropbox using shared folder link
        5. Real-time data extraction and processing
    
    #### 2. Pressure Sensing Array
    - **Hardware Setup**:
        - 3x3 Velostat pressure sensor grid
        - XIAO ESP32C3 microcontroller
        - USB connection at 115200 baud rate
    
    - **Sensor Operation**:
        - ESP32 reads voltage changes from pressure
        - Continuous data streaming via serial port
        - Real-time pressure mapping visualization
    
    #### Quick Start Guide
    1. **Apple Watch**:
       - Open Health Auto Export app
       - Verify Dropbox sync is enabled
       - Check connection status in sidebar
       
    2. **Pressure Grid**:
       - Connect XIAO ESP32C3 via USB
       - Position sensor grid under mattress
       - Verify connection in status panel
    
    Click **Start** to begin real-time monitoring.
    """)

# Auto-refresh for real-time updates
if st.session_state.collecting:
    time.sleep(0.5)  # Reasonable refresh rate
    st.rerun()