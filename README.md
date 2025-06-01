# SleepPose - Real-time Sleep Tracking with Apple Watch

A real-time sleep monitoring application that integrates Apple Watch sensor data with advanced Digital Signal Processing (DSP) techniques for sleep analysis.


### Real-time Data Collection
- Apple Watch heart rate monitoring (PPG sensor)
- 3-axis accelerometer for movement detection
- Breathing rate estimation from HRV
- Live data streaming to Streamlit dashboard

### DSP Processing Pipeline
- **Bandpass Filtering**: Isolate physiological frequency bands (0.01-2 Hz)
- **FFT Analysis**: Frequency domain sleep pattern recognition
- **Moving Average**: Real-time noise reduction with minimal latency
- **Multi-modal Fusion**: Combine HR, movement, and breathing patterns

### Sleep Analysis
- Real-time sleep stage detection (Awake, Light Sleep, Deep Sleep, REM)
- Heart Rate Variability (HRV) analysis
- Movement pattern classification
- Sleep efficiency scoring

### Data Sources
- **Primary**: Apple Watch HealthKit sensors
- **Sampling Rates**: 
  - Heart Rate: 1 Hz (PPG sensor)
  - Accelerometer: 50 Hz (3-axis movement)
  - Breathing: Derived from HRV patterns

### Data Characteristics
- **Heart Rate Range**: 50-120 BPM during sleep
- **Movement Threshold**: 0-2G acceleration
- **Breathing Rate**: 10-25 breaths/minute
- **Buffer Size**: 1000 samples (~20 seconds at 50Hz)

### Data Quality Metrics
- Signal-to-Noise Ratio (SNR) calculation
- Real-time outlier detection
- Missing data interpolation
- Sensor fusion accuracy assessment

## DSP 

1. **Butterworth Bandpass Filter**
   - 4th order filter design
   - Frequency bands: Sleep rhythm (0.01-0.1 Hz), Breathing (0.1-0.5 Hz)
   - Noise reduction: ~15-30%

2. **Fast Fourier Transform (FFT)**
   - Real-time frequency domain analysis
   - Sleep pattern recognition in frequency spectrum
   - Dominant frequency identification

3. **Sleep Stage Classification**
   - Rule-based algorithm using HR variability + movement
   - Processing latency: <100ms per sample
   - Classification accuracy: Demonstrated through confusion matrix


## Error Analysis

### Signal Quality Assessment
- **SNR Calculation**: Continuous signal quality monitoring
- **Artifact Detection**: Movement and sensor disconnection detection
- **Data Validation**: Range checking and outlier removal

### Algorithm Performance
- **Filter Effectiveness**: Before/after signal comparison
- **Classification Confusion Matrix**: True vs predicted sleep stages
- **Temporal Accuracy**: Sleep stage transition timing analysis

### Error Sources & Mitigation
- **Sensor Noise**: Addressed through bandpass filtering
- **Movement Artifacts**: Detected and flagged in real-time  
- **Connection Issues**: Automatic reconnection and data interpolation
- **Individual Variations**: Adaptive thresholds based on personal baselines

## Installation & Setup

```bash
pip install -r requirements.txt
```

```bash
streamlit run src/app.py
```

### Apple Watch Integration
1. Install the iOS companion app (requires Xcode)
2. Grant HealthKit permissions
3. Start the WebSocket server for real-time data streaming
4. Connect to the Streamlit dashboard

##  Real Apple Watch Connection

The project includes iOS Swift code for real HealthKit integration:
- `HealthKitManager.swift`: Core data collection from Apple Watch
- WebSocket connection for real-time streaming
- Automatic reconnection and error handling


## Technical Architecture

```
Apple Watch (HealthKit) → iOS App → WebSocket → Streamlit Dashboard
                                              ↓
                                    DSP Processing Pipeline
                                              ↓
                                    Real-time Analysis & Visualization
```

### Data Flow
1. **Collection**: Apple Watch sensors → HealthKit → iOS App
2. **Transmission**: iOS App → WebSocket → Python Backend  
3. **Processing**: DSP algorithms → Sleep analysis → Metrics calculation
4. **Visualization**: Real-time charts → Sleep metrics → Performance analysis


##  Future Enhancements
- Machine Learning integration for personalized sleep pattern recognition
- Long-term sleep trend analysis and reporting
- Integration with additional health metrics (blood oxygen, skin temperature)
- Cloud storage for historical data analysis
- Advanced sleep disorder detection algorithms

