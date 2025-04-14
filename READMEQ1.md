# EMG-Based Hand Pose Prediction: Comprehensive Project Documentation

## 1. Project Overview

This project implements a machine learning pipeline to predict continuous hand joint angles from surface electromyography (sEMG) signals. The goal is to translate electrical muscle activity measured from the forearm into precise hand pose estimations that could be used to control prosthetic hands or virtual reality interfaces.

The project follows these main steps:
1. Signal filtering (optional)
2. Window creation with overlap
3. Feature extraction from EMG windows
4. Cross-validation strategy implementation
5. Feature selection for multi-output regression
6. Model comparison and selection
7. Test data prediction and submission

## 2. Dataset Description

The dataset consists of two main parts:

### 2.1 Guided Gestures Dataset

- Contains recordings of predefined hand postures (five poses) across 5 sessions
- Raw EMG data: `guided_dataset_X.npy` - shape (5, 8, 230000) for (session, electrode, time)
- Joint angle data: `guided_dataset_y.npy` - shape (5, 51, 230000) for (session, joint angle, time)
- Test data: `guided_testset_X.npy` - shape (5, 332, 8, 500) for (session, window, electrode, time)

### 2.2 Free Gestures Dataset

- Contains recordings of natural hand movements across 5 sessions
- Raw EMG data: `freemoves_dataset_X.npy` - shape (5, 8, 270000)
- Joint angle data: `freemoves_dataset_y.npy` - shape (5, 51, 270000)
- Test data: `freemoves_testset_X.npy` - shape (5, 308, 8, 500)

The EMG data was recorded at 1024 Hz using 8 wireless electrodes placed on the participant's forearm. The joint angle data consists of 51 continuous values representing 3 rotation angles for each of the 17 joints in the hand model, captured using an Oculus Quest VR headset and resampled to match the EMG sampling rate.

## 3. Signal Filtering (Optional)

EMG signals often contain various sources of noise that can affect model performance. We implemented bandpass and notch filtering to improve signal quality:

```python
def filter_emg_signals(emg_data, fs=1024, lowcut=20, highcut=450, notch_freq=50, q=30):
    # Design bandpass filter (20-450 Hz)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Design notch filter (50 Hz for power line interference)
    b_notch, a_notch = signal.iirnotch(notch_freq, q, fs)
    
    # Apply filters to each session and electrode
    filtered_data = np.zeros_like(emg_data)
    for s in range(n_sessions):
        for e in range(n_electrodes):
            # Apply bandpass filter
            filtered = signal.filtfilt(b, a, emg_data[s, e, :])
            # Apply notch filter
            filtered = signal.filtfilt(b_notch, a_notch, filtered)
            filtered_data[s, e, :] = filtered
    
    return filtered_data
```

The bandpass filter (20-450 Hz) removes:
- Low-frequency motion artifacts (<20 Hz)
- High-frequency noise (>450 Hz)

The notch filter (50 Hz) removes power line interference.

## 4. Window Creation with Overlap

To prepare the data for feature extraction and model training, we segmented the continuous EMG signals into overlapping windows of fixed size.

### 4.1 Window Creation Process

```python
def create_windows_with_overlap(data, window_size=500, overlap_percent=50):
    n_sessions, n_channels, n_samples = data.shape
    # Calculate step size based on overlap percentage
    step_size = int(window_size * (1 - overlap_percent/100))
    
    windows = []
    window_indices = []
    
    for session_idx in range(n_sessions):
        n_windows = 1 + (n_samples - window_size) // step_size
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            if end_idx <= n_samples:
                window = data[session_idx, :, start_idx:end_idx]
                windows.append(window)
                window_indices.append((session_idx, start_idx, end_idx))
    
    return np.array(windows), window_indices
```

### 4.2 Overlap Percentage Optimization

We tested different overlap percentages (25%, 50%, 75%) to determine the optimal balance between:
- Dataset size (higher overlap = more windows)
- Computational requirements (higher overlap = more processing time and memory)

Memory usage and window counts for various overlap percentages:

| Overlap | Windows | Memory (MB) | Processing Time (s) |
|---------|---------|-------------|---------------------|
| 25%     | 3,065   | 932.53      | 22.67               |
| 50%     | 4,595   | 1,089.69    | 13.82               |
| 75%     | 9,185   | Memory Error| N/A                 |

We chose **50% overlap** as it provides a good balance of:
1. Creating a substantial number of windows (4,595)
2. Manageable memory usage (~1.1 GB)
3. Reasonable processing time (~14 seconds)

## 5. Memory-Efficient Implementation

Due to memory constraints, we implemented a memory-efficient version of the window creation and feature extraction process:

```python
def memory_efficient_windowing(X_data, y_data, window_size=500, overlap_percent=50, batch_size=500):
    # Count total windows first
    total_windows = 0
    window_indices = []
    for session_idx in range(n_sessions):
        n_windows = 1 + (n_samples - window_size) // step_size
        total_windows += n_windows
        for window_idx in range(n_windows):
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            if end_idx <= n_samples:
                window_indices.append((session_idx, start_idx, end_idx))
    
    # Process windows in batches
    X_features = np.zeros((total_windows, n_features), dtype=np.float32)
    y_targets = np.zeros((total_windows, n_joints), dtype=np.float32)
    
    for batch_start in range(0, total_windows, batch_size):
        batch_end = min(batch_start + batch_size, total_windows)
        batch_indices = window_indices[batch_start:batch_end]
        
        # Create batch arrays
        X_batch = np.zeros((len(batch_indices), n_channels, window_size), dtype=np.float32)
        y_batch = np.zeros((len(batch_indices), n_joints, window_size), dtype=np.float32)
        
        # Extract windows
        for i, (session_idx, start_idx, end_idx) in enumerate(batch_indices):
            X_batch[i] = X_data[session_idx, :, start_idx:end_idx]
            y_batch[i] = y_data[session_idx, :, start_idx:end_idx]
        
        # Extract features
        X_features[batch_start:batch_end] = feature_extractor.transform(X_batch)
        y_targets[batch_start:batch_end] = np.mean(y_batch, axis=2)
        
        # Free memory
        del X_batch, y_batch
    
    return X_features, y_targets, window_indices
```

Additional memory optimization strategies:
1. Processing one session at a time
2. Using float32 instead of float64 (halves memory requirements)
3. Clearing variables immediately after use
4. Extracting features immediately after window creation

## 6. Feature Extraction

We implemented a custom feature extractor that inherits from scikit-learn's BaseEstimator and TransformerMixin:

```python
class EMGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_set='all'):
        self.feature_set = feature_set
        self.all_features = ['MAV', 'RMS', 'VAR', 'STD', 'ZC', 'MPR']
        
        if self.feature_set == 'all':
            self.selected_features = self.all_features
        elif self.feature_set == 'basic':
            self.selected_features = ['MAV', 'RMS', 'VAR']
        elif isinstance(self.feature_set, list):
            self.selected_features = [f for f in self.feature_set if f in self.all_features]
        else:
            raise ValueError("feature_set must be 'all', 'basic', or a list of feature names")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract features from all windows
    
    def _extract_features_from_window(self, window):
        # Extract features from a single window
```

### 6.1 Extracted Features

Six time-domain features were extracted for each electrode:

1. **Mean Absolute Value (MAV)**:
   - Average of absolute EMG amplitude
   - Formula: `MAV = (1/K) * ∑|xᵢ|`

2. **Root Mean Square (RMS)**:
   - Square root of mean squared EMG amplitude
   - Formula: `RMS = √[(1/K) * ∑xᵢ²]`

3. **Variance (VAR)**:
   - Measure of signal dispersion
   - Formula: `VAR = (1/(K-1)) * ∑(xᵢ - x̄)²`

4. **Standard Deviation (STD)**:
   - Square root of variance
   - Formula: `STD = √[(1/(K-1)) * ∑(xᵢ - x̄)²]`

5. **Zero Crossing (ZC)**:
   - Count of signal crossing zero amplitude
   - Formula: `ZC = ∑[1 if xᵢ*xᵢ₊₁ < 0]`

6. **Myopulse Percentage Rate (MPR)**:
   - Percentage of samples exceeding a threshold
   - Formula: `MPR = (1/K) * ∑[1 if |xᵢ| > σ]`

With 8 electrodes and 6 features per electrode, we extracted a total of 48 features per window.

## 7. Cross-Validation Strategy

To ensure robust model evaluation and prevent data leakage, we implemented a leave-one-session-out cross-validation strategy:

```python
def leave_one_session_out_cv(window_indices, n_sessions=5):
    session_indices = np.array([idx[0] for idx in window_indices])
    cv_splits = []
    
    for test_session in range(n_sessions):
        # Windows from the test session
        test_idx = np.where(session_indices == test_session)[0]
        # Windows from all other sessions
        train_idx = np.where(session_indices != test_session)[0]
        cv_splits.append((train_idx, test_idx))
    
    return cv_splits
```

### 7.1 Advantages of Session-Based CV

1. **Prevents Data Leakage**: Ensures windows from the same recording session never appear in both training and test sets
2. **Realistic Evaluation**: Tests the model's ability to generalize to new recording sessions
3. **Robustness**: High session-to-session variability in EMG signals makes this approach more challenging but more realistic

### 7.2 CV Visualization

We visualized the cross-validation splits to verify proper separation between training and test data:

![Cross-Validation Split](cross_validation_split.png)

In the visualization, each row represents a session, with blue areas showing training data and red areas showing test data. The clear separation confirms no leakage between sets.

## 8. Feature Selection for Multi-Output Regression

Since we're predicting 51 joint angles simultaneously (multi-output regression), we needed specialized feature selection approaches:

### 8.1 Approach 1: Aggregate Target Method

```python
# Use the mean across all joints as target for feature selection
y_aggregate = np.mean(y_targets, axis=1)
selector = SelectKBest(f_regression, k=n_features)
selector.fit(X_features, y_aggregate)
```

### 8.2 Approach 2: Per-Target Method

```python
# Apply feature selection for each joint angle separately
n_joints = y_targets.shape[1]
all_selected_indices = []

for joint_idx in range(n_joints):
    y_single = y_targets[:, joint_idx]
    selector = SelectKBest(f_regression, k=n_features)
    selector.fit(X_features, y_single)
    all_selected_indices.extend(selector.get_support(indices=True))

# Count frequency of each feature being selected
from collections import Counter
feature_counts = Counter(all_selected_indices)
top_indices = [idx for idx, _ in feature_counts.most_common(n_features)]
```

### 8.3 Approach 3: Model-Based Approach

```python
# Use RandomForest feature importance
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=50))
rf.fit(X_features, y_targets)
importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
```

After comparing these approaches, we selected the features that provided the lowest RMSE across our cross-validation folds.

## 9. Model Training and Evaluation

We compared several regression models:

```python
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_regression, k=optimal_k)),
        ('regressor', model)
    ])
    
    model_results = evaluate_model(pipeline, X_features, y_targets, window_indices)
    results[name] = model_results
```

### 9.1 Evaluation Metrics

1. **Root Mean Squared Error (RMSE)**:
   ```
   RMSE = sqrt[(1/(Nts*D)) * ∑∑(yi,d - ŷi,d)²]
   ```

2. **Normalized Mean Squared Error (NMSE)**:
   ```
   NMSE = ∑∑(yi,d - ŷi,d)² / ∑∑(yi,d - ȳd)²
   ```

These metrics were calculated across all 51 joint angles to evaluate overall model performance.

## 10. Memory-Efficient Test Prediction

For generating predictions on the test data, we implemented memory-efficient batch processing:

```python
def predict_test_data_efficiently(model, test_data, feature_set='all', batch_size=50):
    n_sessions, n_windows, n_electrodes, window_size = test_data.shape
    
    # Initialize predictions array
    all_predictions = np.zeros((n_sessions * n_windows, n_joints), dtype=np.float32)
    
    # Process windows in batches
    window_count = 0
    for session in range(n_sessions):
        for batch_start in range(0, n_windows, batch_size):
            batch_end = min(batch_start + batch_size, n_windows)
            batch_size_actual = batch_end - batch_start
            
            # Extract batch of windows
            batch_windows = np.zeros((batch_size_actual, n_electrodes, window_size), dtype=np.float32)
            for i, window_idx in enumerate(range(batch_start, batch_end)):
                batch_windows[i] = test_data[session, window_idx]
            
            # Extract features
            batch_features = feature_extractor.transform(batch_windows)
            
            # Generate predictions
            batch_predictions = model.predict(batch_features)
            
            # Store predictions
            for i in range(batch_size_actual):
                all_predictions[window_count] = batch_predictions[i]
                window_count += 1
            
            # Clear memory
            del batch_windows, batch_features, batch_predictions
    
    return all_predictions
```

This approach allows predicting on large test sets without exhausting system memory.

## 11. Prediction Results and Submission

### 11.1 Guided Dataset Predictions

We successfully generated predictions for the guided_testset_X.npy, resulting in a CSV file with:
- 1,659 rows (test windows)
- 51 columns (joint angles)

The predictions represent angles in degrees, stored in scientific notation format:
```
4.040871810913085938e+01,-6.399026107788085938e+01,-2.906723213195800781e+01,...
```

These values can be interpreted as:
- 4.040871810913085938e+01 = 40.41 degrees
- -6.399026107788085938e+01 = -63.99 degrees
- etc.

### 11.2 Submission Format

The final submission requires:
1. Guided dataset predictions (1,660 windows)
2. Free gestures dataset predictions (1,540 windows)
3. Combined in order (guided first, then free) for a total of 3,200 rows

## 12. Complete Pipeline Overview

Here's the complete workflow implemented in this project:

1. **Load and filter EMG data** (optional bandpass and notch filtering)
2. **Create overlapping windows** (50% overlap, 500 samples per window)
3. **Extract time-domain features** (6 features × 8 electrodes = 48 features per window)
4. **Select optimal features** using one of three multi-output approaches
5. **Train regression models** with leave-one-session-out cross-validation
6. **Select best model** based on RMSE across CV folds
7. **Generate predictions** for test data using memory-efficient batch processing
8. **Format and save predictions** for submission

## 13. Performance Results

The pipeline achieves:
- Efficient processing of 4,595 windows with 50% overlap
- Feature extraction (48 features per window)
- Prediction of 51 joint angles
- Memory-efficient handling of large datasets
- Scientific notation format for high-precision angle predictions

## 14. Conclusion

This project successfully implements a complete pipeline for predicting continuous hand poses from EMG signals. The memory-efficient implementation allows processing on systems with limited resources, while the session-based cross-validation ensures robust model evaluation. The final predictions represent meaningful joint angles that can be used to control virtual or prosthetic hands.

The scientific approach combining signal processing, machine learning, and memory optimization techniques demonstrates that EMG signals can be effectively translated into continuous, multi-DOF hand pose estimations.