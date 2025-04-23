import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel
import pandas as pd
from tqdm import tqdm

class EMGFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract time-domain features from EMG windows.
    Inherits from scikit-learn's BaseEstimator and TransformerMixin.
    """
    
    def __init__(self, feature_set='all', use_top_features=False, normalize=True):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        feature_set : str or list
            Specifies which features to extract. Options:
            - 'all': All implemented features
            - 'basic': Only MAV, RMS, and Variance (these are commonly used core features)
            - 'top10': Only the top 10 most important features based on feature importance analysis
            - list of feature names: e.g., ['MAV', 'RMS', 'ZC']
        use_top_features : bool
            If True, only extract the top 10 most important features based on analysis
        normalize : bool
            If True, normalize the signals before feature extraction to reduce inter-session variability
        """
        self.feature_set = feature_set
        self.use_top_features = use_top_features
        self.normalize = normalize
        
        # Define all available features
        self.all_features = ['MAV', 'RMS', 'VAR', 'STD', 'ZC', 'MPR']
        
        # Define top 10 features based on Random Forest importance analysis
        self.top_feature_list = [
            ('E7', 'RMS'), ('E4', 'MAV'), ('E7', 'STD'), ('E7', 'VAR'),
            ('E1', 'MAV'), ('E7', 'MAV'), ('E4', 'VAR'), ('E4', 'STD'),
            ('E2', 'MAV'), ('E4', 'ZC')
        ]
        
        # Define feature sets
        if self.feature_set == 'all' and not self.use_top_features:
            self.selected_features = self.all_features
            self.selected_electrodes = list(range(8))  # All 8 electrodes
        elif self.feature_set == 'basic' and not self.use_top_features:
            # MAV, RMS, VAR are selected as 'basic' because they're the most commonly used
            self.selected_features = ['MAV', 'RMS', 'VAR']
            self.selected_electrodes = list(range(8))  # All 8 electrodes
        elif self.feature_set == 'top10' or self.use_top_features:
            # Use only the top 10 feature-electrode combinations
            # We'll handle this differently in transform
            self.selected_features = self.all_features  # We'll filter later
            self.selected_electrodes = list(range(8))  # All 8 electrodes
        elif isinstance(self.feature_set, list) and not self.use_top_features:
            self.selected_features = [f for f in self.feature_set if f in self.all_features]
            self.selected_electrodes = list(range(8))  # All 8 electrodes
        else:
            raise ValueError("feature_set must be 'all', 'basic', 'top10', or a list of feature names")
    
    def fit(self, X, y=None):
        """
        Fit method (does nothing but needed for scikit-learn pipeline compatibility).
        """
        return self
    
    def transform(self, X):
        """
        Extract features from EMG windows.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data with shape (n_samples, n_channels, window_size)
            
        Returns:
        --------
        features : numpy.ndarray
            Extracted features with shape (n_samples, n_features)
        """
        n_samples, n_channels, window_size = X.shape
        
        # Apply normalization if requested
        if self.normalize:
            X_norm = np.zeros_like(X)
            for i in range(n_samples):
                for c in range(n_channels):
                    # Z-score normalization for each channel in each window
                    channel_data = X[i, c]
                    channel_mean = np.mean(channel_data)
                    channel_std = np.std(channel_data)
                    if channel_std > 0:  # Avoid division by zero
                        X_norm[i, c] = (channel_data - channel_mean) / channel_std
                    else:
                        X_norm[i, c] = channel_data - channel_mean
            X = X_norm
        
        if self.feature_set == 'top10' or self.use_top_features:
            # For top 10 features, we'll create a specific feature extraction 
            n_top_features = len(self.top_feature_list)
            features = np.zeros((n_samples, n_top_features))
            
            for i in range(n_samples):
                for j, (electrode, feature_type) in enumerate(self.top_feature_list):
                    # Extract electrode index (e.g., 'E7' -> 6)
                    e_idx = int(electrode[1]) - 1
                    
                    # Extract the specific feature for this electrode
                    channel_data = X[i, e_idx]
                    
                    # Calculate the feature
                    if feature_type == 'MAV':
                        features[i, j] = np.mean(np.abs(channel_data))
                    elif feature_type == 'RMS':
                        features[i, j] = np.sqrt(np.mean(channel_data**2))
                    elif feature_type == 'VAR':
                        features[i, j] = np.var(channel_data)
                    elif feature_type == 'STD':
                        features[i, j] = np.std(channel_data)
                    elif feature_type == 'ZC':
                        features[i, j] = np.sum(np.diff(np.signbit(channel_data).astype(int)) != 0)
                    elif feature_type == 'MPR':
                        threshold = np.std(channel_data)
                        features[i, j] = np.mean(np.abs(channel_data) > threshold)
        else:
            # Regular feature extraction for all or custom feature sets
            n_features = len(self.selected_features) * len(self.selected_electrodes)
            features = np.zeros((n_samples, n_features))
            
            for i in range(n_samples):
                sample_features = self._extract_features_from_window(X[i])
                features[i] = sample_features
        
        return features
    
    def _extract_features_from_window(self, window):
        """
        Extract features from a single EMG window.
        """
        features = []
        
        for channel in self.selected_electrodes:
            channel_data = window[channel]
            channel_features = []
            
            # Mean Absolute Value (MAV)
            if 'MAV' in self.selected_features:
                mav = np.mean(np.abs(channel_data))
                channel_features.append(mav)
            
            # Root Mean Square (RMS)
            if 'RMS' in self.selected_features:
                rms = np.sqrt(np.mean(channel_data**2))
                channel_features.append(rms)
            
            # Variance (VAR)
            if 'VAR' in self.selected_features:
                var = np.var(channel_data)
                channel_features.append(var)
            
            # Standard Deviation (STD)
            if 'STD' in self.selected_features:
                std = np.std(channel_data)
                channel_features.append(std)
            
            # Zero Crossing (ZC)
            if 'ZC' in self.selected_features:
                zc = np.sum(np.diff(np.signbit(channel_data).astype(int)) != 0)
                channel_features.append(zc)
            
            # Myopulse Percentage Rate (MPR)
            if 'MPR' in self.selected_features:
                threshold = np.std(channel_data)
                mpr = np.mean(np.abs(channel_data) > threshold)
                channel_features.append(mpr)
            
            # Add all features for this channel
            features.extend(channel_features)
        
        return np.array(features)
    
    def get_feature_names(self):
        """
        Get the names of the features that will be extracted.
        """
        if self.feature_set == 'top10' or self.use_top_features:
            return [f"{e}_{f}" for e, f in self.top_feature_list]
        else:
            feature_names = []
            for channel in self.selected_electrodes:
                for feature in self.selected_features:
                    feature_names.append(f'E{channel+1}_{feature}')
            
            return feature_names

def recommended_emg_filter(data, fs=1024):
    """
    Apply recommended filtering to EMG signals.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data with shape (session, electrode, time)
    fs : int
        Sampling frequency in Hz (default: 1024)
        
    Returns:
    --------
    filtered_data : numpy.ndarray
        Filtered data with same shape as input
    """
    n_sessions, n_electrodes, n_samples = data.shape
    filtered_data = np.zeros_like(data)
    
    # EMG typically contains frequencies between 20-500 Hz
    # A bandpass filter of 20-450 Hz is commonly used
    lowcut = 20  # High-pass to remove motion artifacts
    highcut = 450  # Low-pass to remove high frequency noise
    
    # Notch filter at 50Hz (or 60Hz for US) to remove power line interference
    notch_freq = 50
    quality_factor = 30  # Q factor determines notch width
    
    # Create bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    
    # Create notch filter
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
    
    # Apply filters to each session and electrode
    for session in range(n_sessions):
        for electrode in range(n_electrodes):
            # First apply bandpass
            temp = filtfilt(b_band, a_band, data[session, electrode, :])
            # Then apply notch
            filtered_data[session, electrode, :] = filtfilt(b_notch, a_notch, temp)
    
    return filtered_data

def detect_outliers(data, threshold=3.0):
    """
    Detect outliers in EMG data using z-score.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input EMG data with shape (session, electrode, time)
    threshold : float
        Z-score threshold for outlier detection (default: 3.0)
        
    Returns:
    --------
    outlier_mask : numpy.ndarray
        Boolean mask where True indicates an outlier
    """
    n_sessions, n_electrodes, n_samples = data.shape
    outlier_mask = np.zeros_like(data, dtype=bool)
    
    for session in range(n_sessions):
        for electrode in range(n_electrodes):
            signal = data[session, electrode, :]
            
            # Compute z-score
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            z_scores = np.abs((signal - signal_mean) / signal_std)
            
            # Mark outliers
            outlier_mask[session, electrode, :] = z_scores > threshold
    
    return outlier_mask

def create_windows_with_overlap(data, window_size=500, overlap_percent=50):
    """
    Create overlapping windows from continuous EMG data with explicit verification
    of window positioning.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data with shape (session, electrode/joint, time)
    window_size : int
        Size of each window in samples (default: 500)
    overlap_percent : float
        Percentage of overlap between consecutive windows (0-100) (default: 50)
    
    Returns:
    --------
    windows : numpy.ndarray
        Windowed data with shape (n_windows_total, electrode/joint, window_size)
    window_indices : list of tuples
        List containing (session_idx, window_start_idx, window_end_idx) for each window
    """
    n_sessions, n_channels, n_samples = data.shape
    
    # Calculate step size based on overlap percentage
    step_size = int(window_size * (1 - overlap_percent/100))
    print(f"Window size: {window_size}, Overlap: {overlap_percent}%, Step size: {step_size}")
    
    windows = []
    window_indices = []
    
    # Process each session separately
    for session_idx in range(n_sessions):
        # Calculate how many windows we can extract from this session
        n_windows = 1 + (n_samples - window_size) // step_size
        
        for window_idx in range(n_windows):
            # Calculate start and end indices
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            
            # Make sure we don't exceed the data length
            if end_idx <= n_samples:
                # Extract window for all channels
                window = data[session_idx, :, start_idx:end_idx]
                windows.append(window)
                window_indices.append((session_idx, start_idx, end_idx))
    
    # Convert list of windows to numpy array
    windows = np.array(windows)
    
    # Print out the first few window indices to verify correct overlap
    if len(window_indices) > 3:
        print("First few window indices to verify overlap:")
        for i in range(min(5, len(window_indices))):
            session, start, end = window_indices[i]
            print(f"Window {i+1}: Session {session+1}, Samples {start+1}-{end}")
    
    return windows, window_indices

def create_matching_windows(X_data, y_data, window_size=500, overlap_percent=50):
    """
    Create matching windowed datasets for both EMG (X) and joint angles (y).
    
    Parameters:
    -----------
    X_data : numpy.ndarray
        EMG data with shape (session, electrode, time)
    y_data : numpy.ndarray
        Joint angle data with shape (session, joint, time)
    window_size : int
        Size of each window in samples
    overlap_percent : float
        Percentage of overlap between consecutive windows (0-100)
    
    Returns:
    --------
    X_windows : numpy.ndarray
        Windowed EMG data with shape (n_windows, electrode, window_size)
    y_windows : numpy.ndarray
        Windowed joint angle data with shape (n_windows, joint, window_size)
    window_indices : list of tuples
        List containing (session_idx, window_start_idx, window_end_idx) for each window
    """
    # Create windows for EMG data
    X_windows, window_indices = create_windows_with_overlap(X_data, window_size, overlap_percent)
    
    # Create matching windows for joint angle data using the same indices
    n_sessions, n_joints, _ = y_data.shape
    y_windows = []
    
    for session_idx, start_idx, end_idx in window_indices:
        window = y_data[session_idx, :, start_idx:end_idx]
        y_windows.append(window)
    
    y_windows = np.array(y_windows)
    
    return X_windows, y_windows, window_indices

def leave_one_session_out_cv(window_indices, n_sessions=5):
    """
    Create cross-validation splits based on leave-one-session-out strategy.
    
    Parameters:
    -----------
    window_indices : list of tuples
        List containing (session_idx, window_start_idx, window_end_idx) for each window
    n_sessions : int
        Number of sessions in the dataset
    
    Returns:
    --------
    cv_splits : list of tuples
        List of (train_idx, test_idx) tuples for cross-validation
    """
    # Extract session indices for each window
    session_indices = np.array([idx[0] for idx in window_indices])
    cv_splits = []
    
    # For each session, create a train/test split
    for test_session in range(n_sessions):
        # Windows from the test session
        test_idx = np.where(session_indices == test_session)[0]
        # Windows from all other sessions
        train_idx = np.where(session_indices != test_session)[0]
        cv_splits.append((train_idx, test_idx))
        
        # Print some information about this split
        print(f"Split {test_session+1}: {len(train_idx)} training windows, {len(test_idx)} test windows")
    
    return cv_splits

def build_pipeline(feature_set='all', normalize=True, alpha=1.0, model_type='ridge'):
    """
    Build a preprocessing and regression pipeline.
    
    Parameters:
    -----------
    feature_set : str
        Feature set to use ('all', 'basic', 'top10')
    normalize : bool
        Whether to normalize features
    alpha : float
        Regularization strength for Ridge/Lasso regression
    model_type : str
        Model type to use ('ridge', 'lasso', 'rf')
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Complete preprocessing and regression pipeline
    """
    # Create feature extractor
    feature_extractor = EMGFeatureExtractor(feature_set=feature_set, normalize=normalize)
    
    # Choose regressor based on model_type
    if model_type == 'ridge':
        regressor = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        regressor = Lasso(alpha=alpha, max_iter=10000)
    elif model_type == 'rf':
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("model_type must be 'ridge', 'lasso', or 'rf'")
    
    # Build pipeline
    pipeline = Pipeline([
        ('features', feature_extractor),
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    return pipeline

def evaluate_model_cv(X_windows, y_targets, window_indices, pipeline, n_sessions=5):
    """
    Evaluate a model using leave-one-session-out cross-validation.
    
    Parameters:
    -----------
    X_windows : numpy.ndarray
        Windowed EMG data with shape (n_windows, n_channels, window_size)
    y_targets : numpy.ndarray
        Target joint angles with shape (n_windows, n_joints)
    window_indices : list
        List containing (session_idx, start_idx, end_idx) for each window
    pipeline : sklearn.pipeline.Pipeline
        Model pipeline to evaluate
    n_sessions : int
        Number of sessions in the dataset
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation results
    """
    # Create cross-validation splits
    cv_splits = leave_one_session_out_cv(window_indices, n_sessions)
    
    # Prepare results storage
    rmse_folds = []
    r2_folds = []
    predictions = []
    
    # For each fold in cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"Evaluating fold {fold+1}/{len(cv_splits)}...")
        
        # Get training and test data for this fold
        X_train, X_test = X_windows[train_idx], X_windows[test_idx]
        y_train, y_test = y_targets[train_idx], y_targets[test_idx]
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        predictions.append((test_idx, y_pred))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        rmse_folds.append(rmse)
        r2_folds.append(r2)
        
        print(f"  Fold {fold+1} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Calculate average metrics
    avg_rmse = np.mean(rmse_folds)
    avg_r2 = np.mean(r2_folds)
    
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average R²: {avg_r2:.4f}")
    
    # Combine results
    results = {
        'rmse_folds': rmse_folds,
        'r2_folds': r2_folds,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2,
        'predictions': predictions,
        'pipeline': pipeline
    }
    
    return results

def compare_feature_selection_methods(X_windows, y_targets, window_indices, n_sessions=5):
    """
    Compare different feature selection methods for EMG-based hand pose prediction.
    
    Parameters:
    -----------
    X_windows : numpy.ndarray
        Windowed EMG data
    y_targets : numpy.ndarray
        Target values
    window_indices : list
        Window indices
    n_sessions : int
        Number of sessions
        
    Returns:
    --------
    results : dict
        Results of feature selection comparison
    """
    # Create cross-validation splits
    cv_splits = leave_one_session_out_cv(window_indices, n_sessions)
    
    # Extract all features as baseline
    feature_extractor = EMGFeatureExtractor(feature_set='all')
    X_features = feature_extractor.transform(X_windows)
    print(f"Extracted {X_features.shape[1]} features")
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # 1. RFE with Cross-Validation
    print("\nEvaluating Recursive Feature Elimination...")
    
    rfecv_results = {}
    for model_name, model in [('Ridge', Ridge(alpha=1.0)), 
                             ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=42))]:
        print(f"  Using {model_name}...")
        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=3,
            scoring='neg_mean_squared_error',
            min_features_to_select=5
        )
        
        rfecv.fit(X_scaled, y_targets)
        
        selected_features = np.where(rfecv.support_)[0]
        feature_names = feature_extractor.get_feature_names()
        selected_feature_names = [feature_names[i] for i in selected_features]
        
        print(f"  Selected {len(selected_features)} features")
        print(f"  Top features: {selected_feature_names[:5]}...")
        
        # Evaluate with selected features
        cv_rmse = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X_scaled[train_idx][:, selected_features]
            X_test = X_scaled[test_idx][:, selected_features]
            y_train = y_targets[train_idx]
            y_test = y_targets[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_rmse.append(rmse)
        
        avg_rmse = np.mean(cv_rmse)
        print(f"  Average RMSE with RFECV: {avg_rmse:.4f}")
        
        rfecv_results[model_name] = {
            'selected_features': selected_features,
            'selected_feature_names': selected_feature_names,
            'cv_rmse': cv_rmse,
            'avg_rmse': avg_rmse
        }
    
    # 2. Feature Importance Thresholding
    print("\nEvaluating Feature Importance Thresholding...")
    
    importance_results = {}
    for model_name, model in [('Ridge', Ridge(alpha=1.0)), 
                             ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=42))]:
        print(f"  Using {model_name}...")
        
        # Fit model to get feature importances
        model.fit(X_scaled, y_targets)
        
        # Get feature importances
        if model_name == 'Ridge':
            importances = np.abs(model.coef_.mean(axis=0))
        else:
            importances = model.feature_importances_
        
        # Rank features by importance
        ranked_features = np.argsort(importances)[::-1]
        feature_names = feature_extractor.get_feature_names()
        
        # Test different feature counts
        n_features_options = [5, 10, 20, 40]
        feature_results = []
        
        for n_top in n_features_options:
            top_features = ranked_features[:n_top]
            cv_rmse = []
            
            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                X_train = X_scaled[train_idx][:, top_features]
                X_test = X_scaled[test_idx][:, top_features]
                y_train = y_targets[train_idx]
                y_test = y_targets[test_idx]
                
                # Create a fresh model
                if model_name == 'Ridge':
                    model_cv = Ridge(alpha=1.0)
                else:
                    model_cv = RandomForestRegressor(n_estimators=50, random_state=42)
                
                model_cv.fit(X_train, y_train)
                y_pred = model_cv.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_rmse.append(rmse)
            
            avg_rmse = np.mean(cv_rmse)
            feature_results.append((n_top, avg_rmse))
            print(f"  Top {n_top} features: RMSE = {avg_rmse:.4f}")
        
        # Find optimal feature count
        optimal_n, optimal_rmse = min(feature_results, key=lambda x: x[1])
        optimal_features = ranked_features[:optimal_n]
        optimal_feature_names = [feature_names[i] for i in optimal_features]
        
        print(f"  Optimal feature count: {optimal_n}")
        print(f"  Top features: {optimal_feature_names[:5]}...")
        
        importance_results[model_name] = {
            'feature_results': feature_results,
            'optimal_n_features': optimal_n,
            'optimal_rmse': optimal_rmse,
            'selected_features': optimal_features,
            'selected_feature_names': optimal_feature_names
        }
    
    # 3. Compare all methods
    print("\nFeature Selection Summary:")
    summary_data = []
    
    # RFECV results
    for model_name, results in rfecv_results.items():
        summary_data.append({
            'Model': model_name,
            'Method': 'RFECV',
            'Num Features': len(results['selected_features']),
            'RMSE': results['avg_rmse']
        })
    
    # Importance results
    for model_name, results in importance_results.items():
        summary_data.append({
            'Model': model_name,
            'Method': 'Importance',
            'Num Features': results['optimal_n_features'],
            'RMSE': results['optimal_rmse']
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # Find best method overall
    best_idx = summary_df['RMSE'].idxmin()
    best_method = summary_df.iloc[best_idx]
    print(f"\nBest feature selection method: {best_method['Model']} with {best_method['Method']}")
    print(f"Number of features: {best_method['Num Features']}, RMSE: {best_method['RMSE']:.4f}")
    
    # Determine which features to use
    if best_method['Method'] == 'RFECV':
        best_features = rfecv_results[best_method['Model']]['selected_features']
        best_feature_names = rfecv_results[best_method['Model']]['selected_feature_names']
    else:
        best_features = importance_results[best_method['Model']]['selected_features']
        best_feature_names = importance_results[best_method['Model']]['selected_feature_names']
    
    # Combine all results
    results = {
        'rfecv_results': rfecv_results,
        'importance_results': importance_results,
        'summary': summary_df,
        'best_method': best_method,
        'best_features': best_features,
        'best_feature_names': best_feature_names
    }
    
    return results

def optimize_ridge_alpha(X_windows, y_targets, window_indices, n_sessions=5):
    """
    Optimize Ridge regression alpha parameter using cross-validation.
    
    Parameters:
    -----------
    X_windows : numpy.ndarray
        Windowed EMG data
    y_targets : numpy.ndarray
        Target values
    window_indices : list
        Window indices
    n_sessions : int
        Number of sessions
        
    Returns:
    --------
    best_alpha : float
        Best alpha value
    """
    # Create cross-validation splits
    cv_splits = leave_one_session_out_cv(window_indices, n_sessions)
    
    # Extract features
    feature_extractor = EMGFeatureExtractor(feature_set='all')
    X_features = feature_extractor.transform(X_windows)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Define alpha values to test
    alphas = np.logspace(-3, 3, 7)  # 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0
    
    # Store results
    alpha_results = []
    
    # Test each alpha
    for alpha in alphas:
        cv_rmse = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y_targets[train_idx]
            y_test = y_targets[test_idx]
            
            # Train Ridge model
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = ridge.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_rmse.append(rmse)
        
        # Average RMSE across folds
        avg_rmse = np.mean(cv_rmse)
        alpha_results.append((alpha, avg_rmse))
        print(f"Alpha = {alpha:.4f}: RMSE = {avg_rmse:.4f}")
    
    # Find best alpha
    best_alpha, best_rmse = min(alpha_results, key=lambda x: x[1])
    print(f"Best alpha: {best_alpha:.4f} with RMSE = {best_rmse:.4f}")
    
    # Plot alpha vs. RMSE
    alphas_plot = [a for a, _ in alpha_results]
    rmses_plot = [r for _, r in alpha_results]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas_plot, rmses_plot, 'o-')
    plt.axvline(x=best_alpha, color='r', linestyle='--')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Cross-Validation RMSE')
    plt.title('Ridge Regression Alpha Optimization')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return best_alpha, alpha_results

def train_final_pipeline(X_windows, y_targets, feature_selection_results, best_alpha=1.0):
    """
    Train final pipeline using the best feature selection and optimal alpha.
    
    Parameters:
    -----------
    X_windows : numpy.ndarray
        Windowed EMG data
    y_targets : numpy.ndarray
        Target values
    feature_selection_results : dict
        Results from feature selection comparison
    best_alpha : float
        Optimal alpha value for Ridge regression
        
    Returns:
    --------
    final_pipeline : sklearn.pipeline.Pipeline
        Trained final pipeline
    """
    # Extract best method information
    best_method = feature_selection_results['best_method']
    best_features = feature_selection_results['best_features']
    
    # Create feature extractor and extract features
    feature_extractor = EMGFeatureExtractor(feature_set='all')
    X_features = feature_extractor.transform(X_windows)
    
    # Select best features
    X_selected = X_features[:, best_features]
    
    # Create final pipeline
    if best_method['Model'] == 'Ridge':
        regressor = Ridge(alpha=best_alpha)
    else:
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    # Train final pipeline
    pipeline.fit(X_selected, y_targets)
    
    print(f"Trained final pipeline using {best_method['Model']} with {len(best_features)} features")
    
    # Return complete pipeline for inference
    final_pipeline = {
        'feature_extractor': feature_extractor,
        'selected_features': best_features,
        'pipeline': pipeline
    }
    
    return final_pipeline

def predict_with_pipeline(pipeline, X_test):
    """
    Generate predictions using trained pipeline.
    
    Parameters:
    -----------
    pipeline : dict
        Trained pipeline components
    X_test : numpy.ndarray
        Test data
        
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values
    """
    # Extract pipeline components
    feature_extractor = pipeline['feature_extractor']
    selected_features = pipeline['selected_features']
    model_pipeline = pipeline['pipeline']
    
    # Extract features
    X_features = feature_extractor.transform(X_test)
    
    # Select best features
    X_selected = X_features[:, selected_features]
    
    # Generate predictions
    predictions = model_pipeline.predict(X_selected)
    
    return predictions

def predict_test_data(final_pipeline, test_data):
    """
    Generate predictions for test data using a trained pipeline.
    
    Parameters:
    -----------
    final_pipeline : dict
        Trained pipeline components
    test_data : numpy.ndarray
        Test EMG data with shape (n_sessions, n_windows, n_electrodes, window_size)
        
    Returns:
    --------
    predictions : numpy.ndarray
        Joint angle predictions for each test window
    """
    n_sessions, n_windows, n_electrodes, window_size = test_data.shape
    total_windows = n_sessions * n_windows
    
    # Initialize array for predictions
    # Need to determine output dimensionality first by making a sample prediction
    sample_window = test_data[0:1, 0:1].reshape(1, n_electrodes, window_size)
    sample_pred = predict_with_pipeline(final_pipeline, sample_window)
    n_outputs = sample_pred.shape[1] if len(sample_pred.shape) > 1 else 1
    
    predictions = np.zeros((total_windows, n_outputs))
    
    # Process all windows
    window_idx = 0
    
    for session in range(n_sessions):
        print(f"Processing session {session+1}/{n_sessions}...")
        
        for window in tqdm(range(n_windows)):
            # Extract window data
            X_window = test_data[session, window].reshape(1, n_electrodes, window_size)
            
            # Generate prediction
            pred = predict_with_pipeline(final_pipeline, X_window)
            
            # Store prediction
            predictions[window_idx] = pred
            window_idx += 1
    
    return predictions

# Polynomial feature transformation
def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to the input data.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features
    degree : int
        Polynomial degree
        
    Returns:
    --------
    X_poly : numpy.ndarray
        Features with polynomial terms added
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

# Enhanced model evaluation
def evaluate_models(X_windows, y_targets, window_indices, n_sessions=5):
    """
    Evaluate and compare multiple regression models.
    
    Parameters:
    -----------
    X_windows : numpy.ndarray
        Windowed EMG data
    y_targets : numpy.ndarray
        Target values
    window_indices : list
        Window indices
    n_sessions : int
        Number of sessions
        
    Returns:
    --------
    results : dict
        Evaluation results
    """
    # Create cross-validation splits
    cv_splits = leave_one_session_out_cv(window_indices, n_sessions)
    
    # Extract features
    feature_extractor = EMGFeatureExtractor(feature_set='all')
    X_features = feature_extractor.transform(X_windows)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # Models to evaluate
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'RidgePoly': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=10.0))
        ])
    }
    
    # Store results
    model_results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        cv_rmse = []
        cv_r2 = []
        
        for fold, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y_targets[train_idx]
            y_test = y_targets[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            cv_rmse.append(rmse)
            cv_r2.append(r2)
            
            print(f"  Fold {fold+1}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
        
        # Average metrics
        avg_rmse = np.mean(cv_rmse)
        avg_r2 = np.mean(cv_r2)
        
        print(f"  Average RMSE: {avg_rmse:.4f}")
        print(f"  Average R²: {avg_r2:.4f}")
        
        # Store results
        model_results[model_name] = {
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'model': model
        }
    
    # Find best model
    best_model = min(model_results.items(), key=lambda x: x[1]['avg_rmse'])
    print(f"\nBest model: {best_model[0]} with RMSE = {best_model[1]['avg_rmse']:.4f}")
    
    # Plot comparison
    model_names = list(model_results.keys())
    rmses = [model_results[name]['avg_rmse'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, rmses)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Average RMSE')
    plt.title('Model Comparison')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return model_results