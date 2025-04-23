import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFECV, SelectFromModel
from tqdm import tqdm
import seaborn as sns
from emg_processing_pipeline import (
    EMGFeatureExtractor, 
    recommended_emg_filter, 
    create_matching_windows,
    leave_one_session_out_cv,
    detect_outliers,
    compare_feature_selection_methods,
    optimize_ridge_alpha,
    evaluate_models,
    train_final_pipeline, 
    predict_test_data
)

# Set random seed for reproducibility
np.random.seed(42)

def visualize_filtered_signals(original_data, filtered_data, session_id=0, electrode_id=0, time_segment=slice(0, 5000)):
    """
    Visualize the original and filtered EMG signals.
    """
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(original_data[session_id, electrode_id, time_segment])
    plt.title(f'Original EMG Signal - Session {session_id+1}, Electrode {electrode_id+1}')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_data[session_id, electrode_id, time_segment])
    plt.title(f'Filtered EMG Signal - Session {session_id+1}, Electrode {electrode_id+1}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_emg_dataset(data, session_id=0, time_segment=slice(0, 10000)):
    """
    Visualize all electrodes in an EMG dataset for a given session.
    """
    n_electrodes = data.shape[1]
    
    plt.figure(figsize=(16, 12))
    for i in range(n_electrodes):
        plt.subplot(n_electrodes, 1, i+1)
        plt.plot(data[session_id, i, time_segment])
        plt.title(f'Electrode {i+1}')
        plt.grid(True, alpha=0.3)
        plt.ylabel('Amplitude')
        
        if i == n_electrodes - 1:
            plt.xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.show()

def visualize_feature_importance(feature_names, importances, title="Feature Importance"):
    """
    Visualize feature importance.
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plot top 20 features or all if less than 20
    n_features = min(20, len(feature_names))
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(n_features), sorted_importances[:n_features], align='center')
    plt.yticks(range(n_features), sorted_names[:n_features])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_true(y_true, y_pred, joint_idx=0, title="Predictions vs True Values"):
    """
    Plot predicted vs true values for a specific joint.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, joint_idx], label='True')
    plt.plot(y_pred[:, joint_idx], label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Joint Angle {joint_idx}')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_residuals(y_true, y_pred, joint_idx=0):
    """
    Analyze prediction residuals.
    """
    residuals = y_true[:, joint_idx] - y_pred[:, joint_idx]
    
    plt.figure(figsize=(15, 8))
    
    # Residuals vs fitted
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred[:, joint_idx], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # QQ plot
    from scipy.stats import probplot
    plt.subplot(2, 2, 3)
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot')
    
    # Residuals over sample index
    plt.subplot(2, 2, 4)
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual')
    plt.title('Residuals vs Index')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print residual statistics
    print(f"Residual statistics for joint {joint_idx}:")
    print(f"Mean: {np.mean(residuals):.4f}")
    print(f"Std: {np.std(residuals):.4f}")
    print(f"Min: {np.min(residuals):.4f}")
    print(f"Max: {np.max(residuals):.4f}")

def main():
    print("Loading data...")
    try:
        guided_X = np.load('guided_dataset_X.npy')
        guided_y = np.load('guided_dataset_y.npy')
        guided_testset_X = np.load('guided_testset_X.npy')
        
        print("Data loaded successfully.")
        print(f"guided_X shape: {guided_X.shape}")
        print(f"guided_y shape: {guided_y.shape}")
        print(f"guided_testset_X shape: {guided_testset_X.shape}")
    except FileNotFoundError:
        print("Error: Data files not found. Please ensure the dataset files are in the current directory.")
        return
    
    # 1. Data Exploration and Preprocessing
    print("\n====== 1. Data Exploration and Preprocessing ======")
    
    # Check for NaN values
    print("Checking for NaN values...")
    print(f"NaN values in guided_X: {np.isnan(guided_X).sum()}")
    print(f"NaN values in guided_y: {np.isnan(guided_y).sum()}")
    
    # Visualize raw EMG data
    print("Visualizing raw EMG data...")
    visualize_emg_dataset(guided_X)
    
    # Apply filtering
    print("Applying signal filtering...")
    filtered_X = recommended_emg_filter(guided_X)
    
    # Visualize filtered signals
    print("Visualizing filtered signals...")
    visualize_filtered_signals(guided_X, filtered_X)
    
    # Detect outliers
    print("Detecting outliers...")
    outliers = detect_outliers(filtered_X, threshold=5.0)
    print(f"Outliers detected: {outliers.sum()} points ({outliers.sum() / outliers.size * 100:.4f}%)")
    
    # Create windows with overlap
    print("\n====== 2. Window Creation ======")
    window_size = 500
    overlap_percent = 50
    
    print(f"Creating windows with size {window_size} and {overlap_percent}% overlap...")
    X_windows, y_windows, window_indices = create_matching_windows(
        filtered_X, guided_y, window_size=window_size, overlap_percent=overlap_percent
    )
    
    print(f"Created {len(X_windows)} windows")
    print(f"X_windows shape: {X_windows.shape}")
    print(f"y_windows shape: {y_windows.shape}")
    
    # Calculate mean joint angle as target
    print("Calculating mean joint angles as targets...")
    y_targets = np.mean(y_windows, axis=2)
    print(f"Target shape: {y_targets.shape}")
    
    # 3. Feature Selection
    print("\n====== 3. Feature Selection ======")
    print("Comparing feature selection methods...")
    feature_selection_results = compare_feature_selection_methods(X_windows, y_targets, window_indices)
    
    # 4. Model Optimization
    print("\n====== 4. Model Optimization ======")
    print("Optimizing Ridge regression alpha parameter...")
    best_alpha, alpha_results = optimize_ridge_alpha(X_windows, y_targets, window_indices)
    
    # 5. Model Evaluation
    print("\n====== 5. Model Evaluation ======")
    print("Evaluating regression models...")
    model_results = evaluate_models(X_windows, y_targets, window_indices)
    
    # 6. Train Final Model
    print("\n====== 6. Training Final Model ======")
    final_pipeline = train_final_pipeline(
        X_windows, y_targets, feature_selection_results, best_alpha
    )
    
    # 7. Prediction on Test Data
    print("\n====== 7. Prediction on Test Data ======")
    print("Generating predictions for guided_testset_X...")
    predictions = predict_test_data(final_pipeline, guided_testset_X)
    
    print(f"Predictions shape: {predictions.shape}")
    
    # 8. Save predictions
    print("Saving predictions to guided_predictions.csv...")
    np.savetxt('guided_predictions.csv', predictions, delimiter=',')
    print("Predictions saved successfully!")

if __name__ == "__main__":
    main()