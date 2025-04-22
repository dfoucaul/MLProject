# EMG-Based Hand Pose Prediction: Updated Project Documentation

## Project Overview

This project implements a machine learning pipeline to predict hand joint angles from surface electromyography (sEMG) signals. sEMG measures the electrical activity of muscles, which we use to predict continuous joint angles that define hand positioning. The pipeline follows these main steps:

1. Signal filtering (optional)
2. Windowing with overlap
3. Feature extraction 
4. Cross-validation strategy
5. Feature selection methods comparison
6. Model training and comparison
7. Prediction on test data

## 1. Dataset Description

The dataset consists of EMG signals and corresponding hand joint angles:

### Guided Gestures Dataset
- Contains predefined hand postures (five poses) across 5 sessions
- EMG data: `guided_dataset_X.npy` - shape (5, 8, 230000) for (session, electrode, time)
- Joint angle data: `guided_dataset_y.npy` - shape (5, 51, 230000) for (session, joint angle, time)
- Test data: `guided_testset_X.npy` - shape (5, 332, 8, 500) for (session, window, electrode, time)

The EMG data was recorded at 1024 Hz (1024 samples per second) using 8 wireless electrodes placed on the participant's forearm. The joint angle data consists of 51 continuous values representing 3 rotation angles for each of the 17 joints in the hand model.

## 2. Signal Filtering

EMG signals contain various noise sources that can affect model performance. We apply two filters:

### Bandpass Filter (20-450 Hz)
- Removes low-frequency motion artifacts (<20 Hz)
- Removes high-frequency noise (>450 Hz)
- We use a Butterworth filter, which provides a maximally flat frequency response in the passband

### Notch Filter (50 Hz)
- Removes power line interference (electrical noise from power outlets)
- A typical noise source in biomedical signals
- Set at 50 Hz (European standard) with a narrow bandwidth

The filtering process significantly improves signal quality by removing unwanted noise components while preserving the muscle activity information. As shown in the signal visualization, the filtered signal maintains the essential characteristics of the original signal while reducing noise.

## 3. Window Creation with Overlap

EMG signals are continuous, but machine learning models require fixed-size inputs. To address this:

1. We segment the continuous signals into fixed-size windows (500 samples each)
2. We implement 50% overlap between adjacent windows to:
   - Increase the number of training examples
   - Ensure smooth transitions between predictions
   - Capture transient patterns in the signal

With 50% overlap and window size of 500 samples, our implementation created windows as follows:
- Window 1: Session 1, Samples 1-500
- Window 2: Session 1, Samples 251-750
- Window 3: Session 1, Samples 501-1000
- Window 4: Session 1, Samples 751-1250
- Window 5: Session 1, Samples 1001-1500

This approach created a total of 4,595 windows across all 5 sessions. The step size was calculated as 250 samples (50% of the 500-sample window size), resulting in precisely half of each window overlapping with the next.

## 4. Feature Extraction

Raw EMG signals are high-dimensional and noisy. We extract meaningful time-domain features that characterize muscle activity patterns:

### Mean Absolute Value (MAV)
- The average of absolute EMG amplitude
- Represents overall muscle activation level
- Formula: `MAV = (1/K) * ∑|xᵢ|`

### Root Mean Square (RMS)
- Square root of mean squared EMG amplitude
- Related to signal power; robust to noise
- Formula: `RMS = √[(1/K) * ∑xᵢ²]`

### Variance (VAR)
- Measures signal dispersion
- Useful for distinguishing different activity levels
- Formula: `VAR = (1/(K-1)) * ∑(xᵢ - x̄)²`

### Standard Deviation (STD)
- Square root of variance
- Another measure of signal variability
- Formula: `STD = √[(1/(K-1)) * ∑(xᵢ - x̄)²]`

### Zero Crossing (ZC)
- Count of signal crossing zero amplitude
- Related to frequency information
- Formula: `ZC = ∑[1 if xᵢ*xᵢ₊₁ < 0]`

### Myopulse Percentage Rate (MPR)
- Percentage of samples exceeding a threshold
- Measures signal intensity relative to noise level
- Formula: `MPR = (1/K) * ∑[1 if |xᵢ| > σ]`

### Feature Set Options

Our implementation allows for selecting from three feature set options:

- **'all'**: Extracts all 6 features from each electrode (48 features total)
- **'basic'**: Extracts only MAV, RMS, and VAR (24 features total)
- **'top10'**: Extracts only the 10 most important features based on feature importance analysis

## 5. Cross-Validation Strategy

To accurately evaluate model performance and ensure generalizability, we implemented a leave-one-session-out cross-validation strategy:

1. The dataset consists of 5 recording sessions
2. For each fold of cross-validation:
   - One session is used as the test set
   - The remaining 4 sessions are used for training
   - This process is repeated 5 times, using each session once as a test set

From our execution logs, each split had:
- 3,676 training windows (from 4 sessions)
- 919 test windows (from the left-out session)

This approach has several benefits:
- It ensures windows from the same recording session never appear in both training and test sets
- It tests the model's ability to generalize to new recording sessions
- It provides a realistic evaluation by accounting for session-to-session variability in EMG signals

## 6. Feature Selection Methods

We implemented and compared two feature selection methods to determine the optimal feature subset:

### Recursive Feature Elimination with Cross-Validation (RFECV)

RFECV is a recursive process that:
1. Trains the model with all features
2. Computes feature importance/coefficients
3. Removes the least important feature(s)
4. Repeats until finding the optimal number of features

Our results showed:
- **Ridge Regression**: RFECV selected all 48 features as optimal (RMSE = 7.86)
- **Random Forest**: RFECV selected 40 features as optimal (RMSE = 5.49)

The RFECV graphs demonstrated that:
- Ridge Regression performance improved gradually as features were added
- Random Forest showed steep initial improvements, with diminishing returns after ~20 features

### Feature Importance Thresholding

This method:
1. Trains the model with all features
2. Ranks features by their importance scores
3. Tests different thresholds, keeping only the top N most important features
4. Selects the threshold that minimizes RMSE

Our results showed:
- **Ridge Regression**: Selected all 48 features as optimal (RMSE = 7.86)
- **Random Forest**: Selected just 20 features as optimal (RMSE = 5.47)

This method confirmed that Random Forest can achieve optimal performance with fewer features, while Ridge Regression benefits from using all available features.

## 7. Regression Models

We compared different regression models to predict joint angles from EMG features:

### Ridge Regression
Ridge regression is a linear regression method with L2 regularization. It adds a penalty term to the ordinary least squares objective function based on the squared magnitude of the coefficients.

**How Ridge Regression Works:**
1. It fits a linear model: y = Xβ + ε
2. It minimizes the objective function: ||y - Xβ||² + α||β||²
   - The first term is the ordinary least squares error
   - The second term is the L2 penalty (sum of squared coefficients)
   - α is the regularization strength (higher α means more regularization)

**Feature Importance Analysis for Ridge:**
The top 5 features by importance (absolute coefficient values) were:
1. E2_MAV (Mean Absolute Value of Electrode 2): 9.60
2. E7_VAR (Variance of Electrode 7): 4.84
3. E1_MAV (Mean Absolute Value of Electrode 1): 4.08
4. E7_MAV (Mean Absolute Value of Electrode 7): 3.74
5. E8_MAV (Mean Absolute Value of Electrode 8): 2.88

### Random Forest Regression
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees.

**How Random Forest Regression Works:**
1. **Bootstrap Sampling**: It randomly samples the training data with replacement to create multiple subset datasets
2. **Feature Randomization**: For each tree, it considers only a random subset of features at each splitting point
3. **Decision Tree Building**: It builds a decision tree for each bootstrap sample
4. **Aggregation**: It averages the predictions from all trees for regression tasks

**Feature Importance Analysis for Random Forest:**
The top 5 features by importance were:
1. E4_MAV (Mean Absolute Value of Electrode 4): 0.209
2. E7_STD (Standard Deviation of Electrode 7): 0.119
3. E7_RMS (Root Mean Square of Electrode 7): 0.108
4. E7_VAR (Variance of Electrode 7): 0.088
5. E1_MAV (Mean Absolute Value of Electrode 1): 0.060

## 8. Model Evaluation and Results

We evaluated model performance using the Root Mean Squared Error (RMSE):

RMSE = √[(1/(N·D)) · ∑∑(y_i,d - ŷ_i,d)²]

Where:
- N is the number of test samples
- D is the number of joint angles (51)
- y_i,d is the true value for sample i and joint angle d
- ŷ_i,d is the predicted value

Based on our feature selection comparison and cross-validation results:

1. **Ridge Regression with RFECV (48 features)**:
   - Average RMSE: 7.8637
   
2. **Ridge Regression with Importance Thresholding (48 features)**:
   - Average RMSE: 7.8637

3. **Random Forest with RFECV (40 features)**:
   - Average RMSE: 5.4902

4. **Random Forest with Importance Thresholding (20 features)**:
   - Average RMSE: 5.4729 (Best performing model)

The Random Forest model with just 20 carefully selected features demonstrated significantly better performance than both Ridge Regression approaches, confirming that the relationship between EMG features and joint angles is non-linear and benefits from the Random Forest's ability to capture complex patterns.

## 9. Feature Importance and Selection Results

To identify the most important features, we analyzed the feature importance values from both models:

### Key Findings:

1. **Most Important Electrodes**:
   - Electrode 7 features appear frequently in top rankings for both models
   - Electrode 4 features are particularly important for Random Forest
   - Electrodes 1 and 2 also contribute significantly

2. **Most Important Feature Types**:
   - MAV (Mean Absolute Value) is consistently the most important feature type
   - STD, RMS, and VAR also show high importance
   - ZC (Zero Crossing) appears important for electrode 4 in Random Forest

3. **Optimal Feature Set**:
   - For Ridge Regression: All 48 features provide optimal performance
   - For Random Forest: Just 20 features provide optimal performance
   - The 20 selected features for Random Forest include the ones with highest importance scores

This feature selection analysis allowed us to reduce dimensionality by 58% (from 48 to 20 features) while actually improving performance for the Random Forest model.

## 10. Test Data Prediction

After selecting Random Forest with importance thresholding (20 features) as the best-performing model, we trained it on the complete dataset (all 4,595 windows) and generated predictions for the guided_testset_X.npy file:

- Test data consisted of 1,660 windows (5 sessions × 332 windows per session)
- Prediction process took approximately 5.25 seconds
- Output shape was (1660, 51) matching the expected 51 joint angles per window
- Predictions were saved to guided_predictions.csv

## 11. Computational Considerations

The execution times for the different stages of our pipeline were:
- Window creation: ~5.66 seconds
- Feature extraction: Fast (milliseconds)
- Feature selection with RFECV:
  - Ridge: ~15 seconds
  - Random Forest: ~1040 seconds (17.3 minutes)
- Feature selection with importance thresholding:
  - Ridge: ~0.04 seconds (feature importance) + ~10 seconds (threshold testing)
  - Random Forest: ~33 seconds (feature importance) + ~15 seconds (threshold testing)
- Final model training (with 20 features): ~28 seconds
- Test data prediction: ~5.25 seconds

The Random Forest feature selection with RFECV was computationally demanding, while importance thresholding offered a much more efficient approach with comparable or better results.

## 12. Conclusion

Our EMG-based hand pose prediction pipeline successfully predicts continuous joint angles from EMG signals. The key findings include:

1. **Signal filtering** effectively removes noise while preserving important signal characteristics
2. **Windowing with 50% overlap** creates an effective dataset of 4,595 windows
3. **Time-domain features** effectively capture EMG patterns, with MAV, RMS, STD, and VAR being most informative
4. **Feature selection through importance thresholding** identifies the optimal subset of features while reducing dimensionality
5. **Random Forest regression** significantly outperforms Ridge Regression, achieving RMSE of 5.47 vs 7.86
6. **Dimensionality reduction** from 48 to 20 features improves both performance and computational efficiency

The best model configuration (Random Forest with 20 features selected through importance thresholding) demonstrates excellent ability to generalize across sessions, which is crucial for real-world applications like prosthetic control or human-computer interfaces.

Most importantly, the analysis confirms that key electrodes (particularly 4 and 7) and specific feature types (especially MAV) capture the most relevant information for hand pose prediction, providing valuable insights for future EMG-based control system design.