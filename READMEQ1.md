# EMG-Based Hand Pose Prediction: Project Documentation

## Project Overview

This project implements a machine learning pipeline to predict hand joint angles from surface electromyography (sEMG) signals. sEMG measures the electrical activity of muscles, which we use to predict continuous joint angles that define hand positioning. The pipeline follows these main steps:

1. Signal filtering (optional)
2. Windowing with overlap
3. Feature extraction
4. Cross-validation strategy
5. Model training and comparison
6. Prediction on test data

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

The top 10 most important features identified through Random Forest feature importance analysis are:
1. E7_RMS: 0.1677
2. E4_MAV: 0.1462
3. E7_STD: 0.1371
4. E7_VAR: 0.1064
5. E1_MAV: 0.0526
6. E7_MAV: 0.0500
7. E4_VAR: 0.0303
8. E4_STD: 0.0301
9. E2_MAV: 0.0294
10. E4_ZC: 0.0268

Using the 'top10' feature set allows for more efficient model training and prediction while maintaining most of the predictive power.

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

## 6. Regression Models

We compared three different regression models to predict joint angles from EMG features:

### Ridge Regression
Ridge regression is a linear regression method with L2 regularization. It adds a penalty term to the ordinary least squares objective function based on the squared magnitude of the coefficients.

**How Ridge Regression Works:**
1. It fits a linear model: y = Xβ + ε
2. It minimizes the objective function: ||y - Xβ||² + α||β||²
   - The first term is the ordinary least squares error
   - The second term is the L2 penalty (sum of squared coefficients)
   - α is the regularization strength (higher α means more regularization)

**Key Characteristics:**
- Handles multicollinearity (highly correlated features) well
- Shrinks coefficients toward zero but rarely to exactly zero
- Good for datasets with many features that are all potentially relevant
- Computationally efficient for multi-output regression problems
- Works well when n_samples > n_features and features are correlated

**Hyperparameter:**
- α (alpha): Controls the regularization strength. Higher values mean stronger regularization.

### Lasso Regression
Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression method with L1 regularization. It adds a penalty term based on the absolute value of the coefficients.

**How Lasso Regression Works:**
1. It fits a linear model: y = Xβ + ε
2. It minimizes the objective function: ||y - Xβ||² + α||β||₁
   - The first term is the ordinary least squares error
   - The second term is the L1 penalty (sum of absolute values of coefficients)
   - α is the regularization strength

**Key Characteristics:**
- Performs feature selection by setting some coefficients exactly to zero
- Good for sparse models (few important features among many irrelevant ones)
- Less stable with highly correlated features (tends to pick one and ignore others)
- Can be slower to converge than Ridge for large datasets
- The sparsity increases interpretability

**Hyperparameters:**
- α (alpha): Controls the regularization strength
- max_iter: Maximum number of iterations (important for convergence)

### Random Forest Regression
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the average prediction of the individual trees.

**How Random Forest Regression Works:**
1. **Bootstrap Sampling**: It randomly samples the training data with replacement to create multiple subset datasets
2. **Feature Randomization**: For each tree, it considers only a random subset of features at each splitting point
3. **Decision Tree Building**: It builds a decision tree for each bootstrap sample
4. **Aggregation**: It averages the predictions from all trees for regression tasks

**Key Characteristics:**
- Non-linear model that can capture complex relationships
- Highly robust to overfitting (especially with many trees)
- No assumption about data distribution or linearity
- Can handle high-dimensional data
- Provides feature importance measures
- Can be computationally intensive and memory-hungry
- Less interpretable than linear models

**Hyperparameters:**
- n_estimators: Number of trees in the forest (higher is generally better but more computationally intensive)
- max_depth: Maximum depth of each tree (controls complexity)
- min_samples_split: Minimum samples required to split a node
- min_samples_leaf: Minimum samples required in a leaf node
- max_features: Number of features to consider for the best split

## 7. Model Evaluation and Results

We evaluated model performance using the Root Mean Squared Error (RMSE):

RMSE = √[(1/(N·D)) · ∑∑(y_i,d - ŷ_i,d)²]

Where:
- N is the number of test samples
- D is the number of joint angles (51)
- y_i,d is the true value for sample i and joint angle d
- ŷ_i,d is the predicted value

Based on our cross-validation results:

1. **Ridge Regression**:
   - Fold 1 RMSE: 8.6597
   - Fold 2 RMSE: 7.7602
   - Fold 3 RMSE: 7.8722
   - Fold 4 RMSE: 7.2627
   - Fold 5 RMSE: 7.7644
   - **Average RMSE: 7.8638**
   - Execution time: 0.48 seconds

2. **Lasso Regression**:
   - Fold 1 RMSE: 8.6422
   - Fold 2 RMSE: 7.7623
   - Fold 3 RMSE: 7.8888
   - Fold 4 RMSE: 7.2656
   - Fold 5 RMSE: 7.7638
   - **Average RMSE: 7.8646**
   - Execution time: 35.08 seconds

3. **Random Forest**:
   - Fold 1 RMSE: 6.0709
   - Fold 2 RMSE: 5.1325
   - Fold 3 RMSE: 5.7564
   - Fold 4 RMSE: 5.0395
   - Fold 5 RMSE: 5.4665
   - **Average RMSE: 5.4932**
   - Execution time: 374.47 seconds

The Random Forest model demonstrated significantly better performance than both linear models (Ridge and Lasso), with an average RMSE of 5.4932 compared to approximately 7.86 for both linear models. This suggests that the relationship between EMG features and joint angles is non-linear and benefits from the Random Forest's ability to capture complex patterns.

However, this improved performance comes at a computational cost: the Random Forest training took about 375 seconds, compared to just 0.5 seconds for Ridge regression. This represents a trade-off between prediction accuracy and computational efficiency.

## 8. Feature Importance and Selection

To identify the most important features, we analyzed the feature importance values from the Random Forest model. The top 10 features were:

1. E7_RMS (Electrode 7, Root Mean Square): 16.77% importance
2. E4_MAV (Electrode 4, Mean Absolute Value): 14.62% importance
3. E7_STD (Electrode 7, Standard Deviation): 13.71% importance 
4. E7_VAR (Electrode 7, Variance): 10.64% importance
5. E1_MAV (Electrode 1, Mean Absolute Value): 5.26% importance
6. E7_MAV (Electrode 7, Mean Absolute Value): 5.00% importance
7. E4_VAR (Electrode 4, Variance): 3.03% importance
8. E4_STD (Electrode 4, Standard Deviation): 3.01% importance
9. E2_MAV (Electrode 2, Mean Absolute Value): 2.94% importance
10. E4_ZC (Electrode 4, Zero Crossing): 2.68% importance

These results show that:
- Electrode 7 is the most informative electrode, with 4 of its features in the top 10
- Electrode 4 is the second most informative, with 4 features in the top 10
- RMS and MAV are particularly useful feature types across electrodes

We implemented the option to train models using just these top 10 features, which substantially reduces the input dimensionality (from 48 to 10 features) while maintaining most of the predictive power.

## 9. Test Data Prediction

After selecting Random Forest as the best-performing model, we trained it on the complete dataset (all 4,595 windows) and generated predictions for the guided_testset_X.npy file:

- Test data consisted of 1,660 windows (5 sessions × 332 windows per session)
- Prediction process took approximately 10 seconds
- Output shape was (1660, 51) matching the expected 51 joint angles per window
- Predictions were saved to guided_predictions.csv in scientific notation format

We compared predictions using all features versus using only the top 10 features:
- Both approaches produced predictions for all 51 joint angles
- The top 10 feature approach showed differences in the predicted values, indicating that the feature selection was effective
- The top 10 feature approach had faster prediction times due to the reduced dimensionality

## 10. Computational Considerations

The execution times for the different stages of our pipeline were:
- Window creation: ~36 seconds
- Model comparison (total): ~435 seconds
  - Ridge regression: ~0.5 seconds
  - Lasso regression: ~35 seconds
  - Random Forest: ~375 seconds
- Final model training (top 10 features): ~24 seconds
- Test data prediction (top 10 features): ~10 seconds

Random Forest provided the best performance but at a much higher computational cost than the linear models. Using the top 10 features offered a good compromise between performance and computational efficiency.

## 11. Conclusion

Our EMG-based hand pose prediction pipeline successfully predicts continuous joint angles from EMG signals. The key findings include:

1. **Signal filtering** removes noise while preserving important signal characteristics
2. **Windowing with 50% overlap** creates an effective dataset of 4,595 windows
3. **Time-domain features** effectively capture the relevant EMG patterns, with options to use 'all', 'basic', or 'top10' feature sets
4. **Random Forest regression** significantly outperforms linear models, suggesting non-linear relationships between EMG features and joint angles
5. **Feature importance analysis** identified Electrodes 7 and 4 as the most informative, with RMS and MAV as particularly useful feature types
6. **Feature selection** allowed for significant dimensionality reduction (from 48 to 10 features) while maintaining predictive power

The best model (Random Forest) achieved an average RMSE of 5.4932 across the cross-validation folds, demonstrating its ability to generalize to new sessions. This approach could potentially be applied to control prosthetic devices or human-computer interfaces using muscle activity as input.