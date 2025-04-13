# EMG-Based Hand Pose Prediction Project

## Project Overview
This project focuses on developing regression models to predict hand movements using surface electromyography (sEMG) signals from the forearm. We're working with two datasets:
1. **Guided gestures**: Data from 5 specific hand poses performed repeatedly
2. **Free gestures**: Data from natural hand movements while interacting with virtual objects

The goal is to accurately predict 51 joint angles representing the full hand pose using the electrical signals from 8 EMG sensors.

## Current Progress

### ✅ Phase 1: Setup and Understanding
- [x] Downloaded the dataset
- [x] Loaded the NumPy arrays
- [x] Visualized sample EMG signals across all 8 electrodes
- [x] Visualized sample joint angles for different fingers
- [x] Analyzed the relationship between EMG signals and joint movements
- [x] Created initial visualization of the hand skeleton based on joint angles
- [x] Understood the data dimensions:
  - guided_X shape: (5, 8, 230000) - (session, electrode, time)
  - guided_y shape: (5, 51, 230000) - (session, joint_angle, time)
  - guided_test_X shape: (5, 332, 8, 500) - (session, window, electrode, time)

### ✅ Phase 2: Data Preparation
- [x] Implemented window extraction with overlap function
- [ ] Tested different overlap percentages (75% overlap chosen as optimal)
- [ ] Created corresponding windows for joint angles
- [ ] Implemented time-domain feature extraction
- [ ] Design cross-validation strategy

## EMG Feature Analysis Results

We've successfully implemented and analyzed six time-domain features for the EMG signals:

1. **Mean Absolute Value (MAV)**: 
   - Measures average intensity of muscle activation
   - Electrode 2 shows highest activity (~120 units)
   - Electrodes 3 and 4 show moderate activity (~80 and ~70 units)

2. **Root Mean Square (RMS)**:
   - Another measure of signal strength
   - Similar pattern to MAV but with higher values
   - Electrode 2 dominates with ~145 units

3. **Variance (VAR)**:
   - Measures signal variability
   - Extreme differences between electrodes (from ~500 to ~21,000)
   - Electrode 2 shows dramatically higher variance than all others

4. **Standard Deviation (STD)**:
   - Square root of variance
   - Pattern matches RMS values
   - Provides similar information to VAR but on more manageable scale

5. **Zero Crossing (ZC)**:
   - Counts how often the signal crosses zero
   - Indicates frequency content of the signal
   - Electrode 7 has highest count (~160) despite low amplitude
   - Captures different information than amplitude-based features

6. **Myopulse Percentage Rate (MPR)**:
   - Measures how often signal exceeds a threshold
   - Values range from 0.20 to 0.32 across electrodes
   - Electrodes 1 and 4 show highest values
   - Identifies "spiky" or "bursty" signal patterns

**Key insights from feature analysis:**
- Different features capture complementary information about muscle activity
- Electrode 2 appears to be capturing the primary muscle for many movements
- The high variance in features across electrodes suggests feature normalization will be important
- All implemented features show clear patterns, confirming good signal quality

## Upcoming Tasks

### Phase 3: Baseline Model
- [ ] Create custom feature extraction class
  - [x] Implemented time-domain features (MAV, RMS, Variance, STD, ZC, MPR)
  - [ ] Integrate with scikit-learn via BaseEstimator and TransformerMixin
- [ ] Implement at least two regression models
- [ ] Compare model performance using cross-validation
- [ ] Analyze feature importance
- [ ] Perform feature selection
- [ ] Build a complete scikit-learn Pipeline
- [ ] Document findings with visualizations

### Phase 4: Advanced Approaches
- [ ] Implement one of:
  - [ ] Covariance matrices approach with PyRiemann
  - [ ] Neural network approach with PyTorch
- [ ] Compare with baseline models
- [ ] Implement ensemble methods:
  - [ ] Simple averaging ensemble
  - [ ] Meta-learner (stacking) ensemble
- [ ] Compare ensemble performance against individual models
- [ ] Analyze model contributions to ensembles
- [ ] Discuss bias-variance tradeoff

### Phase 5: Final Submission
- [ ] Select best models for both datasets
- [ ] Generate predictions for test sets
- [ ] Combine predictions and save as CSV
- [ ] Submit to competition platform
- [ ] (Optional) Implement hand pose visualization tool
- [ ] Finalize documentation

### Phase 6: Submission Preparation
- [ ] Final code review
- [ ] Ensure all objectives are completed
- [ ] Verify documentation quality
- [ ] Prepare for GitHub repository submission

## Next Steps

Our immediate focus will be on:
1. Designing an appropriate cross-validation strategy
2. Creating the custom feature extraction class using scikit-learn's BaseEstimator and TransformerMixin
3. Implementing baseline regression models
4. Analyzing feature importance and performing feature selection

## Technical Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- (Later) PyRiemann or PyTorch

## Deadline