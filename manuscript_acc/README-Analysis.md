# Analysis Instructions

### LUPE Analysis Notebooks
1. [Download model and dependencies and create IDE (i.e. within PyCharm).](https://github.com/justin05423/LUPE-2.0-AnalysisPackage/blob/main/README.md#installation-guide)

2. Run LUPE-DLC analysis on LUPE 2.0 Video Data (see sample video for demo).

3. Beging LUPE-ASOiD analysis:

    1. Move files from ['template_copy-files-to-notebooks'](https://github.com/justin05423/LUPE-2.0-AnalysisPackage/tree/main/notebooks/template_copy-files-to-notebooks) to your local notebooks folder.
    2. Begin with pre-process files: a) 1_preprocess_data, b) 2_preprocess_get_features, c) 3_preprocess_get_behaviors
  
4. After pre-processing the dataset, follow instructions on the various coding notebooks in the folder to analyze your data.

---

### MATLAB - LUPE-Miniscope Behavior Analysis
<details closed>
    <summary>Section 1: Read data CSVs into the structure oswell</summary>
   oswell.animals(a).sessions(s) – contains behavior, calcium, cell properties, and timing information in order of session.
    The fields in sessions include:
    behavior – raw, frame-by-frame LUPE output (categorical behavior classification)
    calcium – deconvolved dF/F data
    offset – seconds between calcium recording start and behavior recording start
    props – Inscopix software-created spatial ROI data 
    Other variables defined in this section:
    nAnimals – number of animals
    nSesh – number of sessions
    dt – frame rate of calcium videos
    dtB – frame rate of behavior videos
    Behavior & session names and colors for plots
</details>

<details closed>
    <summary>Section 2: Clean data (data for Fig. 2c)</summary>
   Here, behavioral data is downsampled from dtB to dt and converted into a binary matrix, behMat, indicating frames (rows) of engagement (values) for each behavior (columns).
    NaNs are replaced with zeros, and calcium data is z-scored.
    Calcium data is stored in a cell activities of dimensions nSesh x nAnimals.
    Behavior data is stored in a cell behMats of dimensions nSesh x 2 x nAnimals. Column 1 contains binary matrix, and column 2 contains the downsampled categorical array.
    Size of data is stored.
</details>

<details closed>
    <summary>Section 3: Makes representative figure of neuron coefficients along principal components of neural activity</summary>
</details>

<details closed>
    <summary>Section 4: Trains binomial GLMs on first 20PCs of ACC activity (data for Fig. 2g)</summary>
    For non-shuffled (shuff==0) and temporally shuffled (shuff=1) data, establish Bonferroni-corrected p-value threshold alpha, an empty cell for neuron coefficients coeffs, and number of principal components to use for models nDims.
    Then, for each animal and session, render calcium data nonnegative, and take its principal components. Store coefficients. 
    For each behavior, skip if no behavior bouts, then loop through principal components and predict each binary behavior trace from all used principal components with binomial GLM. Calculate auROC. Save coefficients and p-values. Save indices of significant positively and negatively predictive PCs in a cell idx of size: nAnimals x nSessions x nBehaviors x 3 (for all, positively, and negatively predictive PCs respectively). Save significant coefficients in a tensor of nAnimals x nSessions x nBehaviors x 2 x 2. The fourth dimension represents real vs. shuffled data, while the fifth dimension represents positively and negatively predictive PCs respectively. Store fractions of predictive PCs in a matrix fracs of the same dimensions.
</details>

<details closed>
    <summary>Section 5: Collect highly weighted cells along significant PCs (data for Fig. 2h)</summary>
   For each animal, session, behavior, and significantly behavior-predictive PCs, take the cells with PC coefficients of a magnitude greater than 2 z-score. Store in the cell encodingCells of size nAnimals x nSesh x nBehaviors.
</details>

<details closed>
    <summary>Section 6: Get peri-behavior time histograms</summary>
   For each animal, session, and behavior, take the 20 frames before and after each bout onset for all neurons and the behavior itself.
    Store PBTH neuron tensor of nBouts x nCells x nFrames in cell psthStore of size nAnimals x nSesh x nBehaviors. Store PBTH behavior matrix of nBouts x Frames in psthBehStore size nAnimals x nSesh x nBehaviors.
</details>

<details closed>
    <summary>Section 7: Create behavior PBTHs (reproduction, Fig. 2k)</summary>
   For each behavior (right lick with almost no occurrences excluded), plot the average PBTH over all bouts in each session pooled across animals. Store mean area under the curve, standard deviation over bouts, and number of bouts for 0-1 seconds post-onset and 1-2 seconds post-onset in the tensor aucsBeh of size nSessions x nBehaviors x 6.
</details>

<details closed>
    <summary>Section 8: Create behavior PBTHs (reproduction, Fig. 2l,m)</summary>
   For each behavior and session, plot the average neural PBTH of positive and negative behavior-encoding neurons around behavior-onset, z-scored to baseline. Store positive behavior encoding neurons the cell encCell of size nAnimals x nSessions x nBehaviors x 2. Store the prevalence of these neurons in the tensor ratios of the same size.
    Store mean, standard deviation, and sample size of neural activity in lick-encoding neurons 0-1 and 1-2 seconds after behavior-onset in positive and negative lick-encoding neurons in the tensor aucsActs of the same structure as aucsBeh, with additional fourth-dimensional columns for negative lick-encoding neurons.
</details>

<details closed>
    <summary>Section 9: Make heatmaps of encoding neurons for pre-set behavior around every behavior in sessions of interest (reproduction, Fig. 2j)</summary>
   Set behavior of interest  behOfInt (1 = still, 2 = walking, 3 = rearing, 4 = grooming, 5 = left-lick) to generate sorted heatmaps of each set of behavior-encoding neurons around onset of that behavior.
</details>

<details closed>
    <summary>Section 10: Store means z-score of activities to generate behavioral tuning curves (data for Fig. 2m&o)</summary>
  Store z-scores of neural activity in behavior-encoding neurons during each behavior in the cell storeActs of size nAnimals x nSesh x nBehaviors x nBehaviors x nDirections (positive vs. negative).
</details>

<details closed>
    <summary>Section 11: Calculate d’ for a given behavior compared to each other (data for Fig. 2p&q)</summary>
Using the data in storeActs, calculate preference of each neuron for each behavior compared to each other. Store d’ values in cell dPrimes of size nBehaviors x nSesh x directions.
</details>

<details closed>
    <summary>Section 12: 100 fold cross validated fisher decoder (reproduces Fig. 2e)</summary>
    For each animal and session, randomly subsample data (shuffled and unshuffled) to represent same number of samples from each behavior and train Fisher decoder to discriminate each behavior. Train on 50% data test on 50%. Store confusion matrices in tensor conMat of size nAnimals x nSesh x nCrossValidations x nBehaviors x nBehaviors x 2 (shuffled and unshuffled). Run ttests between shuffled and unshuffled data.
</details>

<details closed>
    <summary>Section 13: Make the trace plot in Fig. 2h</summary>
</details>

<details closed>
    <summary>Section 14: Make spatial scatter plots (reproduces Fig. 2i)</summary>
      Maps ROIs, colors positive behavior-encoding cells yellow and negative behavior-encoding cells blue.
</details>

