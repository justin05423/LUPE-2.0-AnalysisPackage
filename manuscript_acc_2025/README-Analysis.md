# Manuscript Analysis Instructions

## LUPE Analysis Notebooks
1. Download [model and dependencies and create IDE (i.e. within PyCharm)](https://github.com/justin05423/LUPE-2.0-AnalysisPackage/blob/main/README.md#installation-guide).

2. To trial LUPE-DLC model on LUPE 2.0 Video Data (see sample video for demo).
> **Note**: See [sample video data](https://upenn.box.com/s/niodmaqcfebiyd0dmnyfutq432co8k4m) for LUPE analysis demo.

3. Within GITHUB, all LUPE coding notebooks utilized for data analysis / figure generation are included. Due to limits on file size, please follow the link to access preprocess files (to run the analysis yourself) and outputs of the LUPE data (for you to look over): [BOX LINK](https://upenn.box.com/s/tmz7v7u73ymvvymi1hy7okc9brx5h7ko)  

4. After pre-processing the dataset, follow instructions on the various coding notebooks in the folder to analyze your data.

5. If you would like to analyze LUPE raw data from start to finish, feel free to download and use the dlc_csvs within the [Box Drive](https://upenn.box.com/s/tmz7v7u73ymvvymi1hy7okc9brx5h7ko).
> **Note**: To analyze data seamlessly, use our [**LUPE 2.0 APP**](https://github.com/justin05423/LUPE-2.0-App)!

---

## LUPE-Miniscope Behavior Analysis -- Behavior State Model Output
This repository contains multiple MATLAB scripts for generating, validating, and analyzing behavioral state models, neural data, and sensory responses. The code is organized into several scripts and sections. Click on the toggles below to view the details for each section.
  
  - **Overview:** This code produces the within-state behavior dynamics and simulations displayed in Figs. 2n, o and S9.
  - **Important:** Run the dataset of interest through Sections 1 and 4 of **Script1** first. **Do not clear your variables.**
  - **Data Access:** Due to limits on file size, please follow the link to access the necessary source data files: [BOX LINK](https://upenn.box.com/s/et3vkipe18l4apoaz38bazkjujgxppr2)
  - **Credits:** Sophie A. Rogers, Corder Lab, University of Pennsylvania, March 24, 2025.

## Table of Contents

- [Script1_GenerateValidateApply_BehavioralStateModel.m](#script1_generatevalidateapply_behavioralstatemodelm)
- [Script2_lickDynamicsInState.m](#script2_lickdynamicsinstatem)
- [Script3_preprocessNeuralDataAndLUPE.m](#script3_preprocessneuraldataandlupem)
- [Script4_neuralDataAndLUPE_ReproduceFigs_2_3_S10_S13_S14.m](#script4_neuraldataandlupe_reproducefigs_2_3_s10_s13_s14m)
- [Script5_sensoryPanels.m](#script5_sensorypanelsm)

---

## Script1_GenerateValidateApply_BehavioralStateModel.m
**Description:**  
This script produces the **Markov-K-Means behavioral state model**. Models were trained on data in `LUPE_statemodel_sourcedata.zip`.  
**Instructions:** To generate models de novo, run the sections in order. To reproduce Figs. 1 and S7–S9, load your data of interest along with `stateModelClassifier.m`, then skip Section 3.

<details>
  <summary><strong>Section 1:</strong> Calculate within-state behavior dynamics and plot CDF (Reproduce Fig. S9c,e; Fig. 2o)</summary>
  
  Calculates the within-state behavior dynamics and plots the cumulative distribution function.
</details>

<details>
  <summary><strong>Section 2:</strong> Plot within-state behavior dynamics as state progresses (Reproduce Fig. 2n, S9b)</summary>
  
  Creates plots to show how the behavior dynamics evolve over the state progression.
</details>

<details>
  <summary><strong>Section 3:</strong> KS test heatmaps (Reproduce Fig. 2o, S9d,f)</summary>
  
  Generates heatmaps based on Kolmogorov–Smirnov tests.
</details>

<details>
  <summary><strong>Section 4:</strong> Markov simulation (Reproduce Fig. S9b top)</summary>
  
  Runs a Markov simulation to reproduce the top portion of Fig. S9b.
</details>

<details>
  <summary><strong>Section 5:</strong> CDF and KS test for simulated State 4 behavior (Reproduce Fig. S9c top)</summary>
  
  Calculates the CDF and performs KS tests on the simulated behavior for State 4.
</details>

---

## Script2_lickDynamicsInState.m
**Description:**  
Processes lick dynamics data.

**Instructions:**  
- Loads and downsamples the data.  
- Generates sliding window transition matrices for different window lengths.  
- Chooses the optimal number of clusters using the silhouette and elbow methods.  
- Classifies states across animals and conditions.  
- Exports state data and calculates a pain scale.

<details>
  <summary><strong>Section 1:</strong> Load and downsample data</summary>
  
  Loads the raw data and applies downsampling.
</details>

<details>
  <summary><strong>Section 2:</strong> Generate sliding window transition matrices and choose k</summary>
  
  Generates sliding window transition matrices for various window lengths and determines the optimal cluster number.
</details>

<details>
  <summary><strong>Section 3:</strong> Generate transition matrices for desired window length and cluster for chosen k</summary>
  
  Produces transition matrices using the selected window length and clusters the data based on the chosen k.
</details>

<details>
  <summary><strong>Section 4:</strong> Classify states in each animal and validate over conditions</summary>
  
  Classifies the states for each animal and validates the results under different conditions.
</details>

<details>
  <summary><strong>Section 5:</strong> Calculate and export state data for each animal</summary>
  
  Calculates state-specific data and exports the results for each animal.
</details>

<details>
  <summary><strong>Section 6:</strong> Generate and calculate pain scale</summary>
  
  Computes a pain scale based on the processed data.
</details>

---

## Script3_preprocessNeuralDataAndLUPE.m
**Description:**  
Preprocesses all raw calcium imaging and behavior label files. It is run with the capsaicin source data due to the smaller experiment size. Preprocessed data for SNI and uninjured mice are provided to allow you to skip directly to Script4.

<details>
  <summary><strong>Section 1:</strong> Load data</summary>
  
  Loads raw calcium and behavior data.
</details>

<details>
  <summary><strong>Section 2:</strong> Clean calcium data, reorganize and downsample behavior data</summary>
  
  Cleans the calcium imaging data and reorganizes/downsamples the behavior data.
</details>

<details>
  <summary><strong>Section 3:</strong> Collect peri-behavioral time histograms (Reproduce Fig. 2k)</summary>
  
  Collects peri-behavioral time histograms to reproduce Fig. 2k.
</details>

---

## Script4_neuralDataAndLUPE_ReproduceFigs_2_3_S10_S13_S14.m
**Description:**  
Produces all source data and figures for neural data presented in Figs. 2, 3, S10, S13, and S14. Only one experimental group (Capsaicin, SNI, or Uninjured) can be run at a time.

<details>
  <summary><strong>Section 1:</strong> Load data</summary>
  
  Loads the neural data required for analysis.
</details>

<details>
  <summary><strong>Section 2:</strong> Generate lick-probabilities for each animal</summary>
  
  Calculates the lick-probabilities for each animal.
</details>

<details>
  <summary><strong>Section 3:</strong> Identify behavior probability-encoding principal components</summary>
  
  Identifies the principal components that encode behavior probability.
</details>

<details>
  <summary><strong>Section 4:</strong> Identify positive- and negative- behavior encoding cells (Reproduce Fig. 2k, S13k,l)</summary>
  
  Detects cells with positive and negative behavior encoding.
</details>

<details>
  <summary><strong>Section 5:</strong> Collect firing rates of behavior-encoding cells (Reproduce Fig. 2g, i, j; Fig. 3k)</summary>
  
  Collects the firing rate data of the identified behavior-encoding cells.
</details>

<details>
  <summary><strong>Section 6:</strong> Collect behavior-evoked activity and selectivity (Reproduce Fig. 2l,m; 3g,h,i,j; S10f,l; S13i,j; S14)</summary>
  
  Gathers data on behavior-evoked activity and the selectivity of these cells.
</details>

<details>
  <summary><strong>Section 7:</strong> Visualize behavior-evoked activity (heatmaps) (Reproduce Fig. S10e)</summary>
  
  Generates heatmaps to visualize the behavior-evoked neural activity.
</details>

<details>
  <summary><strong>Section 8:</strong> Fisher decoder of behaviors (Reproduce Fig. 2e, S10g,h, S13a-d)</summary>
  
  Implements a Fisher decoder to analyze behavioral data.
</details>

<details>
  <summary><strong>Section 9:</strong> Fisher decoder of states (Reproduce Fig. 2e, S10g,h, S13a-d)</summary>
  
  Implements a Fisher decoder to analyze state data.
</details>

<details>
  <summary><strong>Section 10:</strong> Representative image in Fig. 2c</summary>
  
  Generates a representative image for Fig. 2c.
</details>

---

## Script5_sensoryPanels.m
**Description:**  
Analyzes neural responses to sensory stimuli and reproduces Fig. S11. It can also be used to reproduce Fig. S12.

<details>
  <summary><strong>Section 1:</strong> Load data</summary>
  
  Loads the data related to sensory stimuli.
</details>

<details>
  <summary><strong>Section 2:</strong> Calculate and plot PSTHs</summary>
  
  Calculates and plots Peri-Stimulus Time Histograms (PSTHs).
</details>

<details>
  <summary><strong>Section 3:</strong> Identify significantly enhanced or inhibited cells</summary>
  
  Identifies cells that are significantly enhanced or inhibited in response to the stimuli.
</details>

<details>
  <summary><strong>Section 4:</strong> Make heatmaps of neurons</summary>
  
  Creates heatmaps to visualize neuronal activity.
</details>

<details>
  <summary><strong>Section 5:</strong> Calculate stimulus overlaps</summary>
  
  Calculates the overlaps between responses to different sensory stimuli.
</details>

<details>
  <summary><strong>Section 6:</strong> Decode stimuli</summary>
  
  Applies decoding algorithms to classify the sensory stimuli.
</details>
