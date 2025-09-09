# LUPE 2.0 Analysis Package

This package provides the core LUPE analysis pipeline for behavior analysis in Jupyter Notebooks.  
It serves as a resource to run pose-based behavioral classification and downstream analysis with minimal setup.  

> 💡 **Note:** If you would like to use LUPE with a graphical interface instead of coding, check out the [LUPE 2.0 App](https://github.com/justin05423/LUPE-2.0-App).  
This companion Streamlit application requires minimal coding experience and allows you to run LUPE through an interactive interface.

<p align="center">
<img src="public/logo.png" width="400">
</p>

<p align="center">
Introducing LUPE, the innovative analysis pipeline predicting pain behavior in mice. 
With our platform, you can input pose and classify mice behavior inside the LUPE Box. 
With in-depth analysis and interactive visuals, as well as downloadable CSVs for easy integration into your existing workflow, 
Try LUPE today and unlock a new level of insights into animal behavior.
</p>

![Annotated Vids](public/annotated_vids_all.gif)

---

# System Requirements
LUPE-2.0 Analysis Package requires only a standard computer with enough RAM to support the coding notebooks. 

DeepLabCut<sup>1,2</sup> and A-SOiD<sup>3</sup> used to create LUPE-2.0 model for pose estimation and behavior classification, respectively. Refer to GitHub of [DLC](https://github.com/DeepLabCut) or [A-SOiD](https://github.com/YttriLab/A-SOID) for further details. 

#### OS Requirements
- This package is supported for *Windows* and *Mac* but can be run on *Linux* computers given additional installation of require packages.

#### Python Dependencies
- For dependencies please refer to the requirements.txt file. Note that the model is created and tested on Python version 3.11.

---

# Installation Guide
#### Download the LUPE 2.0 A-SOiD Model [HERE](https://upenn.box.com/s/9rfslrvcc7m6fji8bmgktnegghyu88b0) and move the contents of the folder into this 'Model' folder.
> **Note**: Find the LUPE 2.0 DLC Model [HERE](https://upenn.box.com/s/av3i14c64rj6zls9lz6pda0it5b5q7f3) for analyzing pose estimation for LUPE video data.

#### Access the analysis scripts: In a Virutal Environment IDE
```commandline
pip install -r requirements.txt 
```
#### Access the analysis scripts: If no local IDE (not preferred)
```commandline
jupyter lab
```
#### See [README-Analysis](https://github.com/justin05423/LUPE-2.0-AnalysisPackage/blob/main/manuscript_acc_2025/README-Analysis.md) for instructions to run and reproduce some of the analyses from the  [PRE-PRINT manuscript](https://github.com/justin05423/LUPE-2.0-AnalysisPackage/tree/main/manuscript_acc_2025)<sup>4</sup> data. 

---

# Physical System Build
For an overview on building the LUPE 2.0 System, check out the [Build](https://github.com/justin05423/LUPE-2.0-App/wiki/LUPE-2.0-Build-%F0%9F%9B%A0%EF%B8%8F-%F0%9F%A7%B0).

---

# Contacting

#### Project Funding
Collaboration between [Corder Lab](https://corderlab.com/) at University of Pennsylvania and 
[Yttri Lab](https://labs.bio.cmu.edu/yttri/) from Carnegie Mellon. 

#### Contributors
Justin James (Corder Lab) actively develops and maintains this repository/cloud resource.

Other contributors include Alexander Hsu (Yttri Lab).


---

# License
LUPE is released under a Clear BSD License and is intended for research/academic use only.

# References
1. [Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018 Sep;21(9):1281-1289. doi: 10.1038/s41593-018-0209-y. Epub 2018 Aug 20. PubMed PMID: 30127430.](https://www.nature.com/articles/s41593-018-0209-y)
2. [Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using DeepLabCut for 3D markerless pose estimation across species and behaviors. Nat Protoc. 2019 Jul;14(7):2152-2176. doi: 10.1038/s41596-019-0176-0. Epub 2019 Jun 21. PubMed PMID: 31227823.](https://doi.org/10.1038/s41596-019-0176-0)
3. [Tillmann JF, Hsu AI, Schwarz MK, Yttri EA. A-SOiD, an active-learning platform for expert-guided, data-efficient discovery of behavior. Nat Methods. 2024 Apr;21(4):703-711. doi: 10.1038/s41592-024-02200-1. Epub 2024 Feb 21. PMID: 38383746.](https://www.nature.com/articles/s41592-024-02200-1)
4. [James, J. G., McCall, N. M., Hsu, A. I., Oswell, C. S., Salimando, G. J., Mahmood, M., ... & Corder, G. (2024). Mimicking opioid analgesia in cortical pain circuits. bioRxiv.](https://www.biorxiv.org/content/10.1101/2024.04.26.591113v1)
