# Functional Specialization and Integration of the Developing Orbitofrontal Cortex in Early Childhood

**Contact:** yuejuan1016@shanghaitech.edu.cn

---

## 1. Data Processing

- **FSL** v6.0.7 — https://fsl.fmrib.ox.ac.uk/fsl/docs/
- **ANTs** v2.5.3 — https://github.com/ANTsX/ANTs/releases

## 2. Normative Modelling

- **R** v4.2.0 — https://www.r-project.org
- **mgcv (GAMM)** — https://cran.r-project.org/web/packages/mgcv/index.html

## 3. Code

### 3.1 Spatiotemporal OFC parcellation

Each OFC voxel is characterised by its functional connectivity to the rest of the brain across early development, and hierarchical clustering on these spatiotemporal profiles yields functional subregions. The pipeline runs in three steps: voxel-wise Fisher-z functional connectivity (`01_fc_voxelwise.py`), GAMM fitting of each connectivity trajectory across age (`02_gamm_trajectory.py`), and Ward clustering on the resulting profiles to produce the subregion parcellation (`03_clustering.py`).

### 3.2 Functional characterisation

Developmental connectivity trajectories between each OFC subregion and the rest of the brain were modelled and classified into six canonical patterns. This stage runs in two steps: whole-brain GAMM modelling of each subregion's connectivity while controlling for the others (`01_subregion_fc_gamm.py`), and a winner-takes-all assignment of every brain voxel to the subregion it connects to most strongly (`02_winner_takes_all.py`).

## 4. Data

- **`Spatiotemporal_conn_profile.npy`** — the spatiotemporal connectivity matrix used for parcellation. Each row is an OFC voxel; each column concatenates that voxel's functional connectivity to all brain regions across the full developmental age range. Together, these features capture both *where* a voxel connects (spatial connectivity architecture) and *when* those connections mature (developmental trajectory), and form the input to the clustering step.
- **`OFC_Parcellation.nii.gz`** — the resulting six-subregion OFC parcellation.

## 5. Results

`subregion1` through `subregion6` correspond to the six labels in `OFC_Parcellation.nii.gz`. Each folder contains the fitted whole-brain functional connectivity maps of that subregion at every modelled age.

## 6. Neurosynth

Neurosynth meta-analytic maps were downloaded and processed using **NiMARE** — https://nimare.readthedocs.io/en/0.0.1/auto_examples/01_datasets/download_neurosynth.html
