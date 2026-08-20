# Getting started

This page walks through the minimal steps required to run the RegVelo pipeline end to end: preprocessing,
model training, cell fate prediction, and *in silico* transcription factor (TF) perturbation.

To follow along you need:

- An {class}`~anndata.AnnData` object with spliced and unspliced counts stored in
  `adata.layers["spliced"]` and `adata.layers["unspliced"]`. These layers can be generated from a `.loom`
  file with [velocyto](https://velocyto.org/) or [STARsolo](https://cumulus.readthedocs.io/en/latest/starsolo.html).
- A prior gene regulatory network (GRN) as a binary `pandas.DataFrame` or `numpy.ndarray`, with **rows as
  regulators (TFs) and columns as target genes**, where `1` indicates a TF-target regulatory connection and
  `0` indicates no regulation. For details on deriving a prior GRN from single-cell RNA-seq data, see the
  [pySCENIC tutorial](https://regvelo.readthedocs.io/en/latest/tutorials/murine/01_SCENIC_tutorial.html).

## RegVelo workflow at a glance

Import the required packages:

```python
import scvelo as scv
import scanpy as sc
import pandas as pd
import numpy as np
import regvelo as rgv
import matplotlib.pyplot as plt
import scvi
import torch
import cellrank as cr
import anndata as ad

from regvelo import REGVELOVI
```

## 1. Input data preprocessing

### Load input data

```python
adata = ad.read_h5ad("adata.h5ad")
GRN = pd.read_parquet("prior_GRN.parquet")

GRN.head()
```

### scRNA-seq data preprocessing

Before running RegVelo, apply standard scRNA-seq preprocessing: quality control, log-transformation, highly
variable gene selection, and dimensionality reduction, using `scvelo` and `scanpy` {cite:p}`bergen2020generalizing`.

```python
scv.pp.filter_genes(adata, min_shared_counts=20)
scv.pp.normalize_per_cell(adata)
sc.pp.filter_genes_dispersion(adata, n_top_genes=3000)
sc.pp.log1p(adata)

sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_pcs=30, n_neighbors=50)
sc.tl.umap(adata)
scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
```

### Preprocessing and prior GRN integration

Next, preprocess the dataset with {func}`rgv.pp.preprocess_data <regvelo.preprocessing.preprocess_data>`,
which applies velocity gene filtering and min-max scaling, then integrate the prior GRN with
{func}`rgv.pp.set_prior_grn <regvelo.preprocessing.set_prior_grn>`. The resulting skeleton is stored in
`adata.uns["skeleton"]` with regulators as rows and targets as columns.

```{note}
This minimal workflow keeps only genes that pass the velocity gene selection criteria. If you want to keep
all TFs regardless of whether they pass this filter, see the
[RegVelo data preparation tutorial](https://regvelo.readthedocs.io/en/latest/tutorials/murine/02_RegVelo_preparation.html).
```

```python
adata = rgv.pp.preprocess_data(adata)
adata = rgv.pp.set_prior_grn(adata, GRN.T)

TF = adata.var_names[adata.uns["skeleton"].sum(1) != 0]
adata.var["TF"] = adata.var_names.isin(TF)
```

## 2. RegVelo model training

Configure general settings for reproducibility and plotting:

```python
scvi.settings.seed = 0
scv.settings.verbosity = 3
cr.settings.verbosity = 2

plt.rcParams["svg.fonttype"] = "none"
scv.settings.set_figure_params("scvelo", dpi=80, transparent=True, fontsize=14, color_map="viridis")
```

### Set model configuration

Set the TF list and convert the prior GRN to tensor format to configure model training:

```python
# TF and GRN configuration
TF = adata.var_names[adata.var["TF"]]

W = adata.uns["skeleton"].copy()
W = torch.tensor(np.array(W)).int()
W = W.T

REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
vae = REGVELOVI(adata, W=W, regulators=TF)
```

### Training

Train the RegVelo model {cite:p}`wang2026regvelo`, save it, and write the model outputs back to `adata`:

```python
vae.train()
vae.save("regvelo_model")
rgv.tl.set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
```

## 3. Cell fate prediction using CellRank

### Macrostate and terminal-state computation

Define a velocity kernel and compute macrostates with CellRank {cite:p}`lange2022cellrank,weiler2024cellrank`.
The `cluster_key` should correspond to your cell type annotation column and is used to help label and
interpret the macrostates:

```python
vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
vk.write_to_adata()

estimator = cr.estimators.GPCCA(vk)
estimator.compute_macrostates(n_states=9, cluster_key="cell_type")
estimator.plot_macrostates(which="all", legend_loc="right", size=100)
```

Set the terminal states for your biological system (replace the placeholder names below with labels from
your own `cluster_key` annotation) and compute fate probabilities:

```python
TERMINAL_STATES = ["terminal_state_1", "terminal_state_2", "terminal_state_3"]

estimator.set_terminal_states(TERMINAL_STATES)
estimator.plot_macrostates(which="terminal", legend_loc="right", size=100)
```

Next, compute cell fate probabilities towards each terminal state and visualize both the single-cell fate
probabilities and a summary commitment score (higher values indicate more differentiated, lineage-committed
cells):

```python
estimator.compute_fate_probabilities(solver="direct")
estimator.plot_fate_probabilities(same_plot=False, basis="umap")
rgv.pl.commitment_score(
    adata=adata, lineage_key="lineages_fwd", frameon=False,
    s=40, cmap="coolwarm", title="Commitment score",
)
```

## 4. TF perturbation

Finally, for a candidate lineage regulator we can predict its effect on cell fate decisions by performing an
*in silico* regulon knockout: remove all of the TF's outgoing edges from the GRN and recompute the velocity
field and fate probabilities on the perturbed system {cite:p}`wang2026regvelo`. The change relative to the
unperturbed baseline is summarized as a depletion likelihood score per terminal state and visualized as a
bar plot, and bars are drawn with solid, colored borders where the effect is statistically significant
(FDR-adjusted p < 0.05, red dashed line at 0.5).

```python
TF_candidates = ["Gabpa"]
adata_perturb_dict = {}

for tf in TF_candidates:
    adata_perturb, reg_vae_perturb = rgv.tl.in_silico_block_simulation(
        model="regvelo_model", adata=adata, TF=tf, cutoff=0,
    )
    adata_perturb_dict[tf] = adata_perturb

ct_indices = {
    ct: adata.obs["term_states_fwd"][adata.obs["term_states_fwd"] == ct].index.tolist()
    for ct in TERMINAL_STATES
}

for tf, adata_target_perturb in adata_perturb_dict.items():
    vkp = cr.kernels.VelocityKernel(adata_target_perturb).compute_transition_matrix()
    estimator_p = cr.estimators.GPCCA(vkp)
    estimator_p.set_terminal_states(ct_indices)
    estimator_p.compute_fate_probabilities(solver="direct")

df = rgv.mt.cellfate_perturbation(perturbed=adata_perturb_dict, baseline=adata, terminal_state=TERMINAL_STATES)

rgv.pl.cellfate_perturbation(
    adata=adata, df=df, fontsize=14, figsize=(8, 4),
    legend_loc="center left", legend_bbox=(1.02, 0.5),
    color_label="cell_type",
)
```

## Next steps

- Work through the full [tutorials](tutorials/index) for worked examples on real datasets.
- Browse the [API reference](api/index) for the complete set of preprocessing, training, and analysis
  functions.
- Check the [FAQ](faq) if your results (e.g. commitment scores or fate probabilities) look unexpected.
