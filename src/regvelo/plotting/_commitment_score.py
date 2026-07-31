import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Any

from ..metrics._utils import calculate_entropy


def commitment_score(
        adata: AnnData, 
        lineage_key: str = "lineages_fwd",
        **kwargs: Any
        ) -> None:
    r"""Compute and plot cell fate commitment scores based on fate probabilities. 
    
    Parameters
    ----------
    adata
        Annotated data matrix containing cell fate probabilities in `adata.obsm`.
    lineage_key
        Key in `adata.obsm` that stores the cell fate probabilities.
    **kwargs
        Additional keyword arguments passed to :func:`scanpy.pl.umap`
    
    Returns
    -------
    Nothing, just plots the figure. Also updates ``adata`` with the following fields:

    - ``adata.obs["commitment_score"]``.
    """

    if lineage_key not in adata.obsm:
        raise KeyError(f"Key '{lineage_key}' not found in `adata.obsm`.")

    p = pd.DataFrame(adata.obsm[lineage_key], columns=adata.obsm[lineage_key].names.tolist())
    entropy, max_entropy = calculate_entropy(p)
    score = 1 - entropy / max_entropy
    adata.obs["commitment_score"] = np.array(score)

    sc.pl.umap(
        adata,
        color="commitment_score",
        **kwargs
    )

