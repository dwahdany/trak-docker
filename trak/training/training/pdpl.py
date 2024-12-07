import glob
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_ood_scores(args, scores_zarr):
    metadata_dir = "/datasets/datacomp/metadata"
    parquet_files = glob.glob(os.path.join(metadata_dir, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for file in tqdm(parquet_files):
        df = pd.read_parquet(file)
        dfs.append(df)
    metadata_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(metadata_df):,} total samples")
    uid_path = "/datasets/datacomp/present_uids.pkl"
    if os.path.exists(uid_path):
        with open(uid_path, "rb") as f:
            uids = pickle.load(f)
    else:
        raise ValueError("UIDs not found")
    download_mask = metadata_df.uid.isin(uids).to_numpy()
    download_idx = np.where(download_mask)[0]

    ood_scores = scores_zarr[args.curation_method][args.curation_task][
        "ood_scores"
    ][download_idx]
    ood_uids = metadata_df.uid[download_idx]
    return ood_scores, ood_uids
