import numpy as np
import pickle
import tqdm
import os
import pandas as pd 

def splitting_npy(input_file, output_dir, chunk_size):
    data = np.load(input_file, allow_pickle=True, mmap_mode="r")
    # with open(input_file, "rb") as f:
    #     np.lib.format.read_array_header_1_0(f)
    for s in range(0, data.shape[1], chunk_size):
        e = min(s + chunk_size, data.shape[1])
        chunk = data[:, s:e]
        chunk_file = f"{output_dir}/{input_file}_{s}_{e}.npy"
        np.save(chunk_file, chunk)
        print(f"saved {chunk_file} with size {chunk.shape}")

def npy_data(root_dir, output_file, masking_ratio, mean_mask_length, mode, distribution, exclude_feats, chunk_size=1000):
    """
    Preprocess `.npy` files, compute masks, and save the resulting dataset with metadata for use with `ImputationDataset`.
    """
    all_data = {}  
    
    for file_name in tqdm.tqdm(os.listdir(root_dir), desc="Preprocessing .npy files"):
        if file_name.endswith('.npy'):
            file_path = os.path.join(root_dir, file_name)
            key = os.path.splitext(file_name)[0]
            data = np.load(file_path, mmap_mode="r")

            for start in range(0, data.shape[1], chunk_size):
                end = min(start + chunk_size, data.shape[1])
                chunk = data[:, start:end]
                if chunk.shape[0] != 4:
                    raise ValueError(f"doesnt have 4 channels")
                time_points = chunk.shape[1]

                clips = time_points // 250
                clipped = chunk[:, :clips * 250].reshape(4, clips, 250)
                for i in range:
                    clip = clipped[:, i, :]
                    clip_key = f"{key}_{start}_{i}"

                    mask = noise_mask(clip.T, masking_ratio, mean_mask_length, mode, distribution, exclude_feats)
                    all_data[clip_key] = {
                        "feature_df": pd.DataFrame(clip.T),
                        "mask": mask
                    }

    with open(output_file, 'wb') as f:
        pickle.dump(
            {
                "feature_df": pd.concat([entry["feature_df"] for entry in all_data.values()]),
                "FileID": list(all_data.keys()),
                "mask": [entry["mask"] for entry in all_data.values()]
            },
            f
        )

    print(f"Preprocessed data saved to {output_file}")

def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask    

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

splitting_npy(input_file="all_subs.npy", output_dir=npy_data, chunk_size=1000)
npy_data(
    root_dir="npy_data",
    output_file="big_npy_output.pkl",
    masking_ratio=0.15,
    mean_mask_length=3,
    mode='separate',
    distribution='geometric',
    exclude_feats=None,
    chunk_size=1000
)