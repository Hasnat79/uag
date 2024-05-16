# The Oops! Dataset of Unintentional Action

This archive contains the video and annotations for the *Oops!* dataset. It contains two `.tar.gz` files which hold the data. 

## Accessing the data

Extract the contents of the included files with `tar -xvzf *.tar.gz`. Two subdirectories will be created: `annotations` and `oops_video`. `oops_video` has `train` and `val` subdirectories containing the `.mp4` files of video clips after scene detection. 

## Annotations

The `annotations` directory contains seven files with information about the dataset.

### `train.txt`, `val.txt`

These files contain a listing of all video clips in the train and val sets, including those that do not pass our quality control checks but are included for completeness. The filenames are of format `{YouTube video title}{clip index}`.

### `filtered_vids.txt`, `train_filtered.txt`, `val_filtered.txt` 
`X_filtered.txt` is a subset of `X.txt` which contains only filenames of clips that pass the quality control measures described in the paper in Section 3.1. `filtered_vids.txt` is a combination of both files. We use only videos in these files to train and test models. There are 4,712 validation videos and 16,013 training videos. All validation videos are labeled (including a few that did not pass quality control - so there are 4,791 total), as are 6,170 training videos. 

### `transition_times.json` 

A dictionary with keys as filenames (with format described above) and values as dictionaries. Each value dictionary has the following keys:

#### `t`
The list of times, in seconds, where failure (i.e. transition to unintentional action) was labeled. Since each clip was independently labeled by three different workers, this list has three elements. A value of `-1` indicates that the worker labeled that no failure occurred in the video clip.
#### `len`
Video length, in seconds.
#### `stdev`
Standard deviation of labeled failure times (`t`).
#### `rel_t`, `rel_stdev`
`rel_X = X/len`, provided for convenience 
#### `n_notfound`
Number of workers who labeled failure not found, i.e. `sum(1 for _ in t if _ == -1)`.

### `heldout_transition_times.json`

For all videos where the majority of workers labeled that a failure did occur, we ask a separate, fourth labeler to provide a label. In the paper, we use this to quantify human consistency on the tasks we provide. This file has the same format to `transition_times.json` but arrays are of length 1.