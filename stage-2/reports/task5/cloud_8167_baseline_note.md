# Task5 Cloud 8167 Baseline Note

This file records the synchronized AutoDL cloud result used by Task5 and Task6.

- train images: `800000`
- identities: `20000`
- epochs completed: `60`
- actual batch size: `512`
- best LFW accuracy: `0.8166666666666667`
- LFW ROC AUC: `0.8791463333333334`
- target met: `False`

The LFW curve starts around 0.75-0.80 because one epoch already means a full pass
over the 800k-image subset, and LFW is an aligned 1:1 verification benchmark with
threshold selection. Later epochs lower ArcFace classification loss, but LFW
does not keep improving, which points to open-set generalization limits in this
custom subset/pipeline rather than simply too few epochs.
