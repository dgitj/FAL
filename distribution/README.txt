SSL-based Entropy Strategy - Setup Instructions

To use the SSL-based Entropy strategy, you need to place the SSL model checkpoint file in this directory.

REQUIRED FILES:
--------------
- round_99.pt: Copy your SSL model checkpoint here with exactly this name.
  This should be the trained encoder model from the ssl_fl project.

The SSL-based Entropy strategy will:
1. Load the model from this checkpoint
2. Extract the predicted distribution if included in the checkpoint
3. Save the distribution as ssl_distribution.pkl in this folder for future use

IMPORTANT NOTES:
---------------
- The checkpoint MUST be named exactly "round_99.pt"
- The strategy requires this file to be present - it will not fall back to other methods
- If the checkpoint doesn't contain distribution information, it will be generated

If you encounter errors:
1. Make sure the checkpoint file exists and has the correct name
2. Verify that the checkpoint is a valid SSL model checkpoint
3. Check that the model architecture in the checkpoint matches SimpleContrastiveLearning

This implementation uses K-means clustering on SSL-extracted features to generate
pseudo-labels for unlabeled data, which are then used for balanced entropy-based
active learning sampling.
