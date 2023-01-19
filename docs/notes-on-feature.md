## Features
1. Extracted feature ("raw"):
- feature groups: 
  - nuclei_morphology
  - cell_morphology
  - cell_sum
  - cell_ave
  - nuclei_sum
  - nuclei_ave
- attribute: *df_feature*
- folder: feature

2. Log-transformed and normalized by q-th quantile
- attribute: *df_feature_{q}normed* (e.g. *df_feature_75normed*)
- folder: feature_{q}normed (e.g. feature_75normed)

3. Log-transformed, normalized, and scaled
- attribute: *df_feature_{q}normed_scaled* (e.g. *df_feature_75normed*)
- folder: feature_{q}normed_scaled (e.g. feature_75normed_scaled)

4. 
