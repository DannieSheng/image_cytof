## CytofImage

The `CytofImage` class defines a type of object that is targeted for the ease of analyzing CyTOF images. 

#### Attributes
- `df`: the dataframe represents for the CyTOF image  
- `columns`: a list of column names in original CyTOF data (`df`)  
- `markers`: a list of protein markers used in image capturing   
- `labels`: a list of metal isotopes used to tag protein
- `image`: the multi-channel CyTOF image  
- `channels`: a list of channel names correspond to each channel of in `image`  
- `features`: a dictionary of feature groups  
  - nuclei_morphology
  - cell_morphology
  - cell_sum
  - cell_ave
  - nuclei_sum
  - nuclei_ave

#### Methods
##### **get_markers**
##### **preprocess**



## CytofCohort



## Input output files standard
For cohort processing, a .csv file describing all images in the cohort is required. Columns:
- Slide: slide id
- ROI: ROI identifier
- input file: full file name of the input image
After batch processing, the file is updated with a new column:
- output file: full file name of the saved CytofImage object instances (one per input image)

