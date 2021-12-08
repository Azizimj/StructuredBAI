# Python 3 code for the structured bandits BAI paper

## Requirements:

`conda env create -f env.yml`

Please install MatLab engine for Python 
[here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for OD-LinBAI.

`source activate bai-environment`

## Running the experiments:

`cd [code home directory]`

- For the adaptive synthetic experiment run `python jobStBAI.py synt 1`
- For the static synthetic experiment run `python jobStBAI.py synt 5`
- For the randomized linear synthetic experiment run `python jobStBAI.py synt 6`
- For GLM synthetic experiment run `python jobStBAI.py synt 3`
- For auto experiment dataset run `python jobStBAI.py auto -1`
- For motor temperature experiment dataset run `python jobStBAI.py pmsm -1`
- For the randomized experiment for comparison to OD-LinBAI run `python jobStBAI.py synt 6 1`
- For the corner case experiment for OD-LinBAI comparison run `python jobStBAI.py synt 7`

The results would be stored in a csv file, as (key, values) tuples.

##Please do not share or reuse. All rights reserved.
