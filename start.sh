#!/bin/bash
   docker run -it -p 5000:5000 \
   -v /courses/AI_ML/5_cc_dc/1_Dibimbing_ML/Project_rice_classification/RiceLeafClassification/mlruns:/app/mlruns \
   -v /courses/AI_ML/5_cc_dc/1_Dibimbing_ML/Project_rice_classification/RiceLeafClassification/RiceLeafClassification/DataLabelledRice:/app/DataLabelledRice \
   riceleaf