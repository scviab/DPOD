# Code for CVPR 2024 Submission 
# Leveraging-Out-of-Domain-Data-for-Domain-Specific-Prompt-Tuning-in-Multi-Modal-Fake-News-Detection
In this work, we explore
whether out-of-domain data can help to improve out-of
context misinformation detection (termed here as multi
modal fake news detection) of a desired domain, eg. pol-
itics, healthcare, etc. Towards this goal, we propose a novel
framework termed DPOD (Domain-specific Prompt-tuning
using Out-of-Domain data). First, to compute generaliz-
able features, we modify the Vision-Language Model, CLIP
to extract features that helps to align the representations
of the images and corresponding text captions of both the
in-domain and out-of-domain data in a label-aware man-
ner. Further, we propose a domain-specific prompt learning
technique which leverages the training samples of all the
available domains based on the the extent they can be useful
to the desired domain. Extensive experiments on a large-
scale benchmark dataset, namely NewsClippings demon-
strate that the proposed framework achieves state of-the-
art performance, significantly surpassing the existing ap-
proaches for this challenging task.
![cvpr2024sub1](https://github.com/anonymousdragon1729/Leveraging-Out-of-Domain-Data-for-Domain-Specific-Prompt-Tuning-in-Multi-Modal-Fake-News-Detection/assets/151718362/ea0fe17e-090b-4ec3-b576-15bc92487502)

## Dataset Collection
- The query images and captions are found in the NewsCLIPpings datasets (we use the merged balanced dataset).
- The whole NewsCLIPpings dataset was used in our task
## Running Code
### For Creating the Domain Embedding and Storing it into the JSON file
```
python3 make_dom_vector.py
```
