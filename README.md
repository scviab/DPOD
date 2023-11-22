# Code for CVPR 2024 Submission 
# Leveraging-Out-of-Domain-Data-for-Domain-Specific-Prompt-Tuning-in-Multi-Modal-Fake-News-Detection
The spread of fake news using out-of-context images has001
become widespread and is a challenging task in this era002
of information overload. Since annotating huge amounts003
of such data requires significant time of domain experts, it004
is imperative to develop methods which can work in lim-005
ited annotated data scenarios. In this work, we explore006
whether out-of-domain data can help to improve out-of-007
context misinformation detection (termed here as multi-008
modal fake news detection) of a desired domain, eg. pol-009
itics, healthcare, etc. Towards this goal, we propose a novel010
framework termed DPOD (Domain-specific Prompt-tuning011
using Out-of-Domain data). First, to compute generaliz-012
able features, we modify the Vision-Language Model, CLIP013
to extract features that helps to align the representations014
of the images and corresponding text captions of both the015
in-domain and out-of-domain data in a label-aware man-016
ner. Further, we propose a domain-specific prompt learning017
technique which leverages the training samples of all the018
available domains based on the the extent they can be useful019
to the desired domain. Extensive experiments on a large-020
scale benchmark dataset, namely NewsClippings demon-021
strate that the proposed framework achieves state of-the-022
art performance, significantly surpassing the existing ap-023
proaches for this challenging task.
![cvpr2024sub1](https://github.com/anonymousdragon1729/Leveraging-Out-of-Domain-Data-for-Domain-Specific-Prompt-Tuning-in-Multi-Modal-Fake-News-Detection/assets/151718362/ea0fe17e-090b-4ec3-b576-15bc92487502)


