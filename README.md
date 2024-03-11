# Code for ECCV 2024 Submission 
# DPOD: Domain-Specific Prompt Tuning for Multimodal Fake News Detection
![DPOD](https://github.com/anonymouspeacock/DPOD/assets/151718362/5ed1a0d2-6d0c-45fc-abe9-386a6f6f03f2)


## Dataset Collection
- The query images and captions are found in the NewsCLIPpings datasets (we use the merged balanced dataset) [[Link](https://github.com/g-luo/news_clippings)].
- The whole NewsCLIPpings dataset was used in our task.
## Running Code
### Creating the Environment to Run the Code
Create the environment from the given yml file using the command:
```
conda env create -f dpod_env.yml
```
Activate the environment using the command:
```
conda activate dpod_env
```


### For Creating the Domain Embedding and Storing it into the JSON file
```python
python3 make_dom_vector.py
```
### In order to train CLIP-ViT B/32 in the proposed Label Aware Manner to get A-CLIP
```python
python3 make_label_aware_clip.py
```
### Running CoOp + A-CLIP
```python
python3 make_label_aware_clip.py
```
### Running our Proposed DPOD Framework
```python
python3 main_ft_dpod.py
```
### Evaluating our Proposed DPOD Framework
```python
python3 evaluate_dpod.py
```


![Screenshot from 2023-11-22 13-24-08](https://github.com/anonymousdragon1729/Leveraging-Out-of-Domain-Data-for-Domain-Specific-Prompt-Tuning-in-Multi-Modal-Fake-News-Detection/assets/151718362/fd4e9677-6c44-4bcd-a774-7988129565f4)



