### Can Out-of-Domain data help to Learn Domain-Specific Prompts for Multimodal Misinformation Detection?

<div style="text-align:center"><img src="https://github.com/user-attachments/assets/e67bec56-5480-4622-b519-e71e479b58ee" height=75%, width=75%></div>
[Paper](https://arxiv.org/abs/2311.16496)

### Dataset Collection
- The query images and captions are found in the NewsCLIPpings datasets (we use the merged balanced dataset) [[Link](https://github.com/g-luo/news_clippings)].
- The whole NewsCLIPpings dataset was used in our task.
### Running Code
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





