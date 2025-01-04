### Can Out-of-Domain data help to Learn Domain-Specific Prompts for Multimodal Misinformation Detection? (WACV 2025)
[WACV 2025] PyTorch implementation of "Can Out-of-Domain data help to Learn Domain-Specific Prompts for
Multimodal Misinformation Detection?"
[[Paper](https://arxiv.org/abs/2311.16496)]
<div style="text-align:center"><img src="https://github.com/user-attachments/assets/e67bec56-5480-4622-b519-e71e479b58ee"></div>


### Dataset
- The query images and captions are found in the NewsCLIPpings datasets (we use the merged balanced dataset) [[Link](https://github.com/g-luo/news_clippings)].
- The whole NewsCLIPpings dataset was used in our task.
### Running Code
#### Setup
```
git clone https://github.com/scviab/DPOD.git
# Create and activate the environment:
conda env create -f dpod_env.yml
conda activate dpod_env
```


#### For Creating the Domain Embedding and Storing it into the JSON file
```python
python3 make_dom_vector.py
```
#### Train CLIP-ViT B/32 using Label Aware Loss to get A-CLIP
```python
python3 make_label_aware_clip.py
```
#### Running CoOp + A-CLIP
```python
python3 make_label_aware_clip.py
```
#### Running our Proposed DPOD Framework
```python
python3 main_ft_dpod.py
```
#### Evaluating our Proposed DPOD Framework
```python
python3 evaluate_dpod.py
```





