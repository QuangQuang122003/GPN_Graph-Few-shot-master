Graph Prototypical Networks for Few-shot Learning on Attributed Networks 
============

## Graph Prototypical Networks (GPN)

This is the source code of paper ["Graph Prototypical Networks for Few-shot Learning on Attributed Networks"](https://arxiv.org/pdf/2006.12739.pdf).
![The proposed framework](GPN.png)

## Requirements
python==3.6.6 

torch==1.4.0

## Usage
```python train_gpn.py --shot 5 --way 5 --episodes 1000 --dataset dblp --dropout 0.5 --use_cuda```



