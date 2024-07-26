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

You can access the PowerPoint file that I presented to better understand the project I have completed.

This is the result after I ran the train.py file and used the Amazon dataset. 
![image](https://github.com/user-attachments/assets/85da8e4f-005c-41a2-83a9-3e743f444847)

You can access the test.py file to predict the labels of graph-structured data. 
Below is an example result after I executed the test.py file on the Amazon dataset.

![image](https://github.com/user-attachments/assets/94b5f18f-6a54-4cf2-8bb9-df9508ba2be3)

