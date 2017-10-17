# Relation Classification via Attention Model
As you know the attention model can help us to solve many problems.Resently, I have a project which need to recognize the relation from some entities. After reading several paper, I decided to implement this paper: [Relation Classification via Multi-Level Attention CNNs](http://iiis.tsinghua.edu.cn/~weblt/papers/relation-classification.pdf)
I desperately desire to use pytorch to do some awsome things. So it's the only choice for me. And i think you will like it.

some of data handling codes are copied from [ACNN](https://github.com/FrankWork/acnn)
You need an environment:
pytorch 0.2

Git this project to your pycharm or other IDE, then edit the acnn_train.py to satisfied your data
# 10.17 The Final Version
Last version have a heavy bug. I repaired it this time. 
Unfortunately, my model's acc is not as high as the paper's.
# Network Structure
<p align="center"><img width="60%" src="acnn_structure.png" /></p>
