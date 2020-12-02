---
layout:     post
title:      Progressive Sample Mining and Representation Learning for One-Shot Re-ID
subtitle:   PSMA
date:       2020-12-02
author:     JoselynZhao
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Re-ID
    - one-shot
    - Deep Learning
---


**导言**
文章提出了一种新的三元组损失 HSoften-Triplet-Loss，在处理one-shot Re-ID任务中的噪声伪标签样本方面非常强大。文章还提出了一种伪标签采样过程，确保了在保持高可靠性的同时为训练图像形成正对和负对的可行性。与此同时，文章采用对抗学习网络，为训练集提供更多具有相同ID的样本，从而增加了训练集的多样性。 实验表明，文章框架在Market-1501（mAP 42.7％）和DukeMTMC-Reid数据集（mAP 40.3％）取得了最先进的Re-ID性能。

**引用**
>@article{DBLP:journals/pr/LiXSLZ21,
  author    = {Hui Li and
               Jimin Xiao and
               Mingjie Sun and
               Eng Gee Lim and
               Yao Zhao},
  title     = {Progressive sample mining and representation learning for one-shot
               person re-identification},
  journal   = {Pattern Recognit.},
  volume    = {110},
  pages     = {107614},
  year      = {2021}
}

**相关链接**
>paper：https://www.sciencedirect.com/science/article/pii/S0031320320304179?via%3Dihub 
>code：https://github.com/detectiveli/PSMA

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202183823910.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

# 主要挑战
 1) how to design **loss function**s for Re-ID training with pseudo labelled samples;
2) how to **select unlabelled samples** for pseudo label; 
3) how to overcome the **overfitting** problem due to lack of data

# 主要的贡献和创新点

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202155215306.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
>Fig. 1. Example of pseudo labelled person sampling and training with new losses. The upper part is the third iteration step, where we choose 2 similar images with the same pseudo ID. After one more training iteration, in the lower part, we aim to choose one more image with pseudo label for each person, but ignore the wrong sample for ID 2.


- We **identify the necessity of triplet loss in image-based one-shot Re-ID**, where the use of noisy pseudo labels for training is inevitable. Considering the nature of pseudo labels, we introduce an **HSoften-Triplet-Loss** to soften the negative influence of incorrect pseudo label. Meanwhile, **a new batch  formation rule** is designed by taking different nature of labelled samples and pseudo labelled samples into account.

- We propose a **pseudo label sampling mechanism** for one-shot Re-ID task, which is based on the relative sample distance to the feature center of each labelled sample. Our sampling mechanism ensures the feasibility of forming a positive pair and a negative pair of samples for each class label, which paves the way for the utilization of the HSoften-Triplet-Loss.
- We achieve the state-of-the-art mAP score of 42.7% on Market1501 and 40.3% on DukeMTMC-Reid, 16.5 and 11.8 higher than EUG [7] respectively.

> 创新点
> -  triplet loss
> -  HSoften-Triplet-Loss
> - new batch  formation rule
> - pseudo label sampling mechanism


# 提出的方法
##  总体框架与算法
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202155134770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
> Fig. 2. **Overview of our method**. Our training process takes several iterations. Each iteration has two main steps: 1) Add pseudo labelled images for each labelled image.2) Train the model with both CE loss and HSoft-triplet loss. After each iteration, the model should be more discriminative for feature representation and more reliable to generate the next similarity matrix. This is demonstrated by the fact that image features of the same person are clustered in a more compact manner, and features of different person move apart. The new similarity matrix is used to sample more pseudo labelled images for the next iteration training. Best viewed in color.


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202161313944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
##  Vanilla pseudo label sampling (PLS)
Our pseudo label sampling (PLS) process maintains distance ranking for each class label individually, whist EUG only maintains one overall distance ranking for all the class labels. Therefore, the PLS process in EUG cannot ensure the feasibility of forming a positive pair and a negative pair samples for each class label because for some classes there might be only one labelled sample for one class label. Thus, in EUG, it is not compatible to adopt a triplet loss or a contrast loss.
## PLS with adversarial learning
In our framework, we also apply the adversarial learning into the one-shot Re-ID task. To be more specific, **we use the CycleGAN [33] as a data augmentation tool to generate images of different cameras**, and adapt the enhanced dataset to our PLS framework.

**The total CycleGAN loss will be:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202172448342.png)
With the enhanced dataset, we update our PLS process in three aspects: (1) we make full use of the entire enhanced dataset as the training set. (2) more labelled images are available during the initial training process. (3) instead of using the one-shot image feature as sample mining reference, we use the feature centre of that
person under different cameras.
## Training losses
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202161852625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
> Fig. 3. The comparison of different losses.1) In fully-supervised learning, MSMLoss is perfect to distinct the positive and negative samples. 2) In one-shot learning, an incorrect hard positive sample causes strong miss. 3) In one-shot learning, soften hard positive can avoid the fatal impact of the incorrect hard positive sample by averaging the features. Best viewed in color.

**The softmax loss is formulated as:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181317185.png)
**The MSMLoss [24] is formulated as:**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181347819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**we design a soft version of hard positive sample feature representation:**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181416240.png)

**The final HSoften-Triplet-Loss is:**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181436752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)

**The overall loss is the combination of both softmax, and our HSoften-Triplet-Loss with parameter λ.**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181452576.png)

# 实验与结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201202181524708.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L05HVWV2ZXIxNQ==,size_16,color_FFFFFF,t_70)
As can observed from Table 1, among the methods in the lower group (one-shot learning), our model achieves a new state-of-the-art performance on both Market1501 (mAP of 42.7%) and DukeMTMC-ReID (mAP of 40.3%). Compared with the previous state-of-the-art method EUG [7], our method improves the accuracy of mAP by 16.5 on Market1501, and by 11.8 on DukeMTMC-ReID, which shows the robustness of our method on different testing datasets. In terms of the comparison in the second group, our method also achieves competitive results. On both dataset, our method virtually achieves the same accuracy as the best performing method in the upper group (transfer learning), while our method needs much fewer labels for training, which demonstrates the data efficiency of our method.

**Ablation study on components：**
- a) Effect of different network structures
- b) Ablation study on the number of generated samples
- c) Ablation study on the weight parameter λ
- d) Visualization of the feature distribution

#  结论
In this paper, we design a new triplet loss HSoften-Triplet-Loss, which is robust when dealing with the noisy pseudo labelled samples for the one-shot person Re-ID task. To provide compatible input data for the triplet loss, we also propose a pseudo label sampling process, that ensures the feasibility of forming a positive pair and a negative pair for the training images while maintaining high reliability. Extensive experimental results prove that using our new triplet loss leads to much better performance than simply using the softmax loss in existing one-shot person Re-ID methods, as well as conventional triplet loss without a softening mechanism. Besides, we further adopt an adversarial learning network to provide more samples with the same ID for the training set, which increases the diversity of the training set.

We believe our proposed HSoften-Triplet-Loss can be widely used for other identification tasks, where the noisy pseudo labels are involved, for examples, person Re-ID, face recognition with limited and/or weakly annotated labels. In the future work, we plan to study a more sophisticated distance metric to mine pseudo labelled images, and we also plan to deploy our new triplet loss in the one-shot face recognition task.

