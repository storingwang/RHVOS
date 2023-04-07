#Real-time High-Resolution Video Object Segmentation (RHVOS)

Welcome to the RHVOS GitHub repository! We appreciate your interest in our work. We aim to provide a detailed framework and well-structured codebase that will be progressively added to this repository in the coming days and weeks.

##Project Overview
In this paper, we propose an efficient semi-supervised video object segmentation approach that processes the high-resolution video cases at real-time speed with less memory usage. Specifically, we encode video frames and masks independently in the encoding stage, significantly reducing the computation cost caused by large, repeated encoding in previous state-of-the-art methods. On the other hand, we use a lightweight global memory module to represent the feature extracted from all the past video frames, which can be easily updated automatically over time. In addition, to reduce the confusion caused by distractors, the prediction masks of the previous frames are used to constrain the segmentation location of the foreground, and a channel-wise attention operation is adopted to enhance the representation of all foreground objects. In the experiments, our method achieves state-of-the-art on both the DAVIS and YouTube-VOS datasets. It has an accuracy of $84.9\%$ and an average single-frame segmentation speed of $0.04s$ on the DAVIS 2017 dataset. The results demonstrate the proposed method significantly reduces the computation cost and memory usage for high-resolution video cases while maintaining a competitive accuracy compared with other recent methods.

##Framework Details
The framework for RHVOS consists of several key components, including global attention module,mask embedding module and channel-attention module. We are in the process of finalizing the design and implementation of these components, and we will be releasing more information about each of them soon.

##Codebase
As we continue to develop RHVOS, we will be adding well-documented and modular code to this repository. This will enable users to easily understand, modify, and contribute to the project. We will also provide examples and tutorials to help you get started with using RHVOS in your own projects.

##Stay Tuned
Thank you for your patience and interest in RHVOS. We are excited to share our progress with you and welcome any suggestions or contributions you may have. Please feel free to submit issues or pull requests as we continue to build and improve this project together.

For updates and news regarding RHVOS, be sure to watch this repository and follow our development progress. We look forward to collaborating with you!

##DEMO

https://user-images.githubusercontent.com/21287744/190839612-b7d68ca6-0c17-40a2-9973-956c9cc9f9c3.mp4


https://user-images.githubusercontent.com/21287744/190839617-0d79b9c5-c3ef-43c8-875b-2805425cf279.mp4


https://user-images.githubusercontent.com/21287744/190839620-6d8ce608-b68b-41dd-b608-a86d5e87e40c.mp4


https://user-images.githubusercontent.com/21287744/190839640-6abe5e7c-9442-4a24-a375-f4ffca071212.mp4


https://user-images.githubusercontent.com/21287744/190839645-ed7044b5-16cf-49b2-9814-e4684555ad2f.mp4
