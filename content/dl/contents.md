**Deep Learning Contents:**
1. Introduction to Deep Learning. What is Neural Networks?
2. Implement Regression and Classification tasks using Tensorflow, Keras and PyTorch.(High Level)
3. Implementing Neural Netwoorks with NumPy. (and PyTorch tensors)
4. Deep dive into Activation Functions. (With articles)
5. Deep dive into Loss Functions. (With articles)
6. PyTorch Core module: Tensor operations.
7. Pytorch NN module.
8. Pytorch Dataset and DataLoader.

Language Modelling
9. makemore: Bigram Language Model. (With article). Implement with and without PyTorch.
    Introduction to Word Embeddings
10. makemore: MLP (with article). Implement with and without PyTorch.
11. Introduction to Convolutional Neural Networks. Pytorch implementation. Using Pretrained models (AlexNet and Resnet).
12. makemore: CNN (with article).
13. Introduction to Recurrent Neural Networks.
14. makemore: RNN (with article).
15. Introduction to LSTM.
16. makemore: LSTM (with article).
17. Introduction to GRU.
18. makemore: GRU (with article).
19. Introduction to Transformer Artichecture.
20. makemore: Transformer (with article).
21. nanogpt: Bigram, RNN, Transformer
22. Fine tuning for classification.
23. Fine tunig to follow instructions.


---



# **Deep Learning Roadmap**  

## **1. Introduction to Deep Learning**  
- What is Deep Learning?  
- Evolution from Machine Learning to Deep Learning  
- Key Concepts: Weights, Biases, and Neurons  
- Overview of Deep Learning Frameworks (TensorFlow, Keras, PyTorch)  

## **2. Fundamentals of Neural Networks**  
- Understanding Artificial Neural Networks (ANNs)  
- Forward Propagation and Backpropagation  
- Implementing Regression and Classification with TensorFlow, Keras, and PyTorch (High-Level API)  
- Training, Validation, and Testing: Best Practices  

## **3. Building Neural Networks from Scratch**  
- Implementing Neural Networks with NumPy (and PyTorch tensors)  
- Understanding the Computational Graph  
- Gradient Descent and Optimization Techniques  

## **4. Core Deep Learning Components**  
### **4.1 Activation Functions**  
- Introduction to Activation Functions  
- Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish  
- Choosing the right Activation Function  

### **4.2 Loss Functions**  
- Mean Squared Error, Cross-Entropy, Hinge Loss, etc.  
- How to select the right Loss Function  

### **4.3 Optimizers**  
- Stochastic Gradient Descent (SGD), Adam, RMSprop, Adagrad  
- Learning Rate Scheduling and Weight Decay  

## **5. Deep Learning with PyTorch**  
### **5.1 PyTorch Basics**  
- Understanding Tensors and Tensor Operations  
- Autograd and Computational Graphs  

### **5.2 PyTorch nn.Module**  
- Creating Custom Models with `nn.Module`  
- Understanding Model Layers  

### **5.3 Dataset and DataLoader**  
- Working with `torch.utils.data.Dataset` and `DataLoader`  
- Data Augmentation and Normalization  

## **6. Language Modeling and NLP**  
### **6.1 Bigram Language Model (`makemore`)**  
- Understanding Bigram Models  
- Implementing Bigram Model with and without PyTorch  

### **6.2 Word Embeddings**  
- Introduction to Word Embeddings (Word2Vec, GloVe)  
- Using Pretrained Embeddings in PyTorch  

### **6.3 Multi-Layer Perceptron (MLP) for Language Modeling**  
- Implementing `makemore` MLP from scratch  
- Comparison of MLP with other models  

## **7. Convolutional Neural Networks (CNNs)**  
- Introduction to CNNs: Why Convolutions?  
- Implementing CNNs in PyTorch  
- Using Pretrained Models (AlexNet, ResNet, VGG)  
- **makemore: CNN** (With article)  

## **8. Recurrent Neural Networks (RNNs)**  
- Introduction to Sequence Modeling  
- Implementing a Simple RNN in PyTorch  
- **makemore: RNN** (With article)  

## **9. Advanced Recurrent Models**  
### **9.1 Long Short-Term Memory (LSTM)**  
- How LSTMs Improve Over RNNs  
- Implementing LSTM in PyTorch  
- **makemore: LSTM** (With article)  

### **9.2 Gated Recurrent Unit (GRU)**  
- Difference Between LSTM and GRU  
- Implementing GRU in PyTorch  
- **makemore: GRU** (With article)  

## **10. Transformer Models**  
### **10.1 Introduction to Transformers**  
- Why Transformers? Limitations of RNNs  
- Self-Attention Mechanism  
- Transformer Encoder-Decoder Architecture  

### **10.2 Implementing Transformers**  
- **makemore: Transformer** (With article)  
- Understanding `nanogpt`: Bigram, RNN, Transformer  

### **10.3 Advanced Transformer Applications**  
- Fine-tuning Pretrained Transformers (BERT, GPT)  
- Transfer Learning for Classification Tasks  
- Fine-tuning Models for Instruction Following  

## **11. Additional Topics (Optional Advanced Concepts)**  
- Reinforcement Learning with Deep Neural Networks  
- Generative Adversarial Networks (GANs)  
- Autoencoders for Unsupervised Learning  
- Self-Supervised Learning and Contrastive Learning  

### **12. Computer Vision Beyond CNNs**  
- Understanding Vision Transformers (ViTs)  
- Contrastive Learning and Self-Supervised Learning (SimCLR, MoCo)  
- Image Segmentation (UNet, Mask R-CNN)  
- Object Detection (YOLO, Faster R-CNN)  

### **13. Attention Mechanisms and Advanced Transformers**  
- Scaled Dot-Product Attention & Multi-Head Attention  
- BERT and Transformer-Based Text Representation  
- Implementing GPT-Style Autoregressive Models  
- Fine-tuning Large Language Models (LLMs)  

### **14. Hyperparameter Tuning and Model Optimization**  
- Hyperparameter Search: Grid Search, Random Search, Bayesian Optimization  
- Regularization Techniques (Dropout, BatchNorm, L2 Regularization)  
- Model Quantization and Pruning for Efficient Deployment  

### **15. Scalable and Distributed Training**  
- Training on Multiple GPUs (Data Parallelism & Model Parallelism)  
- Using TPUs for Faster Model Training  
- PyTorch Lightning and Hugging Face Trainer for Scalable Workflows  

### **16. Model Deployment and Serving**  
- Exporting Models: ONNX, TorchScript, TensorFlow SavedModel  
- Deploying with Flask, FastAPI, or Django  
- Running Models in Production (TorchServe, TensorFlow Serving)  
- Deploying on Cloud (AWS SageMaker, Google Vertex AI, Azure ML)  
- Edge AI: Running Models on Mobile and IoT Devices (TensorFlow Lite, CoreML)

### **17. Ethical AI and Explainability**  
- Model Interpretability with SHAP and LIME  
- Bias in AI and Fairness Considerations  
- Responsible AI: Avoiding Hallucinations and Misinformation in LLMs  

### **18. End-to-End Deep Learning Projects**  
- **Project 1:** Image Classification with Custom CNN  
- **Project 2:** Sentiment Analysis with LSTMs and Transformers  
- **Project 3:** Speech Recognition with RNNs  
- **Project 4:** Generating Text with GPT-style Models  
- **Project 5:** Reinforcement Learning for Game Playing  

### **19. Research and Advanced Topics**  
- Meta-Learning: Few-Shot and Zero-Shot Learning  
- Diffusion Models (Stable Diffusion, DALL·E)  
- Neural Radiance Fields (NeRF) for 3D Vision  
- Federated Learning and Privacy-Preserving AI  

### **20. Deep Learning in Industry Use Cases**  
- AI in Healthcare: Medical Image Analysis, Drug Discovery  
- AI in Finance: Fraud Detection, Algorithmic Trading  
- AI in Retail: Recommendation Systems, Demand Forecasting  
- AI in Autonomous Vehicles: Perception & Control Systems  
