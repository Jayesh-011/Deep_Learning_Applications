<div align="center">

# 🧠 Deep Learning Applications

**A comprehensive, hands-on collection of 90+ deep learning programs — from single neurons to Transformers, CNNs, RNNs, AI Agents, and real-world applications.**

[Explore Programs](#-program-index) · [Getting Started](#-getting-started) · [Real-World Projects](#-real-world-projects)

</div>

***

## 📌 Overview

This repository is a progressive deep learning curriculum implemented entirely in Python. It covers the full journey from bare-metal neuron math to production-ready architectures — every concept is implemented step-by-step with graphical and animated visualizations wherever possible.

The collection spans **7 major tracks**:

| Track | Files | Topics |
|-------|-------|--------|
| 🔵 **ANN** | `1–12` | Single neurons → multi-layer training → animated backprop |
| 🟢 **RNN** | `1–21`, `43–70` | Tokenization → embeddings → LSTM → sentiment analysis |
| 🔴 **CNN** | `22–29`, `39–42`, `46` | Pixel ops → edge detection → crack detection |
| 🟡 **Loss Functions** | `13–17` | MSE, MAE, Binary Cross-Entropy |
| ⚡ **TensorFlow/Keras** | `30–38` | Tensor ops → neural nets → image classification |
| 🟣 **Transformers** | `71–78` | Self-attention → encoder-decoder → GPT-style decoder |
| 🤖 **AI Agents & Apps** | `79–90` | Streamlit UIs → AI agents → LLM-powered Q&A |

***

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Jayesh-011/Deep_Learning_Applications.git
cd Deep_Learning_Applications

# Install dependencies
pip install tensorflow keras numpy pandas matplotlib scikit-learn streamlit
```

### Run any program

```bash
python 1_ANN_Single_Neuron.py
python 48_Reliance_LSTM_Time_Series.py
streamlit run 86__Intelligent_Document_Question_Answering_System.py
```

***

## 📂 Program Index

### 🔵 ANN — Artificial Neural Networks

| File | Description |
|------|-------------|
| `1_ANN_Single_Neuron.py` | Manual single neuron with weights and bias |
| `2_Activation_Relu.py` | ReLU activation — manual + graphical |
| `3_Activation_Sigmoid.py` | Sigmoid activation — manual + graphical |
| `4_Activation_Sigmpoid_vs_Relu.py` | Side-by-side comparison plot |
| `5_Layered_Artificial_Neural_Network.py` | Full layered ANN from scratch |
| `6_Layered_Artificial_Neural_Network_Simplified.py` | Cleaner modular version |
| `7_Layered_Artificial_Neural_Network_Graphical.py` | Network visualization |
| `8_ANN_Trianing_Steps.py` | Forward pass + manual backprop |
| `9_ANN_Trianing_Steps_Graphical.py` | Training steps with live plots |
| `10_ANN_MultiNeuron_Trianing_Steps.py` | Multi-neuron training walkthrough |
| `10_ANN_MultiNeuron_Trianing_Steps_Multi_Valuee.py` | Batch training extension |
| `11_ANN_MultiNeuron_Trianing_Steps_Graphical.py` | Graphical multi-neuron training |
| `12_ANN_MultiNeuron_Trianing_Steps_Graphical_Animation.py` | Animated weight update visualization |
| `16_Gradient_Backprapogation.py` | Gradient descent & backprop manual |
| `17_Gradient_Backprapogation_Graphical.py` | Graphical gradient flow |

### 🟢 RNN — Recurrent Neural Networks

| File | Description |
|------|-------------|
| `1_RNN_DatasetDisplay.py` | Dataset loading and display |
| `2_RNN_Tokenization.py` | Word tokenization from scratch |
| `3_RNN_VocabularyCreation.py` | Vocabulary builder |
| `4_RNN_KerasTokenization.py` | Tokenization using Keras |
| `5_RNN_TexttoSequences.py` | Text → integer sequences |
| `6_RNN_UseOfPadding.py` | Padding sequences manually |
| `7_RNN_PaddingKeras.py` | Keras `pad_sequences` |
| `8_RNN_VocabularySize.py` | Vocabulary size management |
| `9_RNN_OutputLabels.py` | Label encoding for RNN output |
| `10_RNN_EmbeddingManual.py` | Manual word embedding vectors |
| `11_RNN_EmbeddingKeras.py` | Keras Embedding layer |
| `12_RNN_TimeSteps.py` | Understanding time steps |
| `13_RNN_HiddnState.py` | Hidden state mechanics |
| `14_RNN_ManualCalculations.py` | Full RNN forward pass by hand |
| `15_RNN_Tanh.py` | Tanh activation in RNNs |
| `16_RNN_Sigmoid.py` | Sigmoid gating demo |
| `17_RNN_BinaryCrossEntropyLoass.py` | BCE loss for sequence models |
| `18_RNN_Architecture.py` | Keras RNN architecture build |
| `19_RNN_TrainModel.py` | Training loop |
| `20_RNN_PredictionFlow.py` | End-to-end prediction pipeline |
| `21_RNN_CompleteCode.py` | Full RNN from scratch to prediction |
| `43_RNN_Character_Prediction.py` | Character-level text generation |
| `44_RNN_Sentiment_Analysis.py` | Movie review sentiment classifier |
| `45_RNN_Sequance_Prediction.py` | Numeric sequence forecasting |
| `70_RNN_Graphical_Version.py` | Animated RNN architecture visualizer |

> Files `49–69` mirror files `1–21` as extended/revised versions.

### 🔴 CNN — Convolutional Neural Networks

| File | Description |
|------|-------------|
| `22_Convert28by28_Grayscale_Image.py` | Image resizing to 28×28 grayscale |
| `23_Pixel_Grayscale_Display.py` | Pixel matrix display |
| `24_Color_Image_Pixel_Display.py` | RGB channel visualization |
| `25_CNN_Edge_Detection.py` | Manual convolution edge filter |
| `26_CNN_Edge_Detection_Result_Display.py` | Edge detection output display |
| `27_CNN_Edge_Detection_Result_Animation.py` | Animated kernel sliding |
| `28_CNN_Edge_Detection_Cat.py` | Edge detection on real image |
| `29__Realtime_Image_Classification.py` | Real-time image classifier |
| `39_CNN_Convolution_ReLU_Pooling_FC.py` | Full CNN pipeline |
| `40_CNN_Convolution_ReLU_Pooling_FC_Graphical.py` | Graphical CNN layer-by-layer |
| `41_CNN_Relu_Pooling_flattern.py` | Feature map flattening |
| `42_CNN_Convolution_ReLU_Pooling_FC_Graphical_BigData.py` | CNN on large input |
| `46__CNN_Surface_Crack_Detection.py` | Industrial surface crack classifier |

### 🟡 Loss Functions

| File | Description |
|------|-------------|
| `13_LossFunction_MSE.py` | Mean Squared Error |
| `14_LossFunction_MAE.py` | Mean Absolute Error |
| `15_LossFunction_Binary_Cross_Entropy.py` | Binary Cross-Entropy |

### ⚡ TensorFlow / Keras Fundamentals

| File | Description |
|------|-------------|
| `30_tensorflow_tensor_types.py` | Tensor types: constant, variable, sparse |
| `31_tensorflow_tensor_operations.py` | Add, subtract, multiply ops |
| `32_tensorflow_tensor_shape_reshape.py` | Shape inspection and reshape |
| `33_tensorflow_tensor_variable.py` | Mutable tf.Variable |
| `34_tensorflow_tensor_matrix_multiplication.py` | Matrix multiplication with tf |
| `35_tensorflow_tensor_single_neuron.py` | Single neuron using TensorFlow |
| `36_tensorflow_tensor_neural_network.py` | MLP with TensorFlow |
| `37_tensorflow_tensor_calassification.py` | Binary classification model |
| `38_tensorflow_tensor_ANN_image_calassification.py` | ANN for image classification |

### 🟣 Transformers

| File | Description |
|------|-------------|
| `71_transformer_self_attention.py` | Scaled dot-product self-attention |
| `72_transformer_Multihead_attention.py` | Multi-head attention mechanism |
| `73_transformer_poitionalencoder.py` | Sinusoidal positional encoding |
| `74_transformer_encoder.py` | Full transformer encoder block |
| `75_transformer_decoder.py` | Full transformer decoder block |
| `76_Transformer_Sentiment_Classification_Encoder_Only.py` | Encoder-only sentiment model |
| `77_Transformer_Neural_Machine_Translation_Encoder_Decoder.py` | Seq2seq NMT model |
| `78_Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py` | GPT-style decoder |

### 🤖 AI Agents & Applications

| File | Description |
|------|-------------|
| `79_streamlit_JayGanesh.py` | Hello World Streamlit app |
| `80_streamlit_Input.py` | Streamlit text input widget |
| `81_streamlit_Button.py` | Streamlit button interaction |
| `82_streamlit_fileUpload.py` | File upload handler |
| `83_streamlit_TextExtraction.py` | Extract text from uploaded docs |
| `84_streamlit_Chunking.py` | Text chunking for LLM context |
| `85__Calculator_tkinter.py` | GUI Calculator with tkinter |
| `86__Intelligent_Document_Question_Answering_System.py` | LLM-powered PDF Q&A system |
| `87_AIAgent_RuleBased.py` | Rule-based AI agent |
| `88_AIAGentCalculator.py` | AI agent with calculator tool |
| `89_AIAGenyMultpleTool.py` | Multi-tool AI agent |
| `90_AIAgent_Memory.py` | AI agent with persistent memory |
| `interview_agent.py` | AI interview simulation agent |

***

## 🏆 Real-World Projects

### 📈 Reliance Stock Price Prediction (LSTM)
**File:** `48_Reliance_LSTM_Time_Series.py`  
Time-series forecasting using stacked LSTM on Reliance Industries stock data. Outputs prediction charts and CSV results.

**Assets:** `_reliance_stock_sample.csv` · `_actual_vs_predicted_detailed.png` · `_training_loss_detailed.png`

***

### 🏥 Breast Cancer Classification (FNN)
**File:** `47_IndustrialBreastCancer.py`  
Feedforward neural network on the Wisconsin Breast Cancer dataset for malignant/benign classification.

**Asset:** `breast-cancer-wisconsin.csv`

***

### 🏗️ Surface Crack Detection (CNN)
**File:** `46__CNN_Surface_Crack_Detection.py`  
Convolutional neural network to classify concrete surface images as cracked or intact using a custom image dataset.

**Asset:** `CrackDataset/`

***

### 🎓 Student Placement Prediction (FNN)
**Files:** `19_FNN_Student_Result_Classification.py`, `21_FNN_Placement_Prediction.py`  
Classification models predicting student pass/fail and campus placement probability.

**Asset:** `student.csv` · `placement_data.csv`

***

### 📄 Intelligent Document Q&A System
**File:** `86__Intelligent_Document_Question_Answering_System.py`  
A Streamlit-powered RAG pipeline that lets users upload documents and ask questions answered by an LLM.

***

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Neural network building and training |
| `NumPy` | Matrix math, manual implementations |
| `Pandas` | Data loading and preprocessing |
| `Matplotlib` | Plots, animations, visualizations |
| `Scikit-learn` | Preprocessing, metrics |
| `Streamlit` | Interactive web applications |
| `tkinter` | Desktop GUI |

***

## 📁 Repository Structure

```
Deep_Learning_Applications/
├── 1–12    ─── ANN (Artificial Neural Networks)
├── 13–17   ─── Loss Functions
├── 1–21    ─── RNN Series I (NLP Foundations)
├── 22–29   ─── Image Processing & CNN Basics
├── 30–38   ─── TensorFlow Fundamentals
├── 39–42   ─── CNN Advanced
├── 43–45   ─── RNN Applications
├── 46–48   ─── Industrial / Real-World Projects
├── 49–70   ─── RNN Series II (Extended)
├── 71–78   ─── Transformers
├── 79–90   ─── AI Agents & Streamlit Apps
├── CrackDataset/           ← Surface crack images
├── breast-cancer-wisconsin.csv
├── placement_data.csv
├── student.csv
└── _reliance_stock_sample.csv
```

***

## 🤝 Contributing

Contributions are welcome! To add a new program:

1. Fork the repository
2. Follow the existing naming convention: `NN_Topic_Description.py`
3. Add a brief docstring at the top of your file
4. Submit a pull request

***

## 👨‍💻 Author

**Jayesh** — [@Jayesh-011](https://github.com/Jayesh-011)

***

<div align="center">
⭐ Star this repo if it helped you learn deep learning!
</div>