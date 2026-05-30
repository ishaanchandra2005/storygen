# Story Generator using a GPT-style Small Language Model (SLM)

## Overview

This project implements a GPT-style Small Language Model (SLM) from scratch using PyTorch. The model is trained on text data to learn next-token prediction and generate coherent story continuations from a given prompt.

Unlike using a pre-trained language model, this project builds the Transformer architecture manually, including self-attention, feed-forward networks, positional embeddings, training loops, and text generation logic.

The goal of this project was to gain a deep understanding of how modern language models such as GPT work internally.

---

## Features

- Custom GPT-style Transformer implementation
- Multi-Head Self-Attention
- Positional Embeddings
- Layer Normalization
- Residual Connections
- Feed Forward Networks (MLP)
- Mixed Precision Training
- Gradient Accumulation
- Learning Rate Warmup + Cosine Decay
- Automatic Checkpoint Saving
- Story Generation from User Prompts

---

## Model Architecture

### Configuration

| Parameter | Value |
|------------|--------|
| Vocabulary Size | 50,257 |
| Context Window | 128 Tokens |
| Transformer Layers | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Dropout | 0.1 |

### Components

1. Token Embeddings
2. Positional Embeddings
3. Multi-Head Causal Self-Attention
4. Feed Forward Networks
5. Residual Connections
6. Layer Normalization
7. Language Modeling Head

The model is trained autoregressively, meaning it predicts the next token given all previous tokens.

---

## Dataset Processing

The dataset is tokenized using the GPT-2 tokenizer from tiktoken.

### Workflow

1. Load raw text dataset
2. Convert text into token IDs
3. Store tokenized data efficiently using NumPy memory-mapped files (.bin)
4. Create random input-output training batches

Example:

Input:
The cat sat on

Target:
cat sat on the

---

## Training Strategy

### Optimizer

AdamW

### Learning Rate Schedule

- Linear Warmup
- Cosine Annealing Decay

### Additional Techniques

- Gradient Accumulation
- Gradient Clipping
- Mixed Precision Training
- Validation Loss Monitoring
- Best Model Checkpoint Saving

---

## Text Generation

After training, the model can generate story continuations from a user-provided prompt.

Example:

Prompt:
Once upon a time there was a pumpkin.

Generated Output:
Once upon a time there was a pumpkin that lived in a small village near a magical forest...

Generation is performed autoregressively by repeatedly sampling the next token from the model's output distribution.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Hugging Face Datasets
- tiktoken
- Matplotlib
- tqdm

---

## Learning Outcomes

This project helped develop a practical understanding of:

- Transformer Architectures
- Self-Attention Mechanisms
- Language Modeling
- PyTorch Model Development
- Deep Learning Training Pipelines
- Text Generation Systems
- Modern NLP Fundamentals

---

## Future Improvements

- Fine-tuning on domain-specific datasets
- Larger context windows
- More Transformer layers
- LoRA-based fine-tuning
- Chatbot-style conversational training
- Deployment through a web interface
- Integration with Retrieval-Augmented Generation (RAG)

---

## Author

Ishaan Chandra

Final Year Electronics and Communication Engineering (ECE)

National Institute of Technology Tiruchirappalli (NIT Trichy)

This project was developed as a hands-on exploration of Transformer-based language models and modern generative AI systems.
