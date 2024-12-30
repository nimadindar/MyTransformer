# MyTransformer

An implementation of the Transformer model from scratch

This repository provides an implementation of the Transformer architecture from scratch in Python. The PyTorch library is used for basic functionalities such as Linear layers, the softmax function, and dataset loading. Additionally, the `train.py` file loads the "wikitext" dataset from the Hugging Face Transformers library for a text generation task.

## File Structure

- **transformer.py**  
  This file contains the implementation of the Transformer model. The process begins by building a single head-attention module and progresses to multi-head attention. Notably, the `AttentionHead` class is built for educational purposes and is not used in the rest of the code—multi-head attention is implemented implicitly. The encoder is constructed using multi-head attention, and the decoder is built using masked multi-head attention and cross-attention mechanisms. Finally, the encoder and decoder are combined to create the Transformer model.

- **text_dataset.py**  
  This file implements a class that inherits from PyTorch's `Dataset` to create a custom dataset module for our text dataset.

- **train.py**  
  This script handles dataset loading and splitting into train/test sets. It also contains a training loop to train the model for a text generation task. The "wikitext" dataset is used for training, and model hyperparameters are defined in the `model_params` dictionary.

## Requirements

To run this code, you need to have the following libraries installed:

- PyTorch
- Datasets
- Transformers

## Running the Code

To run the code, use the following command in your terminal:

```bash
python train.py
