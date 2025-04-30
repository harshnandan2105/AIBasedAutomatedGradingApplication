# Automated Grading Application

This Automated Grading Application is designed to grade essays using deep learning and similarity calculations. The application supports both text and photo inputs, employing OCR for text extraction from images. It leverages an LSTM model trained on the ASAP dataset for grading essays and BERT-based sentence transformers for calculating sentence similarity based on stored synoptic.

## Features

- **LSTM-Based Grading**: An LSTM model trained on the Automated Student Assessment Prize (ASAP) dataset is used to grade essays.
- **Sentence Similarity**: BERT sentence transformers are employed to calculate the similarity between synoptic and actual answers, enhancing the grading process by evaluating the coherence and relevance of the content.
- **OCR Integration**: Gemini Vision Pro is used for Optical Character Recognition (OCR) to extract text from photo inputs.
- **Dual Input Support**: The application accepts both text and photo inputs for grading.

## Implementation

### LSTM-Based Essay Grading Module:
- **Purpose**: Grades essays using a machine learning model.
- **Model**: An LSTM (Long Short-Term Memory) model trained on the Automated Student Assessment Prize (ASAP) dataset.
- **Input**: Supports both text and photo inputs. For photo inputs, Optical Character Recognition (OCR) is employed to extract text from images.

### BERT-Based Similarity Calculation Module:
- **Purpose**: Calculates similarity scores based on stored synopses of questions for a variety of applications beyond essay grading.
- **Model**: Utilizes BERT-based sentence transformers.
- **Input**: Similar to the LSTM module, it supports text and photo inputs with OCR used for text extraction from images.

## Dataset

The dataset used for training the model can be found [here](https://paperswithcode.com/dataset/asap). To train the model, download the dataset and run the LSTM Model Training file or use the saved model.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Transformers
- Scikit-learn
- Numpy
- Gemini Vision Pro API
