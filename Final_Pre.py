import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate, Dense, GlobalAveragePooling1D, GlobalAveragePooling2D, Reshape
from transformers import BertTokenizer, TFBertModel, T5Tokenizer, TFT5ForConditionalGeneration
import os
os.chdir("/N/scratch/akorada/VQA_data/")
def preprocess_data(data_file_path, image_size=(224, 224), max_question_length=512, max_answer_length=2):
    # Load data
    df = pd.read_csv(data_file_path)

    # Limit the number of samples
    df = df.iloc[:5000]
    print(df.shape)
    # Load ResNet50 model
    resnet50_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Load BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    Y_true = df['answer'].tolist()
    # Preprocess images
    X_images = np.array([image.load_img(path, target_size=image_size) for path in df['image']])
    X_images = np.array([preprocess_input(image.img_to_array(img)) for img in X_images])

    # Preprocess questions
    df['input_ids'] = df['question'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    df['input_ids'] = df['input_ids'].apply(lambda x: x[:max_question_length])
    df['attention_mask'] = df['input_ids'].apply(lambda x: [1] * len(x) + [0] * (max_question_length - len(x)))
    X_question_ids = np.array([x[:max_question_length] for x in df['input_ids'].values])
    X_question_ids = pad_sequences(X_question_ids, maxlen=max_question_length, padding='post')
    X_question_mask = np.where(X_question_ids == 0, 0, 1)

    # Preprocess answers
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    df['answer_token'] = df['answer'].apply(lambda x: t5_tokenizer.encode(x, add_special_tokens=True, max_length=max_answer_length))
    Y_answer = np.array([[x[0]] for x in df['answer_token'].values])

    # Save preprocessed data to disk
    np.savez('preprocessed_data_5000.npz', X_images=X_images, X_question_ids=X_question_ids, X_question_mask=X_question_mask, Y_answer=Y_answer, Y_true = Y_true)

    return X_images, X_question_ids, X_question_mask, Y_answer

preprocess_data("final.csv")