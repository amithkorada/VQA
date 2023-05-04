from transformers import TFBertModel, BertTokenizer, T5Tokenizer, TFT5ForConditionalGeneration
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
import os
from tensorflow.keras.layers import Reshape
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, GlobalAveragePooling1D, GlobalAveragePooling2D, Reshape
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from bert_score import score

os.chdir("/N/scratch/akorada/VQA_data/")

def load_preprocessed_data(data_file_path):
    data = np.load(data_file_path)
    X_images = data['X_images']
    X_question_ids = data['X_question_ids']
    X_question_mask = data['X_question_mask']
    Y_answer = data['Y_answer']
    Y_true = data['Y_true']
    return X_images, X_question_ids, X_question_mask, Y_answer, Y_true


# Load preprocessed data
X_images, X_question_ids, X_question_mask, Y_answer, Y_true = load_preprocessed_data('preprocessed_data_5000.npz')
X_images_val, X_question_ids_val, X_question_mask_val, Y_answer_val, Y_true_val = load_preprocessed_data('val_preprocessed_data.npz')
# Define model architecture
img_size = (224, 224)
resnet50_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
t5_model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

input_image = Input(shape=(img_size[0], img_size[1], 3))
input_question_ids = Input(shape=(512,), dtype=tf.int32)
input_question_mask = Input(shape=(512,), dtype=tf.int32)
decoder_input = Input(shape=(1,), dtype=np.int32)

x = tf.keras.applications.resnet50.preprocess_input(input_image)
x = resnet50_model(x)
x = Reshape((1, 1, -1))(x)
x = GlobalAveragePooling2D()(x)

question_embeddings = bert_model(input_question_ids, attention_mask=input_question_mask)[0]
question_embeddings = Reshape((-1, 768))(question_embeddings)
question_embeddings = GlobalAveragePooling1D()(question_embeddings)

merged_embeddings = Concatenate(axis=-1)([x, question_embeddings])
dense = Dense(512)
merged_embeddings = dense(merged_embeddings)

answer_logits = t5_model(input_ids=None, inputs_embeds=tf.expand_dims(merged_embeddings, 1),
                         decoder_input_ids=decoder_input, training=True)

model = tf.keras.models.Model(inputs=[input_image, input_question_ids, input_question_mask, decoder_input],
                              outputs=answer_logits.logits)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

decoder_input = np.zeros((len(Y_answer), 1), dtype=np.int32)
decoder_input[:, 0] = t5_tokenizer.pad_token_id

decoder_input_val = np.zeros((len(Y_answer_val), 1), dtype=np.int32)
decoder_input_val[:, 0] = t5_tokenizer.pad_token_id

# Train model
model.fit([X_images, X_question_ids, X_question_mask, decoder_input], Y_answer,
          validation_split=0.2,
          epochs=20, batch_size=16)

print('predicting trained data')
predicted_answer_logits = model.predict([X_images, X_question_ids, X_question_mask, decoder_input])
predicted_answers = [t5_tokenizer.decode(np.argmax(logits, axis=-1), skip_special_tokens=True) for logits in predicted_answer_logits]
results_df = pd.DataFrame({'predicted_answer': predicted_answers, 'ground_truth_answer': Y_true})


results_df.to_csv('vqa_results_train_5000.csv', index=False)

#P, R, F1 = score(predicted_answers, Y_true, lang="en", model_type="bert-base-uncased")

#mean_bert_score = np.mean(F1.numpy())
#print("Mean BERTScore:", mean_bert_score)

print('predicting val data')
predicted_answer_logits = model.predict([X_images_val, X_question_ids_val, X_question_mask_val, decoder_input_val])
predicted_answers = [t5_tokenizer.decode(np.argmax(logits, axis=-1), skip_special_tokens=True) for logits in predicted_answer_logits]
results_df = pd.DataFrame({'predicted_answer': predicted_answers, 'ground_truth_answer': Y_true_val})


results_df.to_csv('vqa_results_val_5000.csv', index=False)

#P, R, F1 = score(predicted_answers, Y_true, lang="en", model_type="bert-base-uncased")

#mean_bert_score = np.mean(F1.numpy())
#print("Mean BERTScore:", mean_bert_score)