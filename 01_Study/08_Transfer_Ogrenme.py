""" Transfer Öğrenme (Transfer Learning)
Transfer öğrenme, daha önce öğrenilmiş bilgiye dayalı olarak yeni bir görevde öğrenmeyi hızlandırır.
 Özellikle derin öğrenmede yaygındır.

Özellik: Önceden eğitilmiş modellerin kullanılması, yeni görevde eğitim sürecini hızlandırır.

Uygulamalar: Görüntü sınıflandırma, dil modelleme

1. Keras ile Transfer Öğrenme: VGG16 Modeli ile Görüntü Sınıflandırma
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# VGG16 modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Üst katmanları oluşturun
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanların eğitilmesini durdurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
2. Keras ile Transfer Öğrenme: ResNet50 Modeli ile Görüntü Sınıflandırma
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# ResNet50 modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
3. Keras ile Transfer Öğrenme: InceptionV3 Modeli ile Görüntü Sınıflandırma
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# InceptionV3 modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Üst katmanları oluşturun
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(299, 299), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
4. Keras ile Transfer Öğrenme: MobileNet Modeli ile Görüntü Sınıflandırma
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# MobileNet modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
5. Keras ile Transfer Öğrenme: Xception Modeli ile Görüntü Sınıflandırma
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Xception modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Yeni katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(299, 299), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
6. Hugging Face Transformers ile Transfer Öğrenme: BERT ile Sentiment Analysis
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset

# BERT tokenizer ve modelini yükleyin
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Veri setini yükleyin (Hugging Face Dataset)
dataset = load_dataset("imdb")

# Tokenize veriyi hazırlayın
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Model eğitimi için ayarları yapılandırın
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Eğitim işlemini başlatın
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()
7. Hugging Face Transformers ile Transfer Öğrenme: GPT-2 ile Metin Tamamlama
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 tokenizer ve modelini yükleyin
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Girdi metni
input_text = "Transfer learning is a technique in which"

# Tokenize veriyi hazırlayın
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Modeli kullanarak metni tamamlayın
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

# Sonuçları yazdırın
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
8. Keras ile Transfer Öğrenme: DenseNet Modeli ile Görüntü Sınıflandırma
from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# DenseNet121 modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
9. Keras ile Transfer Öğrenme: NASNet Modeli ile Görüntü Sınıflandırma
from keras.applications import NASNetMobile
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# NASNet modelini yükleyin, önceden eğitilmiş ağırlıkları kullanarak
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Yeni model
model = Model(inputs=base_model.input, outputs=x)

# Tüm katmanları dondurun
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleyin
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Görüntü verisini hazırlayın
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

# Modeli eğitin
model.fit(train_generator, epochs=10, steps_per_epoch=100)
10. Hugging Face Transformers ile Transfer Öğrenme: BART ile Metin Özetleme
from transformers import BartForConditionalGeneration, BartTokenizer

# BART modelini yükleyin
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Metin verisi
input_text = "Transfer learning is a machine learning technique that allows a model to leverage knowledge learned from one task to improve performance on another task."

# Tokenize ve model ile özetleme yapın
inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

# Sonuçları yazdırın
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

"""