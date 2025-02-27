import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Remove the corrupted Images
files = glob.glob("6 Emotions for image classification/*/*")
for file in files :
   f = open(file, "rb") # open to read binary file
   if not b"JFIF" in f.peek(10) :
          f.close()
          os.remove(file)
   else :
          f.close()


# Load Training and Testing Dataset
training_set = image_dataset_from_directory("6 Emotions for image classification", 
                                            validation_split=0.2, 
                                            subset="training", 
                                            label_mode="categorical", 
                                            seed=0, 
                                            image_size=(100,100))

test_set = image_dataset_from_directory("6 Emotions for image classification", 
                                        validation_split=0.2, 
                                        subset="validation", 
                                        label_mode="categorical", 
                                        seed=0, 
                                        image_size=(100,100))

# Task -1 - Building 2 CNN Model 

# First Model
cnn = Sequential()
cnn.add(Input((100,100,3)))
cnn.add(Rescaling(1/255))
cnn.add(Conv2D(32, kernel_size=(3,3),activation= "relu"))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.2))
cnn.add(Dense(6, activation="softmax"))

cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"])
history  = cnn.fit(training_set, epochs=20,validation_data= (test_set), verbose=2)
history.history['accuracy']

# Evaluation
score = cnn.evaluate(test_set, verbose=2)
print("Test accuracy:", round(score[1],3))

# Visualise the loss propagation over epoch
plt.figure(figsize= [10,7])
plt.title('Training Loss and Validation Loss over Epoch')
plt.plot(history.history['loss'],label = "training")
plt.plot(history.history['val_loss'], label = "validation")
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# Visualise the Accuracy over epoch
plt.figure(figsize= [10,7])
plt.title('Training Accuracy and Validation Accuracy over Epoch')
plt.plot(history.history['accuracy'], label = "training")
plt.plot(history.history['val_accuracy'], label = "validation")
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# Save the First model
cnn.save('cnn1_model.keras')
cnn1_model = load_model('cnn1_model.keras')

# Second Model
cnn = Sequential()

cnn.add(Input((100,100,3)))
cnn.add(Rescaling(1/255))
cnn.add(Conv2D(32, kernel_size=(5,5),activation= "relu"))
cnn.add(MaxPooling2D(pool_size=(3,3)))
cnn.add(Conv2D(64, (5, 5), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(3, 3)))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.2))
cnn.add(Dense(6, activation="softmax"))

cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"])
history  = cnn.fit(training_set, epochs=20,validation_data= (test_set), verbose=2)

history.history['accuracy']
score = cnn.evaluate(test_set, verbose=2)
print("Test accuracy:", round(score[1],3))

# Visualise the loss propagation over epoch
plt.figure(figsize= [10,7])
plt.title('Training Loss and Validation Loss over Epoch')
plt.plot(history.history['loss'],label = "training")
plt.plot(history.history['val_loss'], label = "validation")
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# Visualise the Accuracy over epoch
plt.figure(figsize= [10,7])
plt.title('Training Accuracy and Validation Accuracy over Epoch')
plt.plot(history.history['accuracy'], label = "training")
plt.plot(history.history['val_accuracy'], label = "validation")
plt.legend(loc = 'upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

cnn.save('cnn2_model.keras')
cnn2_model = load_model('cnn2_model.keras')

# Task - 2: Fine tune a pre-trained model -  EfficientNetB0
# Load the base model
base_model = EfficientNetB0(include_top  = False)

x = base_model.output 
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x)
output_layer = Dense(6, activation="softmax")(x)
m = Model(inputs=base_model.input, outputs=output_layer)

for layer in base_model.layers:
        layer.trainable = False

m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
history = m.fit(training_set, epochs=20,validation_data= (test_set), verbose=2)

# Save the fine-tuned model
m.save('pre_trained_model.keras')
pre_trained_model = load_model('pre_trained_model.keras')
score = m.evaluate(test_set, verbose=2)
print("Test accuracy for Pre-trained Model:", round(score[1],3))

# Test Accuracy for different models
result_dict = {
    'Model': ['CNN-1', 'CNN-2', 'Pre-trained'],
    'Test Accuracy': [cnn1_model.evaluate(test_set,verbose= 0)[1],
                          cnn2_model.evaluate(test_set,verbose= 0)[1],
                          pre_trained_model.evaluate(test_set,verbose= 0)[1]]
}
print(f'\nTest Accuracy of Different Models:\n{pd.DataFrame(result_dict)}')

# Task 3: Predicting from Manually Generated Image
true_Test_Labels = [0 , 3, 5, 3, 5, 0, 2, 3, 5, 2]
plt.figure(figsize=[15,10])
plt.title('Prediction using CNN-1')
plt.axis('off')
for i in range(10):
    image = load_img(f'test {i+1}.jpeg', target_size= (100,100))
    image_arr = np.expand_dims(img_to_array(image), axis = 0)
    prediction = cnn1_model.predict(image_arr)
    plt.subplot(5,2,i+1)
    plt.title(f'prediction: {np.argmax(prediction)}, true: {true_Test_Labels[i]}')
    plt.imshow(image_arr[0].astype('uint8'))
    plt.axis('off')

plt.show()

plt.figure(figsize=[15,10])
plt.title('Prediction using Pre-trained Model')
plt.axis('off')
for i in range(10):
    image = load_img(f'test {i+1}.jpeg', target_size= (100,100))
    image_arr = np.expand_dims(img_to_array(image), axis = 0)
    prediction = pre_trained_model.predict(image_arr)
    plt.subplot(5,2,i+1)
    plt.title(f'prediction: {np.argmax(prediction)}, true: {true_Test_Labels[i]}')
    plt.imshow(image_arr[0].astype('uint8'))
    plt.axis('off')

plt.show()