import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Define the dataset path correctly
dataset_path = r"C:\Users\admin\OneDrive\Desktop\project files\Brain Tumor Detection (BTD)\brain tumr\brain tumors resnet\datasets"

# Load ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)  # Flatten the output
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dense(4, activation='softmax')(x)  # Output layer (4 classes)


model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#  Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path + "/training",  #  Use full dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path + "/validation",  #  Use full dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(train_data, validation_data=val_data, epochs=45)

# Save the trained model
model.save("brain_tumor_model.h5")
