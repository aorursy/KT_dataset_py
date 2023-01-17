import tensorflow as tf
train_folder = "../input/training_set/training_set"

test_folder = "../input/test_set/test_set"

target_size = (100, 150, 3)
gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_images = gen.flow_from_directory(train_folder, target_size=target_size[:2], batch_size=40)

test_images = gen.flow_from_directory(test_folder, target_size=target_size[:2], batch_size=40)
def create_model(target_size=(225, 300, 3)):

    model = tf.keras.models.Sequential([

        # Normalizing Input Tensor

        tf.keras.layers.BatchNormalization(input_shape=target_size),

        # Convolutional Layer #1

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer #2

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer #3

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten Layer

        tf.keras.layers.Flatten(),

        # Dense Layer #1

        tf.keras.layers.Dense(128, activation="relu"),

        # Dense Layer #2

        tf.keras.layers.Dense(64, activation="relu"),

        # Dropout Layer

        tf.keras.layers.Dropout(0.5),

        # Output Layer

        tf.keras.layers.Dense(2, activation="softmax")

    ])



    return model
model = create_model(target_size=target_size)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_images, epochs=10, steps_per_epoch=8005 // 40,)
print(f"Accuracy: {model.evaluate_generator(test_images, 2000 // 40)[1]}")