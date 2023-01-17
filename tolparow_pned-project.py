import os
import logging
import keras

# CODEBASE LINK: https://gitlab.com/tolparow/pned/-/archive/dev/pned-dev.zip

# Change directory to import main code
os.chdir('/kaggle/input/pnedcode/pned-dev/pned-dev/')

logging.basicConfig()
logger = logging.getLogger('my')
logger.warning(str(os.listdir()))

from framework.data.generator import DataGenerator
from framework.data.source import DataSource

from framework.model.evaluation import intersection_over_union, mean_intersection_over_union
from framework.model.network import SimpleCNN

# Reset to working directory
os.chdir("/kaggle/working/")


data_source = DataSource(
    '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images',
    '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
)
data_source.split_dataset()

train_data_generator = DataGenerator(
    data_source,
    batch_size=64,
    augment=True,
    dataset=1
)

validate_data_generator = DataGenerator(
    data_source,
    batch_size=16,
    augment=True,
    dataset=2
)

network = SimpleCNN(optimizer='adam',
                    loss=intersection_over_union,
                    metrics=[mean_intersection_over_union, keras.losses.binary_crossentropy],
                    input_size=256,
                    channels=32,
                    n_blocks=2,
                    depth=4)

model = network.get_compiled_model()

save_epochs_callback = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5',
                                                       monitor='val_loss',
                                                       verbose=1,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       period=1)

history = model.fit_generator(train_data_generator,
                              validation_data=validate_data_generator,
                              callbacks=[save_epochs_callback],
                              epochs=15)

model.save('model.h5')

logger.warning(str(os.listdir()))
