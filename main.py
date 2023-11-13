from kubeflow import katib
from kubeflow.katib import search


def train_model(parameters):
    import logging
    import keras
    from keras.models import Model
    from keras.layers import Dense, Flatten
    from keras.applications.mobilenet import MobileNet
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    # Callback class for logging training.
    # Katib parses metrics in this format: <metric-name>=<metric-value>.

    lr = float(parameters["lr"])
    num_epoch = int(parameters["num_epoch"])

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(
                "Epoch {}/{}. accuracy={:.4f} - loss={:.4f}".format(
                    epoch+1, num_epoch, logs["accuracy"], logs["loss"]
                )
            )

    def face_dataset(batch_size):
        import boto3
        import zipfile

        batch_size = batch_size

        bucket_name = 'dataset'
        object_key = 'dataset.zip'
        file_path = 'dataset.zip'

        zip_file_path = 'dataset.zip'
        extract_folder_path = './dataset'

        s3 = boto3.resource('s3', endpoint_url="https://9751378c048c7a61edc23a6e76716451.r2.cloudflarestorage.com",
                            aws_access_key_id="f1734ac33562ff774e86bb6f7289a166",
                            aws_secret_access_key="baf0db2a429042223e9159397ae025c348418bd1df8b5d8136ff9f77577ee4e1")
        s3.Object(bucket_name, object_key).download_file(file_path)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder_path)

    def build_and_compile_mobilenet_model():

        base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)

        for layer in base_model.layers:
            layer.trainable = False

        x = (base_model.output)  # more capture features
        x = Flatten()(x)
        # softmax is an activation that obtains a probability distribuiltion over different classes
        x = Dense(units=7, activation='softmax')(x)

        # creating our model.
        model = Model(base_model.input, x)

        model.summary()

        model.compile(optimizer='adam', loss="categorical_crossentropy",
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            zoom_range=0.2, shear_range=0.2, horizontal_flip=True, rescale=1./255)

        train_data = train_datagen.flow_from_directory(
            directory="./dataset/train", target_size=(224, 224), batch_size=32)

        train_data.class_indices

        val_datagen = ImageDataGenerator(
            zoom_range=0.2, shear_range=0.2, horizontal_flip=True, rescale=1./255)

        val_data = val_datagen.flow_from_directory(
            directory="./dataset/test", target_size=(224, 224), batch_size=32)

        val_data.class_indices

        es = EarlyStopping(monitor='val_accuracy', min_delta=0.01,
                           patience=5, verbose=1, mode='auto')

        model.compile(optimizer='adam',
                      loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(train_data, epochs=5, validation_data=val_data, validation_steps=10, callbacks=[CustomCallback(), es])

        return model

    face_dataset(8)
    build_and_compile_mobilenet_model()


config_file = "./config"

parameters = {
    "lr": katib.search.double(min=0.1, max=0.2),
    "num_epoch": katib.search.int(min=50, max=100),
    "is_dist": False,
    "num_workers": 1
}

# Create Katib Experiment.
katib_client = katib.KatibClient(config_file)
name = "train-cnn"
namespace = "kubeflow"

katib_client.tune(
    name=name,
    namespace=namespace,
    base_image="docker.io/tensorflow/tensorflow:2.12.0-gpu",
    objective=train_model,  # Objective function.
    parameters=parameters,  # HyperParameters to tune.
    algorithm_name="bayesianoptimization",  # Alorithm to use.
    objective_metric_name="accuracy",  # Katib is going to optimize "accuracy".
    # Katib is going to collect these metrics in addition to the objective metric.
    additional_metric_names=["loss"],
    max_trial_count=1,  # Trial Threshold.
    parallel_trial_count=1,
    packages_to_install=["boto3", "pillow", "scipy"]
)
