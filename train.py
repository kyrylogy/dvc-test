import tensorflow as tf
from config import params
from pathlib import Path
from dataset import get_dataset
import io
from contextlib import redirect_stdout

output_dir = Path("./data/model")
output_dir.mkdir(exist_ok=True, parents=True)


def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=params["data"]["img_size"] + [3]),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(params["data"]["classes"]), activation='softmax')
    ])

def load_model():
    return tf.keras.models.load_model(output_dir / "model")

def train():

    model = get_model()

    # write the model summary to a file
    with io.StringIO() as f, redirect_stdout(f):
        model.summary()
        with open(output_dir / "model.txt", "w") as fp:
            fp.write(f.getvalue())

    model.compile(optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ])
    
    train_ds, validation_ds, _ = get_dataset()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./data/model/logs',
        profile_batch=5,
        write_images=True
    )

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=params["train"]["num_epochs"],
        callbacks=[tensorboard_callback]
    )

    model.save(output_dir / "model")
    

if __name__ == "__main__":
    train()