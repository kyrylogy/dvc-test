import tensorflow as tf
from config import params
import os
from pathlib import Path
import matplotlib.pylab as plt

output_dir = Path("./data/samples")
output_dir.mkdir(exist_ok=True, parents=True)


def get_label(file_path):  
    parts = tf.strings.split(file_path, os.path.sep)
    parts = tf.strings.split(parts[-1], ".")
    one_hot = parts[0] == params["data"]["classes"]
    return one_hot
    

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, params["data"]["img_size"])

_HASH_LAYER = tf.keras.layers.Hashing(num_bins=100)
def get_hash(file_path):
    return _HASH_LAYER(file_path)
    

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    hash = get_hash(file_path)
    return img, label, hash

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def filter_and_prepare(ds, hash_filter):
    return (ds        
        .filter(lambda img,label,hash: hash_filter(hash))
        .map(lambda img,label,hash: (img, label))
        .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(buffer_size=1000)
        .batch(params["data"]["batch_size"])
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

def get_dataset():
    ds = tf.data.Dataset.list_files(str(Path(params["data"]["train_path"]) / "*"), shuffle=True)
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # we split the train and validation set using a hash function on the file name
    # this way the outcome is completely deterministic
    train_ds = filter_and_prepare(ds, lambda hash: hash < params["data"]["test_split"] )
    validation_ds = filter_and_prepare(ds, lambda hash: params["data"]["test_split"] <= hash < (params["data"]["test_split"] + params["data"]["validataion_split"]))
    test_ds = filter_and_prepare(ds, lambda hash: hash >= (params["data"]["test_split"] + params["data"]["validataion_split"]))        

    return train_ds, validation_ds, test_ds


def sample_images():
    sets = get_dataset()
    for ds, label in zip(sets, ["training", "validation", "test"]):
        plt.figure(figsize=(20,20))
        plt.suptitle("dataset: %s" % label)
        images, labels = list(ds.take(1))[0]
        for i in range(16):
            plt.subplot(4,4,i+1)        
            plt.imshow(images[i, ...])
            plt.title("%s - %s" % (str(labels[i].numpy().tolist()), params["data"]["classes"][tf.argmax(labels[i])]))
        plt.savefig(output_dir / ("samples_%s.png" % label))
        plt.close()
        

if __name__ == "__main__":
    sample_images()        