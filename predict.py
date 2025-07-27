import argparse
import warnings
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import logging
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image') #dest='image filepath', default='./test_images/wild_pansy.jpg')
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--model_path', type=str, default='./sm_flower_classifier_project.h5', help='Path to saved Keras model')
    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Top K most likely classes')  
    parser.add_argument('--category_names', dest='category_names', default='label_map.json', help='Path to JSON file mapping labels to names')
    parser.add_argument('--gpu', action='store', default='gpu')
    
    return parser.parse_args()



def load_split_data():
    dataset = tfds.load('oxford_flowers102', shuffle_files=True, as_supervised = True, with_info = False)
    training_set, validation_set, testing_set = dataset['train'], dataset['validation'], dataset['test']
    num_training_examples = dataset_info.splits['train'].num_examples
    return training_set, testing_set, validation_set, num_training_examples

image_size = 224 
def format_image(image, label):
    image = tf.cast(image, tf.float32) #from  unit8 to float32
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255 #rescaling images to be between 0 and 1
    return image, label


batch_size = 32
def batch_data(training_set, test_set, valid_set, num_training_examples):
    train_batches = (training_set.cache().map(format_image(image, label)).shuffle(num_train_examples//4).batch(BATCH_SIZE).prefetch(1))
    val_batches = (validation_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1))
    test_batches = (testing_set.cache().map(format_image).batch(BATCH_SIZE).prefetch(1))
    return train_batches, test_batches, val_batches

def map_data():
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)


# Loading the model

#model = "sm_flower_classifier_project.h5"
def load_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_model

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy().squeeze()
    return image


def predict(image, model, top_k = 5):
    if not os.path.exists(image):
        print(f"Error: Image file not found at {image}")
        exit(1)

    try:
        image = Image.open(image)
    except (FileNotFoundError, OSError) as e:
        print(f"Error opening image file: {e}")
        raise
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    predictions = model.predict(image)[0] #predictions is a list of lists, we selected one that only have it

    #top_k_indices = predictions.argsort()[-top_k:][::-1]
    #top_k_probs = predictions[top_k_indices]

    top_k_probs, top_k_indices = tf.math.top_k(predictions, k=top_k)
    probs = top_k_probs.numpy().tolist()
    classes = top_k_indices.numpy().astype(str).tolist()

    return probs, classes



def main():
    args = get_args()
    gpu = args.gpu

    # Load the model
    #model = load_checkpoint(args.checkpoint)
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Predict
    print('\n\nFile selected image_path: ' + args.image_path)
    print(f'selected top_k: {args.top_k}\n')
    probs, classes = predict(args.image_path, model, int(args.top_k))

    # Map to names if provided
    print('selected category_names: ' + args.category_names)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        try:
            class_labels = [class_names[str(index)] for index in classes]
        except KeyError as e:
            print(f"Missing class name for index: {e}")
            raise
    else:
        class_labels = classes

    print('\n probs: ', probs)
    print('Classes: ', classes)
    print('class_labels: ', class_labels, '\n')


    i=0 # this prints out top k classes and probs as according to user
    while i < len(class_labels):
        print("{} with a probability of {}".format(class_labels[i], probs[i]))
        i += 1 # cycle through

    # Display results
    print("\nTop K Predictions:")
    for prob, label in zip(probs, class_labels):
        print(f"{label}: {prob:.4f}")

    # Most likely class
    print(f"\nMost Likely Class: {class_labels[0]} ({probs[0]:.4f})")


if __name__ == "__main__":
    main()