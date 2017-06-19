import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random

show_confusion_matrix=True

batch_size = 32 

img_size = 128 

validation_size = 1. 

classes = ['dog', 'cat']
num_classes = len(classes)
num_channels = 3

img_size_flat = img_size * img_size * num_channels

train_path = '/root/tensorflow_kaggle_cat_dog/cat_dog/test'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

#x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

#y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
#y_true_cls = tf.argmax(y_true, dimension=1)
def plot_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

with tf.Session() as session:
   #x = tf.Variable(-1.0, validate_shape=False, name='x')
   #y = tf.Variable(-1.0, validate_shape=False, name='y')
   #saver = tf.train.import_meta_graph('./checkpoint/model.ckpt-9788.meta', clear_devices=True)
   #saver.restore(session,tf.train.latest_checkpoint('./checkpoint'))
   saver = tf.train.Saver()
   saver.restore(session, './backup.chk')
   graph = tf.get_default_graph()
   #for n in graph.as_graph_def().node:
   #    print n.name
   #layer_fc2 = graph.get_tensor_by_name("accuracy:0")
   x = graph.get_tensor_by_name("x:0")
   y_true = graph.get_tensor_by_name("y_true:0")
   y_pred_cls = graph.get_tensor_by_name("y_pred_cls:0")
   #session.run(tf.global_variables_initializer())
   #session.run(tf.all_variables())
   #print(session.run(tf.all_variables()))
   #print(session.run('bias:0'))
   num_test = len(data.valid.images)
   cls_pred = np.zeros(shape=num_test, dtype=np.int)
   i = 0

   while i < num_test:
       j = min(i + batch_size, num_test)

        #import ipdb ; ipdb.set_trace()
        # Get the images from the test-set between index i and j.
       reshape_size = batch_size
       if data.valid.images[i:j, :].shape[0] < reshape_size:
           reshape_size = data.valid.images[i:j, :].shape[0] 
       try:
           images = data.valid.images[i:j, :].reshape(reshape_size, img_size_flat)
       except:
           print i 
           print j 
           print data.valid.images[i:j, :].shape
           print batch_size 
           print img_size_flat 

        # Get the associated labels.
       labels = data.valid.labels[i:j, :]
        # Create a feed-dict with these images and labels.
       feed_dict = {x: images,
                    y_true: labels}

        # Calculate the predicted class using TensorFlow.
       cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
       i = j

   cls_true = np.array(data.valid.cls)
   cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
   correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
   correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
   acc = float(correct_sum) / num_test

    # Print the accuracy.
   msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
   print(msg.format(acc, correct_sum, num_test))
   if show_confusion_matrix:
       print("Confusion Matrix:")
       plot_confusion_matrix(cls_pred=cls_pred)




