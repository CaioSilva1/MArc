from __future__ import print_function

import tensorflow as tf
import os
import tempfile
import numpy as np
import sys

from glob import glob
from scipy.misc import imresize
from skimage import io
import random



def lerBatchImagens(batch_size, homem_files, mulher_files):
    if batch_size <= len(homem_files):
        imagens = np.zeros((batch_size, 128, 128, 3), dtype='float64')
        shape_labels = (batch_size,2)
        batch_size_homem = batch_size/2
        batch_size_mulher= batch_size - batch_size_homem
    else:
        imagens = np.zeros((len(homem_files)+len(homem_files), 128, 128, 3), dtype='float64')
        shape_labels = (len(homem_files)+len(homem_files),2)
        batch_size_homem = len(homem_files)
        batch_size_mulher= len(mulher_files)
    
    labels = np.zeros(shape_labels)
    qtd = 0
    
    i = 0
    while(i < batch_size_homem):
        name_file = io.imread(homem_files.pop())
        new_img = imresize(name_file, (128, 128, 3))
        imagens[qtd] = new_img
        labels[qtd][0] = 1
        qtd += 1
        i += 1
    i =0
    while(i < batch_size_mulher):
        name_file = io.imread(mulher_files.pop())
        new_img = imresize(name_file, (128, 128, 3))
        imagens[qtd] = new_img
        labels[qtd][1] = 1
        qtd += 1
        i += 1
    
                
    return imagens, labels
    
def lerImagens():    
    homem_files_path = 'treino/homem/'
    mulher_files_path = 'treino/mulher/'
    homem_files_path = os.path.join(homem_files_path, 'homem*.jpg')
    mulher_files_path = os.path.join(mulher_files_path, 'mulher*.jpg')
    
    homem_files = sorted(glob(homem_files_path))
    mulher_files = sorted(glob(mulher_files_path))
    
    homem_files = random.sample(homem_files, len(homem_files))
    mulher_files = random.sample(mulher_files, len(mulher_files))
    
    return homem_files, mulher_files

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_strides(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def CNN_gender(x_image):  
    
    print('\n# 1 camada')
    # 1
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([10, 10, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d_strides(x_image, W_conv1) + b_conv1)
        print(h_conv1)
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
        print(h_pool1)
    print('# 2 camada')
    
    # 2
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([7, 7, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        print(h_conv2)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        print(h_pool2)
    print('# 3 camada')
    

   # 3
    with tf.name_scope('conv3'):
        W_conv2 = weight_variable([7, 7, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv2) + b_conv2)
        print(h_conv3)
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv3)
        print(h_pool2)
        
    print('# dense 1')
    
    
    # dense 1
    with tf.name_scope('dense1'):
        W_fc1 = weight_variable([1152, 1152])
        b_fc1 = bias_variable([1152])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        print(h_fc1)
    
    print('# dense 2')
          
    # dense 2
    with tf.name_scope('dense2'):
        W_fc2 = weight_variable([1152, 768])
        b_fc2 = bias_variable([768])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        print(h_fc2)
    
    print('# dropout')      
    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        final = tf.nn.dropout(h_fc2, keep_prob)
        print(final)
        
    print('# final')
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('final'):
        W_fc2 = weight_variable([768, 2])
        b_fc2 = bias_variable([2])
        y_conv = tf.matmul(final, W_fc2) + b_fc2
        print(y_conv)
    
    print('\n')
    return y_conv, keep_prob
            
def main():
    # Exibir quantidade de dados
    homem_files, mulher_files = lerImagens()
    print('\n# DADOS #')
    print('Imagens Homem',len(homem_files))
    print('Imagens Mulher',len(mulher_files))
    print('Total Imagens', len(homem_files)+len(mulher_files))
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    # Build the graph for the deep net
    y_conv, keep_prob = CNN_gender(x)
    
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
#    graph_location = tempfile.mkdtemp()
#    print('Saving graph to: %s' % graph_location)
#    train_writer = tf.summary.FileWriter(graph_location)
#    train_writer.add_graph(tf.get_default_graph())

    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.Session() as sess:
        homem_files, mulher_files = lerImagens()
        print('\n# Iniciando Treinamento da CNN #')
        
        sess.run(tf.global_variables_initializer())
        
        learning_rate_decaying = 0.005
        for i in range(10000):
            if len(homem_files)==0:
                homem_files, mulher_files = lerImagens() 
                batch_imgs, labels = lerBatchImagens(100, homem_files, mulher_files);
            else:
                batch_imgs, labels = lerBatchImagens(100, homem_files, mulher_files);
            
            #if i != 0 and i % 100 == 0:
            #    learning_rate_decaying = learning_rate_decaying/2
           
            #if i % 3000 == 0:
            #train_accuracy = accuracy.eval(feed_dict={x: batch_imgs, y_: labels, keep_prob: 0.5})
            #print('epoch %d, Treinamento accuracy %g, learning rate: %g' % (i/20, train_accuracy, learning_rate_decaying))
            
            train_accuracy = accuracy.eval(feed_dict={x: batch_imgs, y_: labels, keep_prob: 0.5})
            print('epoch %d, Treinamento accuracy %g, learning rate: %g' % (i, train_accuracy, learning_rate_decaying))
            
            sess.run(train_step, feed_dict={x: batch_imgs, y_: labels, keep_prob: 0.5, learning_rate: learning_rate_decaying})

            
if __name__ == '__main__':
    main()
    
