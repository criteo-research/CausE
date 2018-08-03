# Causal-Prod2vec
from __future__ import absolute_import
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import utils as ut
import models as mo
from tensorflow.contrib.tensorboard.plugins import projector

tf.set_random_seed(42)

# Hyper-Parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_set', 'user_prod_dict.skew.', 'Dataset string.')  # Reg Skew
flags.DEFINE_string('adapt_stat', 'adapt_0', 'Adapt String.')  # Adaptation strategy
flags.DEFINE_string('model_name', 'cp2v', 'Name of the model for saving.')
flags.DEFINE_string('logging_dir', '/tmp/tensorboard', 'Name of the model for saving.')
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('num_steps', 500, 'Number of steps after which to test.')
flags.DEFINE_integer('embedding_size', 50, 'Size of each embedding vector.')
flags.DEFINE_integer('batch_size', 512, 'How big is a batch of training.')
flags.DEFINE_float('cf_pen', 1.0, 'Imbalance loss.')
flags.DEFINE_float('l2_pen', 0.0, 'L2 learning rate penalty.')
flags.DEFINE_string('cf_distance', 'l1', 'Use L1 or L2 for the loss .')
flags.DEFINE_bool('early_stopping_enabled', False, 'Enable early stopping.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')

train_data_set_location = "./Data/" + FLAGS.data_set +  "train." + FLAGS.adapt_stat + ".csv" # Location of train dataset
test_data_set_location = "./Data/" + FLAGS.data_set +  "test." + FLAGS.adapt_stat + ".csv" # Location of the test dataset
validation_test_set_location = "./Data/" + FLAGS.data_set +  "valid_test." + FLAGS.adapt_stat + ".csv" # Location of the validation dataset
validation_train_set_location = "./Data/" + FLAGS.data_set +  "valid_train." + FLAGS.adapt_stat + ".csv" #Location of the validation dataset

model_name = FLAGS.model_name + ".ckpt"
plot_gradients = False # Plot the gradients
cost_val = []

# Number of users and products in dataset
num_products = 1683
num_users = 944

# Create graph object
graph = tf.Graph()
with graph.as_default():

    with tf.device('/cpu:0'):
        
        # Create the model object
        model = mo.CausalProd2Vec(num_users, num_products, FLAGS.embedding_size, FLAGS.l2_pen, FLAGS.learning_rate, FLAGS.cf_pen, cf_distance=FLAGS.cf_distance)

        # Get train data batch from queue
        next_batch = ut.load_train_dataset(train_data_set_location, FLAGS.batch_size, FLAGS.num_epochs)
        test_user_batch, test_product_batch, test_label_batch, test_cr = ut.load_test_dataset(test_data_set_location)
        val_test_user_batch, val_test_product_batch, val_test_label_batch, val_cr = ut.load_test_dataset(validation_test_set_location)
        val_train_user_batch, val_train_product_batch, val_train_label_batch, val_cr = ut.load_test_dataset(validation_train_set_location)

        # create the empirical CR test logits 
        test_logits = np.empty(len(test_label_batch))
        test_logits.fill(test_cr)

# Launch the Session
with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Plot the gradients if required.
    if plot_gradients:
        # Create summaries to visualize weights
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        # Summarize all gradients
        for grad, var in model.grads:
            tf.summary.histogram(var.name + '/gradient', grad)

    # Setup tensorboard
    time_tb = str(time.ctime(int(time.time())))
    train_writer = tf.summary.FileWriter('/tmp/tensorboard' + '/train' + time_tb, sess.graph)
    test_writer = tf.summary.FileWriter('/tmp/tensorboard' + '/test' + time_tb, sess.graph)
    merged = tf.summary.merge_all()

    # Embeddings viz (Possible to add labels for embeddings later)
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = model.product_embeddings.name
    projector.visualize_embeddings(train_writer, config)

    # Variables used in the training loop
    t = time.time()
    step = 0
    average_loss = 0
    average_mse_loss = 0
    average_log_loss = 0

    # Start the training loop---------------------------------------------------------------------------------------------
    print("Starting Training On Causal Prod2Vec")
    print("Num Epochs = ", FLAGS.num_epochs)
    print("Learning Rate = ", FLAGS.learning_rate)
    try:
        while True:

            # Run the TRAIN for this step batch ---------------------------------------------------------------------
            with tf.device('/cpu:0'):
                # Construct the feed_dict
                user_batch, product_batch, label_batch = sess.run(next_batch)
                feed_dict = {model.user_list_placeholder : user_batch, model.product_list_placeholder: product_batch, model.label_list_placeholder: label_batch}

                # Run the graph
                _, sum_str, loss_val, mse_loss_val, log_loss_val = sess.run([model.apply_grads, merged, model.loss, model.mse_loss, model.log_loss], feed_dict=feed_dict)

            step +=1
            average_loss += loss_val
            average_mse_loss += mse_loss_val
            average_log_loss += log_loss_val

            # Every num_steps print average loss
            if step % FLAGS.num_steps == 0:
                if step > FLAGS.num_steps:
                    # The average loss is an estimate of the loss over the last set batches.
                    average_loss /= FLAGS.num_steps
                    average_mse_loss /= FLAGS.num_steps
                    average_log_loss /= FLAGS.num_steps
                print("Average Training Loss on S_c (FULL, MSE, NLL) at step ", step, ": ", average_loss, ": ", average_mse_loss, ": ", average_log_loss, "Time taken (S) = " + str(round(time.time() - t, 1)))

                average_loss = 0
                t = time.time() # reset the time
                train_writer.add_summary(sum_str, step) # Write the summary

                # Run the VALIDATION for this step batch ---------------------------------------------------------------------
                feed_dict_test = {model.user_list_placeholder : val_test_user_batch, model.product_list_placeholder: val_test_product_batch, model.label_list_placeholder: val_test_label_batch}
                feed_dict_train = {model.user_list_placeholder : val_train_user_batch, model.product_list_placeholder: val_train_product_batch, model.label_list_placeholder: val_train_label_batch}
                
                sum_str, loss_val, mse_loss_val, log_loss_val = sess.run([merged, model.loss, model.mse_loss, model.log_loss], feed_dict=feed_dict_test)
                cost_val.append(loss_val)
                print("Validation loss on S_t(FULL, MSE, NLL) at step ", step, ": ", loss_val, ": ", mse_loss_val, ": ", log_loss_val)

                sum_str, loss_val, mse_loss_val, log_loss_val = sess.run([merged, model.loss, model.mse_loss, model.log_loss], feed_dict=feed_dict_train)
                print("Validation loss on S_c (FULL, MSE, NLL) at step ", step, ": ", loss_val, ": ", mse_loss_val, ": ", log_loss_val)
                print("####################################################################################################################")   

                test_writer.add_summary(sum_str, step) # Write the summary

                # If condition for the early stopping condition
                if FLAGS.early_stopping_enabled and step > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                    print("Early stopping...")
                    saver.save(sess, os.path.join(FLAGS.logging_dir, model_name), model.global_step) # Save model
                    break

    except tf.errors.OutOfRangeError:
        print("Reached the required number of epochs")

    finally:
        with tf.device('/cpu:0'):
            saver.save(sess, os.path.join(FLAGS.logging_dir, model_name), model.global_step) # Save model

    train_writer.close()
    print("Training Complete")

    # Run the test set for the trained model --------------------------------------------------------------------------
    print("Running Test Set")
    feed_dict = {model.user_list_placeholder : test_user_batch, model.product_list_placeholder: test_product_batch, model.label_list_placeholder: test_label_batch}
    loss_val, mse_loss_val, log_loss_val = sess.run([model.loss, model.mse_loss, model.log_loss], feed_dict=feed_dict)
    print("Test loss (CE, MSE, NLL) = ", loss_val, ": ", mse_loss_val , ": ",log_loss_val)

    # Run the bootstrap for this model ---------------------------------------------------------------------------------------------------------------
    print("Begin Bootstrap process...")
    ut.compute_bootstraps(sess, model, test_user_batch, test_product_batch, test_label_batch, test_logits, model.ap_mse_loss, model.ap_log_loss)