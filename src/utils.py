from __future__ import absolute_import, print_function

import csv
import random

import numpy as np
import tensorflow as tf
from sklearn import metrics

def load_train_dataset(dataset_location, batch_size, num_epochs):
    """Load the training data using TF Dataset API"""

    with tf.name_scope('train_dataset_loading'):

        record_defaults = [[1], [1], [0.]] # Sets the type of the resulting tensors and default values
        # Dataset is in the format - UserID ProductID Rating
        dataset = tf.data.TextLineDataset(dataset_location).map(lambda line: tf.decode_csv(line, record_defaults=record_defaults))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(5)
        dataset = dataset.cache()
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        user_batch, product_batch, label_batch = iterator.get_next()
        label_batch = tf.expand_dims(label_batch, 1)

    return user_batch, product_batch, label_batch

def load_test_dataset(dataset_location):
    """Load the test and validation datasets"""

    user_list = []
    product_list = []
    labels = []

    with open(dataset_location, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            user_list.append(row[0])
            product_list.append(row[1])
            labels.append(row[2])

    labels = np.reshape(labels, [-1, 1])
    cr = compute_empircal_cr(labels)

    return user_list, product_list, labels, cr

def generate_bootstrap_batch(seed, data_set_size):
    """Generate the IDs for the bootstap"""

    random.seed(seed)
    ids = [random.randint(0, data_set_size-1) for j in range(int(data_set_size*0.8))]

    return ids

def compute_empircal_cr(labels):
    """Compute the cr from the empirical data"""

    labels = labels.astype(np.float)
    clicks = np.count_nonzero(labels)
    views = len(np.where(labels==0)[0])
    cr = float(clicks)/float(views)

    return cr

def compute_2i_regularization_id(prods, num_products):
    """Compute the ID for the regularization for the 2i approach"""

    reg_ids = []
    # Loop through batch and compute if the product ID is greater than the number of products
    for x in np.nditer(prods):
        if x >= num_products:
            reg_ids.append(x)
        elif x < num_products:
            reg_ids.append(x + num_products) # Add number of products to create the 2i representation 

    return np.asarray(reg_ids)

def compute_treatment_or_control(prods, num_products):
    """Compute if product is in treatment or control"""
    # Return the control product places and treatment places as 1's in a binary matrix.
    ids = []
    for x in np.nditer(prods):
        # Greater than the number of products
        if x >= num_products:
            ids.append(0)
        elif x < num_products:
            ids.append(1)
    # create the binary mask and return 
    return np.asarray(ids), np.logical_not(np.asarray(ids)).astype(int)

def compute_bootstraps_2i(sess, model, test_user_batch, test_product_batch, test_label_batch, test_logits, ap_mse_loss, ap_log_loss):
    """Compute the bootstraps for the 2i model"""

    data_set_size = len(test_user_batch)
    mse = []
    llh = []
    ap_mse = []
    ap_llh = []
    auc_list = []
    mse_diff = []
    llh_diff = []

    # Compute the bootstrap values for the test split - this compute the empirical CR as well for comparision
    for i in range(30):

        ids = generate_bootstrap_batch(i*2, data_set_size)
        test_user_batch = np.asarray(test_user_batch)
        test_product_batch = np.asarray(test_product_batch)
        test_label_batch = np.asarray(test_label_batch)

        # Construct the feed-dict for the model and the average predictor
        feed_dict = {model.user_list_placeholder : test_user_batch[ids], model.product_list_placeholder: test_product_batch[ids], model.label_list_placeholder: test_label_batch[ids], model.logits_placeholder: test_logits[ids], model.reg_list_placeholder: test_product_batch[ids]}

        # Run the model test step updating the AUC object
        loss_val, mse_loss_val, log_loss_val, pred = sess.run([model.loss, model.mse_loss, model.log_loss, model.prediction], feed_dict=feed_dict)

        # Run the Average Predictor graph
        ap_mse_val, ap_log_val = sess.run([ap_mse_loss, ap_log_loss], feed_dict=feed_dict)

        mse.append(mse_loss_val)
        llh.append(log_loss_val)
        ap_mse.append(ap_mse_val)
        ap_llh.append(ap_log_val)
        auc_list.append(metrics.roc_auc_score(y_true=test_label_batch[ids].astype(int), y_score=pred))

    for i in range(30):
        mse_diff.append((ap_mse[i]-mse[i]) / ap_mse[i])
        llh_diff.append((ap_llh[i]-llh[i]) / ap_llh[i])

    print("MSE Mean Score On The Bootstrap = ", np.mean(mse))
    print("MSE Mean Lift Over Average Predictor (%) = ", np.round(np.mean(mse_diff)*100, decimals=2))
    print("MSE STD (%) =" , np.round(np.std(mse_diff)*100, decimals=2))

    print("LLH Mean Over Average Predictor (%) =", np.round(np.mean(llh_diff)*100, decimals=2))
    print("LLH STD (%) = ", np.round(np.std(llh_diff)*100, decimals=2))

    print("Mean AUC Score On The Bootstrap = ",  np.round(np.mean(auc_list), decimals=4), "+/-", np.round(np.std(auc_list), decimals=4))

def compute_bootstraps(sess, model, test_user_batch, test_product_batch, test_label_batch, test_logits, ap_mse_loss, ap_log_loss):
    """Compute the bootstraps for the 0 indexed model"""

    data_set_size = len(test_user_batch)
    mse = []
    llh = []
    ap_mse = []
    ap_llh = []
    auc_list = []
    mse_diff = []
    llh_diff = []

    # Compute the bootstrap values for the test split - this compute the empirical CR as well for comparision
    for i in range(30):

        ids = generate_bootstrap_batch(i*2, data_set_size)
        test_user_batch = np.asarray(test_user_batch)
        test_product_batch = np.asarray(test_product_batch)
        test_label_batch = np.asarray(test_label_batch)

        # Construct the feed-dict for the model and the average predictor
        feed_dict = {model.user_list_placeholder : test_user_batch[ids], model.product_list_placeholder: test_product_batch[ids], model.label_list_placeholder: test_label_batch[ids], model.logits_placeholder: test_logits[ids]}

        # Run the model test step updating the AUC object
        loss_val, mse_loss_val, log_loss_val, pred = sess.run([model.loss, model.mse_loss, model.log_loss, model.prediction], feed_dict=feed_dict)

        # Run the Average Predictor graph
        ap_mse_val, ap_log_val = sess.run([ap_mse_loss, ap_log_loss], feed_dict=feed_dict)

        mse.append(mse_loss_val)
        llh.append(log_loss_val)
        ap_mse.append(ap_mse_val)
        ap_llh.append(ap_log_val)
        auc_list.append(metrics.roc_auc_score(y_true=test_label_batch[ids].astype(int), y_score=pred))


    for i in range(30):
        mse_diff.append((ap_mse[i]-mse[i]) / ap_mse[i])
        llh_diff.append((ap_llh[i]-llh[i]) / ap_llh[i])

    print("MSE Mean Score On The Bootstrap = ", np.mean(mse))
    print("MSE Mean Lift Over Average Predictor (%) = ", np.round(np.mean(mse_diff)*100, decimals=2))
    print("MSE STD (%) =" , np.round(np.std(mse_diff)*100, decimals=2))

    print("LLH Mean Over Average Predictor (%) =", np.round(np.mean(llh_diff)*100, decimals=2))
    print("LLH STD (%) = ", np.round(np.std(llh_diff)*100, decimals=2))

    print("Mean AUC Score On The Bootstrap = ",  np.round(np.mean(auc_list), decimals=4), "+/-", np.round(np.std(auc_list), decimals=4))
