from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

class SupervisedProd2vec():
    def __init__(self, FLAGS):
        super().__init__()

        self.num_users = FLAGS.num_users
        self.num_products = FLAGS.num_products
        self.embedding_dim = FLAGS.embedding_size
        self.l2_pen = FLAGS.l2_pen
        self.learning_rate = FLAGS.learning_rate
        self.plot_gradients = FLAGS.plot_gradients
        self.cf_pen = FLAGS.cf_pen
        self.cf_distance = FLAGS.cf_distance
        self.cf_loss = 0

        # Build the graph
        self.create_placeholders()
        self.build_graph()
        self.create_control_embeddings()
        self.create_counter_factual_loss()
        self.create_losses()
        self.add_optimizer()
        self.add_average_predictor()
        self.add_summaries()

    def create_placeholders(self):
        """Create the placeholders to be used """
        
        self.user_list_placeholder = tf.placeholder(tf.int32, [None], name="user_list_placeholder")
        self.product_list_placeholder = tf.placeholder(tf.int32, [None], name="product_list_placeholder")
        self.label_list_placeholder = tf.placeholder(tf.float32, [None, 1], name="label_list_placeholder")
        self.reg_list_placeholder = tf.placeholder(tf.int32, [None], name="reg_list_placeholder")

        # logits placeholder used to store the test CR for the bootstrapping process
        self.logits_placeholder = tf.placeholder(tf.float32, [None], name="logits_placeholder")

    def build_graph(self):
        """Build the main tensorflow graph with embedding layers"""

        with tf.name_scope('embedding_layer'):
            # User matrix and current batch
            self.user_embeddings = tf.get_variable("user_embeddings", shape=[self.num_users, self.embedding_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_list_placeholder)
            self.user_b = tf.Variable(tf.zeros([self.num_users]), name='user_b', trainable=True)
            self.user_bias_embed = tf.nn.embedding_lookup(self.user_b, self.user_list_placeholder)

            # Product embedding
            self.product_embeddings = tf.get_variable("product_embeddings", shape=[self.num_products, self.embedding_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            self.product_embed = tf.nn.embedding_lookup(self.product_embeddings, self.product_list_placeholder)
            self.prod_b = tf.Variable(tf.zeros([self.num_products]), name='prod_b', trainable=True)
            self.prod_bias_embed = tf.nn.embedding_lookup(self.prod_b, self.product_list_placeholder)

        with tf.variable_scope('logits'):

            self.global_bias = tf.get_variable('global_bias', [1], initializer=tf.constant_initializer(0.0, dtype=tf.float32), trainable=True)
            self.alpha = tf.get_variable('alpha', [], initializer=tf.constant_initializer(0.00000001, dtype=tf.float32), trainable=True)
            emb_logits = self.alpha * tf.reshape(tf.reduce_sum(tf.multiply(self.user_embed, self.product_embed), 1), [tf.shape(self.user_list_placeholder)[0], 1])
            logits = tf.reshape(tf.add(self.prod_bias_embed, self.user_bias_embed), [tf.shape(self.user_list_placeholder)[0], 1]) + self.global_bias
            self.logits = emb_logits + logits

            self.prediction = tf.sigmoid(self.logits, name='sigmoid_prediction')

    def create_control_embeddings(self):
        """Create the control embeddings"""

        pass

    def create_counter_factual_loss(self):
        """Create the counter factual loss to add to main loss"""

        pass

    def create_losses(self):
        """Create the losses"""

        with tf.name_scope('losses'):
            # Sigmoid loss between the logits and labels
            self.log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label_list_placeholder))

            # Adding the regularizer term on user vct and prod vct and their bias terms
            reg_term = self.l2_pen * ( tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.product_embeddings) )
            reg_term_biases = self.l2_pen * ( tf.nn.l2_loss(self.prod_b) + tf.nn.l2_loss(self.user_b) )
            self.factual_loss = self.log_loss + reg_term + reg_term_biases

            # Adding the counter-factual loss
            self.loss = self.factual_loss + (self.cf_pen * self.cf_loss) # Imbalance loss
            self.mse_loss = tf.losses.mean_squared_error(labels=self.label_list_placeholder, predictions=self.prediction)

    def add_optimizer(self):
        """Add the required optimiser to the graph"""

        with tf.name_scope('optimizer'):
            # Global step variable to keep track of the number of training steps
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            # Gradient Descent
            if self.plot_gradients:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                # Op to calculate every variable gradient
                grads = tf.gradients(self.loss, tf.trainable_variables())
                grads = list(zip(grads, tf.trainable_variables()))
                # Op to update all variables according to their gradient
                self.apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
            else:
                self.apply_grads = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def add_average_predictor(self):
        """"Add the average predictors to the graph"""

        with tf.variable_scope('ap_logits'):

            ap_logits = tf.reshape(self.logits_placeholder, [tf.shape(self.label_list_placeholder)[0], 1])

        with tf.name_scope('ap_losses'):
            
            self.ap_mse_loss = tf.losses.mean_squared_error(labels=self.label_list_placeholder, predictions=ap_logits)
            self.ap_log_loss =  tf.losses.log_loss(labels=self.label_list_placeholder, predictions=ap_logits)

    def add_summaries(self):
        """Add the required summaries to the graph"""

        with tf.name_scope('summaries'):
            # Add loss to the summaries
            tf.summary.scalar('total_loss', self.loss)
            tf.summary.histogram('histogram_total_loss', self.loss)

            # Add weights to the summaries
            tf.summary.histogram('user_embedding_weights', self.user_embeddings)
            tf.summary.histogram('product_embedding_weights', self.product_embeddings)
            tf.summary.histogram('logits', self.logits)
            tf.summary.histogram('prod_b', self.prod_b)
            tf.summary.histogram('user_b', self.user_b)
            tf.summary.histogram('global_bias', self.global_bias)

            tf.summary.scalar('alpha', self.alpha)

class CausalProd2Vec(SupervisedProd2vec):
    def __init__(self, FLAGS):

        super().__init__(FLAGS)
        
    def create_control_embeddings(self):
        """Create the control embeddings"""

        with tf.name_scope('control_embedding'):
            # Get the control embedding at id 0
            self.control_embed = tf.stop_gradient(tf.nn.embedding_lookup(self.product_embeddings, 0))
    
    def create_counter_factual_loss(self):
        """Create the counter factual loss to add to main loss"""
        
        with tf.name_scope('counter_factual'):
            # Take the mean of the difference between treatment and the control embedding
            if self.cf_distance == "l1":
                print("Using L1 difference between treatment and control embeddings")
                self.cf_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(tf.nn.l2_normalize(self.product_embed, axis=1), tf.nn.l2_normalize(self.control_embed, axis=0))), axis=1))
            
            elif self.cf_distance == "l2":
                print("Using L2 difference between treatment and control embeddings")
                self.cf_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.nn.l2_normalize(self.product_embed,axis=1), tf.nn.l2_normalize(self.control_embed,axis=0)))))

            elif self.cf_distance == "cos":
                print("Using Cosine difference between treatment and control embeddings")
                self.cf_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.control_embed,axis=0), tf.nn.l2_normalize(self.product_embed,axis=1), axis=0)

class CausalProd2Vec2i(SupervisedProd2vec):
    def __init__(self, FLAGS):

        super().__init__(FLAGS)

    def create_control_embeddings(self):
        """Create the control embeddings"""

        with tf.name_scope('control_embedding'):
            # Get the treatment representations for the products
            self.control_embed = tf.stop_gradient(tf.nn.embedding_lookup(self.product_embeddings, self.reg_list_placeholder))
            
    def create_counter_factual_loss(self):
        """Create the counter factual loss to add to main loss"""
        
        with tf.name_scope('counter_factual'):
            # Take the mean of the difference between treatment and the control embedding
            if self.cf_distance == "l1":
                print("Using L1 difference between treatment and control embeddings")
                self.cf_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(self.product_embed, self.control_embed)), axis=1))
            
            elif self.cf_distance == "l2":
                print("Using L2 difference between treatment and control embeddings")
                self.cf_loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.nn.l2_normalize(self.product_embed,axis=1), tf.nn.l2_normalize(self.control_embed,axis=0)))))

            elif self.cf_distance == "cos":
                print("Using Cosine difference between treatment and control embeddings")
                self.cf_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(self.control_embed,axis=0), tf.nn.l2_normalize(self.product_embed,axis=1), axis=0)