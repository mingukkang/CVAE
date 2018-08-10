import tensorflow as tf
from utils import *
from data_utils import *


class CVAE:
    def __init__(self, shape,n_cls,n_hidden,n_z, drop_rate):
        self.shape = shape
        self.n_cls = n_cls
        self.n_hidden = n_hidden
        self.n_z = n_z
        self.drop_rate = drop_rate
        self.n_out = self.shape[1]*self.shape[2]*self.shape[3]

    def conditional_gaussian_encoder(self, X, Y, keep_prob):

        with tf.variable_scope("gaussian_encoder", reuse = tf.AUTO_REUSE):
            X_input = tf.concat((X,Y), axis =1)
            net = drop_out(leaky(dense(X_input, self.n_hidden[0], name = "Dense_1")), keep_prob)
            net = drop_out(leaky(dense(net, self.n_hidden[1], name="Dense_2")), keep_prob)
            net = dense(net, self.n_z*2, name ="Dense_3")
            mean = net[:,:self.n_z]
            std = tf.nn.softplus(net[:,self.n_z:]) + 1e-6

        return mean, std

    def conditional_bernoulli_decoder(self,Z, Y, keep_prob):

        with tf.variable_scope("bernoulli_decoder", reuse = tf.AUTO_REUSE):
            z_input = tf.concat((Z,Y), axis = 1)
            net = drop_out(leaky(dense(z_input, self.n_hidden[2], name = "Dense_1")), keep_prob)
            net = drop_out(leaky(dense(net, self.n_hidden[3], name="Dense_2")), keep_prob)
            net = tf.nn.sigmoid(dense(net, self.n_out, name = "Dense_3"))

        return net


    def Conditional_Variational_AutoEncoder(self, X, X_noised, Y, keep_prob):

        X_flatten = tf.reshape(X, [-1, self.n_out])
        X_flatten_noised = tf.reshape(X_noised, [-1, self.n_out])

        mean, std = self.conditional_gaussian_encoder(X_flatten_noised, Y, keep_prob)
        z = mean + std*tf.random_normal(tf.shape(mean, out_type = tf.int32), 0, 1, dtype = tf.float32)

        X_out = self.conditional_bernoulli_decoder(z, Y, keep_prob)
        X_out = tf.clip_by_value(X_out, 1e-8, 1 - 1e-8)

        likelihood = tf.reduce_mean(tf.reduce_sum(X_flatten*tf.log(X_out) + (1 - X_flatten)*tf.log(1 - X_out), 1))
        KL_Div = tf.reduce_mean(0.5 * tf.reduce_sum(1 - tf.log(tf.square(std) + 1e-8)
                                                    + tf.square(mean)
                                                    + tf.square(std), 1))

        Recon_error = -1*likelihood
        Regul_error = KL_Div

        self.ELBO = Recon_error + Regul_error

        return z, X_out, self.ELBO

    def optim_op(self, loss ,learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        return self.optimizer