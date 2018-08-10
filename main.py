import tensorflow as tf
import numpy as np
from data_utils import *
from utils import *
from plot import *
from CVAE import *
import time

if __name__ =="__main__":
    flag = tf.app.flags
    FLAGS = flag.FLAGS
    flag.DEFINE_float("learning_rate", 0.0001, "learning rate for training")
    flag.DEFINE_integer("n_epoch", 150, "number of Epoch")
    flag.DEFINE_integer("n_z", 2, "Dimension of latent variables")
    flag.DEFINE_integer("keep_prob", 0.6, "Dropout rate")

    data_pipeline = data_pipeline("MNIST")
    train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = data_pipeline.load_preprocess_data()

    _, height, width, channel = np.shape(train_xs)
    n_cls = np.shape(train_ys)[1]

    X = tf.placeholder(dtype = tf.float32, shape = [None, height, width, channel], name ="Input")
    X_noised = tf.placeholder(dtype = tf.float32, shape = [None, height, width, channel], name ="Input_noised")
    Y = tf.placeholder(dtype = tf.float32, shape = [None, n_cls], name = "labels")
    keep_prob = tf.placeholder(dtype = tf.float32, name = "drop_rate")

    CVAE = CVAE([_,height, width, channel], n_cls, [500,500,500,500], FLAGS.n_z, keep_prob)
    z, output, loss = CVAE.Conditional_Variational_AutoEncoder(X, X_noised, Y, keep_prob)
    optim_op = CVAE.optim_op(loss, FLAGS.learning_rate)

    total_batch = data_pipeline.get_total_batch(train_xs,128)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_v_xs, batch_vn_xs, batch_v_ys = data_pipeline.next_batch(valid_xs, valid_ys, 100, make_noise=False)

    start_time = time.time()
    print("training started")
    for i in range(FLAGS.n_epoch):
        loss_val = 0
        for j in range(total_batch):
            batch_xs, batch_noised_xs, batch_ys = data_pipeline.next_batch(train_xs, train_ys, 128, make_noise = True)
            feed_dict = {X:batch_xs, X_noised:batch_noised_xs, Y:batch_ys, keep_prob: FLAGS.keep_prob}
            l, op = sess.run([loss, optim_op], feed_dict = feed_dict)
            loss_val += l/total_batch

        if i % 5 ==0:
            images = sess.run(output, feed_dict = {X:batch_v_xs,
                                                   X_noised:batch_vn_xs,
                                                   Y:batch_v_ys,
                                                   keep_prob:1.0})

            images = np.reshape(images, [-1, height, width, channel])
            name = "Manifold_canvas_" + str(i)
            plot_manifold_canvas(images, 10, type = "MNIST", name = name)

        hour = int((time.time() - start_time) / 3600)
        min = int(((time.time() - start_time) - 3600 * hour) / 60)
        sec = int((time.time() - start_time) - 3600 * hour - 60 * min)
        print("Epoch: %.3d   loss %.5f   Time: %d hour %d min %d sec\n" % (i, loss_val, hour, min, sec))

    ## code for 2D scatter plot
    if FLAGS.n_z == 2:
        test_total_batch = data_pipeline.get_total_batch(test_xs,128)
        latent_holder = []
        for i in range(test_total_batch):
            batch_test_xs, batch_test_noised_xs, batch_test_ys = data_pipeline.next_batch(test_xs, test_ys, 128, make_noise = False)
            feed_dict = {X: batch_test_xs,
                         X_noised: batch_test_noised_xs,
                         Y: batch_test_ys,
                         keep_prob: 1.0}
            latent_vars = sess.run(z, feed_dict = feed_dict)
            latent_holder.append(latent_vars)
        latent_holder = np.concatenate(latent_holder, axis = 0)
        plot_2d_scatter(latent_holder[:,0], latent_holder[:,1], test_ys[:len(latent_holder)])




