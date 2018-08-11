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
    flag.DEFINE_integer("n_epoch", 7, "number of Epoch")
    flag.DEFINE_integer("n_z", 2, "Dimension of latent variables")
    flag.DEFINE_integer("keep_prob", 0.6, "Dropout rate")
    flag.DEFINE_float("decay_rate", 0.98,"learning rate decay rate")
    flag.DEFINE_integer("batch_size", 128, "Batch size for training")
    flag.DEFINE_bool("add_noise", True, "[True/False]")
    flag.DEFINE_bool("PMLR", True, "Boolean for plot manifold learning result")
    flag.DEFINE_bool("PARR", True, "Boolean for plot analogical reasoning result")


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
    latent = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.n_z], name="latent_input")

    global_step = tf.Variable(0, trainable=False)

    if FLAGS.PMLR is True: # code for plot manifold learning Results
        assert FLAGS.n_z == 2, "n_z should be 2!"
        images_manifold = CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)

    if FLAGS.PARR is True: # code for plot analogical reasoning result
        images_PARR = CVAE.conditional_bernoulli_decoder(latent, Y, keep_prob)

    total_batch = data_pipeline.get_total_batch(train_xs, FLAGS.batch_size)
    learning_rate_decayed = FLAGS.learning_rate*FLAGS.decay_rate**(global_step/total_batch)
    optim_op = CVAE.optim_op(loss, learning_rate_decayed, global_step)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    batch_v_xs, batch_vn_xs, batch_v_ys = data_pipeline.next_batch(valid_xs, valid_ys, 100, make_noise= FLAGS.add_noise)
    print("plot original and noised images")
    plot_manifold_canvas(batch_v_xs, 10, "MNIST", "ori_input_images")
    plot_manifold_canvas(batch_vn_xs, 10, "MNIST", "input_image_noised")
    print("_"*80)

    start_time = time.time()
    print("training started")
    for i in range(FLAGS.n_epoch):
        loss_val = 0
        for j in range(total_batch):
            batch_xs, batch_noised_xs, batch_ys = data_pipeline.next_batch(train_xs, train_ys, FLAGS.batch_size, make_noise = True)
            feed_dict = {X:batch_xs, X_noised:batch_noised_xs, Y:batch_ys, keep_prob: FLAGS.keep_prob}
            l, lr, op, g = sess.run([loss,learning_rate_decayed, optim_op, global_step], feed_dict = feed_dict)
            loss_val += l/total_batch

        if i % 5 ==0 or i == (FLAGS.n_epoch -1):
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
        print("Epoch: %.3d   loss: %.5f   lr: %f   Time: %d hour %d min %d sec\n" % (i, loss_val, lr, hour, min, sec))

    if FLAGS.PMLR is True:
        print("-"*80)
        print("plot Manifold Learning Results")
        x_axis = np.linspace(-2,2,10)
        y_axis = np.linspace(-2,2,10)
        z_holder = []
        for i,xi in enumerate(x_axis):
            for j, yi in enumerate(y_axis):
                z_holder.append([xi,yi])
        length = len(z_holder)
        for k in range(n_cls):
            y = [k]*length
            y_one_hot = np.zeros((length, n_cls))
            y_one_hot[np.arange(length), y] = 1
            y_one_hot = np.reshape(y_one_hot, [-1,n_cls])

            MLR = sess.run(images_manifold, feed_dict = {latent: z_holder, Y:y_one_hot, keep_prob: 1.0})
            MLR = np.reshape(MLR, [-1, height, width, channel])
            p_name = "PMLR/labels" +str(k)
            plot_manifold_canvas(MLR, 10, "MNIST", p_name)

    if FLAGS.PARR is True:
        print("-"*80)
        print("plot analogical reasoning result")
        z_holder = []
        z_ = np.random.normal(0, 1, [10, FLAGS.n_z])
        for i in range(n_cls):
            z_holder.append(z_)
        z_holder = np.concatenate(z_holder, axis = 0)
        y = [j for j in range(n_cls)]
        y = y*10
        length = len(z_holder)
        y_one_hot = np.zeros((length, n_cls))
        y_one_hot[np.arange(length), y] = 1
        y_one_hot = np.reshape(y_one_hot, [-1, n_cls])
        PARR = sess.run(images_PARR, feed_dict = {latent: z_holder, Y: y_one_hot, keep_prob: 1.0})
        PARR = np.reshape(PARR, [-1, height, width, channel])
        p_name = "PARR/manifold"
        plot_manifold_canvas(PARR, 10, "MNIST", p_name)

    ## code for 2D scatter plot
    if FLAGS.n_z == 2:
        test_total_batch = data_pipeline.get_total_batch(test_xs,128)
        latent_holder = []
        for i in range(test_total_batch):
            batch_test_xs, batch_test_noised_xs, batch_test_ys = data_pipeline.next_batch(test_xs,
                                                                                          test_ys,
                                                                                          FLAGS.batch_size,
                                                                                          make_noise = False)
            feed_dict = {X: batch_test_xs,
                         X_noised: batch_test_noised_xs,
                         Y: batch_test_ys,
                         keep_prob: 1.0}
            latent_vars = sess.run(z, feed_dict = feed_dict)
            latent_holder.append(latent_vars)
        latent_holder = np.concatenate(latent_holder, axis = 0)
        plot_2d_scatter(latent_holder[:,0], latent_holder[:,1], test_ys[:len(latent_holder)])

    print("learning finished")
    sess.close()