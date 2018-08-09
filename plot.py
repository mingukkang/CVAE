import matplotlib.pyplot as plt
from data_utils import *
import pdb

def plot_2d_scatter(x,y,test_labels):
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, c = np.argmax(test_labels,1), marker ='.', edgecolor = 'none', cmap = discrete_cmap('jet'))
    plt.colorbar()
    plt.grid()
    plt.savefig('2D_Scatter')
    plt.close()

def discrete_cmap(base_cmap =None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0,1,10))
    cmap_name = base.name + str(10)
    return base.from_list(cmap_name,color_list,10)


def plot_manifold_canvas(images, n, type):
    assert images.shape[0] == n**2, "n**2 should be number of images"
    height = images.shape[1]
    width = images.shape[2] # width = height
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)

    if type is "MNIST":
        canvas = np.empty((n * images.height, n * height))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width] = images[n*i + j]
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray")
    else:
        canvas = np.empty((n * height, n * height, 3))
        for i, yi in enumerate(x):
            for j, xi in enumerate(y):
                canvas[height*i: height*i + height, width*j: width*j + width,:] = images[n*i + j]
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas)

    plt.savefig("manifold")
    plt.close()


data_pipeline = data_pipeline("MNIST")
train_xs, train_ys, valid_xs, valid_ys, test_xs, test_ys = data_pipeline.load_preprocess_data()
total_batch = data_pipeline.get_total_batch(train_xs, 100, augment = 2)
batch_xs, batch_ys = data_pipeline.next_batch(train_xs,train_ys,100,augment =2)
plot_manifold_canvas(batch_xs,10, "CIFAR_10")