# %%
import os
import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def select_samples_per_class(images, labels, num_samples=20):
    # Select 20 samples per class to reduce the number of points in the plot
    selected_indices = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        selected_indices.extend(label_indices[:num_samples])

    return images[selected_indices], labels[selected_indices]


# %% Load datasets

# Digits
digits = load_digits()
digits_images = digits.data
digits_labels = digits.target
digits_train_images, digits_test_images, digits_train_labels, digits_test_labels = train_test_split(digits_images, digits_labels, test_size=0.2, random_state=42)


# MNIST
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
train_images_flat = mnist_train_images.reshape((60000, 28 * 28))
test_images_flat = mnist_test_images.reshape((10000, 28 * 28))
mnist_images = np.concatenate([train_images_flat, test_images_flat])
mnist_labels = np.concatenate([mnist_train_labels, mnist_test_labels])


# Semeion
with open("../semeion+handwritten+digit/semeion.data") as textFile:
    semeion = [line.split() for line in textFile]
semeion = np.asarray(semeion)
semeion = semeion.astype(float)

# Divide the Semeion dataset into images and labels (last 10 columns)
semeion_images = semeion[:, :-10]
semeion_labels = semeion[:, -10:]
semeion_lab = [np.argmax(np.array(label)) for label in semeion_labels]
semeion_train_images, semeion_test_images, semeion_train_labels, semeion_test_labels = train_test_split(semeion_images, semeion_lab, test_size=0.2, random_state=42)

# %% Stratified split

# Digits
digits_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
digits_train_indices, _ = next(digits_splitter.split(digits_images, digits_labels))
digits_train_subset = digits_images[digits_train_indices]
digits_labels_subset = digits_labels[digits_train_indices]

# MNIST
mnist_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
mnist_train_indices, _ = next(mnist_splitter.split(mnist_images, mnist_labels))
mnist_train_subset = mnist_images[mnist_train_indices]
mnist_labels_subset = mnist_labels[mnist_train_indices]

# Semeion
semeion_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
semeion_train_indices, _ = next(semeion_splitter.split(semeion_images, semeion_lab))
semeion_train_subset = semeion_images[semeion_train_indices]
semeion_labels_subset = np.array(semeion_lab)[semeion_train_indices]

# %%

digits_train_subset, digits_labels_subset = select_samples_per_class(digits_train_subset, digits_labels_subset, num_samples=20)
mnist_train_subset, mnist_labels_subset = select_samples_per_class(mnist_train_subset, mnist_labels_subset, num_samples=20)
semeion_train_subset, semeion_labels_subset = select_samples_per_class(semeion_train_subset, semeion_labels_subset, num_samples=20)

min_train_size = min(digits_train_images.shape[0], mnist_train_images.shape[0], semeion_train_images.shape[0])

# select a random sample of min_train_size from each dataset
digits_train_images_sampled = digits_train_subset
mnist_train_images_sampled = mnist_train_subset
semeion_train_images_sampled = semeion_train_subset

# reshape mnist images
mnist_train_images_sampled = mnist_train_subset.reshape((200, 28*28))
semeion_train_images_sampled = semeion_train_subset
digits_train_images_sampled = digits_train_subset

assert digits_train_images_sampled.shape[0] == mnist_train_images_sampled.shape[0] == semeion_train_images_sampled.shape[0], 'Train size not equal'


# %% Resize Digits and Semeion images using cv2

mnist_images_28x28_train = np.expand_dims(mnist_train_images.astype(float), axis=1)
mnist_images_28x28_test = np.expand_dims(mnist_test_images.astype(float), axis=1)
mnist_images_28x28_train /= np.max(mnist_images_28x28_train)
mnist_images_28x28_test /= np.max(mnist_images_28x28_test)

digits_images_28x28_train = np.array([cv2.resize(image.reshape((8, 8)), (28, 28), interpolation=cv2.INTER_NEAREST) for image in digits_train_images])
digits_images_28x28_test = np.array([cv2.resize(image.reshape((8, 8)), (28, 28), interpolation=cv2.INTER_NEAREST) for image in digits_test_images])
digits_images_28x28_train = np.expand_dims(digits_images_28x28_train, axis=1)
digits_images_28x28_test = np.expand_dims(digits_images_28x28_test, axis=1)
digits_images_28x28_train /= np.max(digits_images_28x28_train)
digits_images_28x28_test /= np.max(digits_images_28x28_test)

semeion_images_28x28_train = np.array([cv2.resize(image.reshape((16, 16)), (28, 28), interpolation=cv2.INTER_NEAREST) for image in semeion_train_images])
semeion_images_28x28_test = np.array([cv2.resize(image.reshape((16, 16)), (28, 28), interpolation=cv2.INTER_NEAREST) for image in semeion_test_images])
semeion_images_28x28_train = np.expand_dims(semeion_images_28x28_train, axis=1)
semeion_images_28x28_test = np.expand_dims(semeion_images_28x28_test, axis=1)
semeion_images_28x28_train /= np.max(semeion_images_28x28_train)
semeion_images_28x28_test /= np.max(semeion_images_28x28_test)

# save train and test images
os.makedirs('../Dataset_28x28', exist_ok=True)
with open('../Dataset_28x28/mnist_images_28x28_train.pkl', 'wb') as f:
    pkl.dump(mnist_images_28x28_train, f)
with open('../Dataset_28x28/mnist_images_28x28_test.pkl', 'wb') as f:
    pkl.dump(mnist_images_28x28_test, f)
with open('../Dataset_28x28/digits_images_28x28_train.pkl', 'wb') as f:
    pkl.dump(digits_images_28x28_train, f)
with open('../Dataset_28x28/digits_images_28x28_test.pkl', 'wb') as f:
    pkl.dump(digits_images_28x28_test, f)
with open('../Dataset_28x28/semeion_images_28x28_train.pkl', 'wb') as f:
    pkl.dump(semeion_images_28x28_train, f)
with open('../Dataset_28x28/semeion_images_28x28_test.pkl', 'wb') as f:
    pkl.dump(semeion_images_28x28_test, f)

# save labels
with open('../Dataset_28x28/mnist_labels_train.pkl', 'wb') as f:
    pkl.dump(mnist_train_labels, f)
with open('../Dataset_28x28/mnist_labels_test.pkl', 'wb') as f:
    pkl.dump(mnist_test_labels, f)
with open('../Dataset_28x28/digits_labels_train.pkl', 'wb') as f:
    pkl.dump(digits_train_labels, f)
with open('../Dataset_28x28/digits_labels_test.pkl', 'wb') as f:
    pkl.dump(digits_test_labels, f)
with open('../Dataset_28x28/semeion_labels_train.pkl', 'wb') as f:
    pkl.dump(semeion_train_labels, f)
with open('../Dataset_28x28/semeion_labels_test.pkl', 'wb') as f:
    pkl.dump(semeion_test_labels, f)

# plot single image for each dataset
plt.figure(figsize=(10, 10))
plt.imshow(digits_images_28x28_train[0][0], cmap='gray')
plt.title('Digits 28x28', fontsize=50)
plt.axis('off')

plt.figure(figsize=(10, 10))
plt.imshow(mnist_images_28x28_train[0][0], cmap='gray')
plt.title('MNIST 28x28', fontsize=50)
plt.axis('off')

plt.figure(figsize=(10, 10))
plt.imshow(np.array(semeion_images_28x28_train[0][0]), cmap='gray')
plt.title('Semeion 28x28', fontsize=50)
plt.axis('off')

# %%

digits_train_images_sampled = select_samples_per_class(digits_images_28x28_train, digits_train_labels, num_samples=20)
mnist_train_images_sampled = select_samples_per_class(mnist_images_28x28_train, mnist_train_labels, num_samples=20)
semeion_train_images_sampled = select_samples_per_class(semeion_images_28x28_train, np.array(semeion_train_labels), num_samples=20)

# concat train images and labels
all_data = np.vstack((digits_train_images_sampled[0], mnist_train_images_sampled[0], semeion_train_images_sampled[0]))
all_labels = np.hstack((digits_train_images_sampled[1], mnist_train_images_sampled[1], semeion_train_images_sampled[1]))
all_data = all_data.reshape((all_data.shape[0], -1))

# %% t-SNE

tsne = TSNE(n_components=2, random_state=42, perplexity=40)
all_tsne = tsne.fit_transform(all_data)
class_colors = {i: plt.cm.tab10(i / 10) for i in range(10)}
dataset_markers = {'digits': 'o', 'mnist': 's', 'semeion': '^'}
datasets = ['digits', 'mnist', 'semeion']

plt.figure(figsize=(10, 8))
scatter_digits = plt.scatter(all_tsne[: ,0][:200], all_tsne[:, 1][:200], c=[class_colors[label] for label in digits_labels_subset], marker='o',  alpha=0.8)
scatter_mnist = plt.scatter(all_tsne[:, 0][200:400], all_tsne[:, 1][200:400], c=[class_colors[label] for label in mnist_labels_subset], marker='s',  alpha=0.8)
scatter_semeion = plt.scatter(all_tsne[:, 0][400:], all_tsne[:, 1][400:], c=[class_colors[label] for label in semeion_labels_subset], marker='^', alpha=0.8)

for i, color in class_colors.items():
    plt.scatter([], [], c=[color], label=f'{i}')
legend_labels = {'s': 'MNIST', 'o': 'Digits', '^': 'Semeion'}
for marker, label in legend_labels.items():
    plt.scatter([], [], c='white', edgecolors='black', marker=marker, label=label)

plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)
plt.legend(loc='lower left')
plt.savefig('combined_tsne.png')
plt.show()
