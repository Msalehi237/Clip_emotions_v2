import matplotlib.pyplot as plt
from PIL import Image
import random

def sample_visualizer(data_to_show, root_data, n=5):
    random_sample = data_to_show.sample(n=n).sort_values(by='value')
    fig, axes = plt.subplots(1, len(random_sample), figsize=(len(random_sample) * 5, 5))
    for i, value in enumerate(random_sample.values):
        # Open the image file and convert it to a Pillow Image object
        image = Image.open(root_data + '/' + value[0])
        # Plot the image on the corresponding subplot
        axes[i].imshow(image)
        axes[i].set_title(f"{value[1]:.3f}")  # Set a title for each image subplot
    # Adjust spacing and display the plot

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def top_k_visualizer(data_to_show, root_data, k=5):
    random_sample = data_to_show.sort_values(by='value', ascending=False).head(k)
    random_sample = random_sample.sort_values(by='value')
    fig, axes = plt.subplots(1, len(random_sample), figsize=(len(random_sample) * 5, 5))
    for i, value in enumerate(random_sample.values):
        # Open the image file and convert it to a Pillow Image object
        image = Image.open(root_data + '/' + value[0])
        # Plot the image on the corresponding subplot
        axes[i].imshow(image)
        axes[i].set_title(f" {value[0]} \n {value[1]:.3f}")  # Set a title for each image subplot
    # Adjust spacing and display the plot

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def class_visualizer(data_to_show, root_data, n=27):
    # Calculate the number of images to pick from each class
    n_per_class = n // 3

    # Group images by class
    classes = {}
    class_counts = {}
    for _, value in data_to_show.iterrows():
        img = value[0]
        cls = value[1]
        if cls not in classes:
            classes[cls] = []
            class_counts[cls] = 0
        classes[cls].append(img)
        class_counts[cls] += 1

    # Randomly select images from each class
    selected_images = []
    for cls, images in classes.items():
        random.shuffle(images)  # Shuffle the images
        selected_images.extend(images[:n_per_class])

    # Plotting the selected images in a 1x3 grid structure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (cls, images) in enumerate(classes.items()):
        # Create a subplot for each class
        ax = axes[i]
        num_images = class_counts[cls]
        ax.set_title(f"Class {cls} \n Total images: {num_images}")
        ax.axis('off')

        # Create a 3x3 grid structure within each class subplot
        sub_axes = [ax.inset_axes([j % 5 / 5, j // 5 / 5, 1 / 5, 1 / 5]) for j in range(n_per_class)]

        for j, img in enumerate(images[:n_per_class]):
            # Open the image file and convert it to a Pillow Image object
            image = Image.open(root_data + '/' + img)
            # Plot the image on the corresponding subplot within the class subplot
            sub_ax = sub_axes[j]
            sub_ax.imshow(image)
            sub_ax.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()
