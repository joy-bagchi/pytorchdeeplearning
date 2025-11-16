import random
import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


def display_image(image, label, title, num_ticks=6, show_values=True):
    """
    Displays an image with its corresponding label and title.

    This function handles different image formats (PIL Image and PyTorch Tensor),
    normalizes the display range, and optionally overlays pixel values on the image.

    Args:
        image: The image data to be displayed. Can be a PIL Image or a PyTorch Tensor.
        label: The label associated with the image.
        title: The title for the plot.
        num_ticks (int, optional): The number of ticks to display on the color bar. Defaults to 6.
        show_values (bool, optional): If True, overlays the numerical value of each pixel on the image. Defaults to True.
    """
    # Initialize variables for value range and image data.
    vmin_val, vmax_val = None, None
    image_data = None

    # Check if the input is a PIL Image.
    if isinstance(image, Image.Image):
        # Set the value range for a standard 8-bit image.
        vmin_val = 0
        vmax_val = 255
        # Convert the PIL Image to a NumPy array.
        image_data = np.array(image)
    # Check if the input is a PyTorch Tensor.
    elif isinstance(image, torch.Tensor):
        # Convert the tensor to a NumPy array and remove any single-dimensional entries.
        image_np = image.numpy().squeeze()
        # Determine the min and max values from the tensor for normalization.
        vmin_val = image_np.min()
        vmax_val = image_np.max()
        # Assign the NumPy array to image_data.
        image_data = image_np
    # Handle unsupported image types.
    else:
        print("Warning: Unsupported image type.")
        return

    # Create a new figure for the plot.
    plt.figure(figsize=(9, 9))
    # Display the image data as a grayscale image.
    plt.imshow(image_data, cmap='gray', vmin=vmin_val, vmax=vmax_val)
    # Set the title of the plot with the provided title and label.
    plt.title(f'{title} | Label: {label}')

    # Check if pixel values should be displayed on the image.
    if show_values:
        # Calculate a threshold to determine the color of the text (black or white).
        threshold = (vmin_val + vmax_val) / 2.0
        # Get the dimensions of the image.
        height, width = image_data.shape
        
        # Iterate over each pixel to display its value.
        for y in range(height):
            for x in range(width):
                # Get the pixel value.
                value = image_data[y, x]
                # Set text color based on the pixel's brightness.
                text_color = "white" if value < threshold else "black"
                
                # Format the text to display, handling integers and floats differently.
                text_to_display = f"{value:.0f}" if isinstance(value, np.integer) else f"{value:.1f}"
                
                # Add the pixel value as text to the plot.
                plt.text(x, y, text_to_display, 
                         ha="center", va="center", color=text_color, fontsize=6)

    # Add a grid to the plot.
    plt.grid(True, color='red', alpha=0.3, zorder=2)
    # Set the x-axis ticks.
    plt.xticks(np.arange(0, 28, 4))
    # Set the y-axis ticks.
    plt.yticks(np.arange(0, 28, 4))
    
    # Add a color bar to the plot.
    cbar = plt.colorbar()
    # Create evenly spaced ticks for the color bar.
    ticks = np.linspace(vmin_val, vmax_val, num=num_ticks)
    # Set the ticks on the color bar.
    cbar.set_ticks(ticks)
    # Format the tick labels on the color bar.
    cbar.ax.set_yticklabels([f'{t:.2f}' for t in ticks])

    # Show the final plot.
    plt.show()
    
    
def display_predictions(model, test_loader, device):
    """
    Displays a grid of predictions for one random sample from each class.

    Args:
        model: The trained PyTorch model.
        test_loader: The DataLoader for the test set.
        device: The device (e.g., 'cuda' or 'cpu') to run inference on.
    """
    # Ensures the model is on the specified device and in evaluation mode.
    model.to(device)
    model.eval()

    # Creates a dictionary to store indices for each class.
    class_indices = {i: [] for i in range(10)}
    
    # Populates the dictionary with the indices of all samples for each class.
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)
        
    # Selects one random index from the list of indices for each class.
    random_indices = [random.choice(indices) for indices in class_indices.values()]
    
    # Retrieves the images and corresponding labels using the randomly selected indices.
    sample_images = torch.stack([test_loader.dataset[i][0] for i in random_indices])
    sample_labels = [test_loader.dataset[i][1] for i in random_indices]

    # Temporarily disables gradient calculation for inference.
    with torch.no_grad():
        # Passes the selected images through the model to get outputs.
        outputs = model(sample_images.to(device))
        # Gets the predicted class for each image.
        _, predictions = torch.max(outputs, 1)

    # Creates a figure and a grid of subplots for displaying the images.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    # Sets a main title for the entire figure.
    fig.suptitle('Model Predictions for a Sample of Each Class', fontsize=16)

    # Iterates through the subplots to display each image and its prediction.
    for i, ax in enumerate(axes.flat):
        # Extracts and prepares the image, true label, and predicted label for display.
        image = sample_images[i].cpu().squeeze()
        true_label = sample_labels[i]
        predicted_label = predictions[i].item()

        # Displays the image on the current subplot.
        ax.imshow(image, cmap='gray')
        
        # Sets the title of the subplot, with color indicating if the prediction is correct.
        title_color = 'green' if true_label == predicted_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color=title_color)
        
        # Hides the axes for a cleaner visual.
        ax.axis('off')

    # Adjusts the layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Adjusts the vertical spacing between subplots.
    plt.subplots_adjust(hspace=0.3)
    
    # Displays the plot.
    plt.show()
    

def plot_metrics(train_loss, test_acc):
    """
    Displays side-by-side plots for training loss and test accuracy over epochs.

    Args:
        train_loss (list): A list of floating-point numbers representing the
                           average training loss for each epoch.
        test_acc (list): A list of floating-point numbers representing the
                         test accuracy for each epoch.
    """
    # Get the number of epochs from the length of the loss list
    num_epochs = len(train_loss)
    # Create a 1-based epoch range for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Training Loss ---
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color='royalblue')
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True)
    # Ensure the x-axis ticks are integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Plot 2: Test Accuracy ---
    ax2.plot(epochs, test_acc, marker='o', linestyle='-', color='red')
    ax2.set_title('Test Accuracy Over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True)
    # Ensure the x-axis ticks are integers
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to prevent overlap and display the plots
    plt.tight_layout()
    plt.show()



# region = Message =
letter_ref = [
    "Dear Laurence",
    "Hope the PyTorch course is going well",
    "Do not forget to keep the labs interesting and engaging",
    "Maybe the students could decode my messy handwriting",
    "That might be a bit too challenging though",
    "I am impressed you are able to read this",
]


path_data = "./EMNIST_data"


def load_hidden_message_images(file_name="hidden_message_images.pkl"):
    with open(file_name, "rb") as f:
        import pickle

        message_imgs = pickle.load(f)
    return message_imgs


def decode_word_imgs(word_imgs, model, device):
    model.eval()
    decoded_chars = []
    with torch.no_grad():
        for char_img in word_imgs:
            char_img = char_img.unsqueeze(0).to(
                device
            )  # Add batch dimension and move to device
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()
            # uppercase_char = chr(ord("A") + predicted_label)
            lowercase_char = chr(ord("a") + predicted_label)
            # decoded_chars.append(f"{uppercase_char}/{lowercase_char}")
            decoded_chars.append(f"{lowercase_char}")
    decoded_word = "".join(decoded_chars)
    # print("Decoded word:", decoded_word)
    # print("Predicted characters:", " ".join(decoded_chars))
    # return decoded_word, decoded_chars
    return decoded_word


def visualize_image(img, label=None, ax=None):
    """
    Visualizes an EMNIST image with its label. If an axis is provided, plots on that axis; otherwise, creates a new figure.

    Args:
        img (np.ndarray or torch.Tensor): The image to display.
        label (int, optional): The EMNIST label (1-26). If None, no title is shown.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    """
    # Convert to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze()
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img[:, :, 0]

    # Prepare title if label is provided
    if label is not None:
        uppercase_char, lowercase_char = convert_emnist_label_to_char(label)
        title = f"EMNIST Letter: {uppercase_char}/{lowercase_char}"
    else:
        title = None

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show_colorbar = True
    else:
        show_colorbar = False

    im = ax.imshow(img, cmap="gray")
    ax.set_xticks(np.arange(0, 28, 1))
    ax.set_yticks(np.arange(0, 28, 1))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="red", alpha=0.3)
    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax)
        plt.show()


def display_data_loader_contents(data_loader):
    """
    Displays the contents of the data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to display.
    """
    try:
        print("Total number of images in dataset:", len(data_loader.dataset))
        print("Total number of batches:", len(data_loader))
        for batch_idx, (data, labels) in enumerate(data_loader):
            print(f"--- Batch {batch_idx + 1} ---")
            print(f"Data shape: {data.shape}")
            print(f"Labels shape: {labels.shape}")
            break  # display only the first batch.
    except StopIteration:
        print("data loader is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")


def evaluate_per_class(model, test_loader, device):
    """
    Evaluates the model's accuracy for each class (letter).

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test dataset.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: class_accuracies - Dictionary containing accuracy for each class (letter).
    """

    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1
            targets = targets - 1

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    class_accuracies = {}

    for class_idx in range(26):  # 26 classes for A-Z
        class_targets = [
            t for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]
        class_predictions = [
            p for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]

        if len(class_targets) > 0:
            class_accuracies[chr(65 + class_idx)] = accuracy_score(
                class_targets, class_predictions
            )
        else:
            class_accuracies[chr(65 + class_idx)] = 0.0  # Handle empty classes

    return class_accuracies


def save_student_model(model, filename="trained_student_model.pth"):
    """
    Saves the student's trained model and metadata.

    Args:
        model (nn.Module): The student's trained model.
        filename (str): The filename to save to.
    """
    save_dict = {"model": model}
    torch.save(save_dict, filename)
    print(f"Model saved to {filename}")


def convert_emnist_label_to_char(label):
    """
    Converts an EMNIST label to its corresponding uppercase and lowercase letters.

    Args:
        label (int): The EMNIST label (1-26).

    Returns:
        tuple: A tuple containing the uppercase and lowercase letters.
    """
    if not (1 <= label <= 26):
        raise ValueError("Label must be between 1 and 26 inclusive.")

    uppercase_char = chr(64 + label)  # 'A' is at 65, 'B' is at 66, etc.
    lowercase_char = chr(96 + label)  # 'a' is at 97, 'b' is at 98, etc.

    return uppercase_char, lowercase_char


# # region = To generate the images of the secret message =
# def get_message_imgs(letter=letter_ref):
#     sentences_imgs = []

#     for sentence in letter:
#         imgs = get_sentence_imgs(sentence)
#         sentences_imgs.append(imgs)

#     return sentences_imgs


# def get_word_imgs(word):
#     characters = list(word)
#     images = [get_emnist_img(c) for c in characters]
#     return images


# def get_sentence_imgs(sentence):
#     words = sentence.split()
#     images_per_word = [get_word_imgs(word) for word in words]
#     return images_per_word


# def get_emnist_img(character):

#     # Load test emnist dataset

#     # Precomputed mean and std for EMNIST Letters dataset
#     mean = (0.1736,)
#     std = (0.3317,)

#     # Create a transform that converts images to tensors and normalizes them
#     transform = transforms.Compose(
#         [
#             transforms.ToTensor(),  # Converts images to PyTorch tensors and scales pixel values to [0, 1]
#             transforms.Normalize(
#                 mean=mean, std=std
#             ),  # Applies normalization using the computed mean and std
#         ]
#     )

#     emnist_dataset = datasets.EMNIST(
#         root=path_data,
#         split="letters",
#         train=False,
#         download=False,
#         transform=transform,
#     )

#     # Find the image corresponding to the given character
#     if character.islower():
#         target_label = ord(character) - ord("a") + 1  # 'a' is label 1
#     elif character.isupper():
#         target_label = ord(character) - ord("A") + 1  # 'A' is label 1
#     else:
#         raise ValueError("Character must be an uppercase or lowercase letter.")
#     for img, label in emnist_dataset:
#         if label == target_label:
#             return img


# def print_word_imgs(word_imgs):
#     n_words = len(word_imgs)
#     factor = 0.6
#     fig, axes = plt.subplots(1, n_words, figsize=(n_words * factor, factor))
#     if n_words == 1:
#         axes = [axes]
#     for i, character_img in enumerate(word_imgs):
#         img = correct_image_orientation(character_img)
#         visualize_image(img, ax=axes[i])
#         # axes[i].imshow(img, cmap="gray")
#         axes[i].axis("off")
#     plt.show()


# def correct_image_orientation(image):
#     rotated = F.rotate(image, 90)  # Rotate the image 90 degrees clockwise
#     flipped = F.vflip(rotated)  # Flip the image vertically
#     return flipped


# # endregion =


# # region = old stuff =
# def predict_and_visualize(model, test_dataset, device, index_to_predict):
#     """
#     Predicts and visualizes the prediction for a specific transformed image in the test dataset.

#     Args:
#         model: The trained PyTorch model.
#         test_dataset: The transformed EMNIST test dataset.
#         device: Device to run the model on (e.g., 'cpu' or 'cuda').
#         index_to_predict: The index of the image in the test dataset to predict.
#     Returns:
#         predicted_label (int): The predicted label.
#     """
#     model.eval()  # Set the model to evaluation mode

#     # Check if the provided index is within the valid range
#     if not (0 <= index_to_predict < len(test_dataset)):
#         print(
#             f"\033[91mIndex must be between 0 and {len(test_dataset)-1} inclusive.\033[0m"
#         )
#         return None  # Return None if index is out of bounds

#     # Extract the transformed image and label from the specified index
#     image, target = test_dataset[index_to_predict]
#     image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = outputs.max(1)
#         predicted_label = predicted.item()

#     # Visualize the image
#     image_np = image.squeeze().cpu().numpy()
#     image_np = np.transpose(image_np, (1, 0))

#     # Calculate the corresponding uppercase and lowercase letters based on the label
#     uppercase_true = chr(ord("A") + (target - 1))
#     lowercase_true = chr(ord("a") + (target - 1))
#     uppercase_pred = chr(ord("A") + predicted_label)
#     lowercase_pred = chr(ord("a") + predicted_label)

#     plt.figure(figsize=(5, 5))
#     plt.imshow(image_np, cmap="gray")
#     plt.title(
#         f"True: {uppercase_true}/{lowercase_true}, Predicted: {uppercase_pred}/{lowercase_pred}"
#     )
#     plt.axis("off")
#     plt.show()


# # endregion =
