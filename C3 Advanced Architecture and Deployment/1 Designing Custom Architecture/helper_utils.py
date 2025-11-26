import glob
import os
import random
import time
import matplotlib as mpl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from IPython.core.display import HTML
from IPython.display import display
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models as tv_models
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import datasets



# --- Constants for Vegetation Detection ---
LOWER_GREEN_HSV = np.array([35, 40, 40])
UPPER_GREEN_HSV = np.array([85, 255, 255])

# Global plot style
PLOT_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans",  # "sans-serif",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markersize": 6,
}
mpl.rcParams.update(PLOT_STYLE)

def _create_signature_map(base_data_dir, min_real=1, min_fake=1):
    """
    Scans the dataset directory to create a map of signatures for each ID.

    Args:
        base_data_dir (str): The root directory of the signature dataset.
        min_real (int): The minimum number of real images an ID must have.
        min_fake (int): The minimum number of fake images an ID must have.

    Returns:
        dict: A dictionary mapping valid user IDs to their image paths.
    """
    # Construct the path to the directory containing real signatures.
    real_signatures_dir = os.path.join(base_data_dir, 'Real')
    # Construct the path to the directory containing fake signatures.
    fake_signatures_dir = os.path.join(base_data_dir, 'Fake')
    # Initialize a dictionary that will map user IDs to their signature image paths.
    signature_map = defaultdict(lambda: {'real': [], 'fake': []})

    # Check if the directory for real signatures exists to prevent errors.
    if not os.path.isdir(real_signatures_dir):
        print(f"Error: Directory not found at: {real_signatures_dir}")
        return {}

    # Get and sort a list of all items in the real signatures directory.
    all_ids = sorted(os.listdir(real_signatures_dir))
    # Iterate through each potential user ID directory.
    for user_id in all_ids:
        # Process only directories that follow the expected user ID naming convention.
        if user_id.startswith('ID_'):
            # Find all real signature images for the current user.
            real_images = glob.glob(os.path.join(real_signatures_dir, user_id, '*.jpg'))
            
            # Define the path for the corresponding fake signatures directory.
            fake_user_dir = os.path.join(fake_signatures_dir, user_id)
            # Find all fake images if the user's fake directory exists, otherwise use an empty list.
            fake_images = glob.glob(os.path.join(fake_user_dir, '*.jpg')) if os.path.isdir(fake_user_dir) else []
            
            # Filter users based on the minimum required number of real and fake signatures.
            if len(real_images) >= min_real and len(fake_images) >= min_fake:
                # Store the list of real image paths for the valid user.
                signature_map[user_id]['real'] = real_images
                # Store the list of fake image paths for the valid user.
                signature_map[user_id]['fake'] = fake_images
    
    # Return the completed map of user IDs to their signature paths.
    return signature_map



def display_signature_dataset_summary(base_data_dir):
    """
    Displays a dynamic summary and bar charts of the dataset.
    """
    # Create a map of signatures, filtering by minimum image counts.
    signature_map = _create_signature_map(base_data_dir, min_real=2, min_fake=1)

    # Exit the function if no valid data was found.
    if not signature_map:
        print("No valid individuals found for triplet generation.")
        return

    # Convert the map to a list of dictionaries to prepare for DataFrame creation.
    data_list = [{'ID': user_id, 'Real': len(files['real']), 'Fake': len(files['fake'])} for user_id, files in signature_map.items()]
    # Create a pandas DataFrame from the list of data.
    df = pd.DataFrame(data_list)
    # Sort the DataFrame numerically by the ID number for consistent ordering.
    df_sorted = df.sort_values(by="ID", key=lambda x: x.str.split('_').str[1].astype(int)).reset_index(drop=True)

    # Calculate aggregate statistics for the summary.
    num_ids = len(df_sorted)
    total_real = df_sorted['Real'].sum()
    total_fake = df_sorted['Fake'].sum()

    # Display the high level summary statistics.
    print(f"Found {num_ids} valid IDs.")
    print(f"   - Total Real Images: {total_real}")
    print(f"   - Total Fake Images: {total_fake}\n")

    # Set a base font size for all plot elements for better readability.
    plt.rcParams['font.size'] = 14
    # Divide the sorted data into chunks for clearer visualization.
    chunks = {
        "IDs 1-17": df_sorted.iloc[0:17],
        "IDs 18-34": df_sorted.iloc[17:34],
        "IDs 35-51": df_sorted.iloc[34:51]
    }

    # Generate a separate plot for each chunk of data.
    for title, chunk in chunks.items():
        # Skip empty chunks to avoid creating empty plots.
        if chunk.empty:
            continue

        # Extract data series from the chunk for plotting.
        ids = chunk['ID']
        real_counts = chunk['Real']
        fake_counts = chunk['Fake']

        # Define the label locations for the x axis.
        x = np.arange(len(ids))
        # Define the width of the bars in the bar chart.
        width = 0.35

        # Create a new figure and axes for the plot.
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot the bars for real and fake signature counts.
        rects1 = ax.bar(x - width/2, real_counts, width, label='Real Images', color='royalblue')
        rects2 = ax.bar(x + width/2, fake_counts, width, label='Fake Images', color='salmon')

        # Customize plot elements for clarity.
        ax.set_ylabel('Image Count')
        ax.set_title(f'Signature Counts for {title}', fontsize=18, weight='bold')
        ax.set_xticks(x)
        # Rotate x axis labels to prevent overlap.
        ax.set_xticklabels(ids, rotation=45, ha="right")
        ax.legend()
        
        # Add numeric labels on top of each bar for precise values.
        ax.bar_label(rects1, padding=3, fontsize=12)
        ax.bar_label(rects2, padding=3, fontsize=12)

        # Adjust layout to prevent plot elements from being cut off.
        fig.tight_layout()
        # Display the generated plot.
        plt.show()
        


def display_random_signature_pair(base_data_dir):
    """
    Displays a random pair of real and fake signatures.
    """
    # Create a map of signatures, filtering for IDs with at least one of each type.
    signature_map = _create_signature_map(base_data_dir, min_real=1, min_fake=1)

    # Exit the function if no valid data was found.
    if not signature_map:
        print("No valid individuals with both real and fake signatures were found.")
        return
    
    # Randomly select one individual from the list of valid IDs.
    random_id = random.choice(list(signature_map.keys()))
    
    # Get the lists of available images for the selected individual.
    real_images = signature_map[random_id]['real']
    fake_images = signature_map[random_id]['fake']
    
    # Randomly select one real and one fake signature path from the lists.
    real_image_path = random.choice(real_images)
    fake_image_path = random.choice(fake_images)
    
    # Attempt to load the selected images from their respective paths.
    try:
        real_img = Image.open(real_image_path)
        fake_img = Image.open(fake_image_path)
    # Handle cases where an image file cannot be found.
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        return

    # Create a figure with two subplots side by side for comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Set a main title for the entire figure.
    fig.suptitle(f'Signature Comparison for {random_id}', fontsize=16)
    
    # Display the real signature on the left subplot.
    ax1.imshow(real_img, cmap='gray')
    ax1.set_title('Real Signature')
    # Turn off the axis for a cleaner look.
    ax1.axis('off')
    
    # Display the fake signature on the right subplot.
    ax2.imshow(fake_img, cmap='gray')
    ax2.set_title('Fake Signature')
    # Turn off the axis for a cleaner look.
    ax2.axis('off')
    
    # Render the final plot to the screen.
    plt.show()
    

     
def create_signature_datasets_splits(full_dataset, train_split, train_transform, val_transform):
    """
    Splits a pre initialized dataset into training and validation sets,
    applying different transformations to each.

    Args:
        full_dataset: The complete dataset object, initialized with transform=None.
        train_split: The proportion of the dataset to use for training (e.g., 0.8).
        train_transform: Transformations for the training set.
        val_transform: Transformations for the validation set.

    Returns:
        tuple: A tuple containing the (train_dataset, val_dataset).
    """
    
    # Define an internal wrapper class to apply a specific transform to a dataset subset.
    class TransformedSignatureSubset(Dataset):
        # Initialize the subset with its data and a specific transformation pipeline.
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        
        # Retrieve an item by index and apply the transformation.
        def __getitem__(self, index):
            # Get the untransformed data triplet from the base subset.
            anchor, positive, negative = self.subset[index]
            
            # Apply the stored transformation pipeline to each image if it exists.
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            # Return the transformed triplet.
            return anchor, positive, negative
            
        # Return the total number of items in the subset.
        def __len__(self):
            return len(self.subset)

    # Ensure the provided dataset was created without a default transform.
    if full_dataset.transform is not None:
        raise ValueError("The 'full_dataset' must be initialized with transform=None for this function to work correctly.")

    # Calculate the number of samples for the training and validation sets.
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Perform a random split of the original dataset into two subsets.
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Wrap each new subset with its corresponding transformation pipeline.
    train_dataset = TransformedSignatureSubset(train_subset, train_transform)
    val_dataset = TransformedSignatureSubset(val_subset, val_transform)

    # Return the final training and validation datasets.
    return train_dataset, val_dataset


      
def deprocess_signature_image(tensor):
    """
    Reverses the normalization on an image tensor for display.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = np.array([0.861, 0.861, 0.861])
    std = np.array([0.274, 0.274, 0.274])
    
    # Convert the tensor to a NumPy array in the proper format for image processing.
    tensor = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    
    # Reverse the normalization process (denormalize).
    tensor = std * tensor + mean
    
    # Clip the values to the valid range [0, 1] to prevent display errors.
    tensor = np.clip(tensor, 0, 1)
    
    # Return the processed image array ready for visualization.
    return tensor



def show_random_triplet(dataloader):
    """
    Displays a single random triplet from a DataLoader.
    """
    # Check if the dataloader is valid before proceeding.
    if not dataloader:
        print("DataLoader is not available. Cannot display triplet.")
        return

    # Retrieve a single batch of data from the iterator.
    anchor_batch, positive_batch, negative_batch = next(iter(dataloader))
    
    # Select the first triplet from the batch for display.
    anchor = anchor_batch[0]
    positive = positive_batch[0]
    negative = negative_batch[0]
    
    # Prepare the image tensors for visualization by reversing normalization.
    anchor_img = deprocess_signature_image(anchor)
    positive_img = deprocess_signature_image(positive)
    negative_img = deprocess_signature_image(negative)
    
    # Create a figure and a set of subplots for displaying the images.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Set a main title for the entire figure.
    fig.suptitle('Example of a Training Triplet', fontsize=16)
    
    # Display the anchor image in the first subplot.
    axes[0].imshow(anchor_img)
    axes[0].set_title('Anchor (Real)')
    axes[0].axis('off')
    
    # Display the positive image in the second subplot.
    axes[1].imshow(positive_img)
    axes[1].set_title('Positive (Real)')
    axes[1].axis('off')
    
    # Display the negative image in the third subplot.
    axes[2].imshow(negative_img)
    axes[2].set_title('Negative (Fake of Same ID)')
    axes[2].axis('off')
        
    # Adjust subplot parameters for a tight layout.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Render the plot to the screen.
    plt.show()
    
    
    
def show_signature_val_predictions(model, val_loader, threshold, device):
    """
    Visualizes model performance on a sample from the validation set.

    This function fetches a single batch, randomly selects one sample from it,
    and creates two plots: one for a genuine pair (anchor vs. positive) and
    one for a forgery pair (anchor vs. negative). Each plot displays the
    images, the model's prediction, and a graphical bar chart comparing the
    calculated distance to the decision threshold.

    Args:
        model (torch.nn.Module): The trained Siamese network model.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        threshold (float): The distance threshold for classifying pairs.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
    """
    print("--- Displaying Validation Predictions with Distance Visualization ---\n")
    # Set the model to evaluation mode to disable layers like dropout
    model.eval()
    
    # Safely get a single batch of data from the loader
    try:
        anchor, positive, negative = next(iter(val_loader))
    except StopIteration:
        print("DataLoader is empty.")
        return
        
    # Move image tensors to the specified computation device
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
    
    # Safeguard against empty or small batches
    batch_size = anchor.size(0)
    if batch_size < 1:
        print("Batch is empty, cannot show examples.")
        return
    
    # Randomly select one index from the batch to create visualization pairs from
    index = random.choice(range(batch_size))
    
    # Prepare data for one genuine and one forgery pair for display
    # Each tuple contains: (image1, image2, pair_type, title1, title2, true_label)
    pairs_to_show = [
        (anchor[index], positive[index], "Genuine Pair", "Real 1", "Real 2", 1),
        (anchor[index], negative[index], "Forgery Pair", "Real", "Fake", 0)
    ]
    
    # Iterate through the prepared pairs to generate and display predictions
    for img1_tensor, img2_tensor, pair_type, title1, title2, true_label in pairs_to_show:
        # Disable gradient calculations for inference
        with torch.no_grad():
            # Generate embeddings for both images in the pair
            emb1 = model.get_embedding(img1_tensor.unsqueeze(0))
            emb2 = model.get_embedding(img2_tensor.unsqueeze(0))
            # Calculate the Euclidean distance between the two embeddings
            distance = F.pairwise_distance(emb1, emb2).item()
            # Make a prediction based on whether the distance is below the threshold
            prediction = 1 if distance < threshold else 0

        # --- Plotting Setup ---
        # Create a figure and a custom grid layout for the plots
        fig = plt.figure(figsize=(8, 5.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1]) # Top row for images, bottom for bar plot

        ax1 = fig.add_subplot(gs[0, 0])      # Top-left for image 1
        ax2 = fig.add_subplot(gs[0, 1])      # Top-right for image 2
        ax_dist = fig.add_subplot(gs[1, :])  # Bottom row (spanned) for the distance plot

        # Prepare dynamic strings and colors based on the prediction's correctness
        prediction_str = 'Genuine' if prediction == 1 else 'Forgery'
        result_str = 'CORRECT' if prediction == true_label else 'INCORRECT'
        title_color = 'green' if prediction == true_label else 'red'
        bar_color = 'green' if prediction == true_label else 'red'
        operator_str = '<' if distance < threshold else '>='
        
        # Construct the multi-line title for the entire figure
        title = (f"Type: {pair_type}\n"
                 f"Prediction: {prediction_str} -> {result_str}\n"
                 f"Dist: {distance:.2f} {operator_str} Thresh: {threshold:.2f}")
        fig.suptitle(title, color=title_color, fontsize=12, y=0.98)
        
        # --- Display Images (Top Row) ---
        ax1.imshow(deprocess_signature_image(img1_tensor))
        ax1.set_title(title1)
        ax1.axis('off')
        
        ax2.imshow(deprocess_signature_image(img2_tensor))
        ax2.set_title(title2)
        ax2.axis('off')

        # --- Display Distance Plot (Bottom Row) ---
        # Plot the calculated distance as a horizontal bar
        ax_dist.barh([0], [distance], color=bar_color, height=0.5, zorder=1)
        # Draw the threshold as a vertical line on top of the bar
        ax_dist.axvline(x=threshold, color='black', linestyle='--', zorder=2)
        # Add a text label showing the precise distance value
        ax_dist.text(distance + 0.02, 0, f'{distance:.2f}', va='center', fontweight='bold')
        
        # Format the distance plot for clarity
        ax_dist.set_yticks([]) # Hide y-axis ticks
        ax_dist.set_xlim(0, max(distance, threshold) * 1.5) # Set dynamic x-axis limit
        ax_dist.set_xlabel("Euclidean Distance in Embedding Space")
        
        # Adjust layout and display the final plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
        plt.show()
        
        
        
def verify_signature(model, genuine_path, test_path, threshold, transform, device):
    """
    Performs one shot verification for a pair of signatures.
    
    Args:
        model: The trained Siamese network.
        genuine_path (str): Path to a known genuine signature (the anchor).
        test_path (str): Path to the signature being tested.
        threshold (float): The optimal decision threshold from evaluation.
        transform: The image transformations.
        device: The device to run on (CPU or GPU).
    """
    # Set the model to evaluation mode to disable layers like dropout.
    model.eval() 
    
    # Load and process both the genuine and test images.
    try:
        # Apply transformations and move the genuine image tensor to the correct device.
        img_genuine = transform(Image.open(genuine_path).convert("RGB")).unsqueeze(0).to(device)
        # Apply transformations and move the test image tensor to the correct device.
        img_test = transform(Image.open(test_path).convert("RGB")).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        return

    # Disable gradient calculations for inference to save memory and computations.
    with torch.no_grad():
        # Generate embeddings for both images using the model.
        emb_genuine = model.get_embedding(img_genuine)
        emb_test = model.get_embedding(img_test)
        
        # Calculate the euclidean distance between the two embeddings.
        distance = F.pairwise_distance(emb_genuine, emb_test).item()
        
        # Make a prediction based on whether the distance is below the threshold.
        is_genuine = distance < threshold
        
    # Display the numerical results of the verification.
    print(f"--- Verification Result ---")
    print(f"Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    # Print the final prediction outcome.
    if is_genuine:
        print("Prediction: ✅ Genuine Signature\n")
    else:
        print("Prediction: ❌ Forgery Detected\n")
        
    # Create a figure with two subplots for visual comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Set the main title of the figure to show the calculated distance.
    fig.suptitle(f'Distance: {distance:.4f}', fontsize=16)
    
    # Display the known genuine signature in the left subplot.
    ax1.imshow(Image.open(genuine_path))
    ax1.set_title("Known Genuine Signature")
    ax1.axis('off')
    
    # Display the signature to be verified in the right subplot.
    ax2.imshow(Image.open(test_path))
    ax2.set_title("Signature to Verify")
    ax2.axis('off')
    
    # Render the final plot.
    plt.show()
    
    

def _create_change_map(base_data_dir):
    """
    Scans the change detection dataset directory and creates a map of
    'Before' and 'After' image pairs for each change category.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.

    Returns:
        dict: A dictionary mapping change categories to a list of
              (before_path, after_path) tuples.
    """
    # Initialize a dictionary to store the paths for each category.
    change_map = {'Positive': [], 'Negative': [], 'No_Change': []}
    # Get the list of category names to iterate over.
    categories = list(change_map.keys())
    
    # Loop through each category directory (Positive, Negative, No_Change).
    for category in categories:
        # Construct the full paths for the category's subdirectories.
        category_path = os.path.join(base_data_dir, category)
        before_dir = os.path.join(category_path, 'Before')
        after_dir = os.path.join(category_path, 'After')

        # Skip the current category if its 'Before' directory does not exist.
        if not os.path.isdir(before_dir):
            continue

        # Iterate through all files found in the 'Before' directory.
        for filename in os.listdir(before_dir):
            # Construct the full file path for the 'before' and corresponding 'after' image.
            before_path = os.path.join(before_dir, filename)
            after_path = os.path.join(after_dir, filename)
            
            # Check if the corresponding 'after' image exists to form a valid pair.
            if os.path.exists(after_path):
                # If the pair is valid, add the tuple of paths to the map.
                change_map[category].append((before_path, after_path))
    
    # Return the completed map of image pairs.
    return change_map


    
def display_change_dataset_stats(base_data_dir):
    """
    Displays a detailed statistical summary of the change detection dataset.

    This function processes the dataset found in the specified directory,
    calculates the number of image pairs for each change category, and
    presents the statistics in a styled table format.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.
    """
    # Generate a map of change categories to corresponding image pairs.
    change_map = _create_change_map(base_data_dir)

    # Verify that the change map is not empty before proceeding.
    if not any(change_map.values()):
        print("The change map is empty. Cannot display stats.")
        return

    # Restructure the map data into a list of dictionaries for DataFrame creation.
    data_list = []
    for category, pairs in change_map.items():
        data_list.append({
            'Change Category': category,
            'Number of Image Pairs': len(pairs)
        })

    # Create a DataFrame from the list, sort it by category, and reset the index.
    df = pd.DataFrame(data_list).sort_values(by='Change Category').reset_index(drop=True)

    # Calculate the total number of image pairs across all categories.
    total_pairs = df['Number of Image Pairs'].sum()
    
    # Create a new DataFrame for the summary 'Total' row.
    total_row = pd.DataFrame([{
        'Change Category': '<b>Total</b>',
        'Number of Image Pairs': total_pairs
    }])
    
    # Append the total row to the main DataFrame for display.
    df_display = pd.concat([df, total_row], ignore_index=True)

    # Initialize a Styler object to customize the DataFrame's appearance, starting by hiding the index.
    styler = df_display.style.hide(axis="index")
    
    # Apply custom CSS styles to the table elements for improved readability.
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "60%"), ("margin", "0")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
            {"selector": "th", "props": [
                ("text-align", "left"),
                ("padding", "8px"),
                ("background-color", "#4f4f4f"),
                ("color", "white")
            ]}
        ]
    )
    # Set additional properties for table cells to ensure content wraps properly.
    styler.set_properties(**{"white-space": "normal"})
    
    # Render the styled DataFrame in the output.
    display(styler)


    
def display_random_change_pairs(base_data_dir):
    """
    Displays a random 'Before' and 'After' image pair from each change
    category, with the category name centered above each pair.

    This function generates a visual comparison for 'Positive', 'Negative',
    and 'No_Change' categories by selecting a random image pair from each
    and arranging them in a clear, titled grid layout.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.
    """
    # Generate a map of change categories to their corresponding image pairs.
    change_map = _create_change_map(base_data_dir)

    # Verify that the change map is not empty before attempting to display images.
    if not any(change_map.values()):
        print("The change map is empty. Cannot display image pairs.")
        return

    # Initialize the main figure for plotting.
    fig = plt.figure(figsize=(8, 15))

    # Create an outer grid to structure the plots for each category vertically.
    outer_grid = GridSpec(3, 1, figure=fig, hspace=0.2)
    
    # Define the specific order of categories to be displayed.
    categories = ['Positive', 'Negative', 'No_Change']

    # Loop through each category to create and display its corresponding plot.
    for i, category in enumerate(categories):
        # Format the category name for a cleaner display title.
        display_category = category.replace('_', ' ')
        # Retrieve the list of image pairs for the current category.
        image_pairs = change_map.get(category, [])
        
        # Create a nested grid within the outer grid for the category title and the image pair.
        # The height ratio gives more space to the images compared to the title.
        inner_grid = outer_grid[i].subgridspec(2, 2, height_ratios=[0.05, 1], wspace=0.05, hspace=0.1)

        # Add a subplot for the category title, spanning both columns of the inner grid.
        ax_title = fig.add_subplot(inner_grid[0, :])
        # Add and style the category title text.
        ax_title.text(0.5, 0.5, display_category, ha='center', va='center', fontsize=18, weight='bold')
        # Hide the axes for the title subplot.
        ax_title.axis('off')

        # Create subplots for the 'Before' and 'After' images.
        ax_before = fig.add_subplot(inner_grid[1, 0])
        ax_after = fig.add_subplot(inner_grid[1, 1])

        # Set the titles for the individual image subplots.
        ax_before.set_title('Before', fontsize=16)
        ax_after.set_title('After', fontsize=16)
        
        # Check if any image pairs exist for the current category.
        if not image_pairs:
            # Display a message if no images are found.
            ax_before.text(0.5, 0.5, 'No Images Found', ha='center', va='center', fontsize=12)
            ax_after.text(0.5, 0.5, 'No Images Found', ha='center', va='center', fontsize=12)
        else:
            # Randomly select one 'Before' and 'After' image pair.
            before_path, after_path = random.choice(image_pairs)
            # Attempt to open and display images, handling potential file errors.
            try:
                ax_before.imshow(Image.open(before_path))
                ax_after.imshow(Image.open(after_path))
            except FileNotFoundError:
                # Display a message if an image file cannot be found.
                ax_before.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=12)
                ax_after.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=12)

        # Turn off the axes for the image plots for a cleaner look.
        ax_before.axis('off')
        ax_after.axis('off')

    # Render and display the final composite figure.
    plt.show()
    
    

def create_change_datasets_splits(full_dataset, train_split, train_transform, val_transform):
    """
    Splits a pre-initialized change detection dataset into training and
    validation sets, applying different transformations to each.

    This function is designed to take a single, untransformed dataset and
    produce reproducible training and validation splits, each with its own
    set of data augmentation and preprocessing transforms.

    Args:
        full_dataset (Dataset): The complete dataset object, initialized with transform=None.
        train_split (float): The proportion of the dataset to allocate for training (e.g., 0.8).
        train_transform (callable): The transformations to apply to the training set.
        val_transform (callable): The transformations to apply to the validation set.

    Returns:
        tuple: A tuple containing the created (train_dataset, val_dataset).
        
    Raises:
        ValueError: If the `full_dataset` is initialized with any transforms.
    """
    
    # Define an internal wrapper class to apply a specific transform to a dataset subset.
    class TransformedChangeSubset(Dataset):
        def __init__(self, subset, transform):
            # Store the subset of data (e.g., from random_split).
            self.subset = subset
            # Store the transformations to be applied to this specific subset.
            self.transform = transform
        
        def __getitem__(self, index):
            # Retrieve the untransformed data item from the original subset.
            before_img, after_img, label = self.subset[index]
            
            # Apply the specified transformations to both images if a transform is provided.
            if self.transform:
                before_img = self.transform(before_img)
                after_img = self.transform(after_img)
            
            return before_img, after_img, label
            
        def __len__(self):
            # Return the total number of items in the subset.
            return len(self.subset)

    # Validate that the input dataset has not already been assigned a transform.
    if full_dataset.transform is not None:
        raise ValueError("The 'full_dataset' must be initialized with transform=None for this function to work correctly.")

    # Calculate the exact number of samples for the training and validation splits.
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Create a generator with a fixed seed for reproducible splits.
    generator = torch.Generator().manual_seed(42)
    # Perform a random split of the dataset into training and validation subsets.
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap each subset with the custom class to apply the correct set of transforms.
    train_dataset = TransformedChangeSubset(train_subset, train_transform)
    val_dataset = TransformedChangeSubset(val_subset, val_transform)

    return train_dataset, val_dataset


 
def deprocess_change_image(tensor):
    """
    Reverses ImageNet normalization on an image tensor for display.

    This function takes a normalized PyTorch tensor, converts it back to a
    displayable image format by reversing the normalization, and scales it
    to the standard 0-255 pixel value range.

    Args:
        tensor (torch.Tensor): A PyTorch tensor representing an image,
                               typically with shape (C, H, W).

    Returns:
        np.ndarray: A NumPy array representing the deprocessed image in
                    (H, W, C) format with pixel values in the range [0, 255].
    """
    # Define the mean and standard deviation used for ImageNet normalization.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert the tensor to a NumPy array and change channel order from (C, H, W) to (H, W, C).
    tensor = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)

    # Reverse the normalization process (de-standardize).
    tensor = std * tensor + mean

    # Clip the values to ensure they are within the valid [0, 1] range.
    tensor = np.clip(tensor, 0, 1)

    # Scale the pixel values from [0, 1] to [0, 255] and convert to an 8-bit integer.
    return (tensor * 255).astype(np.uint8)



def show_random_pair(dataloader):
    """
    Displays a single random 'Before' and 'After' pair from a DataLoader.

    This function fetches one batch from the provided DataLoader, selects the
    first item, de-processes the image tensors for visualization, and plots
    them side-by-side with appropriate titles.

    Args:
        dataloader: A DataLoader object that yields batches of
                    (before_image, after_image, label) tuples.
    """
    # Ensure the DataLoader is valid before proceeding.
    if not dataloader:
        print("DataLoader is not available. Cannot display a pair.")
        return

    # Retrieve a single batch of data from the dataloader.
    before_batch, after_batch, label_batch = next(iter(dataloader))
    
    # Select the first image pair and its corresponding label from the batch.
    before_tensor = before_batch[0]
    after_tensor = after_batch[0]
    label = label_batch[0].item()
    
    # Convert the image tensors from their normalized format to a displayable format.
    before_img = deprocess_change_image(before_tensor)
    after_img = deprocess_change_image(after_tensor)
    
    # Map the numerical label to its human-readable string representation for the title.
    class_map = {0: 'Positive', 1: 'Negative', 2: 'No Change'}
    class_name = class_map.get(label, 'Unknown')
    
    # Create a plot to display the 'Before' and 'After' images side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Example of a "{class_name}" Training Pair', fontsize=16)
    
    # Display the 'Before' image.
    ax1.imshow(before_img)
    ax1.set_title('Before')
    ax1.axis('off')
    
    # Display the 'After' image.
    ax2.imshow(after_img)
    ax2.set_title('After')
    ax2.axis('off')
          
    # Adjust layout to prevent titles from overlapping and display the plot.
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
 
    
def compute_change_class_weights(train_dataset, full_untransformed_dataset):
    """
    Computes class weights for the training set to handle class imbalance.

    Args:
        train_dataset (Dataset): The training subset, expected to be a wrapper
                                 around a `torch.utils.data.Subset` instance.
        full_untransformed_dataset (Dataset): The original, complete dataset from
                                              which the subset was created.

    Returns:
        torch.Tensor: A tensor of weights for each class, suitable for use in a
                      loss function like `nn.CrossEntropyLoss`.
    """
    # Get the list of indices that belong to the training subset.
    train_indices = train_dataset.subset.indices

    # Extract the ground truth labels for the training samples from the original full dataset.
    train_labels = [full_untransformed_dataset.image_pairs[i][2] for i in train_indices]

    # Use scikit-learn's utility to compute weights that balance the classes.
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    # Convert the NumPy array of weights into a PyTorch tensor.
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return weights_tensor
    
    
    
def get_efficientnet_embedding_backbone(embedding_dim=128, pretrained=True,
    weights_path="./pretrained_efficientnet_weights/efficientnet_b0_rwightman-7f5810bc.pth"):
    """
    Builds an embedding network from an EfficientNet-B0 model, loading weights offline.

    This function configures an EfficientNet-B0 model to act as a feature extractor
    by replacing its final classification layer with a linear layer that outputs
    an embedding of a specified dimension. It supports loading weights from a local file.

    Args:
        embedding_dim (int): The desired dimension of the output embedding vector.
        pretrained (bool): If True, loads weights from the local `weights_path`.
        weights_path (str): The local file path to the .pth weights file.

    Returns:
        torch.nn.Module: The modified EfficientNet-B0 model.
        
    Raises:
        FileNotFoundError: If `pretrained` is True and the weights file is not found.
    """
    # Instantiate the EfficientNet-B0 model architecture with random weights.
    model = tv_models.efficientnet_b0(weights=None)

    # If specified, load the pretrained weights from a local file.
    if pretrained:
        # Raise an error if the weights file is missing to prevent silent failures.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at the specified path: {weights_path}")
        
        # Load the state dictionary from the path, mapping to CPU to avoid device errors.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Load the state dictionary into the model.
        model.load_state_dict(state_dict)
    
    # Modify the final classification layer to produce an embedding.
    # Get the number of input features from the original classifier layer.
    num_ftrs = model.classifier[1].in_features
    # Replace the layer with a new linear layer of the desired embedding dimension.
    model.classifier[1] = nn.Linear(num_ftrs, embedding_dim)
    
    return model



def calculate_vegetation_percentage(image_array):
    """
    Calculates the percentage of an image that contains green vegetation.

    This function identifies vegetation by converting the input image to the
    HSV color space and creating a mask for pixels that fall within a
    pre-defined range for the color green.

    Args:
        image_array (np.ndarray): A NumPy array representing an RGB image,
                                  with shape (height, width, 3).

    Returns:
        float: The percentage of the image classified as vegetation,
               represented as a value between 0.0 and 1.0.
    """
    # Convert the input RGB image to the HSV (Hue, Saturation, Value) color space.
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Create a binary mask where green pixels are white (255) and others are black (0).
    mask = cv2.inRange(hsv_image, LOWER_GREEN_HSV, UPPER_GREEN_HSV)

    # Count the number of pixels identified as vegetation (white pixels in the mask).
    veg_count = np.sum(mask == 255)

    # Calculate the total number of pixels in the image.
    total_pixels = image_array.shape[0] * image_array.shape[1]

    # Return the ratio of vegetation pixels to total pixels, handling division by zero.
    return veg_count / total_pixels if total_pixels > 0 else 0




def classify_greenery_change(input_before, input_after, threshold_percent):
    """
    Analyzes two images and classifies the change in greenery.

    This function can process inputs as either file paths or PyTorch tensors.
    It calculates the percentage of green vegetation in each image and
    determines if the change between them is positive, negative, or negligible
    based on a given threshold.

    Args:
        input_before (str or torch.Tensor): The 'before' image.
        input_after (str or torch.Tensor): The 'after' image.
        threshold_percent (float): The percentage change required to be
                                   considered significant (e.g., 5.0 for 5%).

    Returns:
        str: A classification label: 'Positive', 'Negative', or 'No_Change'.
             Returns an error string if a file path is not found.

    Raises:
        TypeError: If inputs are not of type str or torch.Tensor.
    """
    img_before_array = None
    img_after_array = None

    # Handle input based on whether it is a file path or a tensor.
    if isinstance(input_before, str):
        # If input is a string, treat it as a file path and load the image.
        try:
            img_before_array = np.array(Image.open(input_before).convert("RGB"))
            img_after_array = np.array(Image.open(input_after).convert("RGB"))
        except FileNotFoundError:
            return 'Error: File not found'
            
    elif isinstance(input_before, torch.Tensor):
        # If input is a tensor, de-process it into a NumPy array.
        img_before_array = deprocess_change_image(input_before)
        img_after_array = deprocess_change_image(input_after)
        
    else:
        # Raise an error for unsupported input types.
        raise TypeError("Inputs must be either file paths (str) or PyTorch tensors.")

    # Calculate the percentage of vegetation in each image.
    percent_before = calculate_vegetation_percentage(img_before_array)
    percent_after = calculate_vegetation_percentage(img_after_array)

    # Determine the difference in vegetation and convert the threshold to a decimal.
    veg_change = percent_after - percent_before
    threshold_decimal = threshold_percent / 100.0

    # Classify the change based on the calculated difference and threshold.
    if veg_change >= threshold_decimal:
        return 'Positive'
    elif veg_change <= -threshold_decimal:
        # For a negative change, ensure the initial vegetation was significant.
        if percent_before >= threshold_decimal:
            return 'Negative'
        else:
            return 'No_Change'
    else:
        # If the change is within the threshold, it is considered no change.
        return 'No_Change'
    
    
    
def show_change_val_predictions(model, val_loader, model_threshold, device):
    """
    Displays one random example from each class (Positive, Negative, No_Change)
    from the validation set, showing the model's prediction for each.

    This function iterates through the validation set, selects one random sample
    for each class, runs them through the model to get a prediction, and then
    visualizes the 'before' and 'after' images along with the true and
    predicted labels.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        val_loader (DataLoader): The DataLoader for the validation set.
        model_threshold (float): The distance threshold for classifying change.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run inference on.
    """
    print(f"--- Displaying One Random Prediction for Each Class (Model Threshold: {model_threshold:.4f}) ---\n")
    model.eval()
    
    # First, collect all validation samples and sort them by class.
    positive_samples, negative_samples, no_change_samples = [], [], []
    for before_batch, after_batch, label_batch in val_loader:
        for i in range(len(label_batch)):
            label = label_batch[i].item()
            sample = {
                "before_tensor": before_batch[i],
                "after_tensor": after_batch[i],
                "true_label_int": label,
            }
            if label == 0: positive_samples.append(sample)
            elif label == 1: negative_samples.append(sample)
            else: no_change_samples.append(sample)

    # Ensure samples from all classes were found before proceeding.
    if not all([positive_samples, negative_samples, no_change_samples]):
        print("Validation set is missing samples from one or more classes.")
        return

    # Randomly select one sample from each class list to display.
    samples_to_show = [
        random.choice(positive_samples),
        random.choice(negative_samples),
        random.choice(no_change_samples)
    ]
    
    # Map integer labels to their string representations for plotting.
    class_map = {0: 'Positive', 1: 'Negative', 2: 'No Change'}

    # Process and display each of the three selected samples.
    for sample in samples_to_show:
        before_tensor = sample["before_tensor"].to(device)
        after_tensor = sample["after_tensor"].to(device)
        true_label_int = sample["true_label_int"]

        # Perform inference without calculating gradients for efficiency.
        with torch.no_grad():
            emb1, emb2 = model(before_tensor.unsqueeze(0), after_tensor.unsqueeze(0), triplet_bool=False)
            distance = F.pairwise_distance(emb1, emb2).item()

            # Make a prediction using a two-step process:
            # 1. Use the distance to predict 'Change' vs. 'No Change'.
            # 2. If 'Change' is predicted, use a heuristic to guess the change type.
            if distance <= model_threshold:
                final_prediction_int = 2
            else:
                change_type_str = classify_greenery_change(
                    input_before=before_tensor,
                    input_after=after_tensor,
                    threshold_percent=5.0
                )
                if change_type_str == 'Positive':
                    final_prediction_int = 0
                else:
                    final_prediction_int = 1
        
        # Prepare display strings and colors based on the prediction outcome.
        true_label_str = class_map.get(true_label_int, "Unknown")
        final_prediction_str = class_map.get(final_prediction_int, "Unknown")
        result_str = 'CORRECT' if final_prediction_int == true_label_int else 'INCORRECT'
        title_color = 'green' if final_prediction_int == true_label_int else 'red'
        
        # Construct a detailed title with the results and distance comparison.
        comparison_op = "<=" if distance <= model_threshold else ">"
        title = (f"True Label: {true_label_str} | Prediction: {final_prediction_str} -> {result_str}\n"
                 f"Distance: {distance:.2f} {comparison_op} {model_threshold:.2f} (Threshold)")

        # Create the plot for the 'Before' and 'After' images.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(title, color=title_color, fontsize=12, y=1.03)
        
        ax1.imshow(deprocess_change_image(before_tensor))
        ax1.set_title('Before')
        ax1.axis('off')
        
        ax2.imshow(deprocess_change_image(after_tensor))
        ax2.set_title('After')
        ax2.axis('off')
        
        plt.show()
        
        

def predict_greenery_change(model, before_path, after_path, model_threshold, transform, device):
    """
    Performs one-shot change detection for a pair of images.

    This function loads, preprocesses, and runs a pair of images through a
    trained Siamese network. It then classifies the change and displays the
    images along with the prediction result.

    Args:
        model (torch.nn.Module): The trained Siamese network.
        before_path (str): Path to the 'before' image.
        after_path (str): Path to the 'after' image.
        model_threshold (float): The optimal decision threshold from evaluation.
        transform (callable): The image transformations to apply.
        device (torch.device): The device to run inference on (e.g., 'cpu' or 'cuda').
        
    Returns:
        None. Prints the prediction and displays a plot.
    """
    # Set the model to evaluation mode to disable layers like dropout.
    model.eval()
    
    # Load and preprocess the images, handling potential file errors.
    try:
        img_before = transform(Image.open(before_path).convert("RGB")).unsqueeze(0).to(device)
        img_after = transform(Image.open(after_path).convert("RGB")).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        return

    # Perform inference without calculating gradients for efficiency.
    with torch.no_grad():
        # Get the image embeddings from the model.
        emb_before, emb_after = model(img_before, img_after, triplet_bool=False)
        # Calculate the distance between the two embeddings.
        distance = F.pairwise_distance(emb_before, emb_after).item()
        
        # Use the distance and threshold to make a prediction.
        if distance <= model_threshold:
            # If the distance is below the threshold, classify as no significant change.
            final_prediction = 'No Change'
        else:
            # If a change is detected, use a secondary classifier for the change type.
            final_prediction = classify_greenery_change(
                input_before=before_path, 
                input_after=after_path, 
                threshold_percent=5.0
            )
            
    # Display the results in the console.
    print(f"--- Change Detection Result ---")
    print(f"Model Distance: {distance:.4f}")
    print(f"Decision Threshold: {model_threshold:.4f}")
    
    # Print the final prediction with a corresponding emoji.
    if final_prediction == 'Positive':
        print(f"Prediction: 🌳 {final_prediction}\n")
    elif final_prediction == 'Negative':
        print(f"Prediction: 🪓 {final_prediction}\n")
    else:
        print(f"Prediction: 🔄 {final_prediction}\n")
        
    # Create a plot to visualize the images and the prediction.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Prediction: {final_prediction} (Distance: {distance:.4f})', fontsize=14)
    
    # Display the 'Before' image.
    ax1.imshow(Image.open(before_path))
    ax1.set_title("Before")
    ax1.axis('off')
    
    # Display the 'After' image.
    ax2.imshow(Image.open(after_path))
    ax2.set_title("After")
    ax2.axis('off')
    
    # Show the final plot.
    plt.show()
    
    
    
def plot_confusion_matrix_and_metrics(model, data_loader, threshold, device):
    """
    Computes and plots a confusion matrix and classification metrics for a
    Siamese network on a binary change detection task.

    Args:
        model (nn.Module): The trained Siamese network.
        data_loader (DataLoader): The validation data loader.
        threshold (float): The optimal distance threshold to classify change.
        device (torch.device): The device to run the model on.
    """
    # Set the model to evaluation mode for inference.
    model.eval()
    
    all_labels = []
    all_preds = []

    # Generate predictions for the entire dataset.
    with torch.no_grad():
        for before_img, after_img, labels in tqdm(data_loader, desc="Generating Predictions"):
            # Move data to the specified device.
            before_img = before_img.to(device)
            after_img = after_img.to(device)
            
            # Get embeddings from the model.
            output1, output2 = model(before_img, after_img, triplet_bool=False)
            
            # Calculate the pairwise distance between embeddings.
            distances = F.pairwise_distance(output1, output2)
            
            # Make binary predictions based on the optimal threshold.
            preds = (distances >= threshold).long().cpu().numpy()
            
            # Convert the original multi-class labels to binary ground truth labels.
            binary_labels = (labels != 2).long().cpu().numpy()
            
            # Collect the predictions and labels for all batches.
            all_preds.extend(preds)
            all_labels.extend(binary_labels)

    # --- Metrics Calculation ---
    print("\n--- Classification Report ---")
    target_names = ['No Change (Class 0)', 'Change (Class 1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Macro Score: {f1_macro:.4f}\n")

    # --- Confusion Matrix Plotting ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()



def unnormalize(tensor):
    """
    Reverses the normalization of a PyTorch image tensor.

    This function takes a normalized tensor and applies the inverse
    transformation to return the pixel values to the standard [0, 1] range.
    The mean and standard deviation values used for the original
    normalization are hardcoded within this function.

    Args:
        tensor (torch.Tensor): The normalized input tensor with a shape of
                               (C, H, W), where C is the number of channels.

    Returns:
        torch.Tensor: The unnormalized tensor with pixel values clamped to
                      the valid [0, 1] range.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = torch.tensor([0.378, 0.393, 0.345])
    std = torch.tensor([0.205, 0.173, 0.170])

    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()

    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)

    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)

    # Return the unnormalized tensor.
    return unnormalized_tensor


def display_dataset_stats(base_data_dir):
    """
    Analyzes and displays a statistical summary of an image dataset.

    This function iterates through the subdirectories of a specified base
    directory, where each subdirectory is considered a distinct class. It
    counts the number of image files ('.jpg') within each class and
    renders the statistics in a styled HTML table, including a total count.

    Args:
        base_data_dir (str): The file path to the root directory of the dataset.
    """
    class_counts = {}

    # Iterate through the base directory to find class folders and count images.
    try:
        for class_name in os.listdir(base_data_dir):
            class_path = os.path.join(base_data_dir, class_name)
            if os.path.isdir(class_path):
                # Count only files ending with .jpg (case-insensitive)
                image_count = len([
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith('.jpg')
                ])
                class_counts[class_name] = image_count
    except FileNotFoundError:
        print(f"Error: The directory '{base_data_dir}' was not found.")
        return

    # Verify that some classes were found before proceeding.
    if not class_counts:
        print(f"No class folders with .jpg images found in '{base_data_dir}'.")
        return

    # Restructure the collected data into a list of dictionaries for the DataFrame.
    data_list = []
    for class_name, count in class_counts.items():
        data_list.append({
            'Class Name': class_name,
            'Number of Images': count
        })

    # Create a DataFrame from the list, sort it by class name, and calculate the total.
    df = pd.DataFrame(data_list).sort_values(by='Class Name').reset_index(drop=True)
    total_images = df['Number of Images'].sum()

    # Create a 'Total' summary row and append it to the DataFrame.
    total_row = pd.DataFrame([{
        'Class Name': '<b>Total</b>',
        'Number of Images': total_images
    }])
    df_display = pd.concat([df, total_row], ignore_index=True)

    # Apply CSS styling to the DataFrame for a professional presentation.
    styler = df_display.style.hide(axis="index")
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "60%"), ("margin", "0")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
            {"selector": "th", "props": [
                ("text-align", "left"),
                ("padding", "8px"),
                ("background-color", "#4f4f4f"),
                ("color", "white")
            ]}
        ]
    )
    styler.set_properties(**{"white-space": "normal"})

    # Render the styled DataFrame.
    display(styler)


def create_datasets(dataset_path, train_transform, val_transform, train_split=0.8, seed=42):
    """
    Initializes and splits an image dataset from a directory structure.

    This function loads a dataset using ImageFolder, performs a random split
    to create training and validation subsets, and then applies separate
    data transformations to each. A nested class is used to wrap the subsets,
    ensuring transformations are applied on-the-fly during data loading.

    Args:
        dataset_path (str): The file path to the root of the image dataset.
        train_transform (callable): The transformations to apply to the training set.
        val_transform (callable): The transformations to apply to the validation set.
        train_split (float, optional): The proportion of the dataset to allocate
                                     to the training split. Defaults to 0.8.
        seed (int, optional): A seed for the random number generator to ensure
                              a reproducible split. Defaults to 42.

    Returns:
        tuple: A tuple containing the transformed training and validation datasets.
    """

    # --- Nested Class for Applying Transformations ---
    class TransformedDataset(Dataset):
        """
        A wrapper dataset that applies a given transformation to a subset.

        This allows for different transformations to be applied to datasets that
        have already been split, such as training and validation sets.

        Args:
            subset (torch.utils.data.Subset): The dataset subset to wrap.
            transform (callable): The transformation pipeline to apply to the images.
        """

        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            # Inherit class attributes from the original full dataset
            self.classes = subset.dataset.classes
            self.class_to_idx = subset.dataset.class_to_idx

        def __len__(self):
            """Returns the total number of samples in the subset."""
            return len(self.subset)

        def __getitem__(self, idx):
            """
            Retrieves an image and its label from the subset and applies the
            transformation to the image.

            Returns:
                tuple: A tuple containing the transformed image and its label.
            """
            img, label = self.subset[idx]
            return self.transform(img), label

    # Load the entire dataset from the specified path without applying any transformations yet.
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)

    # Determine the number of samples for the training and validation sets.
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Perform a random split of the dataset using a seeded generator for reproducibility.
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap the subsets with the custom TransformedDataset class to apply the appropriate transformations.
    train_dataset = TransformedDataset(subset=train_subset, transform=train_transform)
    val_dataset = TransformedDataset(subset=val_subset, transform=val_transform)

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, test_dataset, batch_size):
    """
    Initializes and configures DataLoaders for training and testing datasets.

    This function wraps dataset objects into DataLoader instances, which provide
    utilities for batching, shuffling, and iterating over the data during model
    training and evaluation.

    Args:
        train_dataset (Dataset): The dataset object for training.
        test_dataset (Dataset): The dataset object for testing or validation.
        batch_size (int): The number of samples to include in each batch.

    Returns:
        tuple: A tuple containing the configured training and testing DataLoaders.
    """

    # Create the DataLoader for the training set with shuffling enabled.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create the DataLoader for the testing set with shuffling disabled for consistent evaluation.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Return the configured training and testing DataLoaders.
    return train_loader, test_loader


def show_sample_images(dataset):
    """
    Visualizes a random sample image from each class in the dataset.

    This function creates a grid of subplots to display one randomly selected
    image for each available class. It assumes the provided dataset object has
    a `.classes` attribute and is a wrapper around a `torch.utils.data.Subset`,
    which is necessary to efficiently access the original indices and labels.

    Args:
        dataset (Dataset): The dataset to visualize. It must conform to the
                           structure described above.
    """
    # Retrieve the list of class names from the dataset object.
    classes = dataset.classes

    # Build a mapping of class indices to the dataset indices for efficient random sampling.
    class_to_indices = {i: [] for i in range(len(classes))}
    full_dataset_targets = dataset.subset.dataset.targets
    subset_indices = dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)

    # Create a grid of subplots to display the images.
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

    # Iterate over the subplots and populate them with images.
    for i, ax in enumerate(axes.flatten()):
        if i < len(classes):
            class_name = classes[i]

            # Select a random image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])

            # Retrieve the transformed image and its label from the dataset.
            image, label = dataset[random_image_idx]

            # Un-normalize the image for correct color display.
            image = unnormalize(image)

            # Convert the tensor to a NumPy array and transpose dimensions for plotting.
            npimg = image.numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(class_name)

        # Hide the axes for a cleaner look.
        ax.axis('off')

    # Adjust subplot layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()


def show_sample_images_with_class(dataset, class_names):
    """
    Displays a grid of sample images from the dataset.

    This function creates a plot showing one randomly selected image from each
    class and uses the provided `class_names` list for the titles.

    Args:
        dataset (Dataset): The dataset to visualize. Must have a '.classes'
                           attribute and support subset indexing.
        class_names (list of str): A list of formatted class names for the plot titles.
    """
    # Get the total number of classes from the dataset.
    num_classes = len(dataset.classes)

    # Validate that the number of class names matches the number of classes.
    assert len(class_names) == num_classes, "Length of class_names list must match the number of classes in the dataset."

    # Create a mapping of class index to all its image indices.
    class_to_indices = {i: [] for i in range(num_classes)}
    full_dataset_targets = dataset.subset.dataset.targets
    subset_indices = dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)

    # Dynamically calculate the grid size for the plot.
    ncols = 7
    nrows = (num_classes + ncols - 1) // ncols  # Ceiling division for rows.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 2.2))

    # Loop through each class to display one random sample.
    for i, ax in enumerate(axes.flatten()):
        # Hide axes for any empty subplots.
        if i >= num_classes:
            ax.axis('off')
            continue

        # Set the plot title using the provided class names.
        class_name = class_names[i]

        # Pick a random image from the current class.
        random_image_idx = random.choice(class_to_indices[i])

        # Retrieve the image and label from the dataset.
        image, label = dataset[random_image_idx]

        # Un-normalize the image for proper display.
        # Assumes an 'unnormalize' function is available.
        image = unnormalize(image)

        # Prepare the image tensor for plotting.
        npimg = image.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(class_name)
        ax.axis('off')

    # Apply a tight layout and show the plot.
    plt.tight_layout()
    plt.show()



def display_torch_summary(summary_object, attr_names, display_names, depth):
    """
    Formats and displays a torchinfo summary object as a styled HTML table.

    This utility function processes a summary object generated by the torchinfo
    library, extracts specified layer attributes, and formats them into a
    pandas DataFrame. It then renders the DataFrame as a clean, readable
    HTML table within a Jupyter environment. Key summary statistics, such as
    parameter counts and memory usage, are displayed below the main table.

    Args:
        summary_object: The summary object returned by `torchinfo.summary()`.
        attr_names (list of str): A list of layer attribute names to extract from
                                  the summary (e.g., 'input_size', 'num_params').
        display_names (list of str): A list of desired column headers for the
                                     output table that correspond to attr_names.
        depth (int): The maximum depth of the model hierarchy to display.
    """

    # Initialize data structures for building the DataFrame.
    layer_data = []
    display_columns = ["Layer (type:depth-idx)"] + display_names

    # Iterate through each layer in the summary list.
    for layer in summary_object.summary_list:
        # Skip layers that are deeper than the specified maximum depth.
        if layer.depth > depth:
            continue

        # Initialize a dictionary to hold the data for the current layer's row.
        row = {}

        # Construct the hierarchical layer name with appropriate indentation.
        indent = "&nbsp;" * 4 * layer.depth
        if layer.depth > 0:
            layer_name = f"{layer.class_name}: {layer.depth}-{layer.depth_index}"
        else:
            layer_name = layer.class_name

        row["Layer (type:depth-idx)"] = f"{indent}{layer_name}"

        # Populate the row dictionary with the specified layer attributes.
        for attr, name in zip(attr_names, display_names):
            # Handle parameter counts separately to show '--' for non-leaf container modules.
            if attr == "num_params":
                show_params = layer.is_leaf_layer or layer.depth == depth
                if show_params and layer.num_params > 0:
                    value = f"{layer.num_params:,}"
                else:
                    value = "--"
            else:
                # Fetch all other attributes directly from the layer object.
                value = getattr(layer, attr, "N/A")

            row[name] = value
        layer_data.append(row)

    # Create a pandas DataFrame from the collected layer data.
    df = pd.DataFrame(layer_data, columns=display_columns)

    # Apply CSS styling to the DataFrame for a clean HTML presentation.
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("border-bottom", "1px solid #ddd")
        ]},
    ]).set_properties(**{"white-space": "pre", "vertical-align": "top"})

    # Convert the styled table to an HTML string.
    table_html = styler.to_html()

    # Compile summary statistics for parameter counts into an HTML block.
    total_params = f"{summary_object.total_params:,}"
    trainable_params = f"{summary_object.trainable_params:,}"
    non_trainable_params = f"{summary_object.total_params - summary_object.trainable_params:,}"
    total_mult_adds = f"{summary_object.total_mult_adds / 1e9:.2f} G"

    params_html = f"""
    <div style="margin-top: 20px; font-family: monospace; line-height: 1.6;">
        <hr><p><b>Total params:</b> {total_params}</p>
        <p><b>Trainable params:</b> {trainable_params}</p>
        <p><b>Non-trainable params:</b> {non_trainable_params}</p>
        <p><b>Total mult-adds (G):</b> {total_mult_adds}</p><hr>
    </div>"""

    # Compile summary statistics for memory and size estimation.
    input_size_mb = summary_object.total_input / (1024 ** 2)
    fwd_bwd_pass_size_mb = summary_object.total_output_bytes / (1024 ** 2)
    params_size_mb = summary_object.total_param_bytes / (1024 ** 2)
    total_size_mb = (
                            summary_object.total_input +
                            summary_object.total_output_bytes +
                            summary_object.total_param_bytes
                    ) / (1024 ** 2)

    size_html = f"""
    <div style="font-family: monospace; line-height: 1.6;">
        <p><b>Input size (MB):</b> {input_size_mb:.2f}</p>
        <p><b>Forward/backward pass size (MB):</b> {fwd_bwd_pass_size_mb:.2f}</p>
        <p><b>Params size (MB):</b> {params_size_mb:.2f}</p>
        <p><b>Estimated Total Size (MB):</b> {total_size_mb:.2f}</p><hr>
    </div>"""

    # Combine the table and summary statistics into a single HTML object and display it.
    final_html = table_html + params_html + size_html
    display(HTML(final_html))



def training_loop_16_mixed(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Executes a complete training and validation loop for a PyTorch model.

    This function handles the full training process. It uses a 16-bit mixed
    precision training strategy to accelerate performance and reduce memory usage.
    It also tracks does metric tracking (loss, accuracy).

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda', 'mps', 'cpu') to run on.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - history (dict): A dictionary of metrics (loss, accuracy) per epoch.
            - final_cm (numpy.ndarray): The confusion matrix from the final epoch.
    """
    # Determine the device type string for AMP compatibility.
    if device == torch.device("mps"):
        device_str = "mps"
    elif device == torch.device("cuda"):
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Initialize the gradient scaler for AMP, disabled for MPS which does not support it.
    use_scaler = device_str != "mps"
    scaler = GradScaler() if use_scaler else None

    # Move the model to the specified device.
    model.to(device)

    # Initialize a dictionary to store training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Initialize torchmetrics objects for accuracy and confusion matrix calculation.
    num_classes = len(train_loader.dataset.classes)
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        # --- Training Phase ---
        # Set the model to training mode.
        model.train()

        # Initialize accumulators for the training phase.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Create a progress bar for the training loader.
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=False
        )

        for inputs, labels in train_pbar:
            # Move data to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear the gradients from the previous iteration.
            optimizer.zero_grad(set_to_none=True)

            # Use automatic mixed precision for the forward pass to improve performance.
            with autocast(device_type=device_str, dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # Perform backpropagation with the gradient scaler if enabled.
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Perform standard backpropagation if the scaler is not used (e.g., on MPS).
            else:
                loss.backward()
                optimizer.step()

            # Update and display the running loss on the progress bar.
            batch_size = inputs.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples_processed += batch_size
            display_loss = running_train_loss / train_samples_processed
            train_pbar.set_postfix(loss=f"{display_loss:.4f}")

        # Calculate and store the average training loss for the epoch.
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()

        # Reset accumulators and metrics for the validation phase.
        running_val_loss = 0.0
        val_samples_processed = 0
        val_accuracy.reset()
        val_cm.reset()

        # Create a progress bar for the validation loader.
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=False
        )
        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Move data to the specified device.
                inputs, labels = inputs.to(device), labels.to(device)

                # Use automatic mixed precision for the validation forward pass.
                with autocast(device_type=device_str, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)

                # Update validation metrics with the results from the current batch.
                preds = outputs.argmax(dim=1)
                batch_size = inputs.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples_processed += batch_size
                val_accuracy.update(preds, labels)
                val_cm.update(preds, labels)

                # Update the progress bar with current validation loss and accuracy.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                val_pbar.set_postfix(
                    acc=f"{current_acc:.2%}", loss=f"{display_loss:.4f}"
                )

        # Calculate and store the average validation loss and accuracy for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute().item()
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print a summary of the epoch's performance.
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

    # Compute the final confusion matrix after all epochs are complete.
    final_cm = val_cm.compute().cpu().numpy()

    # Return the trained model, metrics history, and confusion matrix.
    return model, history, final_cm



def training_loop_16_mixed_with_scheduler(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None, save_path=None, ):
    """Executes a training and validation loop for a PyTorch model using 16-bit mixed precision.

    This function iterates through epochs, performing a training step and a
    validation step for each. It tracks performance metrics like loss and
    accuracy, and optionally saves the model with the highest validation accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
        loss_function (callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of training epochs.
        device (torch.device): The device to perform training on (e.g., 'cuda', 'cpu').
        scheduler (torch.optim.lr_scheduler, optional): A learning rate scheduler. Defaults to None.
        save_path (str, optional): A file path to save the best model weights. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model with the weights that achieved the best validation accuracy.
            - history (dict): A dictionary containing training and validation loss and validation accuracy for each epoch.
            - best_cm (numpy.ndarray): The confusion matrix from the best validation epoch.
    """
    # Determine the device type as a string for autocast compatibility.
    if device == torch.device("mps"):
        device_str = "mps"
    elif device == torch.device("cuda"):
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Initialize a gradient scaler for mixed-precision training.
    # Gradient scaling is not supported on MPS, so it's conditionally enabled.
    use_scaler = device_str != "mps"
    scaler = GradScaler() if use_scaler else None

    # Create the directory to save the model if a path is provided.
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # Move the model to the specified computation device.
    model.to(device)

    # Initialize variables to track the best validation accuracy and confusion matrix.
    best_val_acc = 0.0
    best_cm = None

    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)
    # Initialize torchmetrics for calculating accuracy and the confusion matrix.
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(
        device
    )
    val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # Set up a single progress bar for the entire training process.
    total_steps = (len(train_loader) + len(val_loader)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    # Begin the main training loop over the specified number of epochs.
    for epoch in range(num_epochs):
        # --- Training Phase ---
        # Set the model to training mode.
        model.train()
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)

            # Use autocast for mixed-precision forward pass.
            with autocast(device_type=device_str, dtype=torch.float16):
                # Forward pass: compute predicted outputs by passing inputs to the model.
                outputs = model(inputs)
                # Calculate the loss.
                loss = loss_function(outputs, labels)

            # If using a scaler (e.g., on CUDA), scale the loss and perform a backward pass.
            if use_scaler:
                scaler.scale(loss).backward()
                # Update the model weights using the scaled gradients.
                scaler.step(optimizer)
                # Update the scaler for the next iteration.
                scaler.update()
            # If not using a scaler (e.g., on CPU/MPS), perform a standard backward pass.
            else:
                loss.backward()
                # Update the model weights.
                optimizer.step()

            # Get the size of the current batch.
            batch_size = inputs.size(0)
            # Accumulate the loss, weighted by the batch size.
            running_train_loss += loss.item() * batch_size
            # Update the count of processed samples.
            train_samples_processed += batch_size
            # Calculate and display the running average loss.
            display_loss = running_train_loss / train_samples_processed
            pbar.set_postfix(loss=f"{display_loss:.4f}")
            # Update the progress bar for the batch.
            pbar.update(1)

        # Calculate the average training loss for the epoch.
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        # Store the epoch's training loss in the history.
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        # Initialize variables to accumulate validation loss.
        running_val_loss = 0.0
        val_samples_processed = 0
        # Reset metric calculators for the new validation epoch.
        val_accuracy.reset()
        val_cm.reset()

        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over the validation data loader.
            for inputs, labels in val_loader:
                # Update the progress bar description for the validation phase.
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Validation]")
                # Move input data and labels to the designated device.
                inputs, labels = inputs.to(device), labels.to(device)

                # Use autocast for mixed-precision forward pass during validation.
                with autocast(device_type=device_str, dtype=torch.float16):
                    # Compute model outputs.
                    outputs = model(inputs)
                    # Calculate the validation loss.
                    loss = loss_function(outputs, labels)

                # Get the predicted class indices by finding the max logit.
                preds = outputs.argmax(dim=1)
                # Get the size of the current batch.
                batch_size = inputs.size(0)
                # Accumulate the validation loss.
                running_val_loss += loss.item() * batch_size
                # Update the count of processed validation samples.
                val_samples_processed += batch_size
                # Update metrics with the current batch's predictions and labels.
                val_accuracy.update(preds, labels)
                val_cm.update(preds, labels)

                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                pbar.set_postfix(
                    acc=f"{current_acc:.2%}",
                    loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate the average validation loss for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        # Compute the final validation accuracy for the epoch.
        epoch_val_acc = val_accuracy.compute().item()
        # Store the epoch's validation loss and accuracy in the history.
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # --- SCHEDULER AND SAVE BEST MODEL ---
        # Adjust the learning rate based on the scheduler's logic, if one is provided.
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_acc)
            else:
                scheduler.step()

        # Check if the current model has the best validation accuracy seen so far.
        if epoch_val_acc > best_val_acc:
            # Update the best validation accuracy.
            best_val_acc = epoch_val_acc
            # Store the confusion matrix from the best performing epoch.
            best_cm = val_cm.compute().cpu().numpy()
            # Save the model's state dictionary if a save path is specified.
            if save_path:
                torch.save(model.state_dict(), save_path)
                # Print a message indicating that a new best model has been saved.
                tqdm.write(
                    f"  -> New best model saved to '{save_path}' with Val Acc: {best_val_acc:.2%}\n"
                )

    # Close the progress bar after the training loop is complete.
    pbar.close()

    # If a save path was provided, load the weights of the best performing model.
    if save_path and os.path.exists(save_path):
        tqdm.write(f"\nBest model saved to '{save_path}' with accuracy {best_val_acc:.2%}")
        model.load_state_dict(torch.load(save_path))

    # Return the trained model, the history of metrics, and the best confusion matrix.
    return model, history, best_cm



def plot_training_logs(history1, history2, model_name1="PlainCNN Model", model_name2="SimpleResNet Model"):
    """
    Plots and compares the training history of two models.

    Args:
        history1 (dict): The training history dictionary for the first model.
        history2 (dict): The training history dictionary for the second model.
        model_name1 (str, optional): The name of the first model for labels.
                                     Defaults to "Plain CNN Model".
        model_name2 (str, optional): The name of the second model for labels.
                                     Defaults to "ResNet Model".
    """
    # Extract the final validation accuracy for each model from the history.
    final_acc1 = history1['val_accuracy'][-1]
    final_acc2 = history2['val_accuracy'][-1]

    # Display a summary of the final validation metrics.
    print("---------- Final Validation Accuracies ---------")
    print(f"{model_name1:<15}     |  Accuracy: {final_acc1:.2%}")
    print(f"{model_name2:<15}  |  Accuracy: {final_acc2:.2%}")
    print("------------------------------------------------\n")

    # Create a figure with two side-by-side subplots for loss and accuracy.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    color1, color2 = 'red', 'blue'

    # Plot the training and validation loss curves for both models.
    ax1.plot(history1['train_loss'], label=f'{model_name1} Train Loss', color=color1, linestyle='-')
    ax1.plot(history1['val_loss'], label=f'{model_name1} Val Loss', color=color1, linestyle='--')
    ax1.plot(history2['train_loss'], label=f'{model_name2} Train Loss', color=color2, linestyle='-')
    ax1.plot(history2['val_loss'], label=f'{model_name2} Val Loss', color=color2, linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot the validation accuracy curves for both models.
    ax2.plot(history1['val_accuracy'], label=f'{model_name1} Val Accuracy', color=color1)
    ax2.plot(history2['val_accuracy'], label=f'{model_name2} Val Accuracy', color=color2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Configure the x-axis ticks to be displayed at a dynamic interval.
    num_epochs = len(history1['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Define the locations for the ticks (0-indexed) and their labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom ticks and labels to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()



def visualize_predictions(model, dataloader, classes, device):
    """
    Displays a grid of predictions for one random image from each class.

    This function sets the model to evaluation mode, intelligently samples one
    random image from each unique class in the dataset, and performs inference.
    It then displays the results in a 3x5 grid, annotating each image with
    its true and predicted label.

    Args:
        model (nn.Module): The trained PyTorch model for inference.
        dataloader (DataLoader): DataLoader for the validation set.
        classes (list of str): A list of all class names for displaying labels.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to run inference on.
    """
    # Move the model to the specified device and set it to evaluation mode.
    model.to(device)
    model.eval()

    # --- Find one random image index for each class ---
    # This creates a map of {class_index: [list_of_sample_indices]}
    class_to_indices = {i: [] for i in range(len(classes))}
    # Access the underlying dataset to get all labels and indices
    full_dataset_targets = dataloader.dataset.subset.dataset.targets
    subset_indices = dataloader.dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)
    # ---

    # Create a 3x5 grid of subplots for the 15 classes.
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

    # Disable gradient calculations for inference.
    with torch.no_grad():
        # Iterate through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # Ensure we don't try to access a class that doesn't exist.
            if i >= len(classes):
                ax.axis('off')
                continue

            # Select a random image from the current class's list of indices.
            random_image_idx = random.choice(class_to_indices[i])

            # Retrieve the transformed image and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]

            # Add a batch dimension and move the image to the specified device for the model.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Perform inference to get the model's prediction.
            outputs = model(image_batch)
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()

            # Determine if the prediction was correct for title coloring.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            ax.set_title(
                f'Predicted: {classes[predicted_label]}\n(True: {classes[true_label]})',
                color=title_color
            )

            # Un-normalize the image tensor for correct color display.
            img_to_plot = unnormalize(image_tensor)

            # Convert tensor to numpy array and transpose for plotting.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            ax.axis('off')

    # Adjust layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm_np, labels):
    """
    Calculates and displays per-class accuracy, then plots a confusion matrix.

    This function first computes the accuracy for each individual class from the
    provided confusion matrix. It displays these scores with a progress bar, then
    uses scikit-learn's ConfusionMatrixDisplay to visualize the full matrix.

    Args:
        cm_np (numpy.ndarray): The confusion matrix to be plotted, where rows
                               represent true labels and columns represent
                               predicted labels.
        labels (list of str): A list of class names that correspond to the
                              matrix indices.
    """
    # --- Per-Class Accuracy Calculation ---
    # The diagonal contains the correct predictions for each class.
    correct_predictions = cm_np.diagonal()

    # The sum of each row is the total number of actual samples for that class.
    total_samples_per_class = cm_np.sum(axis=1)

    # Calculate accuracy, handling division-by-zero for classes with no samples.
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.nan_to_num(correct_predictions / total_samples_per_class)

    # Create a dictionary mapping class labels to their accuracy.
    class_accuracies = {label: acc for label, acc in zip(labels, per_class_acc)}

    # --- Display Per-Class Accuracy with a Progress Bar ---
    print("--- Per-Class Accuracy ---")
    # Use tqdm to create a progress bar while iterating through and printing results.
    for class_name, acc in tqdm(class_accuracies.items(), desc="Calculating Metrics"):
        print(f"{class_name:<20} | Accuracy: {acc:.2%}")
        time.sleep(0.05)  # Pause briefly to make the progress bar visible
    print("-" * 40 + "\n")

    # --- Confusion Matrix Plotting ---
    # Create a confusion matrix display object from the matrix and labels.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)

    # Render the confusion matrix plot with a blue color map.
    disp.plot(cmap=plt.cm.Blues)

    # Rotate the x-axis tick labels for better readability with long names.
    plt.xticks(rotation=45, ha="right")

    # Set the plot's title and axis labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Display the finalized plot.
    plt.show()



def plot_training_history(history, model_name="Custom DenseNet"):
    """Visualizes the training and validation history of a model.

    This function generates and displays two plots: one for training and
    validation loss, and another for validation accuracy. It also highlights
    the epoch where the highest validation accuracy was achieved.

    Args:
        history (dict): A dictionary containing the model's training history.
                        It must include the keys 'val_accuracy', 'val_loss',
                        and 'train_loss'.
        model_name (str, optional): The name of the model, used for plot
                                    titles and labels. Defaults to "Custom DenseNet".
    """
    # Find the index of the epoch with the highest validation accuracy.
    best_epoch_idx = np.argmax(history['val_accuracy'])
    # Get the best validation accuracy and the corresponding validation loss.
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]

    # Print a summary of the model's performance at the best epoch.
    print("---------- Best Epoch Performance ----------")
    print(f"Model: {model_name}")
    print(f"Epoch: {best_epoch_idx + 1}")
    print(f"Validation Accuracy: {best_val_acc:.2%}")
    print(f"Validation Loss:     {best_val_loss:.4f}")
    print("------------------------------------------\n")

    # Set up the figure and subplots for displaying the history.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Define colors for plot elements to ensure consistency.
    train_color = 'blue'
    val_color = 'red'
    best_epoch_color = 'green'

    # Plot training and validation loss on the first subplot.
    ax1.plot(history['train_loss'], label=f'{model_name} Train Loss', color=train_color, linestyle='-')
    ax1.plot(history['val_loss'], label=f'{model_name} Val Loss', color=val_color, linestyle='--')

    # Highlight the validation loss at the best-accuracy epoch with a marker.
    ax1.plot(best_epoch_idx, best_val_loss, marker='o', color=best_epoch_color, markersize=8, label='Loss When Best Acc Was Achieved')
    # Annotate the marker with its precise value.
    ax1.annotate(f'{best_val_loss:.4f}',
                 xy=(best_epoch_idx, best_val_loss),
                 xytext=(best_epoch_idx, best_val_loss + 0.1),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the loss subplot.
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy on the second subplot.
    ax2.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy', color=val_color)

    # Highlight the best validation accuracy with a marker.
    ax2.plot(best_epoch_idx, best_val_acc, marker='o', color=best_epoch_color, markersize=8, label='Best Accuracy Achieved')
    # Annotate the marker with its value.
    ax2.annotate(f'{best_val_acc:.2%}',
                 xy=(best_epoch_idx, best_val_acc),
                 xytext=(best_epoch_idx, best_val_acc - 0.05),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the accuracy subplot.
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Determine an appropriate interval for x-axis ticks for readability.
    num_epochs = len(history['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Generate tick locations (0-indexed) and corresponding labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom x-axis ticks to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust subplot parameters for a tight layout and display the plot.
    plt.tight_layout()
    plt.show()



def load_pretrained_densenet(num_classes, seed=None, pretrained=True, train_classifier_only=True,
                             weights_path="./pretrained_densenet_weights/densenet121-a639ec97.pth"):
    """Initializes a DenseNet-121 model and prepares it for transfer learning.

    This function loads the DenseNet-121 architecture, optionally with weights
    pre-trained on ImageNet. It then replaces the final classifier layer to
    adapt the model for a new task with a specified number of classes and can
    freeze the feature extraction layers.

    Args:
        num_classes (int): The number of output classes for the new classifier head.
        seed (int, optional): A random seed for reproducible weight initialization
                              of the new classifier. Defaults to None.
        pretrained (bool, optional): If True, loads pre-trained weights from the
                                     specified path. Defaults to True.
        train_classifier_only (bool, optional): If True, freezes the feature extractor
                                                layers so only the new classifier is trained.
                                                Defaults to True.
        weights_path (str, optional): The local file path to the pre-trained weights.

    Returns:
        torch.nn.Module: The configured DenseNet-121 model.
    """

    # Set the manual seed for PyTorch for reproducible weight initialization for the classifier head.
    if seed is not None:
        torch.manual_seed(seed)

    # Instantiate the DenseNet-121 model from torchvision without its own pretrained weights.
    model = tv_models.densenet121(weights=None)

    # Conditionally load pretrained weights from a local file if the flag is set.
    if pretrained:
        # Check if the specified weights file exists and raise an error if not.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at path: {weights_path}")

        # Load the state dictionary from the file, mapping it to the CPU to prevent device mismatches.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Apply the loaded weights to the model architecture.
        model.load_state_dict(state_dict)

    # If configured for transfer learning, freeze the parameters of the feature extractor.
    if train_classifier_only:
        # Iterate through all model parameters and disable gradient calculations.
        for param in model.parameters():
            param.requires_grad = False

    # Retrieve the number of input features for the model's original classifier.
    num_ftrs = model.classifier.in_features
    # Replace the original classifier with a new, untrained linear layer.
    # The new layer will have the correct number of output units for the new task.
    # By default, its parameters will have requires_grad=True.
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Return the modified and configured model.
    return model

