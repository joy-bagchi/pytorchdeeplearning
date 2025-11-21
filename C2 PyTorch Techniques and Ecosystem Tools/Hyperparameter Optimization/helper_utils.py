import math
from functools import wraps
from typing import Optional

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import numpy as np

# from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


# Custom colors (reusable)
PINK = "#F65B66"  # Pink
BLUE = "#237B94"  # Blue
DARK_BLUE = "#1C74EB"# Dark Blue
YELLOW = "#FAB901" # Yellow
PURPLE = "#A12F9D" # Purple


# = DLAI plots style =
def apply_dlai_style():
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

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE



color_map, PLOT_STYLE = apply_dlai_style()
mpl.rcParams.update(PLOT_STYLE)



# Custom colors (reusable)
BLUE_COLOR_TRAIN = color_map["blue"]
PINK_COLOR_TEST = color_map["pink"]



def set_seed(seed=42):
    """Sets the seed for random number generators for reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set the seed for all available GPUs
    torch.cuda.manual_seed_all(seed)
    # Ensure that cuDNN's convolutional algorithms are deterministic
    torch.backends.cudnn.deterministic = True
    # Disable the cuDNN benchmark for reproducibility
    torch.backends.cudnn.benchmark = False

    
class NestedProgressBar:
    """A handler for nested tqdm progress bars for training and evaluation loops.

    This class creates and manages an outer progress bar for epochs and an
    inner progress bar for batches. It supports automatic detection for
    terminal and notebook environments and includes a granularity feature to
    control the number of visual updates for very long processes.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        use_notebook=True,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """Initializes the nested progress bars.

        Args:
            total_epochs (int): The absolute total number of epochs.
            total_batches (int): The absolute total number of batches per epoch.
            g_epochs (int, optional): The visual granularity for the epoch bar.
                                      Defaults to total_epochs.
            g_batches (int, optional): The visual granularity for the batch bar.
                                       Defaults to total_batches.
            use_notebook (bool, optional): A flag to control notebook mode.
                                           `tqdm.auto` is used, so this is
                                           often not needed. Defaults to True.
            epoch_message_freq (int, optional): Frequency to log epoch
                                                messages. Defaults to None.
            batch_message_freq (int, optional): Frequency to log batch
                                                messages. Defaults to None.
            mode (str, optional): The operational mode, either 'train' or 'eval'.
                                  Defaults to "train".
        """
        # Store initial configuration
        self.last_batch_step = None
        self.last_epoch_step = None
        self.mode = mode
        self.use_notebook = use_notebook
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

        # Import the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl
        self.tqdm_impl = tqdm_impl

        # Store the absolute total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches
        # Determine the visual granularity, ensuring it doesn't exceed the total count
        self.g_epochs = min(g_epochs or self.total_epochs_raw, self.total_epochs_raw)
        self.g_batches = min(
            g_batches or self.total_batches_raw, self.total_batches_raw
        )
        # Set the progress bar totals to the calculated granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the progress bar widgets
        self._init_bars()

    def _init_bars(self):
        """Initializes or re-initializes the progress bar widgets."""
        # Close existing epoch bar if it exists, ignoring any errors
        if hasattr(self, "epoch_bar") and self.epoch_bar is not None:
            try:
                self.epoch_bar.close()
            except Exception:
                pass
        # Close existing batch bar if it exists, ignoring any errors
        if hasattr(self, "batch_bar") and self.batch_bar is not None:
            try:
                self.batch_bar.close()
            except Exception:
                pass

        # Initialize bars for training mode with nested layout
        if self.mode == "train":
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        # Initialize a single bar for evaluation mode
        elif self.mode == "eval":
            self.epoch_bar = None
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Initialize trackers for the last visualized update step
        self.last_epoch_step = -1
        self.last_batch_step = -1

    def update_epoch(self, epoch, postfix_dict=None, message=None):
        """Updates the epoch-level progress bar.

        Args:
            epoch (int): The current epoch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Skip epoch updates if not in training mode
        if self.mode != "train":
            return
        # Map the raw epoch count to its corresponding visual step based on granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the progress bar only when the visual step changes
        if epoch_step > self.last_epoch_step:
            update_amount = epoch_step - self.last_epoch_step
            self.epoch_bar.update(update_amount)
            self.last_epoch_step = epoch_step
        # Ensure the progress bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(self.g_epochs - self.epoch_bar.n)
            self.last_epoch_step = epoch_step

        # Set the dynamic description for the progress bar
        self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        # Update the postfix with any provided metrics or information
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the inner batch bar at the start of each new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None, message=None):
        """Updates the batch-level progress bar.

        Args:
            batch (int): The current batch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw batch count to its corresponding visual step
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the progress bar only when the visual step changes
        if batch_step > self.last_batch_step:
            update_amount = batch_step - self.last_batch_step
            self.batch_bar.update(update_amount)
            self.last_batch_step = batch_step
        # Ensure the progress bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(self.g_batches - self.batch_bar.n)
            self.last_batch_step = batch_step

        # Set the dynamic description for the progress bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Update the postfix with any provided metrics
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """Logs a message at a specified epoch frequency.

        Args:
            epoch (int): The current epoch number.
            message (str): The message to log.
        """
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """Logs a message at a specified batch frequency.

        Args:
            batch (int): The current batch number.
            message (str): The message to log.
        """
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """Closes all active progress bars and optionally prints a final message.

        Args:
            last_message (str, optional): A final message to print after closing.
                                          Defaults to None.
        """
        # Print a concluding message if one is provided
        if last_message:
            print(last_message)
        # Ensure the batch bar is closed in evaluation mode
        if self.mode == "eval":
            self.batch_bar.close()

    def reset(self):
        """Resets the progress of the bars without re-creating the widgets."""
        # Reset the epoch bar if it exists
        if self.mode == "train" and self.epoch_bar:
            self.epoch_bar.reset()
        # Reset the batch bar if it exists
        if self.batch_bar:
            self.batch_bar.reset()
        # Reset the step trackers
        self.last_epoch_step = -1
        self.last_batch_step = -1



def get_dataset_dataloaders(batch_size=64, subset_size=10_000, imbalanced=False):
    """Prepares and returns training and validation dataloaders.

    This function loads either the standard CIFAR-10 dataset or a custom
    imbalanced dataset, creates a subset, splits it into training and
    validation sets, and returns the corresponding DataLoader objects.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 64.
        subset_size (int, optional): The size of the dataset subset to use. 
                                     Defaults to 10,000.
        imbalanced (bool, optional): If True, loads a custom imbalanced dataset. 
                                     If False, loads standard CIFAR-10. Defaults to False.

    Returns:
        tuple: A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Define the image transformation pipeline
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Conditionally load the appropriate dataset
    if imbalanced:
        # Load a custom imbalanced dataset from a local folder
        full_trainset = ImageFolder(
            root="./cifar10_3class_imbalanced", transform=transform
        )
        # Use the full size of the imbalanced dataset
        subset_size = None
    else:
        # Load the standard CIFAR-10 training dataset, downloading if necessary
        full_trainset = datasets.CIFAR10(
            root="./cifar10", train=True, download=True, transform=transform
        )

    # Use the full dataset size if a subset size is not specified
    if subset_size is None:
        subset_size = len(full_trainset)

    # Calculate the sizes for an 80/20 train-validation split
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size

    # Create a random subset from the full dataset
    subset, _ = torch.utils.data.random_split(
        full_trainset, [subset_size, len(full_trainset) - subset_size]
    )
    # Split the subset into training and validation sets
    train_subset, val_subset = random_split(subset, [train_size, val_size])

    # Create a DataLoader for the training set with shuffling
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation set without shuffling
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader


def plot_learning(history, color, axes=None):
    """Plots the training loss and validation accuracy from a training history.

    This function generates two subplots: one for the training loss per epoch
    and one for the validation accuracy per epoch. It can either create a new
    figure or plot on existing axes.

    Args:
        history (dict): A dictionary containing training metrics. It must have
                        the keys 'train_loss' and 'val_acc'.
        color (str): The color to use for the plot lines.
        axes (matplotlib.axes.Axes, optional): A tuple of two Matplotlib axes
                                               objects to plot on. If None, new
                                               subplots are created. Defaults to None.
    """
    # If no axes are provided, create a new figure with two subplots
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the training loss on the first subplot
    axes[0].plot(history["train_loss"], linestyle="-", color=color)
    # Set the title for the loss subplot
    axes[0].set_title("Training Loss")
    # Set the x-axis label for the loss subplot
    axes[0].set_xlabel("Epochs")
    # Set the y-axis label for the loss subplot
    axes[0].set_ylabel("Loss")

    # Plot the validation accuracy on the second subplot
    axes[1].plot(history["val_acc"], linestyle="-", color=color)
    # Set the title for the accuracy subplot
    axes[1].set_title("Validation Accuracy")
    # Set the x-axis label for the accuracy subplot
    axes[1].set_xlabel("Epochs")
    # Set the y-axis label for the accuracy subplot
    axes[1].set_ylabel("Accuracy")

    # Adjust layout if a new figure was created to prevent overlapping titles
    if axes is None:
        plt.tight_layout()



def plot_learning_curves(colors, labels, histories):
    """Plots and compares the learning curves for multiple training histories.

    This function creates a figure with two subplots (training loss and
    validation accuracy) and overlays the learning curves from several
    training runs, each with a different color and label.

    Args:
        colors (list): A list of color strings, one for each history.
        labels (list): A list of labels for the legend, corresponding to each history.
        histories (list): A list of history dictionaries. Each dictionary should
                          contain keys like 'train_loss' and 'val_acc'.
    """
    # Create a new figure and a set of subplots for the learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Iterate through each training history and its corresponding color
    for history, color in zip(histories, colors):
        # Call a helper function to plot the curves for the current history
        plot_learning(history, color=color, axes=axes)

    # Create custom legend handles for each training run
    color_handles = [
        mlines.Line2D([], [], color=color, label=label)
        for color, label in zip(colors, labels)
    ]

    # Add the combined legend to each of the subplots
    for ax in axes:
        ax.legend(handles=color_handles)

    # Adjust the plot layout to prevent labels from overlapping
    plt.tight_layout()


def evaluate_epoch(model, val_loader, loss_fn, device):
    """Evaluates the model's performance on a validation dataset for one epoch.

    This function iterates over the validation dataloader to calculate the
    average loss and accuracy. Gradient computation is disabled for efficiency.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        val_loader (DataLoader): The DataLoader containing the validation data.
        loss_fn: The loss function used for evaluation.
        device: The device (e.g., 'cuda' or 'cpu') to perform the evaluation on.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize cumulative metrics for the epoch
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Initialize a progress bar for the evaluation loop
    pbar = NestedProgressBar(total_batches=len(val_loader), mode="eval", total_epochs=1)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over the validation data
        for batch, (inputs, labels) in enumerate(val_loader):
            # Update the batch progress bar
            pbar.update_batch(batch + 1)

            # Move input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get model predictions
            outputs = model(inputs)

            # Calculate the loss for the current batch
            loss = loss_fn(outputs, labels)
            # Accumulate the total loss
            total_loss += loss.item() * inputs.size(0)

            # Get the predicted class by finding the index of the maximum logit
            _, predicted = outputs.max(1)
            # Update the count of correctly classified samples
            total_correct += (predicted == labels).sum().item()
            # Update the total number of samples processed
            total_samples += labels.size(0)

    # Calculate the average loss for the epoch
    avg_loss = total_loss / total_samples
    # Calculate the accuracy for the epoch
    accuracy = total_correct / total_samples

    # Close the progress bar
    pbar.close()

    # Return the computed metrics
    return avg_loss, accuracy


def evaluate_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a model on a given dataset.

    This function iterates through the provided data loader, performs a
    forward pass with the model, and compares the predicted labels to the
    true labels to compute the overall accuracy. It operates in evaluation
    mode and disables gradient calculations for efficiency.

    Args:
        model: The neural network model to be evaluated.
        data_loader: The DataLoader providing the evaluation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.

    Returns:
        The accuracy of the model on the dataset as a float.
    """
    # Initialize a progress bar for the evaluation process
    pbar = NestedProgressBar(
        total_epochs=1,
        total_batches=len(data_loader),
        mode="eval",
    )

    # Set the model to evaluation mode
    model.eval()
    # Initialize counters for correct predictions and total samples
    total_correct = 0
    total_samples = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over the batches in the data loader
        for batch_idx, (inputs, labels) in enumerate(data_loader):

            # Update the progress bar for the current batch
            pbar.update_batch(batch_idx + 1)

            # Move the input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get the model's outputs
            outputs = model(inputs)

            # Get the predicted class by finding the index of the maximum logit
            _, predicted = outputs.max(1)
            # Tally the number of correct predictions in the batch
            total_correct += (predicted == labels).sum().item()
            # Tally the total number of samples in the batch
            total_samples += labels.size(0)

    # Close the progress bar and display a completion message
    pbar.close(last_message="Evaluation complete.")

    # Calculate the final accuracy
    accuracy = total_correct / total_samples
    # Return the computed accuracy
    return accuracy


def train_epoch(model, train_dataloader, optimizer, loss_fcn, device, pbar):
    """Trains the model for a single epoch.

    This function iterates over the training dataloader, performs the forward
    and backward passes, updates the model weights, and calculates the loss
    and accuracy for the entire epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The DataLoader containing the training data.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_fcn: The loss function used for training.
        device: The device (e.g., 'cuda' or 'cpu') to perform training on.
        pbar: A progress bar handler object to visualize training progress.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    # Set the model to training mode
    model.train()
    # Initialize metrics for the epoch
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training data
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Update the batch progress bar
        pbar.update_batch(batch_idx + 1)

        # Move input and label tensors to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform a forward pass to get model outputs
        outputs = model(inputs)
        # Calculate the loss
        loss = loss_fcn(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model's weights
        optimizer.step()

        # Accumulate the loss for the epoch
        running_loss += loss.item() * inputs.size(0)
        # Get the predicted class with the highest score
        _, predicted = outputs.max(1)
        # Update the total number of samples
        total += labels.size(0)
        # Update the number of correctly classified samples
        correct += predicted.eq(labels).sum().item()

    # Calculate the average loss for the epoch
    epoch_loss = running_loss / total
    # Calculate the average accuracy for the epoch
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def plot_learning_rates(history, color, label="", ax=None):
    """Plots the learning rate schedule over epochs.

    This function generates a plot showing how the learning rate changed
    during training. It can either create a new figure or plot on an
    existing Matplotlib axes object.

    Args:
        history (dict): A dictionary containing training metrics. It must have
                        the key 'lr'.
        color (str): The color to use for the plot line.
        label (str, optional): A label for the plot line in the legend.
                               Defaults to an empty string.
        ax (matplotlib.axes.Axes, optional): A Matplotlib axes object to plot on.
                                              If None, a new figure and axes are
                                              created. Defaults to None.
    """
    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the learning rate against the epochs
    ax.plot(history["lr"], label=f"{label}", color=color)
    # Set the title of the plot
    ax.set_title("Learning Rate")
    # Set the label for the x-axis
    ax.set_xlabel("Epochs")
    # Set the label for the y-axis
    ax.set_ylabel("Learning Rate")
    # Display the legend on the plot
    ax.legend()


def plot_learning_rates_curves(training_curves_new, colors, labels):
    """Plots and compares multiple learning rate schedules on a single figure.

    This function creates a figure and overlays the learning rate curves from
    several training runs, each with a different color and label.

    Args:
        training_curves_new (list): A list of training history dictionaries.
        colors (list): A list of color strings, one for each history.
        labels (list): A list of labels for the legend, corresponding to each history.
    """
    # Create a new figure and a single axes object for the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # Iterate through the histories, colors, and labels, skipping the first element
    for history, color, label in zip(training_curves_new[1:], colors[1:], labels[1:]):
        # Call a helper function to plot the learning rate curve for the current history
        plot_learning_rates(history, color, label=label, ax=ax)
    # Adjust the plot layout to prevent labels from overlapping
    plt.tight_layout()


def train_and_evaluate_model(model, optimizer, loss_fcn, train_dataloader, val_dataloader, device, n_epochs, p_bar=None):
    """Coordinates the model training and validation process over multiple epochs.

    This function manages the main training loop. For each epoch, it calls
    helper functions to train the model and evaluate it on the validation set.
    It records the loss and accuracy for both and visualizes the progress.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_fcn: The loss function used for training.
        train_dataloader (DataLoader): The DataLoader for the training data.
        val_dataloader (DataLoader): The DataLoader for the validation data.
        device: The device (e.g., 'cuda' or 'cpu') to perform training on.
        n_epochs (int): The total number of epochs to train for.
        p_bar (optional): An existing progress bar handler. If None, a new one
                          is created. Defaults to None.

    Returns:
        dict: A dictionary containing the history of training and validation
              loss and accuracy for each epoch.
    """
    # Initialize a dictionary to store training and validation metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # If no progress bar is provided, create a new one
    if p_bar is None:
        p_bar = NestedProgressBar(
            total_epochs=n_epochs,
            total_batches=len(train_dataloader),
            epoch_message_freq=5,
            mode="train",
        )
    # Otherwise, update the existing progress bar with the correct batch count
    else:
        total_batches = len(train_dataloader)
        p_bar.total_batches_raw = total_batches

    # Loop through the specified number of training epochs
    for epoch in range(n_epochs):
        # Update the outer progress bar for the current epoch
        p_bar.update_epoch(epoch + 1)

        # Call a helper function to train the model for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fcn=loss_fcn,
            device=device,
            pbar=p_bar,
        )

        # Call a helper function to evaluate the model on the validation set
        val_loss, val_acc = evaluate_epoch(model, val_dataloader, loss_fcn, device)

        # Log the training metrics for the current epoch at a set frequency
        p_bar.maybe_log_epoch(
            epoch=epoch + 1,
            message=f"At epoch {epoch + 1}: Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}",
        )

        # Log the validation metrics for the current epoch at a set frequency
        p_bar.maybe_log_epoch(
            epoch=epoch + 1,
            message=f"At epoch {epoch + 1}: Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}",
        )

        # Record the metrics for the current epoch in the history dictionary
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    # Close the progress bar and print a final completion message
    p_bar.close("Training complete\n")

    # Return the collected training history
    return history


def train_model(model, optimizer, loss_fcn, train_dataloader, device, n_epochs):
    """
    Orchestrates the training of a model for a specified number of epochs.

    This function manages the overall training loop. It initializes a progress
    bar to visualize progress and, for each epoch, calls a helper function
    to perform the actual training steps. It also handles periodic logging
    of the training loss.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimization algorithm (e.g., Adam, SGD).
        loss_fcn: The loss function used for training.
        train_dataloader: The DataLoader providing the training data.
        device: The device ('cpu' or 'cuda') to perform training on.
        n_epochs: The total number of epochs to train the model.
    """
    # Initialize a progress bar to visualize the training process
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        epoch_message_freq=5,
        mode="train",
    )

    # Begin the main training loop over the specified number of epochs
    for epoch in range(n_epochs):
        # Update the epoch-level progress bar
        pbar.update_epoch(epoch + 1)

        # Call the helper function to train the model for one epoch
        train_loss, _ = train_epoch(
            model, train_dataloader, optimizer, loss_fcn, device, pbar
        )

        # Log the training loss for the current epoch at specified intervals
        pbar.maybe_log_epoch(
            epoch + 1,
            message=f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}",
        )

    # Close the progress bar and print a final message upon completion
    pbar.close("Training complete!\n")


def get_p_bar(n_epochs):
    """Initializes and returns a pre-configured progress bar handler.

    Args:
        n_epochs (int): The total number of epochs the progress bar should track.

    Returns:
        NestedProgressBar: An instance of the NestedProgressBar class.
    """
    # Create an instance of the progress bar with a default configuration
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=10,
        epoch_message_freq=5,
        mode="train",
    )
    # Return the initialized progress bar object
    return pbar


def plot_metrics_vs_learning_rate(df_metrics):
    """Generates a scatter plot of performance metrics versus learning rates.

    Args:
        df_metrics (pd.DataFrame): A pandas DataFrame containing the results. 
                                   It must have a 'learning_rate' column and 
                                   columns for each metric to be plotted.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))
    
    # Iterate through the metrics and plot each one against the learning rate
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        # Create a scatter plot for the current metric
        plt.scatter(
            df_metrics["learning_rate"],
            df_metrics[metric],
            marker="o",
            label=metric,
        )

    # Set the x-axis to a logarithmic scale
    plt.xscale("log")
    # Set the label for the x-axis
    plt.xlabel("Learning Rate (log scale)")
    # Set the label for the y-axis
    plt.ylabel("Metric Value")
    # Set the title of the plot
    plt.title("Metrics vs Learning Rate")
    # Display the legend to identify each metric's points
    plt.legend()
    # Enable the grid for better readability
    plt.grid(True)


def plot_metrics_vs_batch_size(df_metrics):
    """Generates a scatter plot of performance metrics versus batch sizes.

    Args:
        df_metrics (pd.DataFrame): A pandas DataFrame containing the results.
                                   It must have a 'batch_size' column and
                                   columns for each metric to be plotted.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))
    
    # Iterate through the metrics and plot each one against the batch size
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        # Create a scatter plot for the current metric
        plt.scatter(
            df_metrics["batch_size"],
            df_metrics[metric],
            marker="o",
            label=metric,
        )

    # Set the label for the x-axis
    plt.xlabel("Batch Size")
    # Set the label for the y-axis
    plt.ylabel("Metric Value")
    # Set the title of the plot
    plt.title("Metrics vs Batch Size")
    # Display the legend to identify each metric's points
    plt.legend()
    # Enable the grid for better readability
    plt.grid(True)
    

def plot_results(learning_rates, accuracies):
    """Generates and displays a scatter plot of validation accuracy versus learning rate.

    Args:
        learning_rates (list): A list of learning rates to be plotted on the x-axis.
        accuracies (list): A list of corresponding validation accuracies for the y-axis.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(8, 6))
    # Create a scatter plot of the results
    plt.scatter(learning_rates, accuracies, marker="o", color=BLUE_COLOR_TRAIN)
    # Set the x-axis to a logarithmic scale for better visualization
    plt.xscale("log")
    # Set the label for the x-axis
    plt.xlabel("Learning Rate (log scale)")
    # Set the label for the y-axis
    plt.ylabel("Validation Accuracy")
    # Set the title of the plot
    plt.title("Learning Rate vs Validation Accuracy")
    # Enable the grid for better readability
    plt.grid(True)
    # Display the final plot
    plt.show()
    
def load_resnet_table():
    """Loads ResNet model performance data from a CSV file.

    This function reads a specific CSV file named 'resnet_results.csv'
    and returns its contents as a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the ResNet results.
    """
    # Read the CSV file into a pandas DataFrame, using the first column as the index.
    resnet_results = pd.read_csv("resnet_results.csv", index_col=0)
    # Return the loaded DataFrame.
    return resnet_results


class PbarEpoch:
    """
    Manages a tqdm progress bar for a single training epoch.

    This class encapsulates the creation, updating, and closing of a progress bar,
    making it easier to monitor training progress on a per-epoch basis.
    """

    def __init__(self, train_loader, steps, epoch):
        """
        Initializes the progress bar for the epoch.

        Args:
            train_loader: The DataLoader for the training dataset.
            steps: The total number of steps to display on the progress bar.
            epoch: The current epoch number for the description.
        """
        # Calculate how many batches correspond to a single step in the progress bar.
        self.batches_per_step = len(train_loader) // steps
        # Initialize the tqdm progress bar with its total steps and description.
        self.pbar = tqdm(total=steps, desc=f"Train Epoch {epoch}")

    def update(self, batch_idx, loss):
        """
        Updates the progress bar with the latest batch information.

        Args:
            batch_idx: The index of the current batch being processed.
            loss: The loss value from the current batch.
        """
        # Advance the progress bar by one step.
        self.pbar.update(1)
        # Set the postfix text to display the current loss and batch number.
        self.pbar.set_postfix(current_loss=loss.item(), batch=batch_idx + 1)
        # Print the current loss and batch number to the console.
        print(f"Current Loss: {loss.item():.4f}, Batch: {batch_idx + 1}")

    def close(self):
        """
        Closes the progress bar at the end of the epoch.
        """
        # Finalize and close the tqdm progress bar instance.
        self.pbar.close()


def plot_efficiency_analysis(results_df):
    """
    Generates and displays a scatter plot to analyze model efficiency.

    This function visualizes the relationship between model accuracy, inference time,
    and size. Each model is represented as a point, where the x-axis is the
    inference time, the y-axis is the accuracy, and the size of the point
    corresponds to the model's file size. Annotations are placed dynamically
    based on marker size.

    Args:
        results_df: A pandas DataFrame containing the performance metrics for
                    each model. It must include the columns 'inference_time_ms',
                    'accuracy', and 'model_size_mb'.
    """
    # Set up the plot figure and get its axes for more control over elements.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a base font size for labels.
    label_fontsize = 15 + (10 / 5)

    # Define a list of colors for the plot points.
    colors = [PINK, BLUE, DARK_BLUE, YELLOW, PURPLE]
    # Initialize a counter to cycle through the color list.
    i = 0

    # Iterate over each model's data in the DataFrame.
    for name, row in results_df.iterrows():
        # Define the marker area based on the model's size.
        marker_area = row["model_size_mb"] * 10.15

        # Create a scatter plot point for the current model.
        ax.scatter(
            row["inference_time_ms"],
            row["accuracy"],
            s=marker_area,
            label=str(round(row["model_size_mb"], 1)) + " MB - " + name,
            c=colors[i],
        )

        # Calculate the marker's radius in points to position the text label.
        marker_radius_pts = np.sqrt(marker_area)
        # Calculate the vertical offset to place the text just below the marker.
        vertical_offset_pts = - (marker_radius_pts * 1.02)

        # Add a text annotation showing the exact accuracy value.
        ax.annotate(
            f"{row['accuracy']:.2f}",
            # Set the annotation's anchor point to the center of the marker.
            xy=(row["inference_time_ms"], row["accuracy"]),
            # Set the text's position using an offset in points.
            xytext=(0, vertical_offset_pts),
            # Specify that the offset is measured in points.
            textcoords="offset points",
            fontsize=label_fontsize,
            ha="center",
            # Vertically align the top of the text to the offset point.
            va="top",
        )
        # Move to the next color for the next model.
        i += 1

    # Set the title of the plot.
    ax.set_title("Model Efficiency: Inference Time vs Accuracy")
    # Set the label for the x-axis.
    ax.set_xlabel("Inference Time (ms)")
    # Set the label for the y-axis.
    ax.set_ylabel("Accuracy")

    # Dynamically calculate the x-axis limit with some padding.
    xmax = results_df["inference_time_ms"].max() * 1.15
    # Apply the calculated x-axis limit.
    ax.set_xlim(0, xmax)
    # Set the y-axis limit with a small padding at the top.
    ax.set_ylim(0, 1.05)

    # Explicitly define the tick marks on the y-axis for consistency.
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    # Create the plot's legend with the defined font size.
    legend = ax.legend(fontsize=label_fontsize)
    # Ensure all markers in the legend have a consistent size for clarity.
    for handle in legend.legend_handles:
        handle._sizes = [80]

    # Display a grid on the plot for better readability.
    ax.grid(True)
    # Adjust the plot to ensure all elements fit without overlapping.
    fig.tight_layout()
    # Show the final generated plot.
    plt.show()


