import os
import urllib.request
import tarfile
import torch
import re
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import zipfile
from pathlib import Path
from typing import List, Tuple
import random




def get_imdb_data(data_dir='./imdb_data', max_train_samples=2000, max_test_samples=500):
    """
    Downloads and loads the IMDB movie review dataset.
    
    Parameters:
    -----------
    data_dir : str, optional
        Directory where the dataset will be stored (default: './imdb_data')
    max_train_samples : int, optional
        Maximum number of training samples to load (default: 2000)
    max_test_samples : int, optional
        Maximum number of test samples to load (default: 500)
    
    Returns:
    --------
    tuple
        (train_reviews, train_labels, test_reviews, test_labels)
        - train_reviews: list of training review texts
        - train_labels: list of training labels (1 for positive, 0 for negative)
        - test_reviews: list of test review texts
        - test_labels: list of test labels (1 for positive, 0 for negative)
    """
    # Create data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if dataset already exists
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        # Download the dataset
        print("Downloading IMDB dataset...")
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
        
        try:
            # Download the file
            urllib.request.urlretrieve(url, tar_path)
            print("Download complete!")
            
            # Extract the tar file
            print("Extracting files...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(data_dir)
            
            # Move files to appropriate directories
            source_dir = os.path.join(data_dir, 'aclImdb')
            for split in ['train', 'test']:
                split_source = os.path.join(source_dir, split)
                split_dest = os.path.join(data_dir, split)
                if os.path.exists(split_source) and not os.path.exists(split_dest):
                    os.rename(split_source, split_dest)
            
            # Clean up
            if os.path.exists(tar_path):
                os.remove(tar_path)
            
            # Remove the extracted aclImdb folder if it still exists
            if os.path.exists(source_dir):
                import shutil
                shutil.rmtree(source_dir)
            
            print("IMDB dataset ready!")
            
        except Exception as e:
            print(f"Error downloading or extracting dataset: {e}")
            # Clean up partial downloads
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise
    else:
        print("IMDB dataset already downloaded")
    
    # Load training reviews
    print("\nLoading training data...")
    train_reviews = []
    train_labels = []
    
    # Load positive training reviews
    train_pos_dir = os.path.join(data_dir, 'train', 'pos')
    train_pos_files = os.listdir(train_pos_dir)[:max_train_samples//2]
    print(f"Loading {len(train_pos_files)} positive training reviews...")
    for filename in train_pos_files:
        with open(os.path.join(train_pos_dir, filename), 'r', encoding='utf-8') as f:
            train_reviews.append(f.read())
            train_labels.append(1)  # Positive = 1
    
    # Load negative training reviews
    train_neg_dir = os.path.join(data_dir, 'train', 'neg')
    train_neg_files = os.listdir(train_neg_dir)[:max_train_samples//2]
    print(f"Loading {len(train_neg_files)} negative training reviews...")
    for filename in train_neg_files:
        with open(os.path.join(train_neg_dir, filename), 'r', encoding='utf-8') as f:
            train_reviews.append(f.read())
            train_labels.append(0)  # Negative = 0
    
    print(f"Total training reviews loaded: {len(train_reviews)}")
    
    # Load test reviews
    print("\nLoading test data...")
    test_reviews = []
    test_labels = []
    
    # Load positive test reviews
    test_pos_dir = os.path.join(data_dir, 'test', 'pos')
    test_pos_files = os.listdir(test_pos_dir)[:max_test_samples//2]
    print(f"Loading {len(test_pos_files)} positive test reviews...")
    for filename in test_pos_files:
        with open(os.path.join(test_pos_dir, filename), 'r', encoding='utf-8') as f:
            test_reviews.append(f.read())
            test_labels.append(1)  # Positive = 1
    
    # Load negative test reviews
    test_neg_dir = os.path.join(data_dir, 'test', 'neg')
    test_neg_files = os.listdir(test_neg_dir)[:max_test_samples//2]
    print(f"Loading {len(test_neg_files)} negative test reviews...")
    for filename in test_neg_files:
        with open(os.path.join(test_neg_dir, filename), 'r', encoding='utf-8') as f:
            test_reviews.append(f.read())
            test_labels.append(0)  # Negative = 0
    
    print(f"Total test reviews loaded: {len(test_reviews)}")
    
    return train_reviews, train_labels, test_reviews, test_labels


def print_data_statistics(train_reviews, train_labels, test_reviews, test_labels, sample_size=100):
    """
    Prints comprehensive statistics about the IMDB dataset.
    
    Parameters:
    -----------
    train_reviews : list
        List of training review texts
    train_labels : list
        List of training labels (1 for positive, 0 for negative)
    test_reviews : list
        List of test review texts
    test_labels : list
        List of test labels (1 for positive, 0 for negative)
    sample_size : int, optional
        Number of reviews to use for length statistics (default: 100)
    """
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training set:")
    print(f"  Total reviews: {len(train_reviews)}")
    print(f"  Positive reviews: {sum(train_labels)}")
    print(f"  Negative reviews: {len(train_labels) - sum(train_labels)}")
    print(f"\nTest set:")
    print(f"  Total reviews: {len(test_reviews)}")
    print(f"  Positive reviews: {sum(test_labels)}")
    print(f"  Negative reviews: {len(test_labels) - sum(test_labels)}")
    
    # Show sample reviews
    print("\n=== Sample Reviews ===")
    
    # Find positive review example
    try:
        positive_idx = train_labels.index(1)  # Find first positive review
        print("\n--- Positive Review Example ---")
        print(f"Label: Positive")
        print(f"Text (first 400 chars): {train_reviews[positive_idx][:400]}...")
    except ValueError:
        print("\nNo positive reviews found in training set")
    
    # Find negative review example
    try:
        negative_idx = train_labels.index(0)  # Find first negative review
        print("\n--- Negative Review Example ---")
        print(f"Label: Negative")
        print(f"Text (first 400 chars): {train_reviews[negative_idx][:400]}...")
    except ValueError:
        print("\nNo negative reviews found in training set")
    
    # Check review lengths
    sample_size = min(sample_size, len(train_reviews))  # Ensure we don't exceed available reviews
    review_lengths = [len(review.split()) for review in train_reviews[:sample_size]]
    
    print(f"\n=== Review Length Statistics (first {sample_size} reviews) ===")
    print(f"Average length: {np.mean(review_lengths):.0f} words")
    print(f"Min length: {min(review_lengths)} words")
    print(f"Max length: {max(review_lengths)} words")
    print(f"Median length: {np.median(review_lengths):.0f} words")
    print(f"Std deviation: {np.std(review_lengths):.0f} words")


def print_summary(model, vocab_size=None, embedding_dim=128, num_heads=4):
    """
    Prints a comprehensive summary of the model including device information and architecture details.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to summarize
    vocab_size : int, optional
        Size of the vocabulary (if None, tries to infer from model)
    embedding_dim : int, optional
        Dimension of embeddings (default: 128)
    num_heads : int, optional
        Number of attention heads (default: 4)
    """
    # Check device availability and move model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device available: {device}")
    
    # Move model to device
    model = model.to(device)
    print(f"\nModel moved to {device}")
    print("Ready for training!")
    
    # Try to infer vocab_size if not provided
    if vocab_size is None:
        try:
            # Attempt to get vocab_size from embedding layer
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding):
                    vocab_size = module.num_embeddings
                    break
        except:
            vocab_size = "Unknown"
    
    # Get model class name
    model_name = model.__class__.__name__
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Final model summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Vocabulary size: {vocab_size if vocab_size else 'Unknown'}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of attention heads: {num_heads}")
    print(f"Total parameters: {total_params:,}")
    print("\nThe model is now ready to be trained!")


def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=5, device=None):
    """
    Training function for sentiment analysis models
    Args:
        model: The model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to run on (if None, automatically detects)
    Returns:
        dict: Training history with train and test accuracies, and the best model
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Store history
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    # Best model tracking
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Training on {device}")
    print("="*50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        
        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            # Move to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * labels.size(0)
            
            # Update progress bar
            train_acc = 100 * train_correct / train_total
            avg_loss = train_loss / train_total
            train_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{train_acc:.1f}%'})
        
        # Testing phase
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        # Progress bar for testing
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        with torch.no_grad():
            for inputs, labels in test_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_loss += loss.item() * labels.size(0)
                
                # Update progress bar
                test_acc = 100 * test_correct / test_total
                test_bar.set_postfix({'acc': f'{test_acc:.1f}%'})
        
        # Calculate accuracies and losses
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        avg_train_loss = train_loss / train_total
        avg_test_loss = test_loss / test_total
        
        # Store history
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        
        # Check if this is the best model so far
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_epoch = epoch + 1
            # Deep copy the model state
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'test_loss': avg_test_loss,
                'train_loss': avg_train_loss
            }
            print(f'  🎯 New best model! Test Acc: {test_accuracy:.2f}%')
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%')
        print(f'  Test  - Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%')
        print('-'*50)
    
    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print("\n" + "="*50)
        print(f"Training completed! Best model restored from epoch {best_epoch}")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    else:
        print("\nTraining completed!")
        print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    
    # Add best model info to history
    history['best_epoch'] = best_epoch
    history['best_test_acc'] = best_test_acc
    history['best_model_state'] = best_model_state
    
    return history


def plot_training_history(history, initial_accuracy=50.0):
    """
    Plots training history with accuracy curves.
    
    Args:
        history: Dictionary containing 'train_acc' and 'test_acc'
        initial_accuracy: Initial accuracy before training (default: 50.0 for random baseline)
    """
    # Plot accuracy over epochs
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['test_acc'], 'r-', label='Test Accuracy')
    plt.axhline(y=initial_accuracy, color='gray', linestyle='--', label='Initial Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print accuracy improvement summary
    print("Accuracy Summary:")
    print(f"  Started at: {initial_accuracy:.2f}% (untrained)")
    print(f"  Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"  Total improvement: +{history['test_acc'][-1] - initial_accuracy:.2f}%")


def compare_models(history1, history2,
                   model1_name="Custom Model",
                   model2_name="PyTorch Model",
                   model1_desc="Custom implementation",
                   model2_desc="Built-in implementation",
                   figsize=(14, 5)):
    """
    Compares training histories of two different models side by side.
    Args:
        history1: Training history of first model (dict with 'train_acc' and 'test_acc')
        history2: Training history of second model (dict with 'train_acc' and 'test_acc')
        model1_name: Display name for first model
        model2_name: Display name for second model
        model1_desc: Description of first model architecture
        model2_desc: Description of second model architecture
        figsize: Figure size for the comparison plot
    """
    # Determine number of epochs from histories
    epochs1 = len(history1['train_acc'])
    epochs2 = len(history2['train_acc'])
    epochs = min(epochs1, epochs2)  # Use minimum to ensure fair comparison
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epoch_range = range(1, epochs + 1)
    
    # Plot first model
    ax1.plot(epoch_range, history1['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax1.plot(epoch_range, history1['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(model1_name)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([40, 100])
    if epochs <= 10:
        ax1.set_xticks(range(1, epochs + 1))
    
    # Mark best accuracy for model 1
    # Check if best_epoch is in history (from updated train function)
    if 'best_epoch' in history1 and history1['best_epoch'] <= epochs:
        best_epoch1 = history1['best_epoch']
        best_acc1 = history1['best_test_acc']
    else:
        best_acc1 = max(history1['test_acc'][:epochs])
        best_epoch1 = history1['test_acc'][:epochs].index(best_acc1) + 1
    
    ax1.plot(best_epoch1, best_acc1, 'r*', markersize=12)
    ax1.annotate(f'Best: {best_acc1:.1f}%',
                xy=(best_epoch1, best_acc1),
                xytext=(best_epoch1, best_acc1-5),
                fontsize=9, ha='center')
    
    # Add indicator for selected model epoch
    ax1.axvline(x=best_epoch1, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    # Plot second model
    ax2.plot(epoch_range, history2['train_acc'][:epochs], 'b-', label='Train', linewidth=2)
    ax2.plot(epoch_range, history2['test_acc'][:epochs], 'r-', label='Test', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(model2_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 100])
    if epochs <= 10:
        ax2.set_xticks(range(1, epochs + 1))
    
    # Mark best accuracy for model 2
    if 'best_epoch' in history2 and history2['best_epoch'] <= epochs:
        best_epoch2 = history2['best_epoch']
        best_acc2 = history2['best_test_acc']
    else:
        best_acc2 = max(history2['test_acc'][:epochs])
        best_epoch2 = history2['test_acc'][:epochs].index(best_acc2) + 1
    
    ax2.plot(best_epoch2, best_acc2, 'r*', markersize=12)
    ax2.annotate(f'Best: {best_acc2:.1f}%',
                xy=(best_epoch2, best_acc2),
                xytext=(best_epoch2, best_acc2-5),
                fontsize=9, ha='center')
    
    # Add indicator for selected model epoch
    ax2.axvline(x=best_epoch2, color='green', linestyle='--', alpha=0.3, label='Selected Model')
    
    plt.tight_layout()
    plt.show()
    
    # Get metrics for the SELECTED models (best epoch, not final epoch)
    selected_test_acc1 = history1['test_acc'][best_epoch1-1]
    selected_train_acc1 = history1['train_acc'][best_epoch1-1]
    selected_test_acc2 = history2['test_acc'][best_epoch2-1]
    selected_train_acc2 = history2['train_acc'][best_epoch2-1]
    
    # Final epoch metrics (for comparison)
    final_test_acc1 = history1['test_acc'][epochs-1]
    final_train_acc1 = history1['train_acc'][epochs-1]
    final_test_acc2 = history2['test_acc'][epochs-1]
    final_train_acc2 = history2['train_acc'][epochs-1]
    
    difference = selected_test_acc2 - selected_test_acc1
    
    # Calculate convergence speed
    convergence_threshold = 0.9 * max(best_acc1, best_acc2)
    conv_epoch1 = next((i for i, acc in enumerate(history1['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    conv_epoch2 = next((i for i, acc in enumerate(history2['test_acc'][:epochs])
                        if acc >= convergence_threshold), epochs) + 1
    
    # Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Training Duration: {epochs} epochs")
    
    print(f"\n{model1_name}:")
    print(f"  Architecture: {model1_desc}")
    print(f"  Selected Model from Epoch: {best_epoch1}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc1:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc1:.2f}%")
    if best_epoch1 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc1:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch1}")
    
    print(f"\n{model2_name}:")
    print(f"  Architecture: {model2_desc}")
    print(f"  Selected Model from Epoch: {best_epoch2}")
    print(f"  Selected Model Test Accuracy: {selected_test_acc2:.2f}%")
    print(f"  Selected Model Train Accuracy: {selected_train_acc2:.2f}%")
    if best_epoch2 != epochs:
        print(f"  (Final epoch accuracy was: {final_test_acc2:.2f}%)")
    print(f"  Convergence: Epoch {conv_epoch2}")
    
    print(f"\n" + "-"*60)
    print("PERFORMANCE ANALYSIS")
    print("-"*60)
    print(f"Difference in selected model accuracy: {difference:+.2f}%")
    
    if conv_epoch1 < conv_epoch2:
        print(f"Faster convergence: {model1_name} (by {conv_epoch2-conv_epoch1} epochs)")
    elif conv_epoch2 < conv_epoch1:
        print(f"Faster convergence: {model2_name} (by {conv_epoch1-conv_epoch2} epochs)")
    else:
        print(f"Both models converged at the same speed")
    
    # Interpretation
    print(f"\n" + "-"*60)
    print("INTERPRETATION")
    print("-"*60)
    
    if abs(difference) < 2:
        print("✓ Both models perform similarly! The implementations are comparable.")
    elif difference > 0:
        print(f"✓ {model2_name} performs better by {difference:.2f}%")
        if difference > 5:
            print("  This is a significant improvement.")
    else:
        print(f"✓ {model1_name} performs better by {-difference:.2f}%")
        if difference < -5:
            print("  This is a significant improvement.")
    
    # Check for overfitting using SELECTED model metrics
    overfit1 = selected_train_acc1 - selected_test_acc1
    overfit2 = selected_train_acc2 - selected_test_acc2
    
    print("\n⚠️  Overfitting Analysis (for selected models):")
    print(f"  {model1_name}: {overfit1:.1f}% gap (train-test) at epoch {best_epoch1}")
    print(f"  {model2_name}: {overfit2:.1f}% gap (train-test) at epoch {best_epoch2}")
    
    if overfit1 > 10 or overfit2 > 10:
        if overfit1 > overfit2:
            print(f"  → {model1_name} shows more overfitting")
        else:
            print(f"  → {model2_name} shows more overfitting")
    else:
        print("  → Both models show acceptable generalization (gap < 10%)")
    
    # Early stopping benefit analysis
    if best_epoch1 < epochs or best_epoch2 < epochs:
        print("\n📊 Early Stopping Analysis:")
        if best_epoch1 < epochs:
            prevented_overfit1 = (final_train_acc1 - final_test_acc1) - overfit1
            print(f"  {model1_name}: Early stopping at epoch {best_epoch1} prevented {prevented_overfit1:.1f}% additional overfitting")
        if best_epoch2 < epochs:
            prevented_overfit2 = (final_train_acc2 - final_test_acc2) - overfit2
            print(f"  {model2_name}: Early stopping at epoch {best_epoch2} prevented {prevented_overfit2:.1f}% additional overfitting")
    
    print("="*60)



def get_shakespeare_data(filename="shakespeare.txt", data_dir="./"):
    """
    Downloads and loads the Shakespeare dataset.

    Parameters:
    -----------
    filename : str, optional
        Name for the saved file (default: 'shakespeare.txt')
    data_dir : str, optional
        Directory to save the file (default: current directory)

    Returns:
    --------
    str
        The complete Shakespeare text
    """
    # Create full file path
    filepath = os.path.join(data_dir, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"Shakespeare dataset already exists at {filepath}")
    else:
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the file
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Download complete! Saved to {filepath}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise

    # Read and return the text
    print("Loading Shakespeare text...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text loaded successfully! ({len(text):,} characters)")
    print(f"Preview: {text[:300]}...")  # Show preview

    return text


def train_model_decoder(model, vocab_size, loader, loss_fn, optimizer, epochs=10, device='cpu'):
    """Train the decoder model on Shakespeare text"""
    model.to(device)  # Ensure model is on the right device

    for epoch in range(epochs):
        model.train()  # Set to training mode
        epoch_losses = []  # Track losses for averaging

        with tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for xb, yb in pbar:
                # Move batch to device
                xb, yb = xb.to(device), yb.to(device)

                # Clear gradients
                optimizer.zero_grad()

                # Forward pass through decoder
                logits = model(xb)  # Shape: [batch, seq_len, vocab_size]

                # Reshape for loss calculation
                loss = loss_fn(
                    logits.reshape(-1, vocab_size),  # [batch*seq_len, vocab_size]
                    yb.reshape(-1)  # [batch*seq_len]
                )

                # Backward pass
                loss.backward()

                # Gradient clipping (ADD THIS!)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update parameters
                optimizer.step()

                # Track loss
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())

        # Calculate average loss - simple mean
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1:2d}: avg loss = {avg_loss:.4f}")


# helper_utils.py
class ShakespeareTokenizer:
    """Tokenizer for Shakespeare text that handles contractions and punctuation"""

    def __call__(self, text):
        # Replace line breaks with a special token
        text = text.replace('\n', ' <nl> ')
        # Tokenize words, contractions, <nl>, and punctuation
        return re.findall(r"\w+(?:'\w+)?|<nl>|[^\w\s]", text)


def build_vocabulary(text, vocab_size=5000, tokenizer=None):
    """
    Build vocabulary from Shakespeare text using top-k most frequent tokens.

    Args:
        text: Raw Shakespeare text
        vocab_size: Maximum vocabulary size (default: 5000)
        tokenizer: Tokenizer instance (if None, creates ShakespeareTokenizer)

    Returns:
        vocab: List of vocabulary words
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
        tokenizer: The tokenizer used
    """
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()

    # Count all tokens
    tokens = tokenizer(text)
    token_counts = Counter(tokens)

    # Always include special tokens
    special_tokens = ['<pad>', '<unk>', '<nl>']

    # Get top-k most frequent tokens (excluding space for special tokens)
    most_common = token_counts.most_common(vocab_size - len(special_tokens))

    # Build vocab - special tokens first, then top frequent tokens
    vocab = special_tokens.copy()
    for token, count in most_common:
        if token not in special_tokens:
            vocab.append(token)

    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Calculate coverage statistics
    total_token_occurrences = sum(token_counts.values())
    covered_token_occurrences = sum(token_counts[token] for token in vocab if token in token_counts)
    coverage = covered_token_occurrences / total_token_occurrences

    # Calculate unknown rate
    unknown_count = sum(count for token, count in token_counts.items() if token not in word2idx)
    unknown_rate = unknown_count / total_token_occurrences

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tokens in text: {len(token_counts)}")
    print(f"Coverage: {coverage:.1%} of token occurrences")
    print(f"Unknown token rate: {unknown_rate:.1%}")
    print(f"Most common tokens: {vocab[3:13]}")
    print(f"Least common in vocab: {vocab[-10:]}")

    return vocab, word2idx, idx2word, tokenizer


def create_sequences(text, word2idx, idx2word, tokenizer=None, seq_len=150):
    """
    Create training sequences from text.

    Args:
        text: Raw Shakespeare text
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        tokenizer: Tokenizer instance (if None, creates ShakespeareTokenizer)
        seq_len: Length of each sequence (default: 150)

    Returns:
        inputs: List of input sequences
        targets: List of target sequences (shifted by 1)
    """
    if tokenizer is None:
        tokenizer = ShakespeareTokenizer()

    # Tokenize the full text
    tokens = tokenizer(text)

    inputs = []
    targets = []

    # Create sliding windows
    for i in range(len(tokens) - seq_len):
        # Extract window and target (shifted by 1)
        window = tokens[i:i + seq_len]
        target = tokens[i + 1:i + seq_len + 1]

        # Convert to indices
        input_ids = [word2idx.get(w, word2idx['<unk>']) for w in window]
        target_ids = [word2idx.get(w, word2idx['<unk>']) for w in target]

        inputs.append(input_ids)
        targets.append(target_ids)

    print(f"Created {len(inputs)} sequences of length {seq_len}")

    # Show example with actual tokens for verification
    if inputs:
        # Show the actual tokens
        input_tokens = [idx2word[id] for id in inputs[0][:10]]
        target_tokens = [idx2word[id] for id in targets[0][:10]]

        print(f"Example input tokens: {input_tokens}...")
        print(f"Example target tokens: {target_tokens}...")

        # Verify the shift is correct
        if len(inputs[0]) > 5:
            print(f"\nVerifying shift:")
            for i in range(5):
                input_token = idx2word[inputs[0][i]]
                target_token = idx2word[targets[0][i]]
                expected = idx2word[inputs[0][i + 1]] if i + 1 < len(inputs[0]) else "N/A"
                print(f"  Position {i}: input='{input_token}' → target='{target_token}' (expected: '{expected}')")

    return inputs, targets


class ShakespeareDataset(Dataset):
    """PyTorch Dataset for Shakespeare text"""

    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloaders(inputs, targets, batch_size=32, train_split=0.9, shuffle=True):
    """
    Create train and validation dataloaders.

    Args:
        inputs: List of input sequences
        targets: List of target sequences
        batch_size: Batch size for DataLoader
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle the data

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        dataset: The full dataset
    """
    dataset = ShakespeareDataset(inputs, targets)

    # Split into train and validation if requested
    if train_split < 1.0:
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of train batches: {len(train_loader)}")
        print(f"Number of val batches: {len(val_loader)}")

        return train_loader, val_loader, dataset

    else:
        # Just training data
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(train_loader)}")

        return train_loader, None, dataset


def prepare_shakespeare_data(text_file_or_string, vocab_size=5000, seq_len=150,
                             batch_size=32, train_split=0.9):
    """
    Complete data preparation pipeline.

    Args:
        text_file_or_string: Either a file path or the text string itself
        vocab_size: Maximum vocabulary size
        seq_len: Sequence length for training
        batch_size: Batch size for DataLoader
        train_split: Train/validation split ratio

    Returns:
        Dictionary containing all necessary components
    """
    # Load text if it's a file path
    if isinstance(text_file_or_string, str) and text_file_or_string.endswith('.txt'):
        with open(text_file_or_string, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = text_file_or_string

    # Step 1: Build vocabulary
    print("Step 1: Building vocabulary...")
    vocab, word2idx, idx2word, tokenizer = build_vocabulary(text, vocab_size)

    # Step 2: Create sequences - pass idx2word for proper display
    print(f"\nStep 2: Creating sequences (length={seq_len})...")
    inputs, targets = create_sequences(text, word2idx, idx2word, tokenizer, seq_len)

    # Step 3: Create dataloaders
    print(f"\nStep 3: Creating dataloaders (batch_size={batch_size})...")
    train_loader, val_loader, dataset = create_dataloaders(
        inputs, targets, batch_size, train_split
    )

    return {
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(vocab),
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'dataset': dataset,
        'seq_len': seq_len
    }


# Add these generation functions to helper_utils.py
@torch.no_grad()
def generate_tokens(model, prompt_ids, max_length=100, temperature=1.0,
                    top_k=50, top_p=0.95, repetition_penalty=1.2,
                    eos_token_id=None, device='cpu'):
    """
    Advanced token generation with multiple sampling strategies.

    Args:
        model: The trained model
        prompt_ids: Starting token IDs (list or tensor)
        max_length: Maximum length to generate
        temperature: Controls randomness (0.1=conservative, 2.0=creative)
        top_k: Keep only top k tokens (0=disabled)
        top_p: Nucleus sampling threshold (0.95=default)
        repetition_penalty: Penalty for repeated tokens
        eos_token_id: End of sequence token ID
        device: Device to run on

    Returns:
        Generated token IDs as tensor
    """
    model.eval()

    # Handle different input formats
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    elif len(prompt_ids.shape) == 1:
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
    else:
        prompt_ids = prompt_ids.to(device)

    generated = prompt_ids.clone()
    past_tokens = list(prompt_ids[0].cpu().numpy())

    for step in range(max_length - len(prompt_ids[0])):
        # Get model predictions
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(generated)

        # Get the last token's logits
        next_token_logits = logits[0, -1, :].float()

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Penalize all previously generated tokens
            for token_id in set(past_tokens):
                next_token_logits[token_id] /= repetition_penalty

            # Extra penalty for very recent tokens
            if len(past_tokens) > 3:
                for token_id in past_tokens[-3:]:
                    next_token_logits[token_id] /= 1.5

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < \
                                torch.topk(next_token_logits, min(top_k, len(next_token_logits)))[0][-1]
            next_token_logits[indices_to_remove] = -float('inf')

        # Apply nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')

        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        past_tokens.append(next_token.item())

        # Stop if we hit the EOS token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated.squeeze(0)


def generate_text(model, prompt, tokenizer, word2idx, idx2word,
                  max_length=100, temperature=0.8, top_k=50, top_p=0.95,
                  repetition_penalty=1.2, device='cpu'):
    """
    Generate text from a string prompt using advanced sampling.

    Args:
        model: Trained model
        prompt: String prompt
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        max_length: Maximum generation length
        temperature: Sampling temperature (0.1-2.0 typical)
        top_k: Top-k filtering (50 is good default)
        top_p: Nucleus sampling (0.95 is good default)
        repetition_penalty: Penalty for repetition (1.2 is good default)
        device: Device to run on

    Returns:
        Generated text as string
    """
    # Tokenize prompt
    if not prompt or prompt.isspace():
        # Start with a common word if no prompt
        prompt_tokens = ['the']
    else:
        prompt_tokens = tokenizer(prompt.lower())

    # Convert to indices
    prompt_ids = []
    for token in prompt_tokens:
        if token in word2idx:
            prompt_ids.append(word2idx[token])
        else:
            # Try to find similar token
            token_lower = token.lower()
            if token_lower in word2idx:
                prompt_ids.append(word2idx[token_lower])
            else:
                prompt_ids.append(word2idx['<unk>'])

    # Ensure we have at least one token
    if not prompt_ids:
        prompt_ids = [word2idx.get('the', word2idx['<unk>'])]

    # Generate token IDs
    eos_token_id = word2idx.get('<eos>', None)

    generated_ids = generate_tokens(
        model,
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        device=device
    )

    # Convert to text
    tokens = []
    for idx in generated_ids:
        idx_val = idx.item() if hasattr(idx, 'item') else idx
        token = idx2word.get(idx_val, '<unk>')

        # Handle special tokens
        if token == '<nl>' or token == '<newline>':
            tokens.append('\n')
        elif token not in ['<pad>', '<unk>', '<eos>', '<start>']:
            tokens.append(token)

    # Join and clean up
    text = ' '.join(tokens)

    # Fix punctuation spacing
    text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    text = text.replace(' \'', '\'').replace('\' ', '\'')
    text = text.replace(' \n ', '\n').replace('\n ', '\n')

    return text.strip()


def interactive_generation(model, tokenizer, word2idx, idx2word, device='cpu'):
    """
    Interactive text generation loop.

    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        device: Device to run on
    """
    print("=" * 50)
    print("Interactive Shakespeare Text Generation")
    print("=" * 50)
    print("Enter a prompt to generate text (or 'quit' to exit)")
    print("Commands: 'temp=0.8' to set temperature, 'len=100' to set length")
    print("-" * 50)

    temperature = 0.8
    max_length = 100

    while True:
        prompt = input("\nPrompt: ").strip()

        if prompt.lower() == 'quit':
            break

        # Check for commands
        if prompt.startswith('temp='):
            try:
                temperature = float(prompt[5:])
                print(f"Temperature set to {temperature}")
                continue
            except:
                print("Invalid temperature")
                continue

        if prompt.startswith('len='):
            try:
                max_length = int(prompt[4:])
                print(f"Max length set to {max_length}")
                continue
            except:
                print("Invalid length")
                continue

        # Generate text
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )

        print("\n" + "=" * 50)
        print("Generated:")
        print("-" * 50)
        print(generated)
        print("=" * 50)


# Batch generation for multiple prompts
def generate_batch(model, prompts, tokenizer, word2idx, idx2word,
                   max_length=100, temperature=0.8, device='cpu'):
    """
    Generate text for multiple prompts.

    Args:
        model: Trained model
        prompts: List of prompt strings
        tokenizer: Tokenizer instance
        word2idx: Word to index dictionary
        idx2word: Index to word dictionary
        max_length: Maximum generation length
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        List of generated texts
    """
    results = []
    for prompt in prompts:
        generated = generate_text(
            model, prompt, tokenizer, word2idx, idx2word,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        results.append(generated)
    return results


# Define available language pairs
LANGUAGE_PAIRS = {
    'French': {'code': 'fra', 'file': 'fra.txt'},
    'Spanish': {'code': 'spa', 'file': 'spa.txt'},
    'German': {'code': 'deu', 'file': 'deu.txt'},
    'Italian': {'code': 'ita', 'file': 'ita.txt'},
    'Portuguese': {'code': 'por', 'file': 'por.txt'},
    'Russian': {'code': 'rus', 'file': 'rus.txt'}
}


def load_dataset(languages_dir: str = './languages') -> Tuple[List[Tuple[str, str]], str]:
    """
    Interactively load a translation dataset.

    Args:
        languages_dir: Directory containing language files

    Returns:
        Tuple of (translation_pairs, selected_language_name)
        where translation_pairs is a list of tuples like [('Go.', 'Vai.'), ...]
    """
    # Display available languages
    print("Available translation pairs (to/from English):")
    for i, lang in enumerate(LANGUAGE_PAIRS.keys(), 1):
        print(f"{i}. English ↔ {lang}")

    # Let user choose
    while True:
        try:
            choice = int(input("\nSelect a language (enter number): "))
            if 1 <= choice <= len(LANGUAGE_PAIRS):
                target_language = list(LANGUAGE_PAIRS.keys())[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nYou selected: English ↔ {target_language}")
    lang_info = LANGUAGE_PAIRS[target_language]

    # Set up paths
    languages_dir = Path(languages_dir)
    dataset_file = languages_dir / lang_info['file']
    zip_file = languages_dir / f"{lang_info['code']}-eng.zip"

    # Extract if needed
    if not dataset_file.exists():
        if zip_file.exists():
            print(f"Extracting {target_language} dataset...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(languages_dir)
            print("Extraction complete!")
        else:
            print(f"Error: {zip_file} not found in languages folder!")
            print(f"Please ensure {lang_info['code']}-eng.zip is in the languages folder.")
            return [], target_language
    else:
        print(f"{target_language} dataset already extracted, loading...")

    # Load the translation pairs as list of tuples
    translation_pairs = []  # This will be a list of tuples
    if dataset_file.exists():
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Creating tuple (English, Target Language)
                    translation_pairs.append((parts[0], parts[1]))

    print(f"Loaded {len(translation_pairs)} English-{target_language} translation pairs")

    # Display random examples
    print(f"\nRandom sample English-{target_language} pairs:")
    num_samples = min(5, len(translation_pairs))
    random_pairs = random.sample(translation_pairs, num_samples)
    for eng, target in random_pairs:
        print(f"English: {eng}")
        print(f"{target_language}: {target}")
        print("-" * 50)

    return translation_pairs, target_language



class MultilingualTokenizer:
    """
    Tokenizer that can handle multiple languages
    """

    def __init__(self, language='French'):
        self.language = language

    def __call__(self, text):
        # Convert to lowercase
        text = text.lower()

        # Language-specific handling
        if self.language == 'Russian':
            # Keep Cyrillic characters
            text = re.sub(r"([.!?])", r" \1", text)
            tokens = re.findall(r"[\w]+|[.!?]", text)
        else:
            # Improved handling for Latin-based languages
            # First, add spaces around punctuation (but not apostrophes within words)
            text = re.sub(r"([.!?])", r" \1", text)
            # Find words (including contractions) and punctuation
            # This pattern captures:
            # - Words with internal apostrophes (contractions)
            # - Regular words
            # - Punctuation marks
            tokens = re.findall(r"\b\w+(?:'\w+)*\b|[.!?]", text)

        return tokens


def normalize_string(s, language='French'):
    """
    Normalize a string based on the target language
    """
    # Convert to lowercase
    s = s.lower().strip()

    # Language-specific normalization
    if language == 'Russian':
        # Keep Cyrillic characters
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^а-яА-Яa-zA-Z.!?]+", r" ", s)
    elif language in ['French', 'Spanish', 'Portuguese', 'Italian']:
        # Keep Latin characters with accents and apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÀ-ÿ'.!?]+", r" ", s)  # Added apostrophe
    elif language == 'German':
        # Keep German special characters and apostrophes
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-ZÄäÖöÜüß'.!?]+", r" ", s)  # Added apostrophe
    else:
        # Default handling (including English) - keep apostrophes for contractions
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z'.!?]+", r" ", s)  # Added apostrophe

    # Remove extra spaces
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def prepare_data(translation_pairs: List[Tuple[str, str]],
                 target_language: str,
                 max_pairs: int = 15000,
                 max_length: int = 20) -> Tuple[List[Tuple[str, str]], object]:
    """
    Prepare translation data by normalizing text and creating a tokenizer.

    Args:
        translation_pairs: List of (English, target_language) translation pairs
        target_language: Name of the target language
        max_pairs: Maximum number of pairs to process (default: 15000)
        max_length: Maximum sentence length in words (default: 20)

    Returns:
        Tuple of (normalized_pairs, tokenizer)
    """
    # Apply normalization to all pairs
    normalized_pairs = []

    for eng, target in translation_pairs[:max_pairs]:
        eng_norm = normalize_string(eng, 'English')
        target_norm = normalize_string(target, target_language)

        # Filter by length for more manageable training
        if len(eng_norm.split()) <= max_length and len(target_norm.split()) <= max_length:
            normalized_pairs.append((eng_norm, target_norm))

    # Create tokenizer for the selected language
    tokenizer = MultilingualTokenizer(target_language)

    # Test the tokenizer with examples containing contractions
    print(f"\n=== Tokenizer Test for {target_language} ===")

    # Test with English contractions
    test_sentences = [
        "Hello, how are you?",
        "He's going to the store.",
        "I can't believe it's working!",
        "They're here, aren't they?"
    ]

    for test_sent in test_sentences:
        tokens = tokenizer(test_sent)
        print(f"Original: {test_sent}")
        print(f"Tokenized: {tokens}")
        print()

    # Test with a sample from the target language if available
    if normalized_pairs:
        sample_target = normalized_pairs[0][1]  # Get first target language sentence
        print(f"\n{target_language} sample: {sample_target}")
        print(f"{target_language} tokenized: {tokenizer(sample_target)}")

    print(f"\n=== Data Preparation Complete ===")
    print(f"Normalized pairs: {len(normalized_pairs)} (from {min(max_pairs, len(translation_pairs))} original pairs)")

    return normalized_pairs, tokenizer


def show_model_layers(model):
    """
    Display the 4 main layers of the TranslationEncoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)

    # Show the 4 main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")

    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)


def show_decoder_layers(model):
    """
    Display the main layers of the Decoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Layers")
    print("=" * 70)
    print(f"\n{'Layer':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)

    # Show the main layers
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")

    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)


def show_encoderdecoder_layers(model):
    """
    Display the main components of the EncoderDecoder model.
    """
    print("\n" + "=" * 70)
    print(f" {model.__class__.__name__} - Main Components")
    print("=" * 70)
    print(f"\n{'Component':<30} {'Type':<25} {'Parameters':>15}")
    print("-" * 70)

    # Show the main components
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_type = module.__class__.__name__
        print(f"{name:<30} {module_type:<25} {params:>15,}")

    print("-" * 70)
    total = sum(p.numel() for p in model.parameters())
    print(f"{'TOTAL':<30} {'':<25} {total:>15,}")
    print("=" * 70)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    """
    Simple training function for the translator
    """
    # Store history
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0
        train_batches = 0

        # Progress bar for training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for src_batch, tgt_batch in train_bar:
            # Move to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Prepare decoder input and target
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_input)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(outputs, tgt_output)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track loss
            train_total_loss += loss.item()
            train_batches += 1

            # Update progress bar
            avg_loss = train_total_loss / train_batches
            train_bar.set_postfix({'loss': f'{loss.item():.3f}'})

        # Validation phase
        model.eval()
        val_total_loss = 0
        val_batches = 0

        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]

                outputs = model(src_batch, tgt_input)
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)

                loss = criterion(outputs, tgt_output)

                val_total_loss += loss.item()
                val_batches += 1

        # Calculate average losses
        train_loss = train_total_loss / train_batches
        val_loss = val_total_loss / val_batches

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Print epoch summary
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}\n')

    return history


def plot_training_history(history):
    """
    Plot training and validation loss from training history.

    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
    """
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'],
             label='Training Loss', marker='o')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'],
             label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print final results
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    print(f"Best Validation Loss at Epoch: {best_epoch}")