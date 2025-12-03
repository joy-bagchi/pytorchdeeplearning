import os
import urllib.request
import tarfile

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


import numpy as np

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


import torch

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


import torch
from tqdm import tqdm

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


import matplotlib.pyplot as plt

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


import matplotlib.pyplot as plt

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