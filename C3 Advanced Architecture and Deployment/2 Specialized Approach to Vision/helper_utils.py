import os
import io
import imageio
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DDPMPipeline
from tqdm.auto import tqdm


def plot_image(image_tensor, title=None, **kwargs):
    image_np = image_tensor.squeeze(0).numpy()
    img_transposed = np.transpose(image_np, (1, 2, 0))

    # Use the 'nearest' interpolation to enhance pixelation
    plt.imshow(img_transposed, interpolation='nearest', **kwargs)
    plt.title(title)
    plt.axis('off')


def get_ball():  # Create a structured RGB tensor for a simple circle shape
    height, width = 32, 32

    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    radius = min(center_y, center_x) // 2

    # Create a circular mask
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    # Initialize channels
    red_channel = np.zeros((height, width))
    green_channel = np.zeros((height, width))
    blue_channel = np.zeros((height, width))

    # Apply mask to create a circle in red channel
    red_channel[mask] = 1.0

    # Stack channels to create an RGB image
    structured_image = np.stack([red_channel, green_channel, blue_channel], axis=0)
    return structured_image


def calculate_receptive_field(model, input_size=224):
    """
    Calculates the receptive field and feature map size for each Conv2d/Pool2d layer in a PyTorch model.

    Args:
        model: A PyTorch nn.Module (for example, ResNet50)
        input_size: The spatial size of the square input image (for example, 224 for 224x224)
    Returns:
        List of dictionaries. Each dict contains:
            'layer': Name of the layer
            'rf': Receptive field size up to this layer
            'jump': Effective stride up to this layer
            'map_size': Feature map spatial size (width/height) at this layer
    """
    receptive_field = 1  # Start: Each input pixel only "sees" itself
    jump = 1  # Effective jump/stride (pixels in output to input)
    curr_size = input_size
    results = []

    layer_names = set()  # Helps ensure unique names (if layers repeat)

    def get_layer_name(name, module):
        base = f"{name} ({type(module).__name__})"
        n = 1
        candidate = base
        while candidate in layer_names:
            candidate = f"{base}_{n}"
            n += 1
        layer_names.add(candidate)
        return candidate

    # This function runs for each relevant layer as the model processes a dummy input
    def layer_hook(name):
        def hook(module, inp, outp):
            nonlocal receptive_field, jump, curr_size

            # Get layer parameters (assume square kernels and strides)
            k = getattr(module, "kernel_size", 1)
            s = getattr(module, "stride", 1)
            p = getattr(module, "padding", 0)
            d = getattr(module, "dilation", 1)
            if isinstance(k, tuple): k = k[0]
            if isinstance(s, tuple): s = s[0]
            if isinstance(p, tuple): p = p[0]
            if isinstance(d, tuple): d = d[0]

            # Calculate output spatial size (assume square input/output)
            map_size = outp.shape[-1]

            # Save current layer info
            results.append({
                "layer": get_layer_name(name, module),
                "rf": receptive_field,
                "jump": jump,
                "map_size": curr_size
            })

            # Update receptive field and jump for next layer
            # rf_next = rf + (k-1) * d * jump
            receptive_field = receptive_field + (k - 1) * d * jump
            jump = jump * s
            curr_size = map_size

        return hook

    # Register hooks for all Conv2d and Pool2d layers recursively
    hooks = []

    def register_hooks(net, prefix=''):
        for name, module in net.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(layer_hook(full_name)))
            register_hooks(module, full_name)

    register_hooks(model)

    # Send a dummy input through the model (to activate hooks)
    dummy = torch.zeros(1, 3, input_size, input_size)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return results

activations = {}
def visualize_output_layer(layer):
    output_tensor = activations[layer]
    num_images = output_tensor.shape[0]
    for i in range(num_images):
        grid_size = int(np.ceil(np.sqrt(num_images)))
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(output_tensor[i].detach().numpy(), cmap='gray')
        plt.axis('off')
        #plt.title(f'Filter {i+1}', fontsize=10, pad=10)  
    plt.tight_layout()
    plt.show()


def make_grid(images, grid_size=None):
    # images: (N, H, W)
    num_images, H, W = images.shape
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(num_images)))
    grid = np.zeros((grid_size * H, grid_size * W))
    for idx in range(num_images):
        row = idx // grid_size
        col = idx % grid_size
        grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = images[idx]
    return grid


def visualize_all_layers_grids(activations):
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 8))  # 1 column, n_layers rows

    if n_layers == 1:
        axes = [axes]

    for idx, (layer_name, output) in enumerate(activations.items()):
        acts = output[0]  # (C, H, W)
        n_filters = acts.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_filters)))

        # Create a black canvas for the grid
        h, w = acts.shape[1], acts.shape[2]
        grid = np.zeros((h * grid_size, w * grid_size))

        for i in range(n_filters):
            row = i // grid_size
            col = i % grid_size
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = acts[i].detach().cpu()

        axes[idx].imshow(grid, cmap='gray')
        axes[idx].set_title(layer_name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_receptive_field_summary(rfinfo, figwidth=14):
    names = [x['layer'] for x in rfinfo]
    rf = np.array([x['rf'] for x in rfinfo])
    msize = np.array([x['map_size'] for x in rfinfo])

    # Subsample for key layers
    to_plot_idx = [0]
    for i in range(1, len(names)):
        if msize[i] < msize[i - 1] or i == len(names) - 1 or i % 6 == 0:
            to_plot_idx.append(i)
    to_plot_idx = sorted(set(to_plot_idx))
    names_short = []
    for idx in to_plot_idx:
        n = names[idx]
        n = '.'.join(n.split('.')[-3:]).replace(' (Conv2d)', '').replace(' (MaxPool2d)', '').replace(' (AvgPool2d)', '')
        names_short.append(n)
    rf_plot = rf[to_plot_idx]
    msize_plot = msize[to_plot_idx]

    fig, ax1 = plt.subplots(figsize=(figwidth, 5))
    line_rf, = ax1.plot(rf_plot, '-o', color='tab:blue', label='Receptive Field')
    ax2 = ax1.twinx()
    line_ms, = ax2.plot(msize_plot, '--s', color='tab:red', label='Feature Map Size')

    # Downsampling points
    for i in range(1, len(msize_plot)):
        if msize_plot[i] < msize_plot[i - 1]:
            ax2.plot(i, msize_plot[i], 'v', color='darkred', markersize=9, zorder=3)
            ax2.annotate(
                f"{msize_plot[i]}",
                xy=(i, msize_plot[i]), xytext=(0, 16), textcoords='offset points',
                ha='center', color='darkred', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none"), zorder=5
            )

    ax1.set_xticks(range(len(names_short)))
    ax1.set_xticklabels(names_short, rotation=50, ha='right', fontsize=9)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Receptive Field', color='tab:blue')
    ax2.set_ylabel('Feature Map Size', color='tab:red')
    plt.title("CNN Receptive Field and Feature Map Size per Layer (summary)")
    ax1.grid(axis='y', linestyle=':')
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Make wide room for legend

    # --- Attach legend to the figure, not axes ---
    fig.legend([line_rf, line_ms], ["Receptive Field", "Feature Map Size"],
               loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=True)
    plt.show()


def plot_rfinfo_over_image(
        rfinfo, img_path, input_size=224, layers_to_show=None, figsize=(18, 8)
):
    """
    Visualize receptive field overlays for certain layers, given rfinfo (from calculate_receptive_field)
    and an image.
    rfinfo:         output from calculate_receptive_field(model, input_size)
    img_path:       image filename
    input_size:     input spatial size (should match what was passed to calculate_receptive_field)
    layers_to_show: list of layer names or None to show default interesting ones
    """
    orig_img = Image.open(img_path).convert("RGB")
    img_resized = orig_img.resize((input_size, input_size))
    image_np = np.array(img_resized)
    all_layer_names = [item["layer"] for item in rfinfo]

    if layers_to_show is None:
        # You may adjust this selection logic for your model!
        candidates = []
        for name in [
            "conv1 (Conv2d)",
            "maxpool (MaxPool2d)",
            "layer1.0.conv1 (Conv2d)",
            "layer1.2.conv3 (Conv2d)",
            "layer2.0.conv3 (Conv2d)",
            "layer3.3.conv3 (Conv2d)",
            "layer4.0.conv3 (Conv2d)",
            "layer4.2.conv3 (Conv2d)",
        ]:
            candidates.extend([n for n in all_layer_names if name in n])
        layers_to_show = candidates[:7]  # adjust how many layers you want to show

    def plot_one(info, subplot_idx, image, input_size=224):
        feature_size = info["map_size"]
        rf_size = info["rf"]
        name = info["layer"]

        # Special case: no grid or overlay for first full-size layer (usually conv1)
        if feature_size == input_size and rf_size == 1:
            plt.subplot(2, 4, subplot_idx)
            plt.imshow(image.astype(np.uint8))
            plt.title(
                f"{name}\nFeature Map: {feature_size}×{feature_size}\nReceptive Field: {rf_size}×{rf_size}"
            )
            plt.axis("off")
            return

        img = image.copy().astype(np.float32) / 255.0
        # Draw grid for other layers
        if feature_size > 1:
            cell_size = input_size / feature_size
            for i in range(feature_size):
                y = int(i * cell_size)
                if y < input_size:
                    img[y: y + 1, :, :] = [0, 0, 0]
                x = int(i * cell_size)
                if x < input_size:
                    img[:, x: x + 1, :] = [0, 0, 0]
        # Draw receptive field patch
        center = input_size // 2
        rf_half = min(rf_size // 2, input_size // 2)
        if rf_half > 0:
            start_x = max(0, center - rf_half)
            end_x = min(input_size, center + rf_half)
            start_y = max(0, center - rf_half)
            end_y = min(input_size, center + rf_half)
            img[start_y:end_y, start_x:end_x, 0] = 1.0
            img[start_y:end_y, start_x:end_x, 1] *= 0.7
            img[start_y:end_y, start_x:end_x, 2] *= 0.7
            thickness = max(1, input_size // 100)
            if start_y < end_y:
                img[start_y: start_y + thickness, start_x:end_x] = [1, 0, 0]
                img[end_y - thickness: end_y, start_x:end_x] = [1, 0, 0]
            if start_x < end_x:
                img[start_y:end_y, start_x: start_x + thickness] = [1, 0, 0]
                img[start_y:end_y, end_x - thickness: end_x] = [1, 0, 0]
        plt.subplot(2, 4, subplot_idx)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(
            f"{name}\nFeature Map: {feature_size}×{feature_size}\nReceptive Field: {rf_size}×{rf_size}"
        )
        plt.axis("off")

    plt.figure(figsize=figsize)
    # First: original image
    plt.subplot(2, 4, 1)
    plt.imshow(image_np)
    plt.title(f"Original Image\n({input_size}×{input_size})")
    plt.axis("off")
    # Others: overlays for selected layers
    for i, name in enumerate(layers_to_show):
        matches = [item for item in rfinfo if item["layer"] == name]
        if not matches:
            print(f"(Warning: Layer '{name}' not found in rfinfo.)")
            continue
        plot_one(matches[0], subplot_idx=i + 2, image=image_np, input_size=input_size)
    plt.suptitle("Receptive Field and Feature Map Visualization")
    plt.tight_layout()
    plt.show()


def plot_widget(model):
    import os
    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from PIL import Image as PILImage

    # Download once: (uncomment if not already downloaded)
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
                               "imagenet_classes.txt")

    with open("imagenet_classes.txt") as f:
        imagenet_labels = [line.strip() for line in f.readlines()]

    # --------- CONFIG ---------
    images_folder = 'images'
    canvas_size = (512, 512)
    INPUT_OPTION_NAME = "🟢 Input image"
    # --------------------------

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    feature_maps = {}
    layer_info_list = []  # (pretty_name, module_obj or None)

    def hook_fn(module, input, output):
        feature_maps[module] = output

    # Layer list (first entry is input image)
    layer_info_list.append((INPUT_OPTION_NAME, None))
    for top_name, layer in model.named_children():
        if isinstance(layer, torch.nn.Sequential):
            for subname, sublayer in layer.named_children():
                pretty_name = f"{top_name}.{subname}:{type(sublayer).__name__}"
                short_name = (pretty_name[:60] + '...') if len(pretty_name) > 63 else pretty_name
                layer_info_list.append((short_name, sublayer))
                sublayer.register_forward_hook(hook_fn)
        else:
            pretty_name = f"{top_name}:{type(layer).__name__}"
            short_name = (pretty_name[:60] + '...') if len(pretty_name) > 63 else pretty_name
            layer_info_list.append((short_name, layer))
            layer.register_forward_hook(hook_fn)

    def make_grid(tensor, n_cols=8):
        if tensor.dim() != 4:
            return None
        n_channels = tensor.shape[1]
        n_rows = (n_channels - 1) // n_cols + 1
        grid = np.zeros((n_rows * tensor.shape[2], n_cols * tensor.shape[3]))
        for i in range(n_channels):
            row, col = divmod(i, n_cols)
            grid[row * tensor.shape[2]: (row + 1) * tensor.shape[2],
            col * tensor.shape[3]: (col + 1) * tensor.shape[3]] = tensor[0, i].detach().numpy()
        return grid

    output = widgets.Output()
    layer_params_html = widgets.HTML(
        value="",
        layout=widgets.Layout(
            min_width='320px', max_width='420px',
            font_family='monospace',
            font_size='1em',
            background_color='#181818',
            border='solid 1.5px #444',
            padding='10px 15px 10px 15px',
            color='#CCE4FF',
            overflow='auto'
        )
    )

    def update_plot(image_path, selected_layer_idx):
        with output:
            clear_output(wait=True)
            feature_maps.clear()
            img = Image.open(os.path.join(images_folder, image_path)).convert("RGB")
            img_t = preprocess(img)
            if selected_layer_idx == 0:  # Input Image selected
                input_img = img_t.permute(1, 2, 0).cpu().numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_img = (input_img * std + mean)
                input_img = np.clip(input_img, 0, 1)
                pil_img = PILImage.fromarray((input_img * 255).astype(np.uint8))
                pil_img = pil_img.resize(canvas_size, PILImage.NEAREST)
                plt.figure(figsize=(7, 7))
                plt.imshow(pil_img)
                plt.title("Input image (after preprocessing & crop)", fontsize=16, pad=16)
                plt.axis('off')
                plt.tight_layout(pad=2)
                plt.show()
                # Params/info for input image
                layer_params_html.value = (
                    "<b>Input image</b><br><br>"
                    "<i>Displayed as passed into the model after resizing, cropping, "
                    "and normalization (mean/std = ImageNet).</i>"
                )
                return
            # Forward pass
            batch_t = torch.unsqueeze(img_t, 0)
            output_tensor = model(batch_t)
            layer_obj = layer_info_list[selected_layer_idx][1]
            # Show class prediction if last layer (fc/Linear)
            last_layer = list(model.named_children())[-1][1]
            if layer_obj is last_layer and isinstance(layer_obj, torch.nn.Linear):
                # 1. Get predicted label and probability (=confidence)
                probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                predicted_label = imagenet_labels[predicted_idx] if predicted_idx < len(
                    imagenet_labels) else "(Unknown class)"
                predicted_prob = probabilities[predicted_idx].item()
                confidence_percent = f"{predicted_prob * 100:.1f}%"

                # 2. Show "It is a ..." in the figure (plain, not HTML)
                plt.figure(figsize=(7, 7))
                plt.axis('off')
                plt.text(0.5, 0.5, f"It is a\n{predicted_label}!",
                         ha='center', va='center', fontsize=26, color='#67d800', weight='bold', wrap=True)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.tight_layout(pad=2)
                plt.show()

                # 3. Show class and confidence in HTML in the parameter/sidebar
                param_str = (
                    f"<b>Output:</b><br>"
                    f"<span style='color:#FA4;font-size:1.3em;'>It is a <b>{predicted_label}</b>!</span><br>"
                    f"<b>Confidence:</b> <span style='color:#8FD14A;font-size:1.15em'>{confidence_percent}</span><br><br>"
                    f"<b>{type(layer_obj).__name__}</b><br>"
                    f"<pre>{str(layer_obj)}</pre>"
                )
                layer_params_html.value = param_str
                return

            fmap = feature_maps.get(layer_obj)
            if fmap is not None:
                grid_image = make_grid(fmap, n_cols=8)
                if grid_image is not None:
                    norm_grid = (grid_image - np.min(grid_image)) / (np.ptp(grid_image) + 1e-9)
                    norm_grid = (norm_grid * 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(norm_grid)
                    pil_img = pil_img.resize(canvas_size, resample=PILImage.NEAREST)
                    plt.figure(figsize=(7, 7))
                    plt.imshow(pil_img, cmap='viridis')
                    plt.title(f'Feature maps from: {layer_info_list[selected_layer_idx][0]}', fontsize=16, pad=16)
                    plt.axis('off')
                    plt.tight_layout(pad=2)
                    plt.show()
            else:
                plt.figure()
                plt.title('No feature map for selected layer', fontsize=14)
                plt.axis('off')
                plt.show()
            # Param string (limit lines for very big blocks)
            param_str = str(layer_obj)
            nlines = 23
            param_lines = param_str.split('\n')
            if len(param_lines) > nlines:
                shown_text = '\n'.join(
                    param_lines[:nlines]) + '\n...\n' + '<span style="color:#fbb;">[truncated]</span>'
            else:
                shown_text = '\n'.join(param_lines)
            layer_params_html.value = f"<b>Layer parameters:</b><br><pre style='font-size:0.98em;'>{shown_text}</pre>"

    # --- UI LAYOUT ---

    dropdown_style = {
        'description_width': '80px',
    }

    image_dropdown = widgets.Dropdown(
        options=image_files,
        description='Image',
        value=image_files[0],
        style=dropdown_style,
        layout=widgets.Layout(width='180px', margin='0px 8px 0px 0px')
    )

    layer_dropdown = widgets.Dropdown(
        options=[x[0] for x in layer_info_list],
        description='Layer',
        value=layer_info_list[0][0],
        style=dropdown_style,
        layout=widgets.Layout(width='320px', margin='0px 0px 0px 0px')
    )

    header = widgets.HTML(
        value="<div style='font-size: 1.8em; color:#55AAF2; margin-bottom: 6px; font-family:sans-serif;'><b>Feature Map Visualizer</b></div>"
    )
    caption = widgets.HTML(
        value=(
            "<div style='font-size:1.07em; color:#BBB; margin-bottom: 6px;'>"
            "<b>Select an <span style=\"color:#EEE\">image</span></b> and <b><span style=\"color:#EEE\">network layer</span></b> to view its feature maps.<br>"
            "All feature maps are shown at a fixed display size for easy comparison.</div>"
        )
    )

    controls = widgets.HBox([image_dropdown, layer_dropdown],
                            layout=widgets.Layout(margin="0 0 16px 0", align_items="center"))

    # Here, place image and param text side by side.
    panel = widgets.HBox(
        [output, layer_params_html],
        layout=widgets.Layout(
            justify_content='space-between',
            align_items='flex-start'
        )
    )

    group_box = widgets.VBox(
        [panel],
        layout=widgets.Layout(
            border='solid 2px #444',
            background_color='#111',
            padding='16px',
            align_items='flex-start'
        )
    )

    outer_box = widgets.VBox(
        [header, caption, controls, group_box],
        layout=widgets.Layout(
            background_color="#101014",
            padding="24px 24px 10px 24px",
            border_radius="12px",
            box_shadow="0 1px 8px rgba(0,0,0,0.11)",
            min_width="850px", max_width="1200px"
        )
    )

    def on_any_dropdown_change(change):
        idx = layer_dropdown.options.index(layer_dropdown.value)
        update_plot(image_dropdown.value, idx)

    image_dropdown.observe(on_any_dropdown_change, names='value')
    layer_dropdown.observe(on_any_dropdown_change, names='value')

    display(outer_box)
    update_plot(image_dropdown.value, layer_dropdown.options.index(layer_dropdown.value))


def plot_widget_gradcam(compute_gradcam, visualize_gradcam, model, transform, device, folder="images"):
    """
    Displays a widget to select and visualize GradCAM for images in a folder.

    Args:
        compute_gradcam: function(img_path, model, transform, device) -> (img_display, heatmap, pred_class, pred_score)
        visualize_gradcam: function(img, heatmap, pred_class, pred_score, title)
        model: PyTorch model with weights loaded.
        transform: Torch transform for the input image.
        device: torch.device
        folder: str, folder to look in (default: "images")
    """

    def get_jpg_files(folder=folder):
        """
        Returns a sorted list of .jpg files (case-insensitive) in the chosen folder.
        """
        if not os.path.exists(folder):
            return []
        return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

    out = widgets.Output()

    def gradcam_widget_view(image_name):
        """
        Widget callback: runs GradCAM and visualization for the selected image.
        All plots and outputs are displayed in the widget output area.
        """
        img_path = os.path.join(folder, image_name)
        with out:
            clear_output(wait=True)
            print(f"Showing GradCAM for: {image_name}")
            img_display, heatmap, pred_class, pred_score = compute_gradcam(
                img_path, model, transform, device
            )
            if img_display is not None:
                title = os.path.splitext(image_name)[0]
                visualize_gradcam(img_display, heatmap, pred_class, pred_score, title)
            else:
                print(f"Could not process {image_name}.")

    jpg_list = get_jpg_files(folder)
    if not jpg_list:
        display(widgets.HTML(
            value=f"<b style='color: red;'>No .jpg files found in the <code>{folder}/</code> folder. Please add images and rerun.</b>"
        ))
    else:
        dropdown = widgets.Dropdown(
            options=jpg_list,
            description='Select image:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%'),  # Wider, easier to use
            disabled=False
        )
        display(widgets.HTML(f"<h3>GradCAM Visualizer</h3>Select an image from <code>{folder}</code>:"))
        widgets.interact(gradcam_widget_view, image_name=dropdown)
        display(out)


def remove_all_noise_at_once(noisy_image, predicted_noise, timestep, scheduler):
    """
    Given a noisy image and the model's predicted noise,
    apply the mathematical formula to remove ALL predicted noise in one step (not gradual).

    This uses the original DDPM denoising formula:
    x_0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
    Where:
      - x_t: Noisy image at current timestep
      - epsilon: Predicted noise by the model
      - alpha_cumprod: Cumulative product of noise schedule parameter
    """
    # Get cumulative product of alphas for the current timestep (how much noise is left)
    alpha_prod_t = scheduler.alphas_cumprod[timestep].to(noisy_image.device)
    # Calculate normalization terms for the denoising formula
    sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
    # Remove all predicted noise at once to estimate the original clean image
    clean_image = (noisy_image - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
    return clean_image


def gradual_denoise_step(noisy_image, predicted_noise, timestep, scheduler):
    """
    Perform ONE step of gradual denoising using the scheduler's step function.

    Returns the image after the next denoising step, as the diffusion process normally does.
    """
    step = scheduler.step(predicted_noise, timestep, noisy_image)
    return step.prev_sample


@torch.no_grad()
def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a displayable PIL Image.
    Handles either [-1, 1] or [0, 1] input ranges.
    """
    # Remove batch/channel dimension, rearrange (C, H, W) to (H, W, C)
    img = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    # If input is in [-1, 1], rescale to [0, 1]
    img = (img + 1) / 2 if img.min() < 0 else img
    # Clip any accidental out-of-bounds values
    img = np.clip(img, 0, 1)
    # Convert to uint8 and PIL Image
    return Image.fromarray((img * 255).astype(np.uint8))


@torch.no_grad()
def visualize_ddpm_denoising(pipe, num_inference_steps=100):
    """
    Show the denoising process of a DDPM:
    - Top row: gradual denoising (true DDPM generation)
    - Bottom row: removing all estimated noise in one step at multiple points

    Returns two lists of (step_index, PIL.Image).
    """
    # Get config for image size and channels
    image_size = pipe.unet.config.sample_size
    num_channels = pipe.unet.config.in_channels
    device = pipe.device

    # Start with a single batch of pure noise
    images = torch.randn(1, num_channels, image_size, image_size, device=device)
    scheduler = pipe.scheduler
    model = pipe.unet
    scheduler.set_timesteps(num_inference_steps)

    # Evenly distributed steps to visualize: pick 7 steps (can change n_vis)
    timesteps_to_show = []
    n_vis = 7
    for i in range(n_vis):
        t_idx = int((num_inference_steps - 1) * i / (n_vis - 1))
        timesteps_to_show.append(t_idx)

    # Prepare outputs
    gradual_images = []  # stores (step_index, gradual denoised image)
    full_removal_images = []  # stores (step_index, "full denoise" image)
    timesteps = scheduler.timesteps
    latents = images

    # Denoising loop over all timesteps
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Predict the noise using the model at this step
        noise_pred = model(latents, t).sample

        if i in timesteps_to_show:
            # 1. Compute "full noise removal" for this state
            full_removal = remove_all_noise_at_once(latents, noise_pred, t, scheduler)
            full_removal_img = tensor_to_image(full_removal)
            full_removal_images.append((i, full_removal_img))
            # 2. Save current image for gradual denoising
            current_img = tensor_to_image(latents)
            gradual_images.append((i, current_img))

        # Advance to next time step: normal DDPM sampling
        latents = gradual_denoise_step(latents, noise_pred, t, scheduler)

    # Save the final clean image
    final_img = tensor_to_image(latents)
    gradual_images.append((num_inference_steps, final_img))

    return gradual_images, full_removal_images


def load_widget(pipe):
    """
    Display a pretty interactive Stable Diffusion generator widget.
    pipe: A diffusers StableDiffusionPipeline object, already loaded to GPU/CPU.
    Required Jupyter: ipywidgets, imageio.
    """
    import time

    # Output and image widgets
    output = widgets.Output()
    gif_widget = widgets.Image(format='gif', width=320, height=320)
    final_image_widget = widgets.Image(format='png', width=320, height=320)

    # --- Widget Styles ---
    textbox_style = {'description_width': '120px'}
    slider_style = {'description_width': '140px'}

    heading = widgets.HTML("<h3 style='color:#0EA5E9;font-family:sans-serif'>Stable Diffusion Image Generator</h3>")

    # Generation mode selector
    mode_checkbox = widgets.Checkbox(
        value=True,
        description='Show denoising animation (GIF)',
        indent=False,
        layout=widgets.Layout(width='300px')
    )

    mode_info = widgets.HTML(
        value="""<div style='background-color:#000000;padding:8px;border-radius:6px;margin:8px 0;'>
        <small><b>💡 Tip:</b> Uncheck for faster generation (final image only)</small>
        </div>"""
    )

    prompt_widget = widgets.Text(
        value="A puppy dog riding a skateboard in times square",
        description='Prompt:',
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )
    negative_prompt_widget = widgets.Text(
        value="",
        description='Negative:',
        style=textbox_style,
        layout=widgets.Layout(width='500px')
    )

    steps_slider = widgets.IntSlider(
        value=50, min=10, max=100, step=1,
        description='Inference steps:',
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    steps_value = widgets.Label(f"{steps_slider.value}")

    def update_steps_label(*a):
        steps_value.value = f"{steps_slider.value}"

    steps_slider.observe(update_steps_label, "value")

    gs_slider = widgets.FloatSlider(
        value=7.5, min=4.0, max=16.0, step=0.1,
        description='Guidance scale:',
        style=slider_style,
        continuous_update=False,
        readout=False,
        layout=widgets.Layout(width='350px'),
    )
    gs_value = widgets.Label(f"{gs_slider.value:.2f}")

    def update_gs_label(*a):
        gs_value.value = f"{gs_slider.value:.2f}"

    gs_slider.observe(update_gs_label, "value")

    # Update button text based on mode
    run_button = widgets.Button(
        description="✨ Generate with Animation ✨",
        button_style='info',
        layout=widgets.Layout(width='250px', height='40px')
    )

    def update_button_text(*args):
        if mode_checkbox.value:
            run_button.description = "✨ Generate with Animation ✨"
            run_button.button_style = 'info'
        else:
            run_button.description = "⚡ Quick Generate ⚡"
            run_button.button_style = 'success'

    mode_checkbox.observe(update_button_text, 'value')

    # --- Layout arrangement ---
    input_form = widgets.VBox([
        heading,
        widgets.HBox([mode_checkbox, mode_info]),
        prompt_widget,
        negative_prompt_widget,
        widgets.HBox([steps_slider, steps_value]),
        widgets.HBox([gs_slider, gs_value]),
        run_button,
    ], layout=widgets.Layout(
        border='2px solid #0EA5E9',
        box_shadow="2px 2px 10px #0EA5E9",
        padding='18px 24px 18px 24px',
        border_radius='14px',
        width='600px',
        background='white'
    ))

    # Container for displaying either GIF or final image
    image_container = widgets.VBox([gif_widget, final_image_widget])

    ui = widgets.VBox([
        input_form,
        output,
        image_container
    ])

    def generate_with_animation(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """Generate with full denoising animation"""
        with output:
            output.clear_output()
            print(f"\n🎬 Generating animation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()

            intermediate_images = []
            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=num_inference_steps,
                description='Denoising:',
                bar_style='info',
                orientation='horizontal'
            )
            display(progress)

            def collect_callback(step, timestep, latents):
                with torch.no_grad():
                    scaled_latents = latents / 0.18215
                    image = pipe.vae.decode(scaled_latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    pil_img = Image.fromarray((image * 255).astype('uint8'))
                intermediate_images.append((step, pil_img))
                progress.value = step + 1

            start_time = time.time()
            generator = torch.Generator(pipe.device).manual_seed(42)
            intermediate_images.clear()

            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback=collect_callback,
                callback_steps=1,
                progress_bar=False,
            )

            progress.bar_style = 'success'
            progress.description = 'Done'

            print("\n🎨 Creating animation...")
            gif_bytes = io.BytesIO()
            all_imgs = [img for (step, img) in intermediate_images]
            imageio.mimsave(gif_bytes, all_imgs, format='GIF', duration=0.15)

            # Hide final image widget, show GIF
            final_image_widget.layout.display = 'none'
            gif_widget.layout.display = 'block'
            gif_widget.value = gif_bytes.getvalue()

            elapsed = time.time() - start_time
            print(f"✅ Animation generated in {elapsed:.1f} seconds")

    def generate_final_only(prompt, negative_prompt, num_inference_steps, guidance_scale):
        """Generate only the final image without animation"""
        with output:
            output.clear_output()
            print(f"\n⚡ Quick generation with {num_inference_steps} steps")
            print(f"Prompt: {prompt}")
            if negative_prompt:
                print(f"Negative prompt: {negative_prompt}")
            print()

            progress = widgets.IntProgress(
                value=0,
                min=0,
                max=1,
                description='Generating:',
                bar_style='success',
                orientation='horizontal'
            )
            display(progress)

            start_time = time.time()
            generator = torch.Generator(pipe.device).manual_seed(42)

            # Generate without callbacks for speed
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                progress_bar=False,
            )

            progress.value = 1
            progress.description = 'Done'

            # Convert to PNG bytes
            final_img = result.images[0]
            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format='PNG')

            # Hide GIF widget, show final image
            gif_widget.layout.display = 'none'
            final_image_widget.layout.display = 'block'
            final_image_widget.value = img_bytes.getvalue()

            elapsed = time.time() - start_time
            print(f"✅ Image generated in {elapsed:.1f} seconds")

            # Display generation stats
            print(f"\n📊 Generation Stats:")
            print(f"  • Mode: Quick (final image only)")
            print(f"  • Steps: {num_inference_steps}")
            print(f"  • Guidance Scale: {guidance_scale}")
            print(f"  • Time saved: ~{num_inference_steps * 0.15:.1f}s (animation creation)")

    def run_on_click(b):
        if mode_checkbox.value:
            generate_with_animation(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )
        else:
            generate_final_only(
                prompt_widget.value,
                negative_prompt_widget.value or None,
                steps_slider.value,
                gs_slider.value
            )

    run_button.on_click(run_on_click)
    display(ui)
