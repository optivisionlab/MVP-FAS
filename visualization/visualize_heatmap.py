import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
from loaders.make_dataset import RemoveBlackBorders
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def get_attention_map(model, img_tensor, device):
    """
    Extract attention map from ViT model
    """
    model.eval()

    # Hook to capture attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        # output shape: [batch, num_heads, num_patches, num_patches]
        attention_weights.append(output.detach())

    # Register hook on the last attention layer
    # For CLIP ViT, we need to access the transformer blocks
    hooks = []
    try:
        # Try to hook into CLIP's visual transformer
        for block in model.model.visual.transformer.resblocks[-1:]:
            hook = block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)
    except:
        print("Warning: Could not register attention hooks")

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_weights, outputs


def visualize_attention_map(img, attention_weights, img_size=224):
    """
    Visualize attention map overlaid on image
    """
    if len(attention_weights) == 0:
        print("No attention weights captured")
        return None

    # Get the last layer attention
    attn = attention_weights[-1]  # [batch, num_heads, num_patches, num_patches]

    # Average over heads
    attn = attn.mean(dim=1)  # [batch, num_patches, num_patches]

    # Get attention to CLS token (first token)
    attn = attn[0, 0, 1:]  # [num_patches-1] (exclude CLS token)

    # Reshape to 2D grid
    num_patches = int(np.sqrt(attn.shape[0]))
    attn = attn.reshape(num_patches, num_patches)

    # Normalize
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Resize to image size
    attn = attn.cpu().numpy()
    attn = cv2.resize(attn, (img_size, img_size))

    return attn


def get_gradcam(model, img_tensor, device, target_layer=None):
    """
    Generate Grad-CAM heatmap
    """
    model.eval()

    # Store gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks on target layer
    if target_layer is None:
        # Try multiple possible target layers
        try:
            # Try visual encoder output
            target_layer = model.model.visual.transformer.resblocks[-1]
        except:
            try:
                # Try visual projection
                target_layer = model.model.visual
            except:
                print("Could not find target layer")
                return None

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    img_tensor.requires_grad = True
    outputs = model(img_tensor)

    # Get similarity score for "live" class
    sim = outputs['similarity']
    prob = F.softmax(sim, dim=-1)
    live_score = prob[0, -1]  # Last class is "live"

    # Backward pass
    model.zero_grad()
    live_score.backward(retain_graph=True)

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    if len(gradients) == 0 or len(activations) == 0:
        print("Warning: No gradients or activations captured, trying alternative method")
        return get_simple_activation_map(model, img_tensor, device)

    # Get gradients and activations
    grads = gradients[0]
    acts = activations[0]

    # Handle different tensor shapes
    if len(grads.shape) == 3:
        # Transformer: could be [batch, seq_len, dim] or [seq_len, batch, dim]
        # Check which dimension is batch (should be 1)
        if grads.shape[1] == 1:
            # Shape is [seq_len, batch, dim] - need to transpose
            grads = grads.transpose(0, 1)  # [batch, seq_len, dim]
            acts = acts.transpose(0, 1)

        # Now shape should be [batch, seq_len, dim]
        # Remove CLS token (first token in sequence)
        if grads.shape[1] > 1:
            grads = grads[:, 1:, :]  # Remove CLS token
            acts = acts[:, 1:, :]

        # Global average pooling over channels (dim)
        weights = grads.mean(dim=-1, keepdim=True)  # [batch, seq_len-1, 1]

        # Weighted combination
        cam = (weights * acts).sum(dim=-1)  # [batch, seq_len-1]

        # Reshape to 2D
        seq_len = cam.shape[1]
        num_patches = int(np.sqrt(seq_len))

        if num_patches * num_patches == seq_len:
            cam = cam.reshape(1, num_patches, num_patches)
        else:
            # If not perfect square, try to find closest square
            print(f"Warning: seq_len={seq_len} is not a perfect square, using interpolation")
            # Convert to numpy for easier manipulation
            cam_np = cam[0].cpu().numpy()  # [seq_len]
            target_size = int(np.ceil(np.sqrt(seq_len)))

            # Pad to make it square
            pad_size = target_size * target_size - seq_len
            if pad_size > 0:
                cam_np = np.pad(cam_np, (0, pad_size), mode='constant', constant_values=0)

            cam_np = cam_np.reshape(target_size, target_size)
            cam = torch.from_numpy(cam_np).unsqueeze(0).to(cam.device)
    elif len(grads.shape) == 4:
        # CNN: [batch, channels, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
    else:
        print(f"Warning: Unexpected gradient shape: {grads.shape}")
        return None

    # ReLU and normalize
    cam = F.relu(cam)
    cam = cam[0].cpu().numpy()

    # Check if cam is empty
    if cam.size == 0:
        print("Warning: Empty CAM, using fallback")
        return get_simple_activation_map(model, img_tensor, device)

    # Normalize
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.ones_like(cam) * 0.5

    return cam


def get_simple_activation_map(model, img_tensor, device):
    """
    Fallback: Simple activation map from model output
    """
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)

            # Try to get patch features if available
            if 'patch_features' in outputs:
                features = outputs['patch_features']
            elif 'features' in outputs:
                features = outputs['features']
            else:
                # Create uniform map
                return np.ones((14, 14)) * 0.5

            # Average over channels
            if len(features.shape) == 3:
                # [batch, seq_len, dim]
                features = features[0, 1:, :]  # Remove CLS token
                activation = features.norm(dim=-1)

                # Reshape to 2D
                num_patches = int(np.sqrt(activation.shape[0]))
                if num_patches * num_patches == activation.shape[0]:
                    activation = activation.reshape(num_patches, num_patches)
                else:
                    activation = activation.reshape(1, -1)
            else:
                activation = features[0].mean(dim=0)

            # Normalize
            activation = activation.cpu().numpy()
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

            return activation
    except Exception as e:
        print(f"Error in fallback activation map: {e}")
        return np.ones((14, 14)) * 0.5


def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image
    """
    # Ensure image is numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Resize heatmap to match image size
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)

    return overlayed


def visualize_slot_attention(model, img_tensor, device):
    """
    Visualize slot attention weights
    """
    model.eval()

    # Store slot attention weights
    slot_attn_weights = []

    def hook_fn(module, input, output):
        # Capture attention weights from slot attention
        if hasattr(module, 'attn_weights'):
            slot_attn_weights.append(module.attn_weights.detach())

    # Register hook on MVSlot
    hook = None
    try:
        if hasattr(model, 'MVSlot'):
            hook = model.MVSlot.register_forward_hook(hook_fn)
    except:
        print("Could not register slot attention hook")

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Remove hook
    if hook is not None:
        hook.remove()

    return slot_attn_weights


def create_visualization(img_path, model, cfg, device, save_path, method='gradcam'):
    """
    Create complete visualization with heatmap
    """
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_display = img.copy()

    transform = transforms.Compose([
        RemoveBlackBorders(),
        transforms.Resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.DATASET.Mean, std=cfg.DATASET.Std)
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        sim = outputs['similarity']
        prob = F.softmax(sim, dim=-1).cpu().numpy()[0]
        live_prob = prob[-1]
        spoof_prob = prob[0]

    # Generate heatmap based on method
    if method == 'gradcam':
        heatmap = get_gradcam(model, img_tensor.clone(), device)
        if heatmap is not None:
            heatmap = heatmap.squeeze()
        title = "Grad-CAM Heatmap"
    elif method == 'attention':
        attention_weights, _ = get_attention_map(model, img_tensor, device)
        heatmap = visualize_attention_map(img_display, attention_weights, cfg.MODEL.IMG_SIZE)
        title = "Attention Map"
    else:
        print(f"Unknown method: {method}")
        return None, None

    if heatmap is None:
        print("Failed to generate heatmap, using fallback")
        heatmap = get_simple_activation_map(model, img_tensor, device)
        title = "Activation Map (Fallback)"

    if heatmap is None:
        print("All visualization methods failed")
        return None, None

    # Resize image for display
    img_display = img_display.resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
    img_np = np.array(img_display)

    # Create overlay
    overlayed = overlay_heatmap(img_np, heatmap, alpha=0.5)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f'{title}\nLive: {live_prob:.3f} | Spoof: {spoof_prob:.3f}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    # Add prediction text
    pred_text = f"Prediction: {'LIVE' if live_prob > 0.5 else 'SPOOF'} (confidence: {max(live_prob, spoof_prob):.3f})"
    fig.suptitle(pred_text, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")

    plt.close()

    return heatmap, live_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Heatmap for MVP-FAS")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="Model name")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="Backbone")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_path", type=str, default="runs/visualize", help="Save directory")
    parser.add_argument("--method", type=str, default="gradcam", choices=['gradcam', 'attention'],
                        help="Visualization method")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.weights}")
    net = get_network(cfg=cfg, args=args, net_name=args.model, device=device, backbone=args.backbone)
    net = load_checkpoint(net, weight_path=args.weights)
    net.to(device)
    net.eval()
    print("Model loaded successfully")

    # Create visualization
    save_path = os.path.join(args.save_path, f"{os.path.basename(args.image).split('.')[0]}_heatmap.png")
    create_visualization(args.image, net, cfg, device, save_path, method=args.method)
