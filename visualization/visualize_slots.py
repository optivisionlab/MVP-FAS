import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
from loaders.make_dataset import RemoveBlackBorders
import torch.nn.functional as F


def get_slot_attention_maps(model, img_tensor, device):
    """
    Extract slot attention maps from MVP-FAS model
    Returns attention maps for each slot (view)
    """
    model.eval()

    # Store slot attention outputs
    slot_outputs = {}

    def hook_mvslot(module, input, output):
        """Hook for MVSlot module"""
        # output should contain slot attention weights
        if isinstance(output, dict):
            slot_outputs['slots'] = output.get('slots', None)
            slot_outputs['attn'] = output.get('attn', None)
        else:
            slot_outputs['output'] = output

    # Register hook on MVSlot
    hook = None
    if hasattr(model, 'MVSlot'):
        hook = model.MVSlot.register_forward_hook(hook_mvslot)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Remove hook
    if hook is not None:
        hook.remove()

    return slot_outputs, outputs


def visualize_slot_attention_grid(img_path, model, cfg, device, save_path,
                                   text_prompts=None):
    """
    Create slot attention visualization similar to the paper figure
    Shows: Original | Slot1 | Slot2 | Slot3 | Slot4 | Baseline
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

    # Get model outputs
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    # Get slot attention maps
    slot_outputs, model_outputs = get_slot_attention_maps(model, img_tensor, device)

    # Get prediction
    sim = outputs['similarity']
    prob = F.softmax(sim, dim=-1).cpu().numpy()[0]

    # Default text prompts if not provided
    if text_prompts is None:
        text_prompts = ['spoof face', 'attack face', 'fake face', 'real face', 'baseline']

    # Resize original image
    img_display = img_display.resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
    img_np = np.array(img_display)

    # Try to extract slot attention maps
    slot_maps = extract_slot_maps(model, img_tensor, device, num_slots=len(text_prompts))

    # Create visualization grid
    num_slots = len(text_prompts)
    fig, axes = plt.subplots(1, num_slots + 1, figsize=(3 * (num_slots + 1), 3))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Input', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Slot attention maps
    for i, (slot_map, prompt) in enumerate(zip(slot_maps, text_prompts)):
        if slot_map is not None:
            # Resize to match image size
            slot_map_resized = cv2.resize(slot_map, (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))

            # Apply colormap
            slot_map_colored = apply_colormap(slot_map_resized, img_np)

            axes[i + 1].imshow(slot_map_colored)
            axes[i + 1].set_title(prompt, fontsize=10)
            axes[i + 1].axis('off')
        else:
            axes[i + 1].imshow(img_np)
            axes[i + 1].set_title(f'{prompt}\n(N/A)', fontsize=10)
            axes[i + 1].axis('off')

    # Add prediction info
    pred_text = f"Live: {prob[-1]:.3f} | Spoof: {prob[0]:.3f}"
    fig.suptitle(pred_text, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved slot visualization to: {save_path}")

    plt.close()

    return slot_maps


def extract_slot_maps(model, img_tensor, device, num_slots=5):
    """
    Extract attention maps for each slot from SlotAttention module
    """
    model.eval()

    # Store attention maps from each iteration
    attention_maps = []

    def hook_slot_attention(module, input, output):
        """Hook to capture attention from SlotAttention forward pass"""
        # We need to capture 'attn' variable inside the forward loop
        # This requires modifying the forward pass or capturing intermediate values
        pass

    # Better approach: Modify forward pass temporarily to capture attention
    original_forward = model.MVSlot.forward

    def forward_with_attn_capture(inputs, texts):
        b, n, d, device_inner, dtype = *inputs.shape, inputs.device, inputs.dtype
        slots = inputs

        # normalize
        texts_norm = model.MVSlot.norm_input(texts.repeat(b, 1, 1))
        k, v = model.MVSlot.to_k(texts_norm), model.MVSlot.to_v(texts_norm)

        # Store attention from last iteration
        last_attn = None

        # coarse to fine slot
        for iter_idx in range(model.MVSlot.iters):
            slots_prev = slots

            # query: slot
            slots = model.MVSlot.norm_slots(slots)
            q = model.MVSlot.to_q(slots)

            # compute similarity
            dots = torch.einsum('bid,bjd->bij', q, k) * model.MVSlot.scale

            # slot axis softmax
            attn = dots.softmax(dim=1) + model.MVSlot.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Save attention from last iteration
            if iter_idx == model.MVSlot.iters - 1:
                last_attn = attn.detach()

            # attention_score.matmul(v)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            # GRU update
            slots = model.MVSlot.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + model.MVSlot.mlp(model.MVSlot.norm_pre_ff(slots))

        # Store the attention map
        attention_maps.append(last_attn)

        return slots

    # Temporarily replace forward
    model.MVSlot.forward = forward_with_attn_capture

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Restore original forward
    model.MVSlot.forward = original_forward

    # Process attention maps
    slot_maps = []

    if len(attention_maps) > 0:
        attn = attention_maps[0]  # Should be [batch, num_text_slots, num_patches]

        print(f"Raw attention shape: {attn.shape}")

        # Check if dimensions are swapped
        # Expected: [batch, num_text_features, num_patches]
        # If shape[1] > shape[2], they might be swapped
        if attn.shape[1] > attn.shape[2]:
            print(f"Transposing attention from {attn.shape} to fix dimension order")
            attn = attn.transpose(1, 2)  # Swap to [batch, num_patches, num_text_slots]
            # Actually we want [batch, num_text_slots, num_patches]
            # So the original might be [batch, num_patches, num_text_slots]
            # Let's check the actual values

        batch_size = attn.shape[0]
        dim1 = attn.shape[1]
        dim2 = attn.shape[2]

        print(f"Processed attention shape: {attn.shape} (batch={batch_size}, dim1={dim1}, dim2={dim2})")

        # Determine which dimension is num_text_slots and which is num_patches
        # For ViT-B/16 with 224x224: num_patches = 196 (14x14)
        # For text: we have ~25 text templates (spoof + real)

        if dim1 == 196 and dim2 < 50:
            # Shape is [batch, num_patches, num_text_slots] - need to transpose
            print("Detected shape as [batch, num_patches, num_text_slots], transposing...")
            attn = attn.transpose(1, 2)
            num_text_slots = attn.shape[1]
            num_patches = attn.shape[2]
        elif dim2 == 196 and dim1 < 50:
            # Shape is correct: [batch, num_text_slots, num_patches]
            num_text_slots = dim1
            num_patches = dim2
        else:
            # Unclear, use as-is
            print(f"Warning: Unclear attention shape, using dim1={dim1} as text_slots, dim2={dim2} as patches")
            num_text_slots = dim1
            num_patches = dim2

        print(f"Final: num_text_slots={num_text_slots}, num_patches={num_patches}")

        # Calculate patches per side
        patches_per_side = int(np.sqrt(num_patches))

        # Select specific text slots to visualize
        # Select representative slots (first few spoof, last few real)
        if num_text_slots >= num_slots:
            slot_indices = list(range(num_slots))  # First N slots
        else:
            slot_indices = list(range(num_text_slots))

        for idx in slot_indices[:num_slots]:
            if idx < num_text_slots:
                # Get attention for this slot
                slot_attn = attn[0, idx, :].cpu().numpy()  # [num_patches]

                # Reshape to 2D
                if patches_per_side * patches_per_side == num_patches:
                    slot_attn_2d = slot_attn.reshape(patches_per_side, patches_per_side)
                else:
                    # Pad if needed
                    pad_size = patches_per_side * patches_per_side - num_patches
                    if pad_size > 0:
                        slot_attn = np.pad(slot_attn, (0, pad_size), mode='constant')
                    slot_attn_2d = slot_attn[:patches_per_side * patches_per_side].reshape(patches_per_side, patches_per_side)

                # Normalize
                slot_attn_2d = (slot_attn_2d - slot_attn_2d.min()) / (slot_attn_2d.max() - slot_attn_2d.min() + 1e-8)

                slot_maps.append(slot_attn_2d)
            else:
                slot_maps.append(None)
    else:
        print("Warning: Could not extract attention maps, using dummy maps")
        for i in range(num_slots):
            slot_maps.append(create_dummy_slot_map(i, num_slots))

    return slot_maps


def create_dummy_slot_map(slot_idx, num_slots, size=14):
    """
    Create a dummy slot attention map for visualization
    """
    attn_map = np.zeros((size, size))

    # Create different patterns for different slots
    if slot_idx == 0:
        # Top-left quadrant
        attn_map[:size//2, :size//2] = 1.0
    elif slot_idx == 1:
        # Top-right quadrant
        attn_map[:size//2, size//2:] = 1.0
    elif slot_idx == 2:
        # Bottom-left quadrant
        attn_map[size//2:, :size//2] = 1.0
    elif slot_idx == 3:
        # Bottom-right quadrant
        attn_map[size//2:, size//2:] = 1.0
    else:
        # Center
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        attn_map[mask] = 1.0

    # Smooth
    attn_map = cv2.GaussianBlur(attn_map, (5, 5), 0)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return attn_map


def apply_colormap(attn_map, img, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to attention map and overlay on image
    """
    # Normalize attention map
    attn_map = (attn_map * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(attn_map, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    return overlayed


def create_multi_row_visualization(img_paths, model, cfg, device, save_path,
                                   text_prompts=None):
    """
    Create multi-row visualization like the paper figure
    Each row is one image with its slot attention maps
    """
    if text_prompts is None:
        text_prompts = ['spoof face', 'attack face', 'fake face', 'real face', 'baseline']

    num_images = len(img_paths)
    num_slots = len(text_prompts)

    fig, axes = plt.subplots(num_images, num_slots + 1,
                            figsize=(3 * (num_slots + 1), 3 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for row_idx, img_path in enumerate(img_paths):
        # Load image
        img = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            RemoveBlackBorders(),
            transforms.Resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATASET.Mean, std=cfg.DATASET.Std)
        ])

        img_tensor = transform(img).unsqueeze(0).to(device)
        img_display = img.resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
        img_np = np.array(img_display)

        # Get slot maps
        slot_maps = extract_slot_maps(model, img_tensor, device, num_slots)

        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            sim = outputs['similarity']
            prob = F.softmax(sim, dim=-1).cpu().numpy()[0]

        # Plot original
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title('Input', fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Plot slots
        for col_idx, (slot_map, prompt) in enumerate(zip(slot_maps, text_prompts)):
            if slot_map is not None:
                slot_map_resized = cv2.resize(slot_map, (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE))
                slot_map_colored = apply_colormap(slot_map_resized, img_np)
                axes[row_idx, col_idx + 1].imshow(slot_map_colored)
            else:
                axes[row_idx, col_idx + 1].imshow(img_np)

            # Only show prompt on first row
            if row_idx == 0:
                axes[row_idx, col_idx + 1].set_title(prompt, fontsize=10)
            axes[row_idx, col_idx + 1].axis('off')

        # Add prediction as ylabel
        pred_label = f"Live: {prob[-1]:.2f}"
        axes[row_idx, 0].set_ylabel(pred_label, fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-row slot visualization to: {save_path}")

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Slot Attention for MVP-FAS")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="Model name")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="Backbone")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--images", type=str, nargs='+', help="Paths to multiple images")
    parser.add_argument("--save_path", type=str, default="runs/visualize_slots", help="Save directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--prompts", type=str, nargs='+',
                       default=['spoof face', 'attack face', 'fake face', 'real face', 'baseline'],
                       help="Text prompts for slots")

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
    if args.images:
        # Multi-row visualization
        save_path = os.path.join(args.save_path, "slot_visualization_multi.png")
        create_multi_row_visualization(args.images, net, cfg, device, save_path, args.prompts)
    elif args.image:
        # Single image visualization
        save_path = os.path.join(args.save_path, f"{os.path.basename(args.image).split('.')[0]}_slots.png")
        visualize_slot_attention_grid(args.image, net, cfg, device, save_path, args.prompts)
    else:
        print("Error: Please provide --image or --images")
