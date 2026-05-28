import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
from loaders.make_dataset import RemoveBlackBorders
import torch.nn.functional as F


def analyze_image_prediction(img_path, model, cfg, device):
    """
    Detailed analysis of why model predicts live or spoof
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {img_path}")
    print(f"{'='*80}")

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')

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

    # Get predictions
    sim = outputs['similarity']
    prob = F.softmax(sim, dim=-1).cpu().numpy()[0]

    live_prob = prob[-1]
    spoof_prob = prob[0]

    prediction = "LIVE" if live_prob > 0.5 else "SPOOF"
    confidence = max(live_prob, spoof_prob)

    print(f"\n📊 PREDICTION RESULTS:")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Live probability: {live_prob:.4f}")
    print(f"  Spoof probability: {spoof_prob:.4f}")
    print(f"  Margin: {abs(live_prob - spoof_prob):.4f}")

    # Extract slot attention for analysis
    attention_maps = []

    original_forward = model.MVSlot.forward

    def forward_with_attn_capture(inputs, texts):
        b, n, d, device_inner, dtype = *inputs.shape, inputs.device, inputs.dtype
        slots = inputs

        texts_norm = model.MVSlot.norm_input(texts.repeat(b, 1, 1))
        k, v = model.MVSlot.to_k(texts_norm), model.MVSlot.to_v(texts_norm)

        last_attn = None

        for iter_idx in range(model.MVSlot.iters):
            slots_prev = slots
            slots = model.MVSlot.norm_slots(slots)
            q = model.MVSlot.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * model.MVSlot.scale
            attn = dots.softmax(dim=1) + model.MVSlot.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            if iter_idx == model.MVSlot.iters - 1:
                last_attn = attn.detach()

            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = model.MVSlot.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            slots = slots.reshape(b, -1, d)
            slots = slots + model.MVSlot.mlp(model.MVSlot.norm_pre_ff(slots))

        attention_maps.append(last_attn)
        return slots

    model.MVSlot.forward = forward_with_attn_capture

    with torch.no_grad():
        outputs = model(img_tensor)

    model.MVSlot.forward = original_forward

    # Analyze attention patterns
    if len(attention_maps) > 0:
        attn = attention_maps[0]

        if attn.shape[1] > attn.shape[2]:
            attn = attn.transpose(1, 2)

        # attn shape: [batch, num_text_slots, num_patches]
        num_text_slots = attn.shape[1]
        num_patches = attn.shape[2]

        print(f"\n🔍 ATTENTION ANALYSIS:")
        print(f"  Number of text slots: {num_text_slots}")
        print(f"  Number of patches: {num_patches}")

        # Analyze attention distribution for each slot
        attn_np = attn[0].cpu().numpy()  # [num_text_slots, num_patches]

        # Calculate statistics for each slot
        print(f"\n📈 ATTENTION STATISTICS PER SLOT:")
        print(f"  {'Slot':<5} {'Mean':<10} {'Std':<10} {'Max':<10} {'Min':<10}")
        print(f"  {'-'*50}")

        for i in range(min(10, num_text_slots)):  # Show first 10 slots
            slot_attn = attn_np[i]
            mean_attn = slot_attn.mean()
            std_attn = slot_attn.std()
            max_attn = slot_attn.max()
            min_attn = slot_attn.min()

            print(f"  {i:<5} {mean_attn:<10.6f} {std_attn:<10.6f} {max_attn:<10.6f} {min_attn:<10.6f}")

        # Compare spoof vs real attention
        # Assuming first ~15 slots are spoof, last ~10 are real
        spoof_slots = attn_np[:15]
        real_slots = attn_np[15:]

        spoof_mean = spoof_slots.mean()
        real_mean = real_slots.mean()

        print(f"\n🎯 SPOOF vs REAL ATTENTION:")
        print(f"  Spoof slots (0-14) mean attention: {spoof_mean:.6f}")
        print(f"  Real slots (15-24) mean attention: {real_mean:.6f}")
        print(f"  Ratio (Real/Spoof): {real_mean/spoof_mean:.4f}")

        # Analyze attention concentration
        # High concentration = focused on specific patches
        # Low concentration = distributed across image
        entropy = []
        for i in range(num_text_slots):
            slot_attn = attn_np[i]
            # Normalize to probability distribution
            slot_prob = slot_attn / (slot_attn.sum() + 1e-8)
            # Calculate entropy
            ent = -np.sum(slot_prob * np.log(slot_prob + 1e-8))
            entropy.append(ent)

        avg_entropy = np.mean(entropy)
        print(f"\n🌀 ATTENTION CONCENTRATION:")
        print(f"  Average entropy: {avg_entropy:.4f}")
        print(f"  (Lower = more focused, Higher = more distributed)")

        # Find most attended patches
        total_attn = attn_np.sum(axis=0)  # Sum across all slots
        top_patches = np.argsort(total_attn)[-10:][::-1]  # Top 10 patches

        print(f"\n🎯 TOP 10 MOST ATTENDED PATCHES:")
        patches_per_side = int(np.sqrt(num_patches))
        for rank, patch_idx in enumerate(top_patches, 1):
            row = patch_idx // patches_per_side
            col = patch_idx % patches_per_side
            attn_value = total_attn[patch_idx]
            print(f"  {rank}. Patch {patch_idx} (row={row}, col={col}): {attn_value:.6f}")

    return {
        'prediction': prediction,
        'confidence': confidence,
        'live_prob': live_prob,
        'spoof_prob': spoof_prob,
        'margin': abs(live_prob - spoof_prob)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze MVP-FAS Predictions")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="Model name")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="Backbone")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Images to analyze")
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

    # Analyze each image
    results = []
    for img_path in args.images:
        result = analyze_image_prediction(img_path, net, cfg, device)
        results.append((img_path, result))

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Image':<40} {'Prediction':<10} {'Confidence':<12} {'Live Prob':<12} {'Spoof Prob':<12}")
    print(f"{'-'*90}")
    for img_path, result in results:
        img_name = os.path.basename(img_path)
        print(f"{img_name:<40} {result['prediction']:<10} {result['confidence']:<12.4f} {result['live_prob']:<12.4f} {result['spoof_prob']:<12.4f}")

    print(f"\n{'='*80}")
