import os
import argparse
import pandas as pd
import glob
from visualize_heatmap import create_visualization
from configs.cfg import _C as cfg
from models.make_network import get_network, load_checkpoint
import torch
from tqdm import tqdm


def batch_visualize(args):
    """
    Batch visualize heatmaps for multiple images
    """
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

    # Get list of images
    if args.csv_file:
        # Load from CSV
        df = pd.read_csv(args.csv_file)
        if 'path' in df.columns:
            image_paths = df['path'].tolist()
        else:
            image_paths = df.iloc[:, 0].tolist()

        # Add root_dir if provided
        if args.root_dir:
            image_paths = [os.path.join(args.root_dir, p) for p in image_paths]
    else:
        # Load from directory
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

    # Limit number of images
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"Processing {len(image_paths)} images...")

    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # Process each image
    results = []
    for img_path in tqdm(image_paths, desc="Visualizing"):
        try:
            # Create save path
            img_name = os.path.basename(img_path).split('.')[0]
            save_path = os.path.join(args.save_path, f"{img_name}_heatmap.png")

            # Generate visualization
            heatmap, live_prob = create_visualization(
                img_path, net, cfg, device, save_path, method=args.method
            )

            results.append({
                'image': img_path,
                'live_prob': live_prob,
                'prediction': 'live' if live_prob > 0.5 else 'spoof',
                'heatmap_path': save_path
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'image': img_path,
                'live_prob': None,
                'prediction': 'error',
                'heatmap_path': None
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.save_path, 'visualization_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total images: {len(results)}")
    print(f"Live predictions: {sum(1 for r in results if r['prediction'] == 'live')}")
    print(f"Spoof predictions: {sum(1 for r in results if r['prediction'] == 'spoof')}")
    print(f"Errors: {sum(1 for r in results if r['prediction'] == 'error')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch Visualize Heatmaps")

    parser.add_argument("--model", type=str, default="MVP_FAS", help="Model name")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="Backbone")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--csv_file", type=str, help="CSV file with image paths")
    parser.add_argument("--image_dir", type=str, help="Directory with images")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory for CSV paths")
    parser.add_argument("--save_path", type=str, default="runs/visualize_batch", help="Save directory")
    parser.add_argument("--method", type=str, default="gradcam", choices=['gradcam', 'attention'],
                        help="Visualization method")
    parser.add_argument("--max_images", type=int, default=-1, help="Max number of images (-1 for all)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

    args = parser.parse_args()

    batch_visualize(args)
