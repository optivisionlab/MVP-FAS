"""
Phân tích sự khác biệt giữa dữ liệu Selfie và Reddit
Giúp hiểu tại sao model nhận nhầm spoof thành live khi đổi dữ liệu training
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Thêm path để import modules
sys.path.append('/data/fas/solution/MVP-FAS')


class DataDistributionAnalyzer:
    def __init__(self, base_dir='/data/fas', output_dir='./analysis_output'):
        self.base_dir = base_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define dataset paths
        self.datasets = {
            'reddit_part1': {
                'csv': '/data/fas/csv/vft/reddis_image_260611_live.csv',
                'name': 'Reddit Part1',
                'type': 'reddit1'
            },
            'reddit_part2': {
                'csv': '/data/fas/csv/vft/reddis_image_260615_live.csv',
                'name': 'Reddit Part2',
                'type': 'reddit2'
            },
            'ffhq': {
                'csv': '/data/fas/csv/ffhq/FFHQ_live_260525.csv',
                'name': 'FFHQ',
                'type': 'ffhq'
            },
            'axonlab_selfie': {
                'csv': '/data/fas/csv/axonlab/axonlab_live_release_260320_images.csv',
                'name': 'Axonlab Selfie',
                'type': 'axonlab_live'
            },
            'vft_live_260312': {
                'csv': '/data/fas/csv/vft/vft_images_live_260312.csv',
                'name': 'Axonlab Selfie',
                'type': 'vft_live_260312'
            },
            'vft_live': {
                'csv': '/data/fas/csv/vft/full-vft-live-images.csv',
                'name': 'Axonlab Selfie',
                'type': 'vft_live_full'
            }
        }

    def load_image(self, img_path):
        """Load và xử lý ảnh"""
        full_path = os.path.join(self.base_dir, img_path)
        img = cv2.imread(full_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def compute_image_quality_metrics(self, img):
        """Tính các metric về chất lượng ảnh"""
        if img is None:
            return None

        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        metrics = {}

        # 1. Sharpness (Laplacian variance)
        metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Brightness (mean pixel value)
        metrics['brightness'] = gray.mean()

        # 3. Contrast (std of pixel values)
        metrics['contrast'] = gray.std()

        # 4. Dynamic range
        metrics['dynamic_range'] = gray.max() - gray.min()

        # 5. Edge density (Canny edges)
        edges = cv2.Canny(gray, 100, 200)
        metrics['edge_density'] = np.sum(edges > 0) / edges.size

        # 6. Noise estimation (high frequency content)
        # Using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['noise_estimate'] = np.abs(laplacian).mean()

        # 7. Color saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        metrics['saturation_mean'] = hsv[:,:,1].mean()
        metrics['saturation_std'] = hsv[:,:,1].std()

        # 8. Histogram entropy (texture complexity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        metrics['entropy'] = -np.sum(hist * np.log2(hist))

        # 9. JPEG artifacts detection (block artifacts)
        # Compute frequency of 8x8 block patterns
        dct_blocks = []
        h, w = gray.shape
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct = cv2.dct(block)
                dct_blocks.append(np.abs(dct).flatten())

        if len(dct_blocks) > 0:
            dct_blocks = np.array(dct_blocks)
            # High frequency components indicate artifacts
            metrics['jpeg_artifact_score'] = dct_blocks[:, 32:].mean()
        else:
            metrics['jpeg_artifact_score'] = 0

        # 10. Resolution
        metrics['resolution'] = img.shape[0] * img.shape[1]
        metrics['width'] = img.shape[1]
        metrics['height'] = img.shape[0]
        metrics['aspect_ratio'] = img.shape[1] / img.shape[0]

        return metrics

    def extract_deep_features(self, img, target_size=(224, 224)):
        """Extract simple deep features using histogram and spatial info"""
        if img is None:
            return None

        # Resize
        img_resized = cv2.resize(img, target_size)

        # Compute color histogram features
        features = []
        for channel in range(3):
            hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
            features.extend(hist.flatten())

        # Add gradient features
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient histogram
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist, _ = np.histogram(grad_mag, bins=32, range=(0, 256))
        features.extend(grad_hist)

        # Flatten and normalize
        features = np.array(features, dtype=np.float32)
        if features.sum() > 0:
            features = features / features.sum()

        return features

    def analyze_dataset(self, dataset_key, sample_size=500):
        """Phân tích một dataset"""
        print(f"\n{'='*60}")
        print(f"Analyzing {self.datasets[dataset_key]['name']}...")
        print(f"{'='*60}")

        csv_path = self.datasets[dataset_key]['csv']
        df = pd.read_csv(csv_path)

        # Sample if dataset is too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        results = []
        features_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            img_path = row['path']
            img = self.load_image(img_path)

            if img is None:
                continue

            # Compute quality metrics
            metrics = self.compute_image_quality_metrics(img)
            if metrics is None:
                continue

            metrics['dataset'] = self.datasets[dataset_key]['name']
            metrics['type'] = self.datasets[dataset_key]['type']
            results.append(metrics)

            # Extract deep features
            features = self.extract_deep_features(img)
            if features is not None:
                features_list.append(features)

        print(f"Processed {len(results)} images successfully")
        return pd.DataFrame(results), np.array(features_list)

    def compare_distributions(self, df_all):
        """So sánh distributions giữa các datasets"""
        print(f"\n{'='*60}")
        print("Comparing Distributions...")
        print(f"{'='*60}")

        # Define metrics to compare
        metrics = ['sharpness', 'brightness', 'contrast', 'edge_density',
                   'saturation_mean', 'entropy', 'jpeg_artifact_score', 'noise_estimate']

        # Create subplots
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Plot distribution for each dataset type
            for dtype in df_all['type'].unique():
                data = df_all[df_all['type'] == dtype][metric]
                ax.hist(data, alpha=0.5, label=dtype, bins=30, density=True)

            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: distribution_comparison.png")

    def plot_statistical_comparison(self, df_all):
        """Vẽ boxplot so sánh statistical"""
        metrics = ['sharpness', 'brightness', 'contrast', 'edge_density',
                   'saturation_mean', 'entropy', 'jpeg_artifact_score']

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            sns.boxplot(data=df_all, x='type', y=metric, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boxplot_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: boxplot_comparison.png")

    def visualize_feature_space(self, features_dict):
        """Visualize feature space using t-SNE"""
        print(f"\n{'='*60}")
        print("Computing t-SNE visualization...")
        print(f"{'='*60}")

        # Combine all features
        all_features = []
        all_labels = []

        for dataset_key, features in features_dict.items():
            all_features.append(features)
            all_labels.extend([self.datasets[dataset_key]['type']] * len(features))

        all_features = np.vstack(all_features)

        # Apply PCA first to reduce dimensionality
        print("Applying PCA...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(all_features)

        # Apply t-SNE
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_tsne = tsne.fit_transform(features_pca)

        # Plot
        plt.figure(figsize=(12, 8))

        for dtype in np.unique(all_labels):
            mask = np.array(all_labels) == dtype
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                       label=dtype, alpha=0.6, s=30)

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of Image Features\n(Selfie vs Reddit vs Synthetic)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: tsne_visualization.png")

    def generate_report(self, df_all):
        """Generate statistical report"""
        print(f"\n{'='*60}")
        print("Generating Statistical Report...")
        print(f"{'='*60}")

        report_path = os.path.join(self.output_dir, 'ANALYSIS_REPORT.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Phân tích Distribution: Selfie vs Reddit vs Synthetic\n\n")
            f.write("## Tổng quan\n\n")

            # Count per type
            f.write("### Số lượng mẫu phân tích:\n\n")
            for dtype in df_all['type'].unique():
                count = len(df_all[df_all['type'] == dtype])
                f.write(f"- **{dtype}**: {count} ảnh\n")

            f.write("\n## So sánh các chỉ số chất lượng\n\n")

            metrics = ['sharpness', 'brightness', 'contrast', 'edge_density',
                       'saturation_mean', 'entropy', 'jpeg_artifact_score', 'noise_estimate']

            for metric in metrics:
                f.write(f"### {metric.replace('_', ' ').title()}\n\n")
                f.write("| Dataset Type | Mean | Std | Min | Max |\n")
                f.write("|-------------|------|-----|-----|-----|\n")

                for dtype in df_all['type'].unique():
                    data = df_all[df_all['type'] == dtype][metric]
                    f.write(f"| {dtype} | {data.mean():.2f} | {data.std():.2f} | "
                           f"{data.min():.2f} | {data.max():.2f} |\n")
                f.write("\n")

            f.write("\n## Phân tích và Kết luận\n\n")

            # Compare selfie vs reddit
            selfie_df = df_all[df_all['type'] == 'selfie']
            reddit_df = df_all[df_all['type'] == 'reddit']

            f.write("### Sự khác biệt chính giữa Selfie và Reddit:\n\n")

            # Sharpness
            selfie_sharp = selfie_df['sharpness'].mean()
            reddit_sharp = reddit_df['sharpness'].mean()
            diff_pct = abs(selfie_sharp - reddit_sharp) / selfie_sharp * 100
            f.write(f"1. **Độ nét (Sharpness)**:\n")
            f.write(f"   - Selfie: {selfie_sharp:.2f}\n")
            f.write(f"   - Reddit: {reddit_sharp:.2f}\n")
            f.write(f"   - Chênh lệch: {diff_pct:.1f}%\n\n")

            # Brightness
            selfie_bright = selfie_df['brightness'].mean()
            reddit_bright = reddit_df['brightness'].mean()
            diff_pct = abs(selfie_bright - reddit_bright) / selfie_bright * 100
            f.write(f"2. **Độ sáng (Brightness)**:\n")
            f.write(f"   - Selfie: {selfie_bright:.2f}\n")
            f.write(f"   - Reddit: {reddit_bright:.2f}\n")
            f.write(f"   - Chênh lệch: {diff_pct:.1f}%\n\n")

            # JPEG artifacts
            selfie_jpeg = selfie_df['jpeg_artifact_score'].mean()
            reddit_jpeg = reddit_df['jpeg_artifact_score'].mean()
            diff_pct = abs(selfie_jpeg - reddit_jpeg) / max(selfie_jpeg, 0.01) * 100
            f.write(f"3. **JPEG Artifacts**:\n")
            f.write(f"   - Selfie: {selfie_jpeg:.2f}\n")
            f.write(f"   - Reddit: {reddit_jpeg:.2f}\n")
            f.write(f"   - Chênh lệch: {diff_pct:.1f}%\n\n")

            f.write("### Giả thuyết về nguyên nhân model nhận nhầm:\n\n")
            f.write("1. **Distribution Shift**: Dữ liệu Reddit có distribution khác biệt đáng kể so với selfie\n")
            f.write("2. **Quality Degradation**: Ảnh Reddit thường có chất lượng thấp hơn, giống với một số loại spoof\n")
            f.write("3. **Artifact Patterns**: Compression artifacts khác nhau có thể confuse model\n")
            f.write("4. **Diversity**: Reddit data đa dạng hơn → model khó học pattern rõ ràng\n\n")

            f.write("### Đề xuất:\n\n")
            f.write("1. **Mix Data**: Sử dụng cả selfie và Reddit trong training\n")
            f.write("2. **Data Augmentation**: Augment mạnh hơn để giảm gap\n")
            f.write("3. **Domain Adaptation**: Sử dụng techniques như domain adversarial training\n")
            f.write("4. **Quality Filtering**: Lọc ảnh Reddit có quality quá thấp\n")

        print(f"✓ Saved: ANALYSIS_REPORT.md")
        print(f"\nReport saved to: {report_path}")

    def run_full_analysis(self, sample_size=500):
        """Run toàn bộ pipeline phân tích"""
        print("\n" + "="*60)
        print("STARTING FULL DATA DISTRIBUTION ANALYSIS")
        print("="*60)

        # Analyze each dataset
        all_dfs = []
        features_dict = {}

        for dataset_key in self.datasets.keys():
            df, features = self.analyze_dataset(dataset_key, sample_size)
            all_dfs.append(df)
            features_dict[dataset_key] = features

        # Combine all dataframes
        df_all = pd.concat(all_dfs, ignore_index=True)

        # Save combined data
        csv_path = os.path.join(self.output_dir, 'combined_metrics.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"\n✓ Saved combined metrics to: {csv_path}")

        # Generate visualizations
        self.compare_distributions(df_all)
        self.plot_statistical_comparison(df_all)
        self.visualize_feature_space(features_dict)

        # Generate report
        self.generate_report(df_all)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - distribution_comparison.png")
        print("  - boxplot_comparison.png")
        print("  - tsne_visualization.png")
        print("  - ANALYSIS_REPORT.md")
        print("  - combined_metrics.csv")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze data distribution between Selfie and Reddit datasets')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                       help='Output directory for analysis results')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='Number of samples to analyze per dataset')

    args = parser.parse_args()

    # Run analysis
    analyzer = DataDistributionAnalyzer(output_dir=args.output_dir)
    analyzer.run_full_analysis(sample_size=args.sample_size)
