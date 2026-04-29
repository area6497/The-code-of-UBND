# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from torch.nn import functional as F

# def main():
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 1. Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 2. Load test dataset
#     test_dir = "E:/hutao/demo/Awesome-Backbones-main/datasets/test"
#     if not os.path.exists(test_dir):
#         print(f"Error: Test dataset path {test_dir} does not exist")
#         return
    
#     test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=0
#     )
#     class_names = test_dataset.classes
#     print(f"Classes: {class_names}, Number of samples: {len(test_dataset)}")

#     # 3. Load model and extract features
#     model = models.mobilenet_v2(pretrained=False)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
#     model = model.to(device)

#     checkpoint = torch.load("Val_Epoch088-Acc98.438.pth", map_location=device)
#     state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
#     new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')} or state_dict
#     model.load_state_dict(new_state_dict, strict=False)
#     model.eval()
#     print("Model loaded successfully")

#     # 4. Feature extraction
#     features_list = []
#     labels_list = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             x = model.features(images)
#             x = F.adaptive_avg_pool2d(x, (1, 1))
#             x = torch.flatten(x, 1)
#             x = model.classifier[0](x)
#             features = F.relu(x).cpu().numpy()
#             features_list.append(features)
#             labels_list.append(labels.numpy())

#     all_features = np.vstack(features_list)
#     all_labels = np.hstack(labels_list)
#     print(f"Feature shape: {all_features.shape}")

#     # 5. Feature optimization + LDA dimensionality reduction
#     scaler = StandardScaler()
#     all_features = scaler.fit_transform(all_features)
    
#     lda = LDA(n_components=1)
#     features_lda = lda.fit_transform(all_features, all_labels)
#     print(f"LDA reduced shape: {features_lda.shape}")

#     # 6. Subplot layout: scatter plot (top) + distribution histogram (bottom)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})
#     colors = ['#3498db', '#e74c3c']
#     markers = ['o', '^']
#     bins = 30

#     # 6.1 Top: scatter plot (class-specific regions)
#     for i, cls_name in enumerate(class_names):
#         mask = all_labels == i
#         y_jitter = np.random.normal(i * 0.5, 0.1, size=mask.sum())
#         ax1.scatter(
#             features_lda[mask],
#             y_jitter,
#             c=colors[i],
#             label=cls_name,
#             alpha=0.7,
#             s=80,
#             marker=markers[i],
#             edgecolors='black',
#             linewidths=0.5
#         )
#     ax1.set_title('Breast Ultrasound Features LDA Visualization', fontsize=16, fontweight='bold')
#     ax1.set_xlabel('LDA Feature (Maximizing Class Separation)', fontsize=14)
#     ax1.set_ylabel('Class-specific Region', fontsize=14)
#     ax1.legend(fontsize=12, loc='upper right')
#     ax1.grid(alpha=0.2)

#     # 6.2 Bottom: histogram (distribution visualization)
#     for i, cls_name in enumerate(class_names):
#         mask = all_labels == i
#         ax2.hist(
#             features_lda[mask],
#             bins=bins,
#             alpha=0.5,
#             color=colors[i],
#             density=True,
#             label=f'{cls_name} distribution',
#             edgecolor='black',
#             linewidth=0.8
#         )
#     ax2.set_title('Feature Distribution Histogram', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('LDA Feature Value', fontsize=12)
#     ax2.set_ylabel('Probability Density', fontsize=12)
#     ax2.legend(fontsize=11)
#     ax2.grid(alpha=0.2)

#     # Adjust layout
#     plt.tight_layout()
#     plt.savefig('LDA_visualization_scatter+distribution.png', dpi=300, bbox_inches='tight')
#     print("Optimized visualization saved")
#     plt.show()

#     # Quantitative evaluation
#     inter_dist = np.linalg.norm(lda.means_[0] - lda.means_[1])
#     intra_dist = np.mean([np.std(features_lda[all_labels == i]) for i in range(2)])
#     print(f"\nQuantitative evaluation:")
#     print(f"Inter-class distance: {inter_dist:.4f}")
#     print(f"Average intra-class standard deviation: {intra_dist:.4f}")
#     print(f"Separability index (inter/intra): {inter_dist/intra_dist:.4f}")

# if __name__ == '__main__':
#     main()





import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.nn import functional as F

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load test dataset
    test_dir = "E:/hutao/demo/Awesome-Backbones-main/datasets/test"
    if not os.path.exists(test_dir):
        print(f"Error: Test dataset path {test_dir} does not exist")
        return
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    class_names = test_dataset.classes
    print(f"Classes: {class_names}, Number of samples: {len(test_dataset)}")

    # 3. Load model and extract features
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
    model = model.to(device)

    checkpoint = torch.load("Val_Epoch088-Acc98.438.pth", map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
    new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')} or state_dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded successfully")

    # 4. Feature extraction
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            x = model.features(images)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = model.classifier[0](x)
            features = F.relu(x).cpu().numpy()
            features_list.append(features)
            labels_list.append(labels.numpy())

    all_features = np.vstack(features_list)
    all_labels = np.hstack(labels_list)
    print(f"Feature shape: {all_features.shape}")

    # 5. Feature optimization + LDA dimensionality reduction
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    lda = LDA(n_components=1)
    features_lda = lda.fit_transform(all_features, all_labels)
    print(f"LDA reduced shape: {features_lda.shape}")

    # Colors and style
    colors = ['#3498db', '#e74c3c']
    markers = ['o', '^']
    bins = 30

    # ================= 图1：散点图 =================
    plt.figure(figsize=(10, 6))
    for i, cls_name in enumerate(class_names):
        mask = all_labels == i
        y_jitter = np.random.normal(i * 0.5, 0.1, size=mask.sum())
        plt.scatter(
            features_lda[mask],
            y_jitter,
            c=colors[i],
            label=cls_name,
            alpha=0.7,
            s=80,
            marker=markers[i],
            edgecolors='black',
            linewidths=0.5
        )
    plt.title('', fontsize=16, fontweight='bold')
    plt.xlabel('LDA Feature', fontsize=14)
    plt.ylabel('Class-specific Region', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('LDA_visualization_scatter.png', dpi=300, bbox_inches='tight')
    print("图1：散点图已保存为 LDA_visualization_scatter.png")
    plt.show()

    # ================= 图2：分布直方图 =================
    plt.figure(figsize=(10, 6))
    for i, cls_name in enumerate(class_names):
        mask = all_labels == i
        plt.hist(
            features_lda[mask],
            bins=bins,
            alpha=0.5,
            color=colors[i],
            density=True,
            label=f'{cls_name} distribution',
            edgecolor='black',
            linewidth=0.8
        )
    plt.title('', fontsize=14, fontweight='bold')
    plt.xlabel('LDA Feature Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('LDA_visualization_distribution.png', dpi=300, bbox_inches='tight')
    print("图2：直方图已保存为 LDA_visualization_distribution.png")
    plt.show()

    # Quantitative evaluation
    inter_dist = np.linalg.norm(lda.means_[0] - lda.means_[1])
    intra_dist = np.mean([np.std(features_lda[all_labels == i]) for i in range(2)])
    print(f"\nQuantitative evaluation:")
    print(f"Inter-class distance: {inter_dist:.4f}")
    print(f"Average intra-class standard deviation: {intra_dist:.4f}")
    print(f"Separability index (inter/intra): {inter_dist/intra_dist:.4f}")

if __name__ == '__main__':
    main()






# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from torch.nn import functional as F

# def main():
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 1. Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # 2. Load test dataset
#     test_dir = "E:/hutao/demo/Awesome-Backbones-main/1/datasets/test"
#     if not os.path.exists(test_dir):
#         print(f"Error: Test dataset path {test_dir} does not exist")
#         return
    
#     test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=0
#     )
#     class_names = test_dataset.classes
#     print(f"Classes: {class_names}, Number of samples: {len(test_dataset)}")

#     # 3. Load model and extract features
#     model = models.mobilenet_v2(pretrained=False)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
#     model = model.to(device)

#     checkpoint = torch.load("Val_Epoch114-Acc95.703.pth", map_location=device)
#     state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint.state_dict()
#     new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')} or state_dict
#     model.load_state_dict(new_state_dict, strict=False)
#     model.eval()
#     print("Model loaded successfully")

#     # 4. Feature extraction
#     features_list = []
#     labels_list = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             x = model.features(images)
#             x = F.adaptive_avg_pool2d(x, (1, 1))
#             x = torch.flatten(x, 1)
#             x = model.classifier[0](x)
#             features = F.relu(x).cpu().numpy()
#             features_list.append(features)
#             labels_list.append(labels.numpy())

#     all_features = np.vstack(features_list)
#     all_labels = np.hstack(labels_list)
#     print(f"Feature shape: {all_features.shape}")

#     # 5. Feature optimization + LDA dimensionality reduction
#     scaler = StandardScaler()
#     all_features = scaler.fit_transform(all_features)
    
#     lda = LDA(n_components=1)
#     features_lda = lda.fit_transform(all_features, all_labels)
#     print(f"LDA reduced shape: {features_lda.shape}")

#     # Colors and style
#     colors = ['#3498db', '#e74c3c']
#     markers = ['o', '^']
#     bins = 30

#     # ================= 图1：散点图 =================
#     plt.figure(figsize=(10, 6))
#     for i, cls_name in enumerate(class_names):
#         mask = all_labels == i
#         y_jitter = np.random.normal(i * 0.5, 0.1, size=mask.sum())
#         plt.scatter(
#             features_lda[mask],
#             y_jitter,
#             c=colors[i],
#             label=cls_name,
#             alpha=0.7,
#             s=80,
#             marker=markers[i],
#             edgecolors='black',
#             linewidths=0.5
#         )
#     plt.title('', fontsize=16, fontweight='bold')
#     plt.xlabel('LDA Feature', fontsize=14)
#     plt.ylabel('Class-specific Region', fontsize=14)
#     plt.legend(fontsize=12, loc='upper right')
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.savefig('LDA_visualization_scatter_BUSI.png', dpi=300, bbox_inches='tight')
#     print("图1：散点图已保存为 LDA_visualization_scatter.png")
#     plt.show()

#     # ================= 图2：分布直方图 =================
#     plt.figure(figsize=(10, 6))
#     for i, cls_name in enumerate(class_names):
#         mask = all_labels == i
#         plt.hist(
#             features_lda[mask],
#             bins=bins,
#             alpha=0.5,
#             color=colors[i],
#             density=True,
#             label=f'{cls_name} distribution',
#             edgecolor='black',
#             linewidth=0.8
#         )
#     plt.title('', fontsize=14, fontweight='bold')
#     plt.xlabel('LDA Feature Value', fontsize=12)
#     plt.ylabel('Probability Density', fontsize=12)
#     plt.legend(fontsize=11)
#     plt.grid(alpha=0.2)
#     plt.tight_layout()
#     plt.savefig('LDA_visualization_distribution_BUSI.png', dpi=300, bbox_inches='tight')
#     print("图2：直方图已保存为 LDA_visualization_distribution.png")
#     plt.show()

#     # Quantitative evaluation
#     inter_dist = np.linalg.norm(lda.means_[0] - lda.means_[1])
#     intra_dist = np.mean([np.std(features_lda[all_labels == i]) for i in range(2)])
#     print(f"\nQuantitative evaluation:")
#     print(f"Inter-class distance: {inter_dist:.4f}")
#     print(f"Average intra-class standard deviation: {intra_dist:.4f}")
#     print(f"Separability index (inter/intra): {inter_dist/intra_dist:.4f}")

# if __name__ == '__main__':
#     main()














