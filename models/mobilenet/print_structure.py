# # print_structure.py
# import torch
# from torchsummary import summary
# from torchviz import make_dot
# from mobilenet_v4 import MobileNetV4

# # 创建模型
# model = MobileNetV4(num_classes=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # 打印网络结构
# summary(model, (3, 224, 224), device=str(device))

# # 生成可视化图
# x = torch.randn(1, 3, 224, 224).to(device)
# y = model(x)
# make_dot(y, params=dict(list(model.named_parameters()))).render("mobilenetv4_structure", format="png")

# print("\n✅ 网络结构图已保存为 mobilenetv4_structure.png")


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block(ax, xy, width, height, text, color="#5B8FF9"):
    """绘制单个模块块图"""
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=color, facecolor='white'
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha='center', va='center', fontsize=9, color=color, fontweight='bold')

def plot_mobilenetv4_architecture():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # ==== 输入层 ====
    draw_block(ax, (0.2, 2), 1.5, 1, "Input\n(3×224×224)", "#333333")

    # ==== 第一层 Conv2D ====
    draw_block(ax, (2, 2), 1.6, 1, "Conv2D 3×3\n→ 32", "#5B8FF9")

    # ==== Fused Inverted Bottleneck ====
    draw_block(ax, (3.8, 2), 1.8, 1, "Fused IB\n(32→64)", "#5AD8A6")

    # ==== Depthwise Separable Conv ====
    draw_block(ax, (5.8, 2), 1.8, 1, "DWConv 5×5\n→128", "#F6BD16")

    # ==== Inverted Bottleneck x2 ====
    draw_block(ax, (7.8, 2), 1.6, 1, "IB×2\n(128→128)", "#E8684A")

    # ==== ECA 注意力 ====
    draw_block(ax, (9.4, 2), 1.4, 1, "ECA\nAttention", "#6DC8EC")

    # ==== ConvNext Block ====
    draw_block(ax, (2, 0.5), 1.8, 1, "ConvNext\n3×3 →384", "#5B8FF9")

    # ==== Depthwise Separable Conv 2 ====
    draw_block(ax, (3.9, 0.5), 1.8, 1, "DWConv 3×3\n→256", "#F6BD16")

    # ==== Inverted Bottleneck ====
    draw_block(ax, (5.8, 0.5), 1.8, 1, "IB (256→256)", "#E8684A")

    # ==== 全局平均池化 ====
    draw_block(ax, (7.8, 0.5), 1.8, 1, "Global\nAvgPool", "#A7A7A7")

    # ==== 全连接层 ====
    draw_block(ax, (9.7, 0.5), 1.8, 1, "FC (512)\nReLU", "#A7A7A7")

    # ==== 输出层 ====
    draw_block(ax, (11.6, 0.5), 1.3, 1, "Output\n(2 classes)", "#333333")

    # ==== 连线 ====
    for x1, y1, x2, y2 in [
        (1.7, 2.5, 2, 2.5),
        (3.6, 2.5, 3.8, 2.5),
        (5.6, 2.5, 5.8, 2.5),
        (7.6, 2.5, 7.8, 2.5),
        (9.2, 2.5, 9.4, 2.5),
        (2, 1.0, 2, 2.0),
        (3.8, 1.0, 3.8, 2.0),
        (5.7, 1.0, 5.7, 2.0),
        (7.7, 1.0, 7.7, 2.0),
        (9.6, 1.0, 9.6, 2.0),
        (11.5, 1.0, 11.5, 2.0)
    ]:
        ax.arrow(x1, y1, x2-x1, y2-y1, width=0.01, head_width=0.1, color="#666666", length_includes_head=True)

    ax.text(0.2, 4.6, "MobileNetV4 Architecture for Breast Ultrasound Classification",
            fontsize=11, fontweight='bold', color="#000000", ha='left')
    plt.tight_layout()
    plt.savefig("mobilenetv4_architecture_breast_ultrasound.png", dpi=400, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_mobilenetv4_architecture()

