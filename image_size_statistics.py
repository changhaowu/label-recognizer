import os
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


def get_image_sizes(image_folder):
    widths = []
    heights = []
    sizes = []

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)
                        sizes.append((width, height))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return widths, heights, sizes


def print_statistics(widths, heights, sizes, label):
    widths = np.array(widths)
    heights = np.array(heights)

    print(f"\nStatistics for cluster {label}:")
    print(f"  Mean Width: {np.mean(widths):.2f}")
    print(f"  Std Dev Width: {np.std(widths):.2f}")
    print(f"  Min Width: {np.min(widths)}")
    print(f"  Max Width: {np.max(widths)}")

    print(f"  Mean Height: {np.mean(heights):.2f}")
    print(f"  Std Dev Height: {np.std(heights):.2f}")
    print(f"  Min Height: {np.min(heights)}")
    print(f"  Max Height: {np.max(heights)}")

    print("\nMost Common Sizes:")
    size_counts = Counter(sizes)
    for size, count in size_counts.most_common(10):
        print(f"  {size}: {count} images")


def plot_size_distribution(widths, heights):
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title("Image Size Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.grid(True)
    plt.savefig("size_distribution.png")


if __name__ == "__main__":
    image_folder = input("Enter the path to the image folder: ")
    widths, heights, sizes = get_image_sizes(image_folder)
    # plot_size_distribution(widths, heights)

    # 转换为numpy数组
    widths = np.array(widths)
    heights = np.array(heights)

    # 初步估计两个斜率
    k1_initial = 0.66  # 调整这个初值
    k2_initial = 2.0  # 调整这个初值

    # 定义分段线性函数
    def piecewise_linear(x, k1, k2, xb):
        return np.piecewise(x, [x < xb, x >= xb], [lambda x: k1 * x, lambda x: k2 * x])

    # 初步估计分界点
    xb_initial = 1000  # 根据你的数据调整这个值

    # 使用curve_fit进行拟合
    params, _ = curve_fit(
        piecewise_linear, widths, heights, p0=[k1_initial, k2_initial, xb_initial]
    )
    k1_opt, k2_opt, xb_opt = params

    # 打印优化后的参数
    print(f"优化后的斜率 k1: {k1_opt}")
    print(f"优化后的斜率 k2: {k2_opt}")
    print(f"优化后的分界点 xb: {xb_opt}")

    # Plot the original data points and the fitted lines
    plt.scatter(widths, heights, alpha=0.5)
    plt.plot(
        widths, piecewise_linear(widths, *params), color="red", label="Piecewise Fit"
    )
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Image Size Distribution with Optimized Piecewise Linear Fit")
    plt.legend()
    plt.savefig("size_distribution.png")
