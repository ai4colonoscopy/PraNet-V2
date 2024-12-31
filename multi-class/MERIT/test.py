import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os

def crop_and_zoom(image, label):
    """
    裁剪image和label的黑色边缘，并放大到原始尺寸
    """
    # 找到非黑色区域的边界
    non_zero_coords = np.argwhere(image > 0)
    min_coord = non_zero_coords.min(axis=0)
    max_coord = non_zero_coords.max(axis=0)
    
    # 裁剪图像和标签
    image_cropped = image[:, min_coord[1]:max_coord[1]+1, min_coord[2]:max_coord[2]+1]
    label_cropped = label[:, min_coord[1]:max_coord[1]+1, min_coord[2]:max_coord[2]+1]
    
    # 缩放回原始尺寸
    zoom_factors = (
        1,  # 不缩放切片数目维度
        image.shape[1] / image_cropped.shape[1],  # 高度缩放比例
        image.shape[2] / image_cropped.shape[2]   # 宽度缩放比例
    )
    
    # 使用双线性插值法放大图像
    image_zoomed = zoom(image_cropped, zoom_factors, order=1)  # 对图像使用双线性插值
    
    # 使用最近邻插值法放大标签
    label_zoomed = zoom(label_cropped, zoom_factors, order=0)  # 对标签使用最近邻插值
    
    return image_zoomed, label_zoomed

# 定义可视化函数
def visualize_data(image, label, slice_index, save_dir):
    """
    可视化给定切片索引的图像和标签，并保存到指定目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 图像数据可视化
    axes[0].imshow(image[slice_index, :, :], cmap='gray')
    axes[0].set_title('Image Slice')
    axes[0].axis('off')
    
    # 标签数据可视化
    axes[1].imshow(label[slice_index, :, :], cmap='jet', alpha=0.5)
    axes[1].set_title('Label Slice')
    axes[1].axis('off')
    
    # 保存图片
    save_path = os.path.join(save_dir, f'slice_{slice_index}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"Saved slice {slice_index} to {save_path}")

if __name__ == '__main__':
    
    testlist=[1,2,3]
    if testlist:
        print("testlist is not empty")
    exit()
    # 定义保存路径
    save_dir = './'
    os.makedirs(save_dir, exist_ok=True)  # 创建保存文件夹，如果不存在则创建
    
    # 加载数据
    data_path = '/defaultShare/archive/zhuzixuan/cascade_dataset/ACDC/test/case_092_volume_ES.npz'
    data = np.load(data_path)
    image, label = data['img'], data['label']

    # 查看数据形状
    print(f"Original Image shape: {image.shape}")
    print(f"Original Label shape: {label.shape}")

    # 裁剪并缩放图像和标签
    image_processed, label_processed = crop_and_zoom(image, label)

    # 查看处理后的数据形状
    print(f"Processed Image shape: {image_processed.shape}")
    print(f"Processed Label shape: {label_processed.shape}")

    # 选择一个切片进行可视化
    slice_index = 2 # 选择中间切片（第一维度的中间切片）
    visualize_data(image_processed, label_processed, slice_index, save_dir)
