# 导入各用于文件操作、数据处理、图像处理、绘图以及构建和训练深度学习的模型
import os
import numpy as np
import pandas as pd
from skimage import io, transform, measure
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 读取训练数据
# 定义训练图像和掩码所在目录，并对包含掩码信息的CSV文件进行读取
train_images_dir = "E:/深度学习竞赛/train_dataset/img"
mask_csv = "E:/深度学习竞赛/train_dataset/mask.csv"

# 读取CSV文件
mask_df = pd.read_csv(mask_csv)

# 将RLE转换为二进制掩码——RLE：压缩编码方式，常用于存储稀疏矩阵
def rle_to_mask(rle, img_shape):
    w, h = img_shape # 获取图像的宽度和高度
    mask = np.zeros((w, h), dtype=np.uint8) # 创建一个全零的二维数组，用于存储掩码，数据类型为无符号8位整数
    array = np.asarray([int(x) for x in rle.split()]) # 将RLE字符串按空格分割并转换为整数数组
    starts = array[0::2] # 提取起始位置数组（偶数索引处的值）
    lengths = array[1::2] # 提取长度数组（奇数索引处的值）
    ends = starts + lengths # 计算结束位置数组（起始位置加上长度）
    # 遍历起始位置和结束位置，将对应位置的掩码值设为1
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

# 设置尺寸
target_size = (128, 128) # 统一输入大小

# 加载和处理图像及对应的掩码
# 初始化两个空列表，用于存储训练图像和对应的掩码
train_images = []
train_masks = []

# 遍历CSV文件中的每一行数据
for index, row in mask_df.iterrows():
    filename = row['filename'] # 从当前行中提取文件名
    w, h = map(int, row['w,h'].split())  # 从当前行中提取图像的宽度和高度，并将其转换为整数
    rle = row['rle'] # 从当前行中提取RLE编码的掩码信息
    img_path = os.path.join(train_images_dir, filename) # 构建图像文件的完整路径
    img = io.imread(img_path) # 读取图像文件
    img = transform.resize(img, target_size)
    img = img[:, :, 0] if len(img.shape) == 3 else img # 如果图像是彩色图像（即有3个通道），则只保留第一个通道（灰度图）
    mask = rle_to_mask(rle, (h, w))  # 生成掩码
    mask = transform.resize(mask, target_size, order=0, preserve_range=True).astype(np.uint8)  # 调整掩码大小
    # 将处理后的图像和掩码分别添加到对应的列表中
    train_images.append(img)
    train_masks.append(mask)

# 将图像列表堆叠成一个NumPy数组，形状为 (num_samples, height, width, channels)
train_images = np.stack(train_images)

# 将掩码列表堆叠成一个NumPy数组，形状为 (num_samples, height, width)
train_masks = np.stack(train_masks)

# 扩展维度以适应模型输入
train_images = np.expand_dims(train_images, axis=-1)
train_masks = np.expand_dims(train_masks, axis=-1)

def unet_model(input_size=(128, 128, 1)):
    # 定义输入层，输入尺寸为 (128, 128, 1)
    inputs = layers.Input(input_size)
    # 第一层卷积和池化
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    # 第二层卷积和池化
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    # 第三层卷积和池化
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    # 第四层卷积和池化
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    # 第五层卷积（没有池化）
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    # 上采样和拼接（解码路径）
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    c6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    c7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    c8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    c9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)# 输出层，使用 sigmoid 激活函数生成单通道输出
    # 创建模型对象
    model = models.Model(inputs, outputs)
    return model


# 定义一个函数来创建U-Net模型
model = unet_model()
# 编译模型，指定优化器为'adam'，损失函数为'binary_crossentropy'，评估指标为'accuracy'
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 使用训练数据训练模型，设置训练周期为10个epochs，每个batch的大小为8，将10%的训练数据用作验证集
history = model.fit(train_images, train_masks, epochs=10, batch_size=8, validation_split=0.1)
# 绘制训练过程中准确率和验证准确率的变化曲线
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')  # x轴标签为'Epoch'
plt.ylabel('Accuracy')  # y轴标签为'Accuracy'
plt.legend(loc='upper left')  # 图例位置在左上角
plt.show()  # 显示图像
test_images_dir = "E:/深度学习竞赛/test_dataset_A/img" # 定义测试图片的目录路径
# 初始化空列表用于存储测试图片、文件名和宽高信息
test_images = []
test_filenames = []
test_wh = []

# 定义测试图像目录路径
test_images_dir = "E:/深度学习竞赛/test_dataset_A/img"

# 初始化列表用于存储处理后的测试图像、文件名和宽高信息
test_images = []
test_filenames = []
test_wh = []

# 遍历测试图像目录中的所有图像文件
for image_file in os.listdir(test_images_dir):
    img = io.imread(os.path.join(test_images_dir, image_file)) # 读取图像文件
    img = transform.resize(img, (128, 128)) # 调整图像大小
    img = img[:, :, 0] if len(img.shape) == 3 else img # 如果图像是彩色的（即有3个通道），则转换为灰度图
    test_images.append(img) # 将处理后的图像添加到test_images列表中
    test_filenames.append(os.path.splitext(image_file)[0]) # 获取图像文件的文件名（不包括扩展名）并添加到test_filenames列表中
    test_wh.append(f"{img.shape[0]} {img.shape[1]}") # 记录图像的宽高信息并添加到test_wh列表中
test_images = np.stack(test_images) # 将test_images列表中的图像堆叠成一个numpy数组
test_images = np.expand_dims(test_images, axis=-1) # 在最后一个维度上增加一个维度，使其形状变为(num_images, height, width, 1)
predictions = model.predict(test_images) > 0.5 # 使用模型对测试图像进行预测，并将结果二值化（大于0.5的设为True，否则设为False）
# 创建一个DataFrame用于保存预测结果
results = pd.DataFrame({'filename': [f"{fn}.png" for fn in test_filenames]})  # 添加图片名+.png后缀
results['wh'] = test_wh  # 添加宽高信息列
# 将预测结果转换为RLE格式，并添加到DataFrame中
results['rle'] = list(map(lambda x: ' '.join(str(v) for v in measure.label(x).ravel()), predictions))
# 将结果保存到CSV文件中
results.to_csv("E:/深度学习竞赛/precdict.csv", index=False)
