import os
import glob
import numpy as np
from PIL import Image
import imageio
import cv2

def process_and_combine_raw_images(
    raw_file_path,
    width,
    height,
    channels=1,
    combined_image_suffix='.png'
):
    """
    讀取原始二進制圖像數據，分割為兩張圖像，保存為 PNG，並將兩張圖像水平拼接保存為一張新的 PNG 圖片。

    :param raw_file_path: str, 原始二進制圖像文件的路徑（.raw）。
    :param width: int, 單張圖像的寬度（像素）。
    :param height: int, 單張圖像的高度（像素）。
    :param channels: int, 每像素的通道數（默認為 1，表示灰度圖；3 表示 RGB 彩色圖）。
    :param output_image1_suffix: str, 第一張分割後圖像的文件後綴（默認為 '_1.png'）。
    :param combined_image_suffix: str, 拼接後圖像的文件後綴（默認為 '_combined.png'）。
    :return: None
    """
    try:
        # 計算每張圖像的字節數
        bytes_per_image = width * height * channels
        total_bytes_expected = bytes_per_image * 4  # 每個 .raw 文件包含兩張圖像

        # 讀取 raw 文件
        with open(raw_file_path, 'rb') as f:
            raw_data = f.read()

        # 檢查數據長度是否與預期匹配
        if len(raw_data) != total_bytes_expected:
            raise ValueError(
                f"文件 {raw_file_path} 的數據長度與指定的寬度、高度和通道數不匹配。\n"
                f"預期字節數: {total_bytes_expected}, 實際字節數: {len(raw_data)}"
            )

        # 將數據轉換為 NumPy 陣列
        image_array = np.frombuffer(raw_data, dtype=np.uint8)

        # 分割為兩張圖像的數據
        img1_array = image_array[:bytes_per_image]
        img2_array = image_array[bytes_per_image:2 * bytes_per_image]
        img3_array = image_array[2 * bytes_per_image:3 * bytes_per_image]
        img4_array = image_array[3 * bytes_per_image:4 * bytes_per_image]


        # 重塑為 2D 或 3D 陣列
        if channels == 1:
            img1_reshaped = img1_array.reshape((height, width))
            img2_reshaped = img2_array.reshape((height, width))
            img3_reshaped = img3_array.reshape((height, width))
            img4_reshaped = img4_array.reshape((height, width))
            mode = 'L'  # 灰度圖像
        elif channels == 3:
            img1_reshaped = img1_array.reshape((height, width, channels))
            img2_reshaped = img2_array.reshape((height, width, channels))
            img3_reshaped = img3_array.reshape((height, width, channels))
            img4_reshaped = img4_array.reshape((height, width, channels))
            mode = 'RGB'  # RGB 彩色圖像
        else:
            raise ValueError(f"不支持的通道數: {channels}")

        # 將 NumPy 陣列轉換為 PIL 圖片對象
        image1 = Image.fromarray(img1_reshaped, mode=mode)
        image2 = Image.fromarray(img2_reshaped, mode=mode)
        image3 = Image.fromarray(img3_reshaped, mode=mode)
        image4 = Image.fromarray(img4_reshaped, mode=mode)

        # 構造輸出文件名
        base_name = os.path.splitext(os.path.basename(raw_file_path))[0]
        directory = os.path.dirname(raw_file_path) + "/png/"
        combined_image_path = os.path.join(directory, f"{base_name}{combined_image_suffix}")

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"資料夾 '{directory}' 已建立")

        # 將兩張圖片轉換為 NumPy 陣列以進行拼接
        img1_np = np.array(image1)
        img2_np = np.array(image2)
        img3_np = np.array(image3)
        img4_np = np.array(image4)

        # # 確保兩張圖片的高度一致
        # if img1_np.shape[0] != img2_np.shape[0]:
        #     raise ValueError("兩張圖片的高度不一致，無法進行水平拼接。")

        # 使用 NumPy 的 hstack 拼接圖片
        combined_array = np.vstack((img1_np, img2_np))
        combined_array = np.vstack((combined_array, img3_np))
        combined_array = np.vstack((combined_array, img4_np))

        # 使用 imageio 保存拼接後的圖片
        imageio.imwrite(combined_image_path, combined_array)
        print(f"已保存拼接的 PNG 圖片：{combined_image_path}")

    except Exception as e:
        print(f"處理文件 {raw_file_path} 時出錯: {e}")

def rgb_raw_to_png(image_file, width, height):
    """
    Load the bmp image and convert it to rgb image
    """
    frame_size = width * height

    print(f"file: '{image_file}'")
    with open(image_file, 'rb') as f:
        raw_img = f.read(frame_size)

    raw_img_array = np.frombuffer(raw_img, dtype=np.uint8)
    img = raw_img_array.reshape((874, 3492))
    
    img_size = (1164, 874, 3)
    y = img[: img_size[1], : img_size[0]]
    u = img[: img_size[1], img_size[0] + 0 : img_size[0] * 3 + 0][:, ::2]
    v = img[: img_size[1], img_size[0] + 1 : img_size[0] * 3 + 1][:, ::2]
    img = np.zeros([img_size[1], img_size[0], 3], dtype="uint8")
    img[:, :, 0] = y
    img[:, :, 1] = u
    img[:, :, 2] = v
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    

    # 構造輸出文件名
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    directory = os.path.dirname(image_file) + "/png/"
    combined_image_suffix = '.png'
    combined_image_path = os.path.join(directory, f"{base_name}{combined_image_suffix}")
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"資料夾 '{directory}' 已建立")

    cv2.imwrite(combined_image_path, img)
    print(f"已保存RGB PNG 圖片：{combined_image_path}")
    
def convert_all_raw_to_png(directory_path, width, height, rgb_width, rgb_height, channels=1):
    """
    遍歷指定目錄下的所有 .raw 文件，將其轉換為兩張 .png 文件並保留原始名稱，同時生成拼接後的 .png 文件。

    :param directory_path: str, 目標目錄的路徑。
    :param width: int, 單張圖像的寬度（像素）。
    :param height: int, 單張圖像的高度（像素）。
    :param channels: int, 每像素的通道數（默認為 1，表示灰度圖像；3 表示 RGB 彩色圖像）。
    :return: None
    """
    # 獲取所有 ir .raw 文件
    ir_img_path = directory_path + "/frame_5fps/ir"

    raw_files = glob.glob(os.path.join(ir_img_path, "*.raw"))

    if not raw_files:
        print(f"在目錄 {ir_img_path} 中未找到任何 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 個 .raw 文件。開始轉換...")

    for raw_file in raw_files:
        process_and_combine_raw_images(
            raw_file_path=raw_file,
            width=width,
            height=height,
            channels=channels
        )

    print("所有ir img處理完成。")

    # 獲取所有 slam .raw 文件
    slam_img_path = directory_path + "/frame_30fps/slam"

    raw_files = glob.glob(os.path.join(slam_img_path, "*.raw"))

    if not raw_files:
        print(f"在目錄 {slam_img_path} 中未找到任何 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 個 .raw 文件。開始轉換...")

    for raw_file in raw_files:
        process_and_combine_raw_images(
            raw_file_path=raw_file,
            width=width,
            height=height,
            channels=channels
        )

    print("所有slam img處理完成。")

    # rgb l.raw files
    rgb_img_path = directory_path + "/frame_5fps/rgb_raw_l"
    raw_files = glob.glob(os.path.join(rgb_img_path, "*.raw"))
    if not raw_files:
        print(f"在目錄 {rgb_img_path} 中未找到任何 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 個 .raw 文件。開始轉換...")

    for raw_file in raw_files:
        rgb_raw_to_png(image_file=raw_file, width=rgb_width, height=rgb_height)

    print("所有rgb l img處理完成。")

    # rgb r.raw files
    rgb_img_path = directory_path + "/frame_5fps/rgb_raw_r"
    raw_files = glob.glob(os.path.join(rgb_img_path, "*.raw"))
    if not raw_files:
        print(f"在目錄 {rgb_img_path} 中未找到任何 .raw 文件。")
        return

    print(f"找到 {len(raw_files)} 個 .raw 文件。開始轉換...")

    for raw_file in raw_files:
        rgb_raw_to_png(image_file=raw_file, width=rgb_width, height=rgb_height)
    print("所有rgb r img處理完成。")

if __name__ == "__main__":
    
    directory_path = '../realtime_scene_record' 

    width = 640    
    height = 480   
    rgb_width=3492
    rgb_height=874
    channels = 1   
    
    convert_all_raw_to_png(
        directory_path=directory_path,
        width=width,
        height=height,
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        channels=channels
    )


