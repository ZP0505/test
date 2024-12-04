import time
import numpy as np
import cv2
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
from PIL import ImageGrab
import math
import os
# 加载YOLO模型
model = YOLO("best.pt")  # 请替换为你的训练模型文件路径

# 获取并推理浏览器窗口截图
def get_detections(image):
    # 使用YOLO模型进行推理
    results = model(image)  # 将图像传递给YOLO模型
    return results[0]  # 获取推理后的结果


# 获取浏览器窗口的坐标和尺寸
def get_browser_window(title="Game - Crystal Caves"):
    windows = gw.getWindowsWithTitle(title)
    if len(windows) > 0:
        return windows[0]  # 返回第一个匹配窗口
    else:
        print(f"未找到标题为 '{title}' 的窗口")
        return None

# 捕获浏览器窗口的截图
def capture_browser_window(window):
    # 获取窗口的位置和大小
    x, y, w, h = window.left, window.top, window.right, window.bottom

    screenshot = ImageGrab.grab(bbox=[x, y, x+w, y+h])  # 截取窗口截图
    image_src = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return image_src, x, y  # 返回截图和窗口的左上角坐标

def print_detections(results):
    person_pos = None  # 存储人物坐标
    blocks = []  # 存储方块的坐标和类别

    # 遍历每个检测框
    for i, result in enumerate(results.boxes):
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # 获取坐标 (左上角, 右下角)
        class_id = int(result.cls[0].item())  # 获取类别ID
        class_name = results.names[class_id]  # 获取类别名称
        
        # 假设人物的类别ID为 0，方块类别ID为 1, 2, 3, 4
        if class_id == 0:  # 如果是人物
            person_pos = (x1 + x2) / 2, (y1 + y2) / 2  # 获取人物的中心坐标
        elif class_id in [1, 2, 3, 4]:  # 如果是方块
            blocks.append(((x1 + x2) / 2, (y1 + y2) / 2, class_id))  # 存储方块的中心坐标和类别ID

    # 如果有方块，选择与人物最近的方块
    closest_block = None
    min_distance = float('inf')  # 初始化一个非常大的距离

    if person_pos and blocks:
        for block in blocks:
            block_pos = block[0], block[1]  # 方块的坐标
            # 计算人物与方块之间的欧几里得距离
            distance = math.sqrt((person_pos[0] - block_pos[0]) ** 2 + (person_pos[1] - block_pos[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_block = block

    return person_pos, closest_block  # 返回人物位置和最近的方块位置

def find_and_double_click(results, img, browser_window, person_pos=None):
    # 找到离人物最近的方块
    person_pos, closest_block = print_detections(results)
    
    if closest_block is not None:
        block_x, block_y, block_id = closest_block
        print(f"正在双击方块ID: {block_id}, 坐标: ({block_x}, {block_y})")
        
        # 模拟点击，实际操作可能使用 pyautogui 或其他工具
        click_position = (block_x, block_y)
        
        # 点击操作，模拟鼠标点击 (双击)
        # 这里是一个伪代码，实际需要使用一个模拟点击工具如 pyautogui 或 selenium 等
        pyautogui.click(click_position[0], click_position[1])  # 单击
        time.sleep(0.1)
        pyautogui.click(click_position[0], click_position[1])  # 再单击，实现双击
        return True
    else:
        print("没有找到最近的方块，跳过双击操作")
        return False


# 找图函数：在截图中查找目标图像
def find_image(template_path, image):
    # 检查模板路径是否有效
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板图像未找到，请检查路径: {template_path}")
    
    # 读取模板图像
    template = cv2.imread(template_path, 0)
    if template is None:
        raise FileNotFoundError(f"无法读取模板图像，请检查路径: {template_path}")
    
    # 检查截图图像是否有效
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("传入的图像无效，请检查截图函数返回的内容。")
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用模板匹配找到图像中的位置
    res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # 匹配阈值，0-1之间，越高要求越严格
    loc = np.where(res >= threshold)

    if len(loc[0]) > 0:
        # 找到图像匹配位置，返回第一个匹配点的坐标
        return loc[::-1][0][0], loc[::-1][1][0]
    else:
        return None  # 如果没有找到，返回 None

# 点击函数：根据找到的图像位置进行点击
def click_image_location(image_location, browser_window):
    if image_location:
        # 将窗口内坐标转换为屏幕坐标
        screen_x = image_location[0] + browser_window.left
        screen_y = image_location[1] + browser_window.top

        pyautogui.moveTo(screen_x, screen_y)
        pyautogui.click()
        print(f"点击位置: ({screen_x}, {screen_y})")
    else:
        print("未找到匹配的图像，跳过点击。")

def main():
    browser_window = get_browser_window("Game - Crystal Caves")  # 获取浏览器窗口
    print(f"已找到浏览器窗口: {browser_window}")
    if browser_window is None:
        return

    person_pos = None  # 初始人物位置为空

    while True:
        # 捕获浏览器窗口截图并进行推理
        img, window_x, window_y = capture_browser_window(browser_window)
        results = get_detections(img)

        # 获取人物坐标和最近的方块信息
        person_pos, closest_block = print_detections(results)

        # 打印人物和最近方块的信息
        if person_pos:
            print(f"人物坐标: {person_pos}")
        if closest_block:
            print(f"离人物最近的方块ID: {closest_block[2]}, 坐标: {closest_block[0]}, {closest_block[1]}")

        # 找到离人物最近的方块并执行双击
        if closest_block:
            if find_and_double_click(results, img, browser_window, person_pos):
                # 在双击后等待 5 秒
                time.sleep(5)

        # 暂停一段时间，避免过度占用资源
        time.sleep(0.1)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)

if __name__ == "__main__":
    main()    