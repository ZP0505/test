import time
import numpy as np
import cv2
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
from PIL import ImageGrab
import math
import os
import sys
from version_updater import VersionUpdater

update_manager = UpdateManager(
        local_version="1.0.0",  # 你的当前版本
        version_url="https://raw.githubusercontent.com/ZP0505/test/main/version.txt",  # 远程版本文件URL
        script_url="https://raw.githubusercontent.com/ZP0505/test/main/Game.py"  # 远程脚本文件URL
    )
# 运行更新
update_manager.run_update()
model = YOLO("best.pt")

def get_detections(image):
    results = model(image)
    return results[0]

def get_browser_window(title="Game - Crystal Caves"):
    windows = gw.getWindowsWithTitle(title)
    if len(windows) > 0:
        return windows[0]
    else:
        print(f"未找到标题为 '{title}' 的窗口")
        return None

def capture_browser_window(window):
    x, y, w, h = window.left, window.top, window.right, window.bottom
    screenshot = ImageGrab.grab(bbox=[x, y, x+w, y+h])
    image_src = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return image_src, x, y

def print_detections(results):
    person_pos = None
    blocks = []
    for i, result in enumerate(results.boxes):
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        class_id = int(result.cls[0].item())
        class_name = results.names[class_id]
        if class_id == 0:
            person_pos = (x1 + x2) / 2, (y1 + y2) / 2
        elif class_id in [1, 2, 3, 4]:
            blocks.append(((x1 + x2) / 2, (y1 + y2) / 2, class_id))
    closest_block = None
    min_distance = float('inf')
    if person_pos and blocks:
        for block in blocks:
            block_pos = block[0], block[1]
            distance = math.sqrt((person_pos[0] - block_pos[0]) ** 2 + (person_pos[1] - block_pos[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_block = block
    return person_pos, closest_block

def find_and_double_click(results, img, browser_window, person_pos=None):
    person_pos, closest_block = print_detections(results)
    if closest_block is not None:
        block_x, block_y, block_id = closest_block
        print(f"正在双击方块ID: {block_id}, 坐标: ({block_x}, {block_y})")
        click_position = (block_x, block_y)
        pyautogui.click(click_position[0], click_position[1])
        time.sleep(0.1)
        pyautogui.click(click_position[0], click_position[1])
        return True
    else:
        print("没有找到最近的方块，跳过双击操作")
        return False

def find_and_click_image(image_path):
    try:
        print("开始查找图像...")
        location = pyautogui.locateOnScreen(image_path, confidence=0.8)
        if location:
            center = pyautogui.center(location)
            pyautogui.click(center)
            print(f"点击位置：{center}")
            return center
        else:
            print("未找到目标图像！")
            return None
    except pyautogui.ImageNotFoundException:
        print("未找到图像。")

def handle_post_double_click():
    template_path = 'confirm.png'
    image_location = find_and_click_image("confirm.png")
    if image_location is None:
        print("未找到 confirm.png，跳过点击。")
    else:
        print(f"找到 confirm.png，点击位置: {image_location}")
        for i in range(10):
            image_location_qr = find_and_click_image("qrjy.png")
            if image_location_qr is None:
                print("未找到 qrjy.png，跳过点击。")
            else:
                print(f"找到 qrjy.png，点击位置: {image_location_qr}")
                break
            time.sleep(2)
        while True:
            image_location_ok = find_and_click_image("ok.png")
            if image_location_ok is None:
                print("未找到 ok.png，跳过点击。")
            else:
                print(f"找到 ok.png，点击位置: {image_location_ok}")
                break
            time.sleep(2)

def main():
    browser_window = get_browser_window("Game - Crystal Caves")
    print(f"已找到浏览器游戏进程: {browser_window}")
    if browser_window is None:
        return
    person_pos = None
    print("请最小化我们游戏窗口，并将焦点切换到浏览器游戏窗口。")
    time.sleep(5)
    counter = 0  # 初始化计数器
    while True:
        if counter % 5 == 0:
            image_location_qr = find_and_click_image("dw.png")
        img, window_x, window_y = capture_browser_window(browser_window)
        results = get_detections(img)
        person_pos, closest_block = print_detections(results)
        if person_pos:
            print(f"人物坐标: {person_pos}")
        if closest_block:
            print(f"离人物最近的方块ID: {closest_block[2]}, 坐标: {closest_block[0]}, {closest_block[1]}")
        if closest_block:
            find_and_double_click(results, img, browser_window, person_pos)
        if counter % 5 == 0:
           handle_post_double_click() 
        time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counter += 1
        time.sleep(1)

if __name__ == "__main__":
    main()
