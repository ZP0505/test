import time
import numpy as np
import cv2
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
from PIL import ImageGrab
import math
import os
import requests
import logging
import sys
import threading
from datetime import datetime
from colorama import init, Fore, Style
from version_updater import UpdateManager

def run_update_and_restart():
    update_manager = UpdateManager(
            local_version="1.0.3",
            version_url="https://raw.githubusercontent.com/ZP0505/test/main/version.txt",
            script_url="https://raw.githubusercontent.com/ZP0505/test/main/Game.py"
        )
    update_manager.run_update()

run_update_and_restart()

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        log_format = (
            f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - "
            f"{color}%(levelname)s{Style.RESET_ALL}: "
            f"{Fore.WHITE}%(message)s{Style.RESET_ALL}"
        )
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_colored_logging():
    logger = logging.getLogger('GameBot')
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    colored_formatter = ColoredFormatter()
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_colored_logging()

model = YOLO("best.pt")

def get_detections(image):
    results = model(image)
    return results[0]

def Get_FeeGas():
    url = "https://base.blockpi.network/v1/rpc/public"
    payload = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_feeHistory\",\"params\":[\"0xa\",\"latest\",[25,75]]}"
    headers = {
        'Accept': "application/json, text/plain, */*",
        'Sec-Fetch-Site': "cross-site",
        'Origin': "https://ct.app",
        'Sec-Fetch-Mode': "cors",
        'sec-ch-ua': "\"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
        'sec-ch-ua-mobile': "?0",
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        'Content-Type': "application/json",
        'Accept-Encoding': "gzip, deflate, br, zstd",
        'Host': "base.blockpi.network",
        'Referer': "https://ct.app/",
        'Connection': "keep-alive",
        'sec-ch-ua-platform': "\"Windows\"",
        'Accept-Language': "zh-CN,zh;q=0.9",
        'Sec-Fetch-Dest': "empty",
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    data = response.json()
    baseFeePerGas = data["result"]["baseFeePerGas"]
    base_fee_per_gas = [int(value, 16) for value in baseFeePerGas]
    max_value = max(base_fee_per_gas)
    gas = max_value/100000000+0.01
    gas = round(gas, 3)
    return gas

def monitor_gas():
    while True:
        gas = Get_FeeGas()
        logger.info(f"当前Gas值: {gas}")
        if gas <= 0.35:
            break
        time.sleep(5)

def get_browser_window(title="Game - Crystal Caves"):
    windows = gw.getWindowsWithTitle(title)
    if len(windows) > 0:
        return windows[0]
    else:
        logger.warning(f"未找到游戏的窗口")
        return None

def capture_browser_window(window):
    x, y, w, h = window.left, window.top, window.right, window.bottom
    screenshot = ImageGrab.grab(bbox=[x, y, x+w, y+h])
    image_src = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return image_src, x, y

def logstr_detections(results):
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
    person_pos, closest_block = logstr_detections(results)
    if closest_block is not None:
        block_x, block_y, block_id = closest_block
        click_position = (block_x, block_y)
        pyautogui.doubleClick(click_position[0], click_position[1])
        return True
    else:
        logger.warning("没有找到最近的方块，跳过双击操作")
        return False

def find_and_click_image(image_path):
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=0.8)
        if location:
            center = pyautogui.center(location)
            pyautogui.click(center)
            return center
        else:
            return None
    except pyautogui.ImageNotFoundException:
        logger.warning("未找到图像。")

def handle_post_double_click():
    template_path = 'confirm.png'
    image_location = find_and_click_image("confirm.png")
    for i in range(10):
        image_location_qr = find_and_click_image("qrjy.png")
        if image_location_qr:
            break
        time.sleep(1)

def main():
    logger.info("游戏机器人启动")
    browser_window = get_browser_window("Game - Crystal Caves")
    logger.info(f"已找到游戏窗口")
    if browser_window is None:
        return
    person_pos = None
    counter = 0
    while True:
        monitor_gas()
        image_location_ok = find_and_click_image("ok.png")
        if counter % 5 == 0:
            find_and_click_image("dw.png")
            find_and_click_image("dw2.png")
        img, window_x, window_y = capture_browser_window(browser_window)
        results = get_detections(img)
        person_pos, closest_block = logstr_detections(results)
        if person_pos:
            logger.info(f"人物坐标: {person_pos}")
        if closest_block:
            logger.info(f"离人物最近的方块ID: {closest_block[2]}, 坐标: {closest_block[0]}, {closest_block[1]}")
        if closest_block:
            find_and_double_click(results, img, browser_window, person_pos)
        else:
            logger.warning("未找到最近的方块")
            find_and_click_image("dw.png")
            find_and_click_image("dw2.png")
            handle_post_double_click()
        counter += 1
        time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
