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
import win32gui
import win32con
import win32api

def run_update_and_restart():
    update_manager = UpdateManager(
            local_version="8.0",
            version_url="https://raw.githubusercontent.com/ZP0505/test/main/version.txt",
            script_url="https://raw.githubusercontent.com/ZP0505/test/main/Game.py"
        )
    update_manager.run_update()

run_update_and_restart()

def get_detections(image):
    results = model(image)
    return results[0]

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

def get_window_handle(title_pattern="Game - Crystal Caves"):
    def enum_window_callback(hwnd, lParam):
        window_title = win32gui.GetWindowText(hwnd)
        if title_pattern in window_title:
            lParam.append(hwnd)

    hwnd_list = []
    win32gui.EnumWindows(enum_window_callback, hwnd_list)
    if hwnd_list:
        return hwnd_list[0]
    return None

def get_browser_window(title_pattern="Game - Crystal Caves"):
    windows = gw.getAllTitles()
    matched_windows = [win for win in windows if title_pattern in win]

    if len(matched_windows) > 0:
        return gw.getWindowsWithTitle(matched_windows[0])[0]
    else:
        logger.warning(f"未找到匹配的窗口")
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

def background_double_click(hwnd, x, y, click_count=3):
    client_x, client_y = win32gui.ScreenToClient(hwnd, (int(x), int(y)))
    lParam = win32api.MAKELONG(client_x, client_y)
    for _ in range(click_count):
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
        time.sleep(0.05)  
        win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
        time.sleep(0.1) 
        
def find_and_double_click(results, img, browser_window, window_x, window_y, hwnd, person_pos=None):
    person_pos, closest_block = logstr_detections(results)
    if closest_block is not None:
        block_x, block_y, block_id = closest_block
        click_x = window_x + block_x
        click_y = window_y + block_y
        if hwnd:
            background_double_click(hwnd, click_x, click_y)
            return True
        else:
            logger.warning("无法找到窗口句柄")
            return False
    else:
        logger.warning("没有找到最近的方块，跳过双击操作")
        return False

def locate_image_multi_scale(image_path, screenshot=None, scales=[0.8, 1.0, 1.2], confidence=0.7):
    if screenshot is None:
        screenshot = np.array(ImageGrab.grab())
    template = cv2.imread(image_path, cv2.IMREAD_COLOR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    for scale in scales:
        resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        result = cv2.matchTemplate(screenshot_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        
        locations = np.where(result >= confidence)
        
        if len(locations[0]) > 0:
            top_left = (locations[1][0], locations[0][0])
            w, h = resized_template.shape[:2]
            center = (top_left[0] + w // 2, top_left[1] + h // 2)
            return center
    
    return None

def robust_background_click_image(image_path, hwnd, confidence=0.8):
    try:
        center = locate_image_multi_scale(image_path, confidence=confidence)
        
        if center:
            client_x, client_y = win32gui.ScreenToClient(hwnd, center)
            lParam = win32api.MAKELONG(client_x, client_y)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(client_x, client_y))
            time.sleep(0.05)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(client_x, client_y))    
            logger.info(f"成功点击图像: {image_path}")
            return center
        
        logger.warning(f"未找到图像: {image_path}")
        return None
    
    except Exception as e:
        logger.error(f"出错: {e}")
        return None

def click_image(image_path, hwnd):
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=0.8)
        if location:
            center = pyautogui.center(location)
            client_x, client_y = win32gui.ScreenToClient(hwnd, center)
            lParam = win32api.MAKELONG(client_x, client_y)     
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(client_x, client_y))
            time.sleep(0.05)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(client_x, client_y))    
            return center
        else:
            return None
    except pyautogui.ImageNotFoundException:
        logger.warning(f"未找到图像: {image_path}")
        return None


# 替换原有的 background_click_image 函数
background_click_image = robust_background_click_image

def handle_post_double_click(hwnd):
    click_image("confirm.png", hwnd)
    for i in range(3):
        hwnd1=get_window_handle(title_pattern="OKX Wallet")
        if hwnd1:
            win32gui.ShowWindow(hwnd1, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd1)
            time.sleep(1.5)
            if click_image("qrjy.png", hwnd1):
                break
            else:
                time.sleep(0.3)


def main():
    user_input = input("请输入挖矿的链(b为:Base链，S为:Skale)")
    logger.info("游戏机器人启动")
    hwnd = get_window_handle()
    browser_window = get_browser_window()
    if not hwnd or not browser_window:
        logger.error("无法找到游戏窗口")
        return
    logger.info(f"已找到游戏窗口，句柄: {hwnd}")
    person_pos = None
    counter = 0
    while True:
        if user_input.lower() == 'b':
            try:
                monitor_gas()
            except Exception as e:
                logger.error(f"Gas监控出错: {e}")
                time.sleep(5)
                continue
        click_image("ok.png", hwnd)
        background_click_image("x.png", hwnd)
        background_click_image("x1.png", hwnd)
        # background_click_image("dw.png", hwnd)
        img, window_x, window_y = capture_browser_window(browser_window)
        results = get_detections(img)
        person_pos, closest_block = logstr_detections(results)
        if person_pos:
            logger.info(f"人物坐标: {person_pos}")
        if closest_block:
            logger.info(f"离人物最近的方块ID: {closest_block[2]}, 坐标: {closest_block[0]}, {closest_block[1]}")
        if closest_block:
            find_and_double_click(results, img, browser_window, window_x, window_y, hwnd, person_pos)
            handle_post_double_click(hwnd)
        else:
            logger.warning("未找到最近的方块")
            handle_post_double_click(hwnd)
        counter += 1
        time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)

def Get_FeeGas():
    url = "https://base.blockpi.network/v1/rpc/public"
    payload = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_feeHistory\",\"params\":[\"0xa\",\"latest\",[25,75]]}"
    headers = {
        'Accept': "application/json, text/plain, */*",
        'Sec-Fetch-Site': "cross-site",
        'Origin': "https://ct.app",
        'Sec-Fetch-Mode': "cors",
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
        if gas <= 0.25:
            break
        time.sleep(5)

if __name__ == "__main__":
    main()
