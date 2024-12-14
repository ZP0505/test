import time
import numpy as np
import cv2
import pygetwindow as gw
import pyautogui
from ultralytics import YOLO
from mss import mss
import math
import os
import requests
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import win32gui
import win32con
import win32api
import configparser
from version_updater import UpdateManager
import tenacity

def run_update_and_restart():
    update_manager = UpdateManager(
            local_version="11.0",
            version_url="https://raw.githubusercontent.com/ZP0505/test/main/version.txt",
            script_url="https://raw.githubusercontent.com/ZP0505/test/main/Game.py"
        )
    update_manager.run_update()

run_update_and_restart()

try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'
    torch = None

# 使用更简洁的日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
    
    def get_mining_config(self):
        return {
            'chain': self.config.get('Mining', 'chain', fallback='Base'),
            'gas_threshold': self.config.getfloat('Mining', 'gas_threshold', fallback=0.2)
        }

class RequestsSession:
    def __init__(self):
        self.session = requests.Session()
    
    def get_fee_gas(self):
        url = "https://base.blockpi.network/v1/rpc/public"
        payload = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_feeHistory\",\"params\":[\"0xa\",\"latest\",[25,75]]}"
        headers = {
            'Accept': "application/json, text/plain, */*",
            'Sec-Fetch-Site': "cross-site",
            'Origin': "https://ct.app",
            'Sec-Fetch-Mode': "cors",
        }
        
        try:
            response = self.session.post(url, data=payload, headers=headers)
            response.raise_for_status()
            return self._parse_gas(response.json())
        except requests.RequestException as e:
            logger.error(f"Gas 获取失败: {e}")
            return None
    
    def _parse_gas(self, data):
        try:
            base_fee_per_gas = [int(value, 16) for value in data["result"]["baseFeePerGas"]]
            max_value = max(base_fee_per_gas)
            return round(max_value/100000000 + 0.01, 3)
        except (KeyError, ValueError) as e:
            logger.error(f"Gas 解析错误: {e}")
            return None

# 加载 YOLO 模型
model = YOLO("best.pt")

# 如果可用，将模型移动到 CUDA 设备
if torch and torch.cuda.is_available():
    model = model.to('cuda')

def get_window_handle(title_pattern="Game - Crystal Caves"):
    def enum_window_callback(hwnd, lParam):
        window_title = win32gui.GetWindowText(hwnd)
        if title_pattern in window_title:
            lParam.append(hwnd)

    hwnd_list = []
    win32gui.EnumWindows(enum_window_callback, hwnd_list)
    return hwnd_list

def get_browser_windows(title_pattern="Game - Crystal Caves"):
    windows = gw.getAllWindows()
    matched_windows = [win for win in windows if title_pattern in win.title]

    if len(matched_windows) > 0:
        return matched_windows
    else:
        logger.warning(f"未找到匹配的窗口")
        return []

def capture_browser_window(window):
    with mss() as sct:
        monitor = {"top": window.top, "left": window.left, 
                   "width": window.width, "height": window.height}
        screenshot = np.array(sct.grab(monitor))
        image_src = screenshot[:, :, :3]  # 去除 alpha 通道
        return image_src, window.left, window.top

def get_detections(image):
    results = model(image)
    return results[0]

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

def click_image(image_path, hwnd, confidence=0.8):
    try:
        locations = list(pyautogui.locateAllOnScreen(image_path, confidence=confidence))
        
        if locations:
            for location in locations:
                center = pyautogui.center(location)
                client_x, client_y = win32gui.ScreenToClient(hwnd, center)
                
                win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, 
                win32api.MAKELONG(client_x, client_y))
                time.sleep(0.05)
                win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, 
                win32api.MAKELONG(client_x, client_y))
                time.sleep(0.2)
            return True, None
        else:
            return False, f"未找到图像: {image_path}"
    
    except Exception as e:
        return False, f"点击图像时发生错误: {str(e)}"

def handle_post_double_click(hwnd):
    # 检查 OKX Wallet 是否已经在运行
    hwnd_list = get_window_handle(title_pattern="OKX Wallet")
    
    if not hwnd_list:
        # 如果 OKX Wallet 没有运行，则点击确认按钮
        click_image("images/confirm.png", hwnd)
    
    # 获取 OKX Wallet 窗口句柄
    hwnd_list = get_window_handle(title_pattern="OKX Wallet")
    
    if hwnd_list:
        for window_handle in hwnd_list:
            win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(window_handle)
            
            time.sleep(1.5)
            
            if click_image("images/qrjy.png", window_handle):
                logger.info(f"成功激活并点击窗口: {window_handle}")
            else:
                logger.warning(f"无法在窗口 {window_handle} 上点击图像")

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
)
def process_window(hwnd, browser_window, user_input):
    logger.info(f"处理窗口，句柄: {hwnd}")
    
    try:
        window_handle = hwnd if isinstance(hwnd, int) else browser_window._hWnd
        
        if user_input.lower() == 'b':
            monitor_gas()
        click_image("images/x.png", window_handle)
        click_image("images/x1.png", window_handle)
        click_image("images/x2.png", window_handle)
        img, window_x, window_y = capture_browser_window(browser_window)
        results = get_detections(img)
        person_pos, closest_block = logstr_detections(results)

        if person_pos:
            logger.info(f"窗口人物坐标: {person_pos}")
        if closest_block:
            logger.info(f"离人物最近的方块ID: {closest_block[2]}, 坐标: {closest_block[0]}, {closest_block[1]}")

        if closest_block:
            find_and_double_click(results, img, browser_window, window_x, window_y, window_handle, person_pos)
            handle_post_double_click(window_handle)
        else:
            logger.warning("未找到最近的方块")
            click_image("images/ok.png", window_handle)
            handle_post_double_click(window_handle)

    except Exception as e:
        logger.error(f"处理窗口时发生错误: {e}")

def monitor_gas(threshold=0.2):
    request_session = RequestsSession()
    while True:
        gas = request_session.get_fee_gas()
        if gas is not None:
            logger.info(f"当前Gas值: {gas}")
            if gas <= threshold:
                break
        time.sleep(5)

def main():
    config_manager = ConfigManager()
    mining_config = config_manager.get_mining_config()
    
    user_input = input("请输入挖矿的链(b为:Base链，S为:Skale)")
    time.sleep(1.5)
    logger.info("游戏机器人启动")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        while True:
            hwnd_list = get_window_handle()
            browser_windows = get_browser_windows()
            if not hwnd_list or not browser_windows:
                logger.error("无法找到游戏窗口")
                time.sleep(5)
                continue

            min_windows = min(len(hwnd_list), len(browser_windows))
            logger.info(f"找到 {min_windows} 个游戏窗口")
            
            futures = []
            for i in range(min_windows):
                future = executor.submit(
                    process_window, 
                    hwnd_list[i], 
                    browser_windows[i], 
                    user_input
                )
                futures.append(future)

            for future in futures:
                future.result()

            time.sleep(3.5)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
