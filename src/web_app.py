import base64
import datetime
import json
import os
import sys
import time

from loguru import logger
from starlette.responses import JSONResponse

from src.client_component.painter_context import start_painting, init_painter, manager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from fastapi import WebSocket

app = FastAPI()
from pydantic import BaseModel, validator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)



class ColorPaletteRequest(BaseModel):
    colors: list
    old_colors: list = None
    
    @validator('colors', 'old_colors')
    def validate_colors(cls, v):
        if v is None:
            return v
        if not isinstance(v, list) or len(v) != 3 or not all(isinstance(c, int) and 0 <= c <= 255 for c in v):
            raise ValueError('颜色必须是包含3个0-255之间整数的RGB数组')
        return v
# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/paint")
async def start_paint_endpoint(file: UploadFile = File(None)):
    try:
        config = load_config()
        config_values = {
            'render_width': config['canvas']['render_width'],
            'render_height': config['canvas']['render_height'],
            'num_strokes': config['painting']['num_strokes'],
            'ink': config['painting']['ink'],
            'color_palette': config['colors']['palettes'],
            'how_often_to_get_paint': config['painting']['how_often_to_get_paint'],
            'num_adaptations': config['painting']['num_adaptations'],
            'objective_data': config['objective']['data'],
            'objective': config['objective']['type'],
            'objective_weight': config['objective']['weight'],
            'init_optim_iter': config['optimization']['init_optim_iter'],
            'lr_multiplier': config['optimization']['lr_multiplier'],
            'cache_dir': config['system']['cache_dir'],
            'robot': config['system']['robot'],
            'use_cache': config['system']['use_cache'],
        }
        if file:
            # 将文件保存到项目的outputs目录
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, file.filename)
            with open(file_path, 'wb') as f:
                f.write(await file.read())
            logger.info(f"Image uploaded: {file_path}")
            config_values['objective_data'] = ['../uploads/' + file.filename]
        painter, opt = init_painter(config_values)
        result = manager.start_task('painting_task', painter, opt, config_values)
        return {
            "status": "success",
            "data": {
                "message": result,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": str(e)
            }
        )
        
        
@app.post("/color-palette")
async def add_color_palette(request: ColorPaletteRequest):
    try:
        config = load_config()
        if 'palettes' not in config['colors']:
            config['colors']['palettes'] = []
        
        # 检查是否已存在相同的RGB值
        if request.colors not in config['colors']['palettes']:
            config['colors']['palettes'].append(request.colors)
            save_config(config)
        
        return {
            "status": "success",
            "data": request.colors
        }
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/color-palette")
async def get_color_palettes():
    try:
        config = load_config()
        palettes = config['colors'].get('palettes', [])
        return {
            "status": "success",
            "data": {
                "count": len(palettes),
                "palettes": palettes
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.put("/color-palette")
async def update_color_palette(request: ColorPaletteRequest):
    try:
        config = load_config()
        palettes = config['colors'].get('palettes', [])
        
        try:
            if request.old_colors:
                index = palettes.index(request.old_colors)
                palettes[index] = request.colors
                save_config(config)
            return {
                "status": "success",
                "data": request.colors
            }
        except ValueError:
            # 如果找不到旧的RGB值，直接返回成功
            return {
                "status": "success",
                "data": request.colors
            }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.delete("/color-palette")
async def delete_color_palette(request: ColorPaletteRequest):
    try:
        config = load_config()
        palettes = config['colors'].get('palettes', [])
        
        try:
            palettes.remove(request.colors)
            save_config(config)
        except ValueError:
            pass  # 如果找不到要删除的RGB值，直接忽略
        
        return {
            "status": "success",
            "message": "颜色已删除"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/config")
async def get_config():
    try:
        config = load_config()
        return {
            "status": "success",
            "data": config
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/config")
async def update_config(config: dict):
    try:
        current_config = load_config()
        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        updated_config = update_dict(current_config, config)
        save_config(updated_config)
        
        return {
            "status": "success",
            "data": updated_config
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.get("/paint/progress")
async def get_paint_progress():
    try:
        progress = manager.get_task_status('painting_task')
        return {
            "status": "success",
            "data": progress
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post("/paint/stop")
async def stop_paint():
    try:
        result = manager.stop_task('painting_task')
        return {
            "status": "success",
            "data": {
                "stopped": result,
                "message": "绘画已停止" if result else "当前没有进行中的绘画"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)}
        )

@app.post('/start_painting')
async def start_painting_endpoint(file: UploadFile = File(...)):
    config = load_config()
    config_values = {
        'render_width': config['canvas']['render_width'],
        'render_height': config['canvas']['render_height'],
        'num_strokes': config['painting']['num_strokes'],
        'ink': config['painting']['ink'],
        'color_palette': config['colors']['palettes'],
        'how_often_to_get_paint': config['painting']['how_often_to_get_paint'],
        'num_adaptations': config['painting']['num_adaptations'],
        'objective_data': config['objective']['data'],
        'objective': config['objective']['type'],
        'objective_weight': config['objective']['weight'],
        'init_optim_iter': config['optimization']['init_optim_iter'],
        'lr_multiplier': config['optimization']['lr_multiplier'],
        'cache_dir': config['system']['cache_dir'],
        'robot': config['system']['robot'],
        'use_cache': config['system']['use_cache'],
    }
    if file:
        # 将文件保存到项目的outputs目录
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        logger.info(f"Image uploaded: {file_path}")
        config_values['objective_data'] = ['../uploads/' + file.filename]

    painter, opt = init_painter(config_values)
    result = manager.start_task('painting_task', painter, opt, config_values)
    return {
        "status": "success",
        "data": {
            "message": "task start success",
            "timestamp": datetime.datetime.now().isoformat()
        }
    }

@app.post('/stop_painting')
async def stop_painting_endpoint():
    success = manager.stop_task('painting_task')
    return {
        "status": "success" if success else "failed",
        "message": "Painting task stopped" if success else "No running task to stop"
    }

@app.get('/painting_progress')
async def painting_progress_endpoint():
    task_status = manager.get_task_status('painting_task')
    return task_status

import asyncio

@app.websocket("/ws/canvas")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    import cv2
    from starlette.websockets import WebSocketState
    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.info("WebSocket client disconnected")
                break

            # 假设有一个函数 get_latest_canvas_image() 返回最新的canvas图像数据
            image_data = manager.get_latest_canvas_image()
            if image_data is None:
                logger.warning("image_data is None")
                await asyncio.sleep(1)
                continue

            try:
                # 将画布数据转换为图像格式
                _, buffer = cv2.imencode('.jpg', image_data)
                image_data = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_text(image_data)
            except Exception as send_error:
                logger.error(f"Error sending data: {str(send_error)}\nError details: {repr(send_error)}")
                break

            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # 确保在连接结束时进行清理
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

