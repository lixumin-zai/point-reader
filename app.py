# -*- coding: utf-8 -*-
# @Time    :   2025/09/29 14:53:52
# @Author  :   lixumin1030@gmail.com
# @FileName:   app.py


import logging
import base64
import traceback
import io
import aiohttp

from PIL import Image
from fastapi import FastAPI, Request, Depends, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from setup_logging import setup_logging, request_id_ctx_var, generate_request_id, set_request_id, get_request_id
from test import PointReader
 

# 全局初始化logging
setup_logging(log_level="INFO", log_file="point-reader.log")
logger = logging.getLogger(__name__)

# 初始化模型
point_reader = PointReader("./ckpt")

app = FastAPI(title="Point Reader API", version="0.0.1")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求生成和设置request_id的中间件"""
    
    async def dispatch(self, request: Request, call_next):
        # 从请求头获取request_id，如果没有则生成新的
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        
        # 设置到上下文变量中
        set_request_id(request_id)
        
        # 记录请求开始
        logger.info(f"** Request started **: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            # 将request_id添加到响应头
            response.headers["X-Request-ID"] = request_id
            logger.info(f"Request completed: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"** Request failed **: {str(e)}", exc_info=True)
            raise

# 添加request_id中间件
app.add_middleware(RequestIdMiddleware)

class UploadData(BaseModel):
    image_base64: str = Field(default="", description="base64")
    image_url: str = Field(default="", description="url")
    prompt: str = Field(default="", description="prompt")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    logger.info("Health check requested")
    return {"status": "healthy", "request_id": get_request_id()}

@app.post("/point-reader")
async def point_reader_exc(upload_data: UploadData, request: Request):
    """点读接口"""
    logger.info(f"** 点读处理 **")
    logger.info(f"prompt: {upload_data.prompt}")
    
    try:
        # 验证输入数据
        if not upload_data.image_base64 and not upload_data.image_url:
            logger.warning("错误输入")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="必须提供 image_base64 或 image_url"
            )
        
        # 准备图像数据
        
        if upload_data.image_base64:
            logger.info("处理 base64 image")
            image_data = upload_data.image_base64
            image_bytes = base64.b64decode(image_data)

        elif upload_data.image_url:
            logger.info(f"处理 image url: {upload_data.image_url}")
            image_url = upload_data.image_url
            # 异步下载图像
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    image_bytes = await resp.read()
        
        image = Image.open(io.BytesIO(image_bytes))
        # 调用模型处理图像
        generated_text = point_reader(upload_data.prompt, image)
        # </s><s><loc_275><loc_120><loc_348><loc_91><loc_357><loc_131><loc_284><loc_161></s>
        logger.info(f"生成文本: {generated_text}")

        return {
                "code": 0, 
                "message": "success", 
                "data": {
                    "generated_text": generated_text,
                },
                "request_id": get_request_id()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"服务错误 : \n******** error \n{traceback.format_exc()}********")
        return {
            "code": -1,
            "message": "Internal server error",
            "request_id": get_request_id()
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Diagram Vie server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=20070, 
        reload=False,
        log_config=None  # 使用我们自定义的logging配置
    )
    # nohup python main.py > main.log &