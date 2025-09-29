# -*- coding: utf-8 -*-
# @Time    :   2025/09/29 14:20:26
# @Author  :   lixumin1030@gmail.com
# @FileName:   setup_logging.py


import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

# 创建request_id的上下文变量
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class RequestIdFilter(logging.Filter):
    """添加request_id到日志记录中"""
    
    def filter(self, record):
        request_id = request_id_ctx_var.get()
        record.request_id = request_id or 'no-request-id'
        return True

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置全局logging配置"""
    
    # 创建自定义格式器
    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(request_id)s] - %(levelname)s - %(name)s - line : %(lineno)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加request_id过滤器
    request_filter = RequestIdFilter()
    
    # 控制台输出handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_filter)
    root_logger.addHandler(console_handler)
    
    # 文件输出handler（如果指定了文件）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_filter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging setup completed with level: {log_level}")

def generate_request_id() -> str:
    """生成新的request_id"""
    return str(uuid.uuid4())

def set_request_id(request_id: str):
    """设置当前请求的request_id"""
    request_id_ctx_var.set(request_id)

def get_request_id() -> Optional[str]:
    """获取当前请求的request_id"""
    return request_id_ctx_var.get()