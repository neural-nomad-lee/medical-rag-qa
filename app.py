# 该脚本用于: 基于FastAPI搭建智能问答系统Web服务, 提供Http接口, WebSocket流式问答, 静态资源服务, 会话历史管理, 跨域配置
#           健康探针等完整Web能力, 底层对接集成 RAG + MySQL的问答核心系统.

# todo 0. 导包
# 导入 FastAPI 相关模块，用于构建 API 和 WebSocket  -> 构建Http接口, WebSocket长连接, 异常抛出, 请求参数依赖注入...
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入 FastAPI 响应类型，用于流式响应和文件服务 -> 流式字节输出, 返回本地静态文件.
from fastapi.responses import StreamingResponse, FileResponse
# 导入 CORS 中间件，支持跨域请求 -> 解决前端浏览器访问后端接口产生的CORS跨域拦截问题.
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入 WebSocket 断开异常
from starlette.websockets import WebSocketDisconnect
# 导入系统操作模块，用于文件目录管理
import os
# 导入 Pydantic 模型，用于请求验证
from pydantic import BaseModel
# 导入异步事件循环模块
import asyncio
# 导入 JSON 处理模块
import json
# 导入 UUID 模块，生成唯一会话 ID
import uuid
# 导入类型注解模块
from typing import Optional, List, Dict, Any
# 导入时间模块，记录处理时间
import time
# 导入正则表达式模块，用于匹配日常问候
import re
# 导入异步上下文管理器, 用于 FastAPI lifespan 生命周期管理
from contextlib import asynccontextmanager
# 导入优化后的问答系统
from new_main import IntegratedQASystem



# todo 1. FastAPI应用实例初始化 和 全局中间件配置.
# 【启动容错】问答核心系统改为"惰性实例化": 首次请求时才连接 MySQL/Redis/Milvus 并加载模型.
# 原实现(模块导入时即实例化)导致任一外部依赖未就绪, 服务直接启动失败且无任何接口可用.
qa_system: Optional[IntegratedQASystem] = None


def get_qa_system() -> IntegratedQASystem:
    """获取全局问答系统单例, 首次调用时才初始化(连接数据库/加载模型等重操作)."""
    global qa_system
    if qa_system is None:
        qa_system = IntegratedQASystem()
    return qa_system


async def _expired_knowledge_cleanup_loop(interval_hours: int = 24):
    """
    后台任务: 定期清理MySQL中已到期的沉淀知识(半年到期自动删除).
    说明: FAQ种子数据 expire_at 为 NULL, 不受影响; 清理失败不影响服务运行.
    """
    while True:
        try:
            # 惰性初始化: 若外部依赖未就绪, 捕获异常后等待下个周期重试, 不阻断服务.
            system = get_qa_system()
