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
