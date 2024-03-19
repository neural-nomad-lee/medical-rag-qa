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
            system.cleanup_expired_knowledge()
        except Exception as e:
            print(f"[到期知识清理] 任务异常(将在下个周期重试): {e}")
        await asyncio.sleep(interval_hours * 3600)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """FastAPI 生命周期: 启动时创建清理后台任务, 关闭时取消."""
    cleanup_task = asyncio.create_task(_expired_knowledge_cleanup_loop())
    yield
    cleanup_task.cancel()


# 1. 创建FastAPI应用实例, 设置接口文档标题, 接口整体业务描述, 自动生成Swagger在线文档.
app = FastAPI(title="MedRAG 医疗智慧问答系统API", description="集成MySQL精准问答和RAG检索的医疗智能问答系统",
              lifespan=lifespan)

# 2. 全局注入CORS跨域中间件, 配置前后端跨域访问规则.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有前端域名来源访问, 生产环境需配置具体前端域名. 避免安全风险.
    allow_credentials=True,     # 允许凭证 -> 允许请求携带Cookie, Token等身份凭证.
    allow_methods=["*"],        # 允许所有 HTTP 方法 -> 放行全部Http请求: GET/POST/PUT/DELETE....
    allow_headers=["*"],        # 允许客户端所有自定义请求头, 如: 鉴权Token...
)

# 3. 静态资源目录: 基于当前文件位置计算绝对路径, 避免依赖启动时的工作目录(cwd).
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)


# todo 2. 全局固定问候语 正则规则配置.
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "你好！我是 MedRAG 医疗智能助手，专注于健康咨询答疑，很高兴为你服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是 MedRAG 医疗智能助手，致力于提供医疗健康相关的咨询解答！"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是 MedRAG 医疗智能助手，随时为你解答健康问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为你解答医疗健康相关的问题！有什么我可以帮你的？"
    }
]


# todo 3. Pydantic请求/响应数据模型定义
# todo 3.1 非流式查询接口入参结构体 -> 接收Post请求Json参数.
# 定义查询请求模型
class QueryRequest(BaseModel):
    query: str                           # 查询内容，必填
    source_filter: Optional[str] = None  # 科室过滤，可选
    session_id: Optional[str] = None     # 会话 ID，可选

# todo 3.2 非流式查询接口标准返回结构体 -> 统一接口出参格式.
# 定义查询响应模型
class QueryResponse(BaseModel):
    answer: str                 # 答案内容 -> 问答最终回复内容.
    is_streaming: bool          # 是否流式响应 -> 标签当前回复是否需要流式输出, True代表需切换WebSocket接口
    session_id: str             # 会话 ID -> 用于上下文历史关联
    processing_time: float      # 处理时间 -> 本次问答完整处理耗时, 单位: 秒.


# todo 4. 静态资源挂载与页面路由配置.
# 1. 将本地static目录挂载至服务器根路径.
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

# todo根路径 GET接口, 访问首页HTML页面.
@app.get("/")               # 这个/的意思是: http://ip地址:端口号/
async def read_root():
    # 返回static目录下的首页html文件 -> 实现域名根路径直接打开前端页面.
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))
