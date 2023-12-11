# 该脚本用于: 检索策略选择器.

# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base.config import config
from base.logger import logger
# 导入 OpenAI
from openai import OpenAI


# todo 1. 定义StrategySelector类: 用于根据用户查询选择最合适的检索增强策略.
class StrategySelector:
    # todo 1.1 初始化方法 -> 创建大模型客户端, 加载策略选择的提示模板.
    def __init__(self):
        # 1. 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
                             base_url=config.DASHSCOPE_BASE_URL)
        # 2. 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

    # todo 1.2 调用大模型的API -> 向DashScope发送请求, 获取模型返回结果.
