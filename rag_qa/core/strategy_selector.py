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
    def call_dashscope(self, prompt):
        """
        调用DashScope大模型API, 获取模型对输入提示的响应.
        :param prompt: 发给大模型的提示文本(字符和窜), 包含: 用户查询和任务要求.
        :return: 大模型返回的文本结果(字符串), 这里是选择的策略名称, 如果调用失败, 直接返回: 直接检索
        """
        try:
            # 1. 创建聊天完成请求 -> 调用大模型, 传入模型名称, 对话内容, 温度参数.
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1
            )
            # 2. 返回完成结果 -> 如果有结果, 返回第1个回答的内容, 否则返回默认策略.
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 3. 异常处理.
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
