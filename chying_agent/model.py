import os
import json
from typing import Any, Dict, List, Optional
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from chying_agent.common import log_system_event
from chying_agent.config import AgentConfig


class CustomChatDeepSeek(ChatDeepSeek):
    """
    自定义 ChatDeepSeek 类，用于修复 reasoning_content 丢失的问题。
    DeepSeek API 在开启 thinking 模式后，要求历史消息中的 assistant 消息
    必须包含 reasoning_content 字段，否则会报错 400。
    """
    def _get_request_payload(self, input_: Any, *, stop: List[str] | None = None, **kwargs: Any) -> Dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        try:
            messages = input_
            if hasattr(input_, "to_messages"):
                messages = input_.to_messages()
            
            if "messages" in payload and isinstance(messages, list):
                # 过滤出源消息中的 AIMessage
                ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                
                # 过滤出 payload 中的 assistant 消息
                payload_ai_msgs = [m for m in payload["messages"] if m.get("role") == "assistant"]
                
                # 如果数量一致，尝试注入 reasoning_content
                if len(ai_msgs) == len(payload_ai_msgs):
                    for src, tgt in zip(ai_msgs, payload_ai_msgs):
                        # 尝试从 additional_kwargs 或 response_metadata 获取 reasoning_content
                        reasoning = src.additional_kwargs.get("reasoning_content")
                        if not reasoning and hasattr(src, "response_metadata"):
                            reasoning = src.response_metadata.get("reasoning_content")
                            
                        # 如果存在 reasoning_content 且 payload 中缺失，则注入
                        if reasoning and "reasoning_content" not in tgt:
                            tgt["reasoning_content"] = reasoning
                            
        except Exception as e:
            # 仅打印警告，不中断流程
            print(f"Warning: Failed to inject reasoning_content: {e}")
            
        return payload


def create_model(
    config: AgentConfig,
    temperature: float = 0.5,
    max_tokens: int = 12800,
    timeout: int = 600,
    max_retries: int = 20  # ⭐ 提升重试次数：2 → 10（应对并发速率限制）
) -> BaseChatModel:
    """
    创建模型实例

    Args:
        config: AgentConfig实例，包含LLM配置。
        temperature: 温度参数。
        max_tokens: 最大token数。
        timeout: 超时时间。
        max_retries: 重试次数。

    Returns:
        BaseChatModel: 模型实例。
    """
    model_name = config.llm_model_name
    
    # 使用自定义的 CustomChatDeepSeek
    model = CustomChatDeepSeek(
        api_base=config.llm_base_url,
        api_key=config.llm_api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        streaming=False,  # 禁用流式输出以支持结构化输出
        extra_body={
            "thinking": {
                "type": "enabled",
                "enable_search": True,
            }
        }
    )

    log_system_event(
        "✅ 创建CHYing Agent模型实例 (CustomDeepSeek)",
        {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
        },
    )

    return model
