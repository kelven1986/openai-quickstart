from typing import Any
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator
from langchain.schema import (
    BaseMessage,
    ChatMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ChatResult,
    ChatGeneration,
)
from transformers import AutoTokenizer, AutoModel

class ChatGLM2(BaseChatModel):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).float()
    model.eval()

    @root_validator()
    def validate_environment(cls, values):
        return values

    def _generate(
        self,
        messages: [BaseMessage],
        stop: [[str]] = None,
        run_manager: [CallbackManagerForLLMRun] = None,
        **kwargs: dict,
    ) -> ChatResult:
        message = self._convert_messages(messages)
        print(message)

        response = self._chat(message)
        print(response)

        generations = self._get_generations(response)
        return ChatResult(generations=generations)

    def _chat(self, message: str) -> str:
        response, history = self.model.chat(self.tokenizer, message, history=[])
        return response

    def _convert_messages(self, messages: [BaseMessage]) -> str:
        return " ".join([m.content for m in messages])

    def _get_generations(self, response: str) -> [ChatGeneration]:
        return [ChatGeneration(message=AIMessage(content=response))]

    @property
    def _llm_type(self) -> str:
        return "chatGLM2-6b"