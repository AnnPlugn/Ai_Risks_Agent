# langchain-openai injector concoction

from typing import Optional
from typing import Union


import ssl
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import httpx
from concoction import Configuration
from httpx import create_ssl_context
from httpx._types import CertTypes
from injector import Module, provider, singleton
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from openai._base_client import make_request_options
from openai._streaming import Stream
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai._utils import maybe_transform, required_args
from openai.resources.chat.completions import (
    async_maybe_transform,
    validate_response_format,
)
from openai.resources.completions import AsyncStream
from openai.types.chat import completion_create_params
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_audio_param import (
    ChatCompletionAudioParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_modality import ChatCompletionModality
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_reasoning_effort import (
    ChatCompletionReasoningEffort,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,
)
from openai.types.chat_model import ChatModel
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal


class SSLConfig(BaseModel):
    """pydantic wrapper around httpx._config.SSLConfig"""

    #cert: CertTypes | None = None
    cert: Union[CertTypes, None] = None
    verify: Union[str, bool] = True
    trust_env: bool = True
    http2: bool = False


class WithSSLConfig(BaseModel):
    """Inherit this with BaseModel to add SSL to config"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #ssl_config: SSLConfig | None = None
    ssl_config: Union[SSLConfig, None] = None
    #ssl_context: ssl.SSLContext | None = Field(init=False, default=None)
    ssl_context: Union[ssl.SSLContext, None] = Field(init=False, default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.ssl_config is None:
            self.ssl_context = None
            return

        self.ssl_context = create_ssl_context(
            cert=self.ssl_config.cert,
            verify=self.ssl_config.verify,
            trust_env=self.ssl_config.trust_env,
        )


@required_args(["messages", "model"], ["messages", "model", "stream"])
def custom_create(
    self,
    *,
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    #audio: Optional[ChatCompletionAudioParam] | NotGiven = NOT_GIVEN,
    audio: Union[ Optional[ChatCompletionAudioParam], NotGiven] = NOT_GIVEN,
    frequency_penalty: Union [Optional[float], NotGiven] = NOT_GIVEN,
    function_call: (
        Union[completion_create_params.FunctionCall, NotGiven]
    ) = NOT_GIVEN,
    functions: (
        Union[Iterable[completion_create_params.Function], NotGiven]
    ) = NOT_GIVEN,
    logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
    logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
    max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
    max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
    metadata: Union[Optional[Dict[str, str]], NotGiven] = NOT_GIVEN,
    modalities: Union[Optional[List[ChatCompletionModality]], NotGiven] = NOT_GIVEN,
    n: Union[Optional[int], NotGiven] = NOT_GIVEN,
    parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
    prediction: (
        Union[Optional[ChatCompletionPredictionContentParam], NotGiven]
    ) = NOT_GIVEN,
    presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
    reasoning_effort: Union[ChatCompletionReasoningEffort, NotGiven] = NOT_GIVEN,
    response_format: (
        Union[completion_create_params.ResponseFormat, NotGiven]
    ) = NOT_GIVEN,
    seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
    service_tier: Union[Optional[Literal["auto", "default"]], NotGiven] = NOT_GIVEN,
    stop: Union[ Union[Optional[str], List[str]], NotGiven] = NOT_GIVEN,
    store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
    stream: Union[ Optional[Literal[False]], Literal[True], NotGiven] = NOT_GIVEN,
    stream_options: (
        Union[Optional[ChatCompletionStreamOptionsParam], NotGiven]
    ) = NOT_GIVEN,
    temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
    tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
    tools: Union[ Iterable[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
    top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
    top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    user: Union[str, NotGiven] = NOT_GIVEN,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Union[Headers, None] = None,
    extra_query: Union[Query, None] = None,
    extra_body: Union[Body, None] = None,
    timeout: Union[ float,  httpx.Timeout, None, NotGiven] = NOT_GIVEN,
) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    validate_response_format(response_format)
    return self._post(
        "/predict",
        body=maybe_transform(
            {
                "messages": messages,
                "model": model,
                "audio": audio,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_completion_tokens": max_completion_tokens,
                "max_tokens": max_tokens,
                "metadata": metadata,
                "modalities": modalities,
                "n": n,
                "parallel_tool_calls": parallel_tool_calls,
                "prediction": prediction,
                "presence_penalty": presence_penalty,
                "reasoning_effort": reasoning_effort,
                "response_format": response_format,
                "seed": seed,
                "service_tier": service_tier,
                "stop": stop,
                "store": store,
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "user": user,
            },
            completion_create_params.CompletionCreateParams,
        ),
        options=make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        ),
        cast_to=ChatCompletion,
        stream=stream or False,
        stream_cls=Stream[ChatCompletionChunk],
    )


@required_args(["messages", "model"], ["messages", "model", "stream"])
async def async_custom_create(
    self,
    *,
    messages: Iterable[ChatCompletionMessageParam],
    model: Union[str, ChatModel],
    audio: Union[ Optional[ChatCompletionAudioParam], NotGiven] = NOT_GIVEN,
    frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
    function_call: (
        Union[completion_create_params.FunctionCall, NotGiven]
    ) = NOT_GIVEN,
    functions: (
        Union[ Iterable[completion_create_params.Function], NotGiven]
    ) = NOT_GIVEN,
    logit_bias: Union[ Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
    logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
    max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
    max_tokens: Union[ Optional[int], NotGiven] = NOT_GIVEN,
    metadata: Union[Optional[Dict[str, str]], NotGiven] = NOT_GIVEN,
    modalities: Union[Optional[List[ChatCompletionModality]], NotGiven] = NOT_GIVEN,
    n: Union[ Optional[int], NotGiven] = NOT_GIVEN,
    parallel_tool_calls: Union[ bool, NotGiven] = NOT_GIVEN,
    prediction: (
        Union[ Optional[ChatCompletionPredictionContentParam], NotGiven]
    ) = NOT_GIVEN,
    presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
    reasoning_effort: Union[ChatCompletionReasoningEffort, NotGiven] = NOT_GIVEN,
    response_format: (
        Union[ completion_create_params.ResponseFormat, NotGiven]
    ) = NOT_GIVEN,
    seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
    service_tier: Union[ Optional[Literal["auto", "default"]], NotGiven] = NOT_GIVEN,
    stop: Union[ Union[Optional[str], List[str]], NotGiven] = NOT_GIVEN,
    store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
    stream: Union[Optional[Literal[False]], Literal[True], NotGiven] = NOT_GIVEN,
    stream_options: (
        Union[ Optional[ChatCompletionStreamOptionsParam], NotGiven]
    ) = NOT_GIVEN,
    temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
    tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
    tools: Union[ Iterable[ChatCompletionToolParam] , NotGiven] = NOT_GIVEN,
    top_logprobs: Union[ Optional[int], NotGiven] = NOT_GIVEN,
    top_p: Union[ Optional[float], NotGiven] = NOT_GIVEN,
    user: Union[ str, NotGiven] = NOT_GIVEN,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Union[Headers, None] = None,
    extra_query: Union[Query, None] = None,
    extra_body: Union[Body, None] = None,
    timeout: Union[ float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
) -> Union[ ChatCompletion, AsyncStream[ChatCompletionChunk]]:
    validate_response_format(response_format)
    return await self._post(
        "/predict",
        body=await async_maybe_transform(
            {
                "messages": messages,
                "model": model,
                "audio": audio,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_completion_tokens": max_completion_tokens,
                "max_tokens": max_tokens,
                "metadata": metadata,
                "modalities": modalities,
                "n": n,
                "parallel_tool_calls": parallel_tool_calls,
                "prediction": prediction,
                "presence_penalty": presence_penalty,
                "reasoning_effort": reasoning_effort,
                "response_format": response_format,
                "seed": seed,
                "service_tier": service_tier,
                "stop": stop,
                "store": store,
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "user": user,
            },
            completion_create_params.CompletionCreateParams,
        ),
        options=make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        ),
        cast_to=ChatCompletion,
        stream=stream or False,
        stream_cls=AsyncStream[ChatCompletionChunk],
    )


class MefChatOpenAI(ChatOpenAI):
    """
    Обертка для OpenAI совместимого API, развернутого в MEF
    Идет 'переименование' эндпоинта /chat/completions в /predict
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client.create = types.MethodType(custom_create, self.client)
        self.async_client.create = types.MethodType(
            async_custom_create, self.async_client
        )


@Configuration("app.chat-openai")
class ChatOpenAIConfig(WithSSLConfig):
    model_name: str
    extra_body: dict = Field(default_factory=dict)
    openai_api_base: str
    openai_api_key: str = "abc"
    max_retries: Optional[int] = None


class MEFChatOpenAIModule(Module):

    @provider
    @singleton
    def provide_chat_openai_client(
        self, config: ChatOpenAIConfig
    ) -> MefChatOpenAI:
        client = httpx.Client(
            verify=config.ssl_context,  # type: ignore
            headers={"Content-Type": "application/json"},
        )
        async_client = httpx.AsyncClient(
            verify=config.ssl_context,  # type: ignore
            headers={"Content-Type": "application/json"},
        )

        openai_client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
            http_client=client,
        )

        async_openai_client = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base,
            http_client=async_client,
        )

        return MefChatOpenAI(
            client=openai_client.chat.completions,
            async_client=async_openai_client.chat.completions,
            model_name=config.model_name,  # type: ignore
            openai_api_base=config.openai_api_base,  # type: ignore
            openai_api_key=config.openai_api_key,  # type: ignore
            extra_body=config.extra_body,  # type: ignore
            max_retries=config.max_retries,  # type: ignore
        )

if __name__ == "__main__":
    print(str(Path(__file__).parent.parent / "client_cert.pem"))
    chat_openai_config = ChatOpenAIConfig(
        ssl_config=SSLConfig(
            cert=(
                #str(Path(__file__).parent.parent / "client_cert.pem"),
                #str(Path(__file__).parent.parent / "client_key.pem"),
                "client_cert.pem",
                "client_key.pem",
            ),
            verify=False,
        ),  # type: ignore
        model_name="Qwen2.5-Coder-32B-Instruct-AWQ",
        openai_api_base="https://sberorm-llm-deploy.ci03039946-eiftmefds-"
                        "vectorizer-service.apps.ift-mef-ds.delta.sbrf.ru/"
                        "sberorm-llm-deploy",
        max_retries=1,
    )
    chat_openai = MEFChatOpenAIModule().provide_chat_openai_client(
        chat_openai_config
    )
    #print(chat_openai.invoke("hello there"))
