import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .client import GrokClient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str
    function_call: dict[str, Any] | None = None


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = 1.0
    max_tokens: int | None = None
    functions: list[Function] | None = None
    function_call: str | dict[str, str] | None = None
    response_format: dict[str, str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]


class GrokAPI:
    def __init__(self, cookies: dict[str, str]):
        self.client = GrokClient(cookies)

    @staticmethod
    def prepare_system_message(request: ChatCompletionRequest) -> str:
        system_content = (
            "You are a helpful assistant. Provide direct, simple answers to questions."
        )

        if request.functions:
            system_content = "You are a helpful assistant that provides structured data."
            system_content += f" Available functions: {[f.name for f in request.functions]}"
            system_content += (
                f" Function schemas: {json.dumps([f.model_dump() for f in request.functions])}"
            )

        elif request.response_format and request.response_format.get("type") == "json_object":
            system_content = (
                "You are a helpful assistant that always responds in valid JSON format."
            )

        return system_content

    def stream_chat(self, request: ChatCompletionRequest):
        try:
            system_msg = self.prepare_system_message(request)
            conversation = (
                    f"system: {system_msg}\n"
                    + "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            )

            logger.debug(f"Sending conversation to Grok: {conversation}")

            response_stream = self.client.send_message(conversation)
            logger.debug(f"Got response stream from Grok: {response_stream}")

            for token in response_stream.split():
                chunk = ChatCompletionChunk(
                    id="chatcmpl-" + str(int(time.time())),
                    created=int(time.time()),
                    model="grok-3",
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": token + " "},
                            "finish_reason": None,
                        }
                    ],
                )
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

            final_chunk = ChatCompletionChunk(
                id="chatcmpl-final",
                created=int(time.time()),
                model="grok-3",
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )
            yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in stream_chat: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "grok-3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai",
                "permission": [],
                "root": "grok-3",
                "parent": None,
            }
        ]
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    try:
        body = await raw_request.json()
        logger.debug(f"Received request body: {body}")

        request = ChatCompletionRequest(**body)

        headers = dict(raw_request.headers)
        logger.debug(f"Received headers: {headers}")

        cookies = (
            {"Cookie": headers.get("cookie", "")} if headers.get("cookie") else {}
        )
        logger.debug(f"Extracted cookies: {cookies}")

        if not cookies:
            raise HTTPException(
                status_code=401, detail="No authentication cookies provided"
            )

        grok = GrokAPI(cookies)

        if request.stream:
            return StreamingResponse(
                grok.stream_chat(request),
                media_type="text/event-stream",
            )

        system_msg = grok.prepare_system_message(request)
        conversation = (
                f"system: {system_msg}\n"
                + "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        )
        logger.debug(f"Sending conversation to Grok: {conversation}")

        response = grok.client.send_message(conversation)
        logger.debug(f"Received response from Grok: {response}")

        if not response:
            logger.error("Empty response from Grok API")
            raise HTTPException(status_code=500, detail="Empty response from Grok API")

        if request.functions and request.function_call:
            try:
                parsed_response = json.loads(response)

                function_name = (
                    request.function_call.get("name", request.functions[0].name)
                    if isinstance(request.function_call, dict)
                    else request.functions[0].name
                )

                message = ChatMessage(
                    role="assistant",
                    content="",
                    function_call={
                        "name": function_name,
                        "arguments": json.dumps(parsed_response),
                    },
                )
            except json.JSONDecodeError:
                function_name = (
                    request.function_call.get("name", request.functions[0].name)
                    if isinstance(request.function_call, dict)
                    else request.functions[0].name
                )
                message = ChatMessage(
                    role="assistant",
                    content="",
                    function_call={
                        "name": function_name,
                        "arguments": json.dumps({"result": response}),
                    },
                )
        else:
            if (
                    request.response_format
                    and request.response_format.get("type") == "json_object"
            ):
                try:
                    json.loads(response)
                    message = ChatMessage(role="assistant", content=response)
                except json.JSONDecodeError:
                    message = ChatMessage(
                        role="assistant",
                        content=json.dumps({"response": response}),
                    )
            else:
                message = ChatMessage(role="assistant", content=response)

        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{str(int(time.time()))}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(message=message, finish_reason="stop")
            ],
        )

        logger.debug(f"Sending response: {chat_response.model_dump()}")
        return chat_response

    except Exception as e:
        logger.error(f"Error in create_chat_completion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Failed to process request"},
        )
