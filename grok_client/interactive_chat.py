import argparse
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from .grok_openai_client import GrokOpenAIClient, _StreamWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with Grok API using OpenAI-compatible interface"
    )
    parser.add_argument("--model", help="Model name (default: grok-3)")
    parser.add_argument("--sso", help="SSO token (default: GROK_SSO env var)")
    parser.add_argument("--sso-rw", help="SSO-RW token (default: GROK_SSO_RW env var)")
    parser.add_argument("--json", action="store_true", help="Request responses in JSON format")
    parser.add_argument("--system", help="Custom system message")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for response generation (default: 1.0)",
    )
    return parser.parse_args()


def setup_client(args: argparse.Namespace) -> GrokOpenAIClient:
    load_dotenv()

    sso = args.sso or os.getenv("GROK_SSO")
    sso_rw = args.sso_rw or os.getenv("GROK_SSO_RW")

    cookies = None
    if sso or sso_rw:
        cookies = {"sso": sso or "", "sso-rw": sso_rw or ""}

    try:
        return GrokOpenAIClient(
            cookies=cookies if cookies else None,
            model_name=args.model or "grok-3",
        )
    except ValueError as e:
        logger.error(f"Error initializing client: {e}")
        sys.exit(1)


def interactive_chat() -> None:
    args = parse_arguments()
    client = setup_client(args)
    model_name = client.model_name

    system_message = args.system or ""
    if args.json and not system_message:
        system_message = "You are a helpful assistant that always responds in valid JSON format."
    elif not system_message:
        system_message = "You are a helpful assistant."

    print(f"\n===== Grok Interactive Chat ({model_name}) =====")
    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Type '/help' to see available commands.")
    print("==============================\n")

    conversation: list[dict[str, str]] = [{"role": "system", "content": system_message}]

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting chat. Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation = []
                if system_message:
                    conversation.append({"role": "system", "content": system_message})
                print("\nConversation history cleared.")
                continue

            if user_input.lower() == "/help":
                print("\nAvailable commands:")
                print("  exit, quit - Exit the chat")
                print("  clear - Clear conversation history")
                print("  /help - Show this help message")
                print("  /json - Toggle JSON response format")
                print("  /temp <value> - Set temperature (0.0-2.0)")
                print("  /system <message> - Set system message")
                continue

            if user_input.lower() == "/json":
                args.json = not args.json
                print(f"\nJSON response format: {'enabled' if args.json else 'disabled'}")
                continue

            if user_input.lower().startswith("/temp "):
                try:
                    new_temp = float(user_input.split(" ", 1)[1])
                    if 0.0 <= new_temp <= 2.0:
                        args.temperature = new_temp
                        print(f"\nTemperature set to: {args.temperature}")
                    else:
                        print("\nTemperature must be between 0.0 and 2.0")
                except (ValueError, IndexError):
                    print("\nInvalid temperature value. Format: /temp 0.7")
                continue

            if user_input.lower().startswith("/system "):
                system_message = user_input.split(" ", 1)[1]
                conversation = [msg for msg in conversation if msg["role"] != "system"]
                conversation.insert(0, {"role": "system", "content": system_message or ""})
                print(f"\nSystem message updated.")
                continue

            conversation.append({"role": "user", "content": user_input})

            try:
                print("\nGrok: ", end="", flush=True)

                params: dict[str, Any] = {
                    "messages": conversation,
                    "stream": True,
                    "temperature": args.temperature,
                }

                if args.json:
                    params["response_format"] = {"type": "json_object"}

                raw = client.chat_completion(**params)
                assert isinstance(raw, _StreamWrapper)
                full_response = client.process_streaming_response(raw)

                conversation.append({"role": "assistant", "content": full_response})

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")

    except KeyboardInterrupt:
        print("\n\nExiting chat. Goodbye!")


def main() -> None:
    load_dotenv()
    interactive_chat()


if __name__ == "__main__":
    main()
