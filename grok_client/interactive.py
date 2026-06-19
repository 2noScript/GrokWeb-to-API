import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_client():
    """Set up and return the OpenAI client with Grok API configuration"""
    load_dotenv()

    api_url = os.getenv("GROK_API_URL", "http://127.0.0.1:8000")
    model_name = os.getenv("MODEL_NAME", "grok-3")
    grok_sso = os.getenv("GROK_SSO")
    grok_sso_rw = os.getenv("GROK_SSO_RW")

    if not all([grok_sso, grok_sso_rw]):
        logger.error("Missing required environment variables. Please check your .env file.")
        logger.info("Required variables: GROK_SSO, GROK_SSO_RW")
        logger.info("Optional variables: GROK_API_URL, MODEL_NAME")
        sys.exit(1)

    client = OpenAI(
        base_url=f"{api_url}/v1",
        api_key="dummy-key",
        default_headers={
            "Cookie": f"sso={grok_sso}; sso-rw={grok_sso_rw}",
        },
    )

    return client, model_name


def interactive_chat():
    """Run an interactive chat session with Grok"""
    client, model_name = setup_client()

    print("\n===== Grok Interactive Chat ====="
          "\nType 'exit', 'quit', or Ctrl+C to end the conversation."
          "\nType 'clear' to start a new conversation."
          "\n==============================\n")

    conversation = []

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting chat. Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation = []
                print("\nConversation history cleared.")
                continue

            conversation.append({"role": "user", "content": user_input})

            try:
                print("\nGrok: ", end="", flush=True)

                stream = client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    stream=True,
                )

                full_response = ""
                for chunk in stream:
                    c: Any = chunk
                    if c.choices[0].delta.content:
                        print(c.choices[0].delta.content, end="", flush=True)
                        full_response += c.choices[0].delta.content

                print()
                conversation.append({"role": "assistant", "content": full_response})

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")

    except KeyboardInterrupt:
        print("\n\nExiting chat. Goodbye!")


if __name__ == "__main__":
    interactive_chat()
