import json
import logging
import os
import re
import uuid

import requests

logger = logging.getLogger(__name__)


class GrokClient:
    def __init__(self, cookies_str: str = ""):
        """
        Initialize the Grok client with a cookie string.

        Args:
            cookies_str: Raw Cookie header string, e.g.
                "sso=xxx; sso-rw=yyy". If empty, no cookies are set.
        """
        self.base_url = "https://grok.com/rest/app-chat"
        self.proxies = None
        proxy_url = os.getenv("PROXY_URL")
        if proxy_url:
            proxy_username = os.getenv("PROXY_USERNAME")
            proxy_password = os.getenv("PROXY_PASSWORD")
            if proxy_username and proxy_password:
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    proxy_url = f"{scheme}://{proxy_username}:{proxy_password}@{rest}"

            no_proxy = os.getenv("NO_PROXY", "")
            target_host = self.base_url.split("://", 1)[-1].split("/", 1)[0]
            if any(target_host.endswith(host.strip()) for host in no_proxy.split(",") if host.strip()):
                self.proxies = None
            else:
                self.proxies = {
                    "http": proxy_url,
                    "https": proxy_url,
                }

        self.headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Cookie": cookies_str,
        }
        logger.debug(f"Initialized GrokClient with headers: {self.headers}")

    @staticmethod
    def _prepare_payload(
            message,
            temperature: float | None = None,
            max_tokens: int | None = None,
    ):
        """Prepare the default payload with the user's message"""
        payload = {
            "temporary": False,
            "modelName": "grok-3",
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": False,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": False,
            "imageGenerationCount": 0,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.debug(f"Prepared payload: {payload}")
        return payload

    @staticmethod
    def _clean_json_response(response):
        """Clean up JSON response by removing markdown and code blocks"""
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)

        try:
            json_data = json.loads(response)

            if isinstance(json_data, dict):
                if "response" in json_data:
                    json_data = json_data["response"]
                elif "function_call" in json_data:
                    json_data = json_data["function_call"]["arguments"]
                    if isinstance(json_data, str):
                        json_data = json.loads(json_data)

            return json.dumps(json_data, indent=2)
        except json.JSONDecodeError:
            return response

    def send_message(self, message, temperature=None, max_tokens=None):
        """
        Send a message to Grok and collect the streaming response

        Args:
            message (str): The user's input message
            temperature (float, optional): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate

        Returns:
            str: The complete response from Grok
        """
        try:
            logger.debug(f"Sending message to Grok: {message}")
            payload = self._prepare_payload(message, temperature, max_tokens)

            conversation_id = str(uuid.uuid4())
            url = f"{self.base_url}/conversations/{conversation_id}"

            logger.debug(f"Making POST request to {url}")

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True,
                proxies=self.proxies,
            )

            logger.debug(f"Response status code: {response.status_code}")
            if response.status_code == 404:
                logger.debug("Primary endpoint returned 404, trying fallback /new endpoint")
                fallback_url = "https://grok.com/rest/app-chat/conversations/new"
                response = requests.post(
                    fallback_url,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    proxies=self.proxies,
                )
                logger.debug(f"Fallback response status code: {response.status_code}")
            response.raise_for_status()

            full_response = ""
            last_response = None

            logger.debug("Processing response stream...")
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        logger.debug(f"Received line: {decoded_line}")

                        json_data = json.loads(decoded_line)
                        logger.debug(f"Parsed JSON: {json_data}")

                        if "error" in json_data:
                            error_msg = json_data["error"]
                            logger.error(f"Error in response: {error_msg}")
                            raise Exception(error_msg)

                        result = json_data.get("result", {})
                        response_data = result.get("response", {})
                        logger.debug(f"Response data: {response_data}")

                        if "modelResponse" in response_data:
                            complete_response = response_data["modelResponse"].get("message", "")
                            if complete_response:
                                logger.debug(f"Got complete response: {complete_response}")
                                return self._clean_json_response(complete_response)

                        token = response_data.get("token", "")
                        if token:
                            full_response += token
                            last_response = full_response
                            logger.debug(f"Current response: {full_response}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing response line: {e}")
                        if str(e):
                            raise Exception(f"Error in response: {str(e)}")
                        continue

            if last_response:
                logger.debug(f"Returning last valid response: {last_response.strip()}")
                return self._clean_json_response(last_response.strip())

            logger.error("No valid response received from Grok API")
            raise Exception("No valid response received from Grok API")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            raise Exception(f"Failed to process response: {str(e)}")
