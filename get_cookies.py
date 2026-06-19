import json
import logging
import os
import sys
import time
from typing import Optional
from urllib.parse import urlparse

import undetected_chromedriver as uc
from dotenv import load_dotenv
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("GrokAuth")


def parse_proxy_settings() -> Optional[str]:
    """Safely parse proxy from environment and prepare Chromium argument."""
    proxy_url = os.getenv("PROXY_URL")
    if not proxy_url:
        return None

    parsed = urlparse(proxy_url)
    proxy_address = parsed.netloc if parsed.netloc else parsed.path
    return proxy_address


def verify_login() -> None:
    """
    Launch a secure Chrome instance for manual authentication on grok.com.
    Eliminates false positives on guest pages by checking authorization markers.
    """
    global login_detected, captured_cookies, grok_cookies
    load_dotenv()

    chrome_user_data_dir = os.getenv("CHROME_USER_DATA_DIR")

    options = uc.ChromeOptions()
    options.add_argument("--disable-notifications")
    options.add_experimental_option("prefs", {
        "translate_enabled": False,
        "translate_deny_list": ["*"],
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "signin.promotion": False,
        "signin.enabled": False,
        "signin.SyncPromo": False,
        "signin.GoogleOfflineSignInPromo": False,
        "signin.GoogleSignInPromo": False,
        "signin.GoogleSignInRequiredPromo": False,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
    })
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-save-password-bubble")
    options.add_argument("--disable-translate")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-download-notification")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-component-extensions-with-background-pages")
    options.add_argument("--disable-client-side-phishing-detection")
    options.add_argument("--disable-hang-monitor")
    options.add_argument("--disable-prompt-on-repost")
    options.add_argument("--disable-domain-reliability")
    options.add_argument("--disable-features=site-per-process")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")

    if chrome_user_data_dir:
        abs_user_data = os.path.abspath(chrome_user_data_dir)
        options.user_data_dir = abs_user_data
        logger.info(f"Using specified Chrome profile: {abs_user_data}")

    proxy_netloc = parse_proxy_settings()
    if proxy_netloc:
        options.add_argument(f"--proxy-server={proxy_netloc}")
        logger.info(f"Proxy server configured: {proxy_netloc}")

    logger.info("Starting secure browser instance (Undetected Chromedriver)...")
    try:
        driver = uc.Chrome(version_main=149, options=options, headless=False)
    except Exception as e:
        logger.critical(f"Failed to initialize Chromium: {e}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.abspath(os.path.join(script_dir, "cookies/grok_cookies.json"))

    try:
        logger.info("Navigating directly to xAI sign-in page ...")
        driver.get("https://accounts.x.ai/sign-in?redirect=grok-com&return_to=%2F%3Fq%3D%26reasoning")

        print("\n" + "=" * 75)
        print("[INSTRUCTION] Log in to your account in the opened browser window.")
        print("The script will capture the session ONLY after login completion and chat redirect.")
        print("=" * 75 + "\n")

        login_detected = False
        visited_login_page = False
        captured_cookies = {}

        while not login_detected:
            try:
                current_url = driver.current_url
            except WebDriverException:
                logger.info("Browser window closed by user. Session ended.")
                driver.quit()
                return

            if "accounts.x.ai" in current_url:
                if not visited_login_page:
                    logger.info("Authorization page detected: accounts.x.ai. Waiting for login completion...")
                    visited_login_page = True

            is_grok = "grok.com" in current_url
            is_chat_url = "reasoningMode" in current_url or "voice=" in current_url or "/chat" in current_url
            is_logged_out_zone = False
            if is_grok:
                try:
                    login_elements = driver.find_elements(
                        By.XPATH,
                        "//a[contains(@href, 'accounts.x.ai')] | "
                        "//button[contains(text(), 'Sign in') or contains(text(), 'Log in')]"
                    )
                    if len(login_elements) > 0:
                        is_logged_out_zone = True
                        logger.debug(f"Found {len(login_elements)} login element(s) - user appears logged out")
                except WebDriverException as e:
                    logger.debug(f"Error checking login elements: {e}")

            if is_grok and not is_logged_out_zone and (is_chat_url or visited_login_page):
                logger.info("=" * 60)
                logger.info("SUCCESS: Login verified!")
                logger.info(f"Current URL: {current_url}")
                logger.info("Capturing session cookies...")
                logger.info("=" * 60)
                login_detected = True
                break

            if int(time.time()) % 10 == 0 and not login_detected:
                logger.info(f"Waiting for login... Current: {current_url[:80]}")

            time.sleep(1)

        if login_detected:
            try:
                selenium_cookies = driver.get_cookies()
                captured_cookies = {c["name"]: c["value"] for c in selenium_cookies}

                grok_cookies = {k: v for k, v in captured_cookies.items() if
                                "grok.com" in str(k) or k in ["x-anonuserid", "x-challenge", "x-signature", "sso",
                                                              "sso-rw"]}

                with open(resolved_path, "w", encoding="utf-8") as f:
                    json.dump(grok_cookies, f, indent=2, ensure_ascii=False)

                logger.info(f"Session captured successfully. Total cookies: {len(captured_cookies)}")
                logger.info(f"Grok cookies filtered: {len(grok_cookies)}")
                logger.info(f"Cookies saved to: {resolved_path}")

            except (WebDriverException, OSError, json.JSONDecodeError) as cookie_err:
                logger.critical(f"Failed to save cookies: {cookie_err}")
                driver.quit()
                return

    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user (Ctrl+C).")
    finally:
        logger.info("Shutting down browser...")
        try:
            driver.quit()
        except WebDriverException:
            pass

        if login_detected:
            print("\n" + "=" * 75)
            print("SESSION CAPTURE COMPLETE")
            print("=" * 75)
            print(f"Output file : {resolved_path}")
            print(f"Cookies     : {len(grok_cookies)} captured")
            print("=" * 75)
            print("\nPress ENTER to close this window...")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                pass

        logger.info("Verification script finished.")


if __name__ == "__main__":
    try:
        verify_login()
    except KeyboardInterrupt:
        sys.exit(0)
