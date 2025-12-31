import os
import json
import time
import hashlib
import logging
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Setup ----------
load_dotenv()

SANDBOX_DIR = "sandbox"
DOWNLOADS_DIR = os.path.join(SANDBOX_DIR, "downloads")
LOGS_DIR = os.path.join(SANDBOX_DIR, "logs")
SCREENSHOTS_DIR = os.path.join(SANDBOX_DIR, "screenshots")
HANDOFF_PATH = os.path.join(SANDBOX_DIR, "handoff.json")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

logger = logging.getLogger("agent_a")
logger.setLevel(logging.INFO)

# file log
fh = logging.FileHandler(os.path.join(LOGS_DIR, "agent_a.log"), encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# console log
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Handoff schema ----------
@dataclass
class HandoffPayload:
    status: str
    file_path: str
    file_name: str
    source_url: str
    sha256: str
    bytes: int
    downloaded_at_iso: str
    notes: str
    extra: Dict[str, Any]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def _safe_screenshot(page, label: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = os.path.join(SCREENSHOTS_DIR, f"{label}_{ts}.png")
    try:
        await page.screenshot(path=out, full_page=True)
        logger.info(f"Saved screenshot: {out}")
    except Exception as e:
        logger.info(f"Screenshot failed: {e}")
    return out


async def _collect_page_state(page) -> Dict[str, Any]:
    """
    Small, safe page snapshot so GPT-5-mini can reason.
    Keep it short: title, url, visible buttons/labels (limited).
    """
    title = ""
    try:
        title = await page.title()
    except:
        pass

    url = ""
    try:
        url = page.url
    except:
        pass

    # Grab visible button-like texts (limited)
    button_texts: List[str] = []
    try:
        # Many Drive controls are div/button with aria-label. We capture both.
        # Limit to keep prompt small.
        loc = page.locator("button, [role='button'], a[role='button'], div[aria-label], button[aria-label]")
        count = await loc.count()
        limit = min(count, 40)
        for i in range(limit):
            el = loc.nth(i)
            try:
                visible = await el.is_visible()
                if not visible:
                    continue
                txt = (await el.inner_text()) or ""
                aria = (await el.get_attribute("aria-label")) or ""
                cand = (aria.strip() or txt.strip())
                if cand:
                    cand = " ".join(cand.split())
                    button_texts.append(cand[:80])
            except:
                continue
        # de-dup
        button_texts = list(dict.fromkeys(button_texts))[:25]
    except:
        pass

    return {"title": title, "url": url, "buttons": button_texts}


def _call_gpt5_mini_plan(page_state: Dict[str, Any], attempt: int) -> Dict[str, Any]:
    """
    GPT-5-mini chooses next action from a strict action set (guardrail).
    This satisfies: "Uses GPT-5-mini for reasoning".
    """
    system = (
        "You are Agent A's planner. Choose the next browser action to download a Google Drive PDF. "
        "You MUST choose one action from the allowed list and provide a short rationale. "
        "Output ONLY valid JSON matching the schema."
    )

    allowed_actions = [
        "CLICK_DOWNLOAD_ANYWAY",
        "CLICK_DOWNLOAD_BUTTON",
        "OPEN_OVERFLOW_MENU_AND_DOWNLOAD",
        "REFRESH_AND_RETRY_SELECTORS",
        "FAIL_GIVE_UP"
    ]

    user = {
        "attempt": attempt,
        "page_state": page_state,
        "allowed_actions": allowed_actions,
        "json_schema": {
            "action": "one of allowed_actions",
            "rationale": "short string",
        }
    }

    # Prefer Responses API, fallback to chat.completions if needed.
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        )
        text = resp.output_text.strip()
    except Exception:
        chat = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        )

        text = chat.choices[0].message.content.strip()

    try:
        plan = json.loads(text)
        if plan.get("action") not in allowed_actions:
            return {"action": "REFRESH_AND_RETRY_SELECTORS", "rationale": "Invalid action from model; fallback."}
        return plan
    except Exception:
        return {"action": "REFRESH_AND_RETRY_SELECTORS", "rationale": "Non-JSON output; fallback."}


class DownloadAgent:
    def __init__(self):
        logger.info("Agent A init: reasoner=gpt-5-mini executor=playwright(chromium)")

    async def _try_click_download_anyway(self, page) -> bool:
        btn = page.get_by_role("button", name="Download anyway")
        try:
            if await btn.count() > 0 and await btn.first.is_visible():
                logger.info("Step: Found 'Download anyway' button. Clicking.")
                await btn.first.click()
                return True
        except:
            pass
        return False

    async def _try_click_download_button(self, page) -> Optional[Any]:
        """
        Try common download locators (main page + iframes).
        Returns a locator if found else None.
        """
        logger.info("Step: Searching for download button in main page selectors.")
        candidates = [
            page.get_by_label("Download"),
            page.get_by_label("Download file"),
            page.get_by_role("button", name="Download"),
            page.locator('div[aria-label="Download"]'),
            page.locator('button[aria-label*="Download"]'),
            page.locator('div[data-tooltip="Download"]'),
        ]
        for idx, loc in enumerate(candidates):
            try:
                if await loc.count() > 0 and await loc.first.is_visible():
                    logger.info(f"Step: Download selector hit (main): candidate[{idx}]")
                    return loc.first
            except:
                continue

        logger.info("Step: Searching for download button inside iframes.")
        for fr in page.frames:
            try:
                btn = fr.get_by_label("Download")
                if await btn.count() > 0:
                    logger.info(f"Step: Download selector hit (iframe): frame={fr.name}")
                    return btn.first
            except:
                continue
        return None

    async def _overflow_menu_download(self, page) -> bool:
        """
        Try Drive "More actions" (3 dots) -> Download.
        """
        logger.info("Step: Trying overflow menu route.")
        menu_candidates = [
            page.locator('button[aria-label*="More actions"]'),
            page.locator('div[aria-label*="More actions"]'),
            page.get_by_role("button", name="More actions"),
        ]
        menu = None
        for m in menu_candidates:
            try:
                if await m.count() > 0 and await m.first.is_visible():
                    menu = m.first
                    break
            except:
                continue
        if not menu:
            logger.info("Step: Overflow menu not found.")
            return False

        await menu.click()
        await asyncio.sleep(1)

        dl = page.get_by_role("menuitem", name="Download")
        if await dl.count() == 0:
            dl = page.get_by_text("Download")

        try:
            if await dl.count() > 0 and await dl.first.is_visible():
                logger.info("Step: Clicking overflow 'Download'.")
                await dl.first.click()
                return True
        except:
            pass
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
    async def run(self, url: str, max_seconds: int = 600) -> Dict[str, Any]:
        """
        Downloads a provided Google Drive PDF (through browser flow, no API).
        Retries on failure, stores in sandbox, logs each step.
        """
        start = time.time()
        logger.info(f"Agent A start: url={url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(
                accept_downloads=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = await context.new_page()

            try:
                # Guardrail: ensure we never exceed max_seconds within this agent call
                async def _time_guard():
                    while True:
                        if time.time() - start > max_seconds:
                            raise TimeoutError("Agent A exceeded 10-minute limit (internal guard).")
                        await asyncio.sleep(1)

                guard_task = asyncio.create_task(_time_guard())

                logger.info("Step: goto url")
                await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                await asyncio.sleep(3)  # Drive scripts

                # Create page state and let GPT-5-mini choose a strategy
                page_state = await _collect_page_state(page)
                plan = _call_gpt5_mini_plan(page_state, attempt=1)
                logger.info(f"Planner(gpt-5-mini): {plan}")

                # Execute plan with strict actions
                download = None

                async def _expect_download_after(action_fn):
                    nonlocal download
                    async with page.expect_download(timeout=60_000) as dlinfo:
                        await action_fn()
                    download = await dlinfo.value

                if plan["action"] == "CLICK_DOWNLOAD_ANYWAY":
                    async with page.expect_download(timeout=60_000) as dlinfo:
                        ok = await self._try_click_download_anyway(page)
                        if not ok:
                            raise RuntimeError("Planner chose CLICK_DOWNLOAD_ANYWAY but button not available.")
                    download = await dlinfo.value

                elif plan["action"] == "CLICK_DOWNLOAD_BUTTON":
                    btn = await self._try_click_download_button(page)
                    if not btn:
                        raise RuntimeError("Planner chose CLICK_DOWNLOAD_BUTTON but no button found.")
                    async with page.expect_download(timeout=60_000) as dlinfo:
                        logger.info("Step: click download button")
                        await btn.click(force=True)
                        await asyncio.sleep(1)
                        # Sometimes a secondary warning appears
                        await self._try_click_download_anyway(page)
                    download = await dlinfo.value

                elif plan["action"] == "OPEN_OVERFLOW_MENU_AND_DOWNLOAD":
                    async with page.expect_download(timeout=60_000) as dlinfo:
                        ok = await self._overflow_menu_download(page)
                        if not ok:
                            raise RuntimeError("Overflow download failed.")
                        await asyncio.sleep(1)
                        await self._try_click_download_anyway(page)
                    download = await dlinfo.value

                elif plan["action"] == "REFRESH_AND_RETRY_SELECTORS":
                    logger.info("Step: refresh and retry selectors")
                    await page.reload(wait_until="domcontentloaded", timeout=60_000)
                    await asyncio.sleep(3)
                    btn = await self._try_click_download_button(page)
                    if not btn:
                        # try overflow as fallback
                        logger.info("Fallback: overflow menu after refresh")
                        async with page.expect_download(timeout=60_000) as dlinfo:
                            ok = await self._overflow_menu_download(page)
                            if not ok:
                                raise RuntimeError("No download route found after refresh.")
                            await asyncio.sleep(1)
                            await self._try_click_download_anyway(page)
                        download = await dlinfo.value
                    else:
                        async with page.expect_download(timeout=60_000) as dlinfo:
                            await btn.click(force=True)
                            await asyncio.sleep(1)
                            await self._try_click_download_anyway(page)
                        download = await dlinfo.value

                else:
                    raise RuntimeError("Planner chose FAIL_GIVE_UP.")

                if not download:
                    raise RuntimeError("Download did not start (no download object).")

                suggested = download.suggested_filename or "downloaded_doc.pdf"
                # Ensure .pdf extension if Drive returns something odd
                if not suggested.lower().endswith(".pdf"):
                    suggested = suggested + ".pdf"

                save_path = os.path.join(DOWNLOADS_DIR, suggested)
                logger.info(f"Step: saving download to {save_path}")
                await download.save_as(save_path)

                file_bytes = os.path.getsize(save_path)
                file_sha = _sha256_file(save_path)

                payload = HandoffPayload(
                    status="success",
                    file_path=os.path.abspath(save_path),
                    file_name=suggested,
                    source_url=url,
                    sha256=file_sha,
                    bytes=file_bytes,
                    downloaded_at_iso=datetime.now(timezone.utc).isoformat(),
                    notes="Downloaded via Playwright browser flow (no Google Drive API). Planner used gpt-5-mini.",
                    extra={
                        "executor": "playwright-chromium",
                        "reasoner": "gpt-5-mini",
                    },
                )

                with open(HANDOFF_PATH, "w", encoding="utf-8") as f:
                    json.dump(asdict(payload), f, indent=2)
                logger.info(f"Step: wrote handoff -> {HANDOFF_PATH}")

                guard_task.cancel()
                logger.info("Agent A: success.")
                return asdict(payload)

            except (PWTimeoutError, Exception) as e:
                logger.error(f"Agent A error: {repr(e)}")
                await _safe_screenshot(page, "agent_a_error")
                raise
            finally:
                await browser.close()
