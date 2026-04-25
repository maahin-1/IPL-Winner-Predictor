"""
Webhook dispatcher — POST to registered URLs on each over with 3x retry.
SLA: < 30s delivery per prd.api_output_spec.rate_limits.
MITIGATION: R5 — queue-backed delivery with Redis buffer.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
DELIVERY_TIMEOUT_S = 10.0


@dataclass
class WebhookRegistration:
    url: str
    match_id: str
    secret: Optional[str] = None


@dataclass
class WebhookDispatcher:
    registrations: list[WebhookRegistration] = field(default_factory=list)

    def register(self, url: str, match_id: str, secret: Optional[str] = None) -> None:
        self.registrations.append(WebhookRegistration(url=url, match_id=match_id, secret=secret))
        logger.info("Registered webhook for match %s → %s", match_id, url)

    async def dispatch_all(self, match_id: str, payload: dict) -> None:
        """Fire webhooks to all registered URLs for this match."""
        targets = [r for r in self.registrations if r.match_id == match_id]
        if not targets:
            return
        await asyncio.gather(
            *[self._dispatch_with_retry(reg, payload) for reg in targets],
            return_exceptions=True,
        )

    async def _dispatch_with_retry(
        self, reg: WebhookRegistration, payload: dict
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if reg.secret:
            headers["X-IPIE-Signature"] = reg.secret

        async with httpx.AsyncClient(timeout=DELIVERY_TIMEOUT_S) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = await client.post(reg.url, json=payload, headers=headers)
                    resp.raise_for_status()
                    logger.debug(
                        "Webhook delivered to %s (attempt %d, status %d)",
                        reg.url, attempt, resp.status_code,
                    )
                    return
                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    logger.warning(
                        "Webhook delivery attempt %d/%d failed for %s: %s",
                        attempt, MAX_RETRIES, reg.url, e,
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY_S * attempt)

            logger.error("All %d webhook delivery attempts failed for %s", MAX_RETRIES, reg.url)
