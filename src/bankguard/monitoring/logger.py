"""Structured JSON logging for BankGuard operational monitoring (Task 7.1).

Enforces:
- Machine-readable JSON structured log output.
- Standard operational fields: timestamp, level, event, transaction_id, latency_ms, score, tier.
- Strict PII privacy: never logs raw card numbers, raw emails, IP addresses, or device fingerprints.
"""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, Optional, Set

# Sensitive fields that must NEVER appear in raw logs (PII / security)
SENSITIVE_FIELDS: Set[str] = {
    "ip",
    "ip_address",
    "email",
    "card_number",
    "device_info",
    "device_id",
    "mac_address",
    "phone",
    "ssn",
    "password",
    "token",
    "secret",
}


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge extra structured attributes if provided
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            for key, val in record.structured_data.items():
                if key.lower() not in SENSITIVE_FIELDS:
                    log_entry[key] = val

        # Handle exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_structured_logger(name: str = "bankguard.monitoring", level: int = logging.INFO) -> logging.Logger:
    """Returns a logger configured with the structured JSON formatter.

    Args:
        name: Logger hierarchy name.
        level: Logging level threshold.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, StructuredJsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)

    # Do not propagate to root to avoid duplicate unformatted stderr logging
    logger.propagate = False
    return logger


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes transaction input payload by redacting any potential PII keys.

    Args:
        payload: Input dictionary.

    Returns:
        Sanitized dictionary safe for operational telemetry.
    """
    sanitized: Dict[str, Any] = {}
    for k, v in payload.items():
        if k.lower() in SENSITIVE_FIELDS:
            continue
        sanitized[k] = v
    return sanitized

