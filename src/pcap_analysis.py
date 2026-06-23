"""Correlate predicted error logs with raw WiFi PCAP evidence.

This module exposes callable analysis functions for FastAPI/notebook use. It
does not parse CLI arguments or execute work at import time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REASON_CODE_HINTS: dict[int, dict[str, object]] = {
    1: {
        "reason": "unspecified reason",
        "possible_root_causes": [
            "generic AP/client disconnect",
            "vendor-specific behavior",
            "requires supporting logs for root cause",
        ],
    },
    3: {
        "reason": "station leaving",
        "possible_root_causes": [
            "normal client roam or disconnect",
            "client intentionally left BSS",
            "client sleep or interface reset",
        ],
    },
    4: {
        "reason": "disassociated due to inactivity",
        "possible_root_causes": [
            "client inactivity",
            "client sleep or power-save behavior",
            "missed keepalives",
            "network or DHCP failure causing disconnect",
        ],
    },
    7: {
        "reason": "class 3 frame from nonassociated station",
        "possible_root_causes": [
            "client/AP association state mismatch",
            "roaming transition issue",
            "client sent data after disassociation",
        ],
    },
    8: {
        "reason": "station leaving BSS",
        "possible_root_causes": [
            "client roaming away",
            "client intentionally disconnected",
            "client radio reset",
        ],
    },
    15: {
        "reason": "4-way handshake timeout",
        "possible_root_causes": [
            "wrong WiFi password or PSK",
            "WPA/WPA2/WPA3 security mismatch",
            "PMF compatibility issue",
            "client/AP key negotiation failure",
        ],
    },
    23: {
        "reason": "IEEE 802.1X authentication failed",
        "possible_root_causes": [
            "RADIUS authentication failure",
            "invalid client credentials or certificate",
            "RADIUS reachability or shared secret issue",
        ],
    },
    34: {
        "reason": "disassociated because of poor channel conditions",
        "possible_root_causes": [
            "weak signal",
            "RF interference",
            "poor channel quality",
            "client too far from AP",
        ],
    },
}


@dataclass
class PcapAnalysisConfig:
    """Settings for PCAP/log correlation."""

    errors_jsonl: Path
    pcap: Path
    output: Path | None = None
    window_seconds: float = 3.0


@dataclass
class ErrorLog:
    """One model-predicted error log row."""

    ts: float
    log: str
    macs: list[str]


@dataclass
class TeardownEvent:
    """Deauthentication or disassociation frame evidence from the packet."""

    ts: float
    kind: str
    packet_type: int
    packet_subtype: int
    reason_code: int
    packet_summary: str = ""


@dataclass
class PcapSession:
    """Packet evidence accumulated for one station MAC."""

    mac: str
    first_seen: float
    last_seen: float
    packet_count: int = 0
    teardown_events: list[TeardownEvent] = field(default_factory=list)

    def observe(self, ts: float) -> None:
        """Update packet counters and time bounds."""

        self.first_seen = min(self.first_seen, ts)
        self.last_seen = max(self.last_seen, ts)
        self.packet_count += 1


def load_error_logs(path: Path) -> list[ErrorLog]:
    """Load predicted error rows with MACs already extracted by inference."""

    errors: list[ErrorLog] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prediction") != "error":
                continue
            log = str(row.get("log", ""))
            macs = [str(mac).lower() for mac in row.get("mac_addresses", []) if mac]
            if not macs:
                continue
            try:
                ts = float(row["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} missing numeric timestamp") from exc
            errors.append(ErrorLog(ts=ts, log=log, macs=macs))
    return errors


def parse_pcap(path: Path) -> dict[str, PcapSession]:
    """Parse useful WiFi evidence from an 802.11 PCAP."""

    try:
        from scapy.all import PcapReader
        from scapy.layers.dot11 import Dot11, Dot11Deauth, Dot11Disas
        from scapy.layers.eap import EAPOL
    except ImportError as exc:
        raise ImportError("PCAP analysis requires Scapy: pip install scapy") from exc

    sessions: dict[str, PcapSession] = {}
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if Dot11 not in packet:
                continue

            dot11 = packet[Dot11]
            station = station_from_packet(packet, dot11, EAPOL)
            if not station:
                continue

            ts = float(packet.time)
            session = sessions.setdefault(
                station,
                PcapSession(mac=station, first_seen=ts, last_seen=ts),
            )
            session.observe(ts)

            if Dot11Deauth in packet:
                session.teardown_events.append(
                    TeardownEvent(
                        ts=ts,
                        kind="deauth",
                        packet_type=int(dot11.type),
                        packet_subtype=int(dot11.subtype),
                        reason_code=int(packet[Dot11Deauth].reason),
                        packet_summary=packet.summary(),
                    )
                )
            if Dot11Disas in packet:
                session.teardown_events.append(
                    TeardownEvent(
                        ts=ts,
                        kind="disassoc",
                        packet_type=int(dot11.type),
                        packet_subtype=int(dot11.subtype),
                        reason_code=int(packet[Dot11Disas].reason),
                        packet_summary=packet.summary(),
                    )
                )

    return sessions


def station_from_packet(packet: Any, dot11: Any, eapol_layer: Any) -> str | None:
    """Return the client/station MAC for supported 802.11 frame layouts."""

    if eapol_layer in packet:
        return station_from_data(dot11)
    if int(getattr(dot11, "type", 0)) == 2:
        return station_from_data(dot11)
    return station_from_management(dot11)


def station_from_management(dot11: Any) -> str | None:
    """Pick the non-BSSID endpoint from a management frame."""

    bssid = normalize_mac(getattr(dot11, "addr3", None))
    for attr in ("addr2", "addr1"):
        mac = normalize_mac(getattr(dot11, attr, None))
        if mac and mac != bssid and not is_broadcast_mac(mac):
            return mac
    return None


def station_from_data(dot11: Any) -> str | None:
    """Pick the client/station endpoint from infrastructure data frames."""

    fcfield = int(getattr(dot11, "FCfield", 0))
    to_ds = bool(fcfield & 0x1)
    from_ds = bool(fcfield & 0x2)

    if to_ds and not from_ds:
        return normalize_mac(getattr(dot11, "addr2", None))
    if from_ds and not to_ds:
        return normalize_mac(getattr(dot11, "addr1", None))

    bssid = normalize_mac(getattr(dot11, "addr3", None))
    for attr in ("addr1", "addr2"):
        mac = normalize_mac(getattr(dot11, attr, None))
        if mac and mac != bssid and not is_broadcast_mac(mac):
            return mac
    return None


def normalize_mac(mac: str | None) -> str | None:
    """Normalize Scapy MAC values."""

    return mac.lower() if mac else None


def is_broadcast_mac(mac: str) -> bool:
    """Return True for broadcast addresses."""

    return mac == "ff:ff:ff:ff:ff:ff"


def group_errors_by_mac(
    errors: list[ErrorLog],
    known_station_macs: set[str],
) -> tuple[dict[str, list[ErrorLog]], list[ErrorLog]]:
    """Group error logs by station MACs found in the PCAP."""

    grouped: dict[str, list[ErrorLog]] = defaultdict(list)
    unmatched: list[ErrorLog] = []
    for error in errors:
        station_macs = [mac for mac in error.macs if mac in known_station_macs]
        if not station_macs:
            unmatched.append(error)
            continue
        for mac in station_macs:
            grouped[mac].append(error)
    return dict(grouped), unmatched


def nearby_errors(
    errors: list[ErrorLog],
    session: PcapSession | None,
    window_seconds: float,
) -> list[ErrorLog]:
    """Keep errors close to a PCAP session time range."""

    if session is None:
        return errors
    start = session.first_seen - window_seconds
    end = session.last_seen + window_seconds
    return [error for error in errors if start <= error.ts <= end]


def unique_values(values: list[int]) -> list[int]:
    """Return values in first-seen order without duplicates."""

    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def reason_code_hints(reason_codes: list[int]) -> list[dict[str, object]]:
    """Build possible root-cause hints from packet reason codes."""

    hints: list[dict[str, object]] = []
    for code in unique_values(reason_codes):
        hint = REASON_CODE_HINTS.get(
            code,
            {
                "reason": f"reason code {code}",
                "possible_root_causes": [
                    "unknown or vendor-specific disconnect reason",
                    "requires log and packet context",
                ],
            },
        )
        hints.append(
            {
                "reason_code": code,
                "reason": hint["reason"],
                "possible_root_causes": hint["possible_root_causes"],
            }
        )
    return hints


def build_evidence_record(
    mac: str,
    errors: list[ErrorLog],
    session: PcapSession,
) -> dict[str, Any]:
    """Build one neutral evidence record for downstream LLM diagnosis."""

    first_teardown = session.teardown_events[0]
    return {
        "mac": mac,
        "timestamp": f"{first_teardown.ts:.6f}",
        "error_log_count": len(errors),
        "error_logs": [error.log for error in errors],
        "pcap_session": session_summary(session),
    }


def session_summary(session: PcapSession | None) -> dict[str, Any] | None:
    """Return JSON-serializable session details."""

    if session is None:
        return None
    reason_codes = [event.reason_code for event in session.teardown_events]
    return {
        "first_seen": session.first_seen,
        "last_seen": session.last_seen,
        "packet_count": session.packet_count,
        "reason_code_hints": reason_code_hints(reason_codes),
        "teardown_events": [
            {
                "ts": event.ts,
                "kind": event.kind,
                "packet_type": event.packet_type,
                "packet_subtype": event.packet_subtype,
                "reason_code": event.reason_code,
                "packet_summary": event.packet_summary,
            }
            for event in session.teardown_events
        ],
    }


def analyze(
    errors_path: Path,
    pcap_path: Path,
    window_seconds: float,
) -> list[dict[str, Any]]:
    """Build per-MAC evidence records from error logs and PCAP."""

    errors = load_error_logs(errors_path)
    sessions = parse_pcap(pcap_path)
    grouped_errors, _ = group_errors_by_mac(errors, set(sessions))

    records = []
    for mac, mac_errors in sorted(grouped_errors.items()):
        session = sessions.get(mac)
        if session is None or not session.teardown_events:
            continue
        correlated_errors = nearby_errors(mac_errors, session, window_seconds)
        if not correlated_errors:
            correlated_errors = mac_errors
        records.append(build_evidence_record(mac, correlated_errors, session))

    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def run_pcap_analysis(config: PcapAnalysisConfig) -> list[dict[str, Any]]:
    """Run PCAP analysis, optionally write JSONL, and return evidence rows."""

    records = analyze(config.errors_jsonl, config.pcap, config.window_seconds)
    if config.output is not None:
        write_jsonl(config.output, records)
    return records
