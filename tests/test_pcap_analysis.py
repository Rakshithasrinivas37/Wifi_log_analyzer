from __future__ import annotations

import json
from pathlib import Path

from src import pcap_analysis as pcap


def test_load_error_logs_keeps_only_error_rows_with_macs(tmp_path: Path) -> None:
    input_path = tmp_path / "output.jsonl"
    rows = [
        {
            "timestamp": "1782095400.100000",
            "log": "hostapd error",
            "prediction": "error",
            "mac_addresses": ["3C:22:FB:10:24:38"],
        },
        {
            "timestamp": "1782095400.200000",
            "log": "normal",
            "prediction": "normal",
            "mac_addresses": ["3c:22:fb:10:24:38"],
        },
        {
            "timestamp": "1782095400.300000",
            "log": "missing mac",
            "prediction": "error",
            "mac_addresses": [],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    errors = pcap.load_error_logs(input_path)

    assert len(errors) == 1
    assert errors[0].macs == ["3c:22:fb:10:24:38"]
    assert errors[0].ts == 1782095400.1


def test_reason_code_hints_maps_known_and_unknown_codes() -> None:
    hints = pcap.reason_code_hints([15, 15, 999])

    assert hints[0]["reason_code"] == 15
    assert hints[0]["reason"] == "4-way handshake timeout"
    assert hints[1]["reason_code"] == 999
    assert hints[1]["reason"] == "reason code 999"


def test_nearby_errors_uses_session_window() -> None:
    session = pcap.PcapSession(
        mac="3c:22:fb:10:24:38",
        first_seen=100.0,
        last_seen=110.0,
    )
    errors = [
        pcap.ErrorLog(ts=96.9, log="too early", macs=[session.mac]),
        pcap.ErrorLog(ts=97.0, log="inside start", macs=[session.mac]),
        pcap.ErrorLog(ts=113.0, log="inside end", macs=[session.mac]),
        pcap.ErrorLog(ts=113.1, log="too late", macs=[session.mac]),
    ]

    nearby = pcap.nearby_errors(errors, session, window_seconds=3.0)

    assert [error.log for error in nearby] == ["inside start", "inside end"]


def test_build_evidence_record_contains_teardown_summary() -> None:
    session = pcap.PcapSession(
        mac="3c:22:fb:10:24:38",
        first_seen=100.0,
        last_seen=101.0,
        packet_count=2,
        teardown_events=[
            pcap.TeardownEvent(
                ts=100.5,
                kind="deauth",
                packet_type=0,
                packet_subtype=12,
                reason_code=15,
                packet_summary="Dot11Deauth",
            )
        ],
    )
    errors = [pcap.ErrorLog(ts=100.4, log="EAPOL timeout", macs=[session.mac])]

    record = pcap.build_evidence_record(session.mac, errors, session)

    assert record["mac"] == session.mac
    assert record["timestamp"] == "100.500000"
    assert record["error_logs"] == ["EAPOL timeout"]
    assert record["pcap_session"]["teardown_events"][0]["reason_code"] == 15


def test_run_pcap_analysis_writes_records_with_mocked_parser(
    tmp_path: Path,
    monkeypatch,
) -> None:
    errors_path = tmp_path / "output.jsonl"
    output_path = tmp_path / "diagnosis.jsonl"
    mac = "3c:22:fb:10:24:38"
    errors_path.write_text(
        json.dumps(
            {
                "timestamp": "100.400000",
                "log": "EAPOL timeout",
                "prediction": "error",
                "mac_addresses": [mac],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    session = pcap.PcapSession(
        mac=mac,
        first_seen=100.0,
        last_seen=101.0,
        packet_count=1,
        teardown_events=[
            pcap.TeardownEvent(100.5, "deauth", 0, 12, 15, "Dot11Deauth")
        ],
    )
    monkeypatch.setattr(pcap, "parse_pcap", lambda _: {mac: session})

    records = pcap.run_pcap_analysis(
        pcap.PcapAnalysisConfig(
            errors_jsonl=errors_path,
            pcap=tmp_path / "fake.pcap",
            output=output_path,
        )
    )

    assert len(records) == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["mac"] == mac
