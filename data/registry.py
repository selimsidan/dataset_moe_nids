"""DATASET_REGISTRY: the single source of truth for how each raw NIDS dataset
maps onto the shared harmonized schema.

Adding a new dataset means adding a `DatasetSpec` entry here (and, if it
introduces genuinely new raw features, extending CANONICAL_FEATURES in
harmonization.py) — it must never require touching encoder/expert/gate code.

IMPORTANT: the `feature_alias` / `label_alias` maps below are a first-pass
best effort based on each dataset's published documentation (NetFlow v2/v3
format for the NF-* families, CICFlowMeter for CICIDS2017/CSE-CIC-IDS2018,
the native UNSW-NB15 train/test schema, and CICIoT2023's folder-derived
labels). They have NOT been validated against the actual CSV headers on disk
for this project and should be checked with `python -m data.registry
inspect <name>` (see bottom of this file) before a real training run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import paths as _paths

# Columns that identify *where a flow came from* or *which physical
# endpoints/timestamps it involved* rather than describing traffic behavior.
# These are always dropped before harmonization -- never handed to the
# encoder/experts/gate, and never used as "features" even under an alias.
# IDENTITY_LIKE_HINTS is used defensively in harmonization.py to catch any
# such column that slips into a per-dataset alias map by mistake.
IDENTITY_LIKE_HINTS = (
    "ipv4", "ipv6", "ip_addr", "ipaddr", "src_ip", "dst_ip", "srcip", "dstip",
    "_port", "port_", "flow_id", "flow id", "timestamp", "time_stamp",
    "src_mac", "dst_mac", "mac_addr",
)
# NOTE: deliberately NOT a bare "ip" or "port" substring check -- legitimate
# traffic features like MIN_IP_PKT_LEN ("IP packet length") or Down/Up
# Ratio-style names can contain "ip"/"port" as a substring without being an
# address/port identity column. Hints above target the specific address/
# port-column naming patterns actually used across the registered datasets.


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str  # 'file' | 'files' | 'directory'
    default_paths: list[str]
    label_col: str
    benign_label: str
    feature_alias: dict[str, str]  # canonical_name -> raw_column_name
    label_alias: dict[str, str | None] = field(default_factory=dict)  # raw label -> canonical class name (None = exclude); low-priority default, override in config's data.label_mapping
    drop_columns: tuple[str, ...] = ()  # raw identity-like columns to strip explicitly
    folder_label_fn: str | None = None  # name of a function in loaders.py, for folder-labeled datasets
    notes: str = ""

    @property
    def paths(self) -> list[str]:
        return _paths.dataset_paths(self.name, self.default_paths)


# ---------------------------------------------------------------------------
# NetFlow v3 family (NF-UNSW-NB15-v3, NF-CICIDS2018-v3, NF-BoT-IoT-v3) share
# one schema (Sarhan et al. NetFlow feature set), so one alias map covers all
# three.
# ---------------------------------------------------------------------------
NETFLOW_V3_ALIAS = {
    "protocol": "PROTOCOL",
    "l7_proto": "L7_PROTO",
    "in_bytes": "IN_BYTES",
    "out_bytes": "OUT_BYTES",
    "in_pkts": "IN_PKTS",
    "out_pkts": "OUT_PKTS",
    "tcp_flags": "TCP_FLAGS",
    "client_tcp_flags": "CLIENT_TCP_FLAGS",
    "server_tcp_flags": "SERVER_TCP_FLAGS",
    "flow_duration_ms": "FLOW_DURATION_MILLISECONDS",
    "duration_in": "DURATION_IN",
    "duration_out": "DURATION_OUT",
    "min_ttl": "MIN_TTL",
    "max_ttl": "MAX_TTL",
    "longest_flow_pkt": "LONGEST_FLOW_PKT",
    "shortest_flow_pkt": "SHORTEST_FLOW_PKT",
    "min_ip_pkt_len": "MIN_IP_PKT_LEN",
    "max_ip_pkt_len": "MAX_IP_PKT_LEN",
    "src_to_dst_second_bytes": "SRC_TO_DST_SECOND_BYTES",
    "dst_to_src_second_bytes": "DST_TO_SRC_SECOND_BYTES",
    "retransmitted_in_bytes": "RETRANSMITTED_IN_BYTES",
    "retransmitted_in_pkts": "RETRANSMITTED_IN_PKTS",
    "retransmitted_out_bytes": "RETRANSMITTED_OUT_BYTES",
    "retransmitted_out_pkts": "RETRANSMITTED_OUT_PKTS",
    "src_to_dst_avg_throughput": "SRC_TO_DST_AVG_THROUGHPUT",
    "dst_to_src_avg_throughput": "DST_TO_SRC_AVG_THROUGHPUT",
    "tcp_win_max_in": "TCP_WIN_MAX_IN",
    "tcp_win_max_out": "TCP_WIN_MAX_OUT",
}
NETFLOW_V3_DROP = ("IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "NF-UNSW-NB15-v3": DatasetSpec(
        name="NF-UNSW-NB15-v3",
        kind="file",
        default_paths=[os.path.join(_paths.DRIVE_BASE, "NF-UNSW-NB15-v3", "data", "NF-UNSW-NB15-v3.csv")],
        label_col="Attack",
        benign_label="Benign",
        feature_alias=NETFLOW_V3_ALIAS,
        drop_columns=NETFLOW_V3_DROP,
        notes="NetFlow v3 schema shared with NF-CICIDS2018-v3 / NF-BoT-IoT-v3.",
    ),
    "NF-CICIDS2018-v3": DatasetSpec(
        name="NF-CICIDS2018-v3",
        kind="file",
        default_paths=[os.path.join(_paths.DRIVE_BASE, "NF-CICIDS2018-v3", "data", "NF-CICIDS2018-v3.csv")],
        label_col="Attack",
        benign_label="Benign",
        feature_alias=NETFLOW_V3_ALIAS,
        drop_columns=NETFLOW_V3_DROP,
    ),
    "NF-BoT-IoT-v3": DatasetSpec(
        name="NF-BoT-IoT-v3",
        kind="file",
        default_paths=[os.path.join(_paths.DRIVE_BASE, "NF-BoT-IoT-v3", "data", "NF-BoT-IoT-v3.csv")],
        label_col="Attack",
        benign_label="Benign",
        feature_alias=NETFLOW_V3_ALIAS,
        drop_columns=NETFLOW_V3_DROP,
    ),
    "CSE-CIC-IDS2018": DatasetSpec(
        name="CSE-CIC-IDS2018",
        kind="files",
        default_paths=[
            os.path.join(_paths.DRIVE_BASE, "CSE-CIC-IDS2018", f)
            for f in [
                "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
                "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
                "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
                "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
                "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
                "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
                "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
                "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
                "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
                "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            ]
        ],
        label_col="Label",
        benign_label="Benign",
        feature_alias={
            "protocol": "Protocol",
            "flow_duration_ms": "Flow Duration",
            "in_pkts": "Tot Bwd Pkts",
            "out_pkts": "Tot Fwd Pkts",
            "out_bytes": "TotLen Fwd Pkts",
            "in_bytes": "TotLen Bwd Pkts",
            "min_ip_pkt_len": "Pkt Len Min",
            "max_ip_pkt_len": "Pkt Len Max",
            "flow_iat_mean": "Flow IAT Mean",
            "flow_iat_std": "Flow IAT Std",
            "fwd_psh_flags": "Fwd PSH Flags",
            "bwd_psh_flags": "Bwd PSH Flags",
            "syn_flag_count": "SYN Flag Cnt",
            "ack_flag_count": "ACK Flag Cnt",
            "fin_flag_count": "FIN Flag Cnt",
            "rst_flag_count": "RST Flag Cnt",
            "down_up_ratio": "Down/Up Ratio",
            "init_win_bytes_fwd": "Init Fwd Win Byts",
            "init_win_bytes_bwd": "Init Bwd Win Byts",
        },
        drop_columns=("Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"),
        notes="CICFlowMeter output; header naming has historically varied by release -- verify against actual CSVs.",
    ),
    "UNSW-NB15-train-test": DatasetSpec(
        name="UNSW-NB15-train-test",
        kind="files",
        default_paths=[
            os.path.join(_paths.DRIVE_BASE, "UNSW_NB15", "UNSW_NB15_training-set.csv"),
            os.path.join(_paths.DRIVE_BASE, "UNSW_NB15", "UNSW_NB15_testing-set.csv"),
            os.path.join(_paths.DRIVE_BASE, "UNSW-NB15-train-test", "UNSW_NB15_training-set.csv"),
            os.path.join(_paths.DRIVE_BASE, "UNSW-NB15-train-test", "UNSW_NB15_testing-set.csv"),
        ],
        label_col="attack_cat",
        benign_label="Normal",
        feature_alias={
            "protocol": "proto",
            "flow_duration_ms": "dur",
            "out_pkts": "spkts",
            "in_pkts": "dpkts",
            "out_bytes": "sbytes",
            "in_bytes": "dbytes",
            "min_ttl": "sttl",
            "max_ttl": "dttl",
            "src_to_dst_avg_throughput": "sload",
            "dst_to_src_avg_throughput": "dload",
            "tcp_win_max_in": "swin",
            "tcp_win_max_out": "dwin",
        },
        label_alias={"": "Normal", "normal": "Normal"},
        drop_columns=("id", "label"),
        notes="Uses the official train/test split CSVs (already engineered features, not raw NetFlow).",
    ),
    "CICIDS2017": DatasetSpec(
        name="CICIDS2017",
        kind="files",
        default_paths=[
            os.path.join(_paths.DRIVE_BASE, "CICIDS2017", f)
            for f in [
                "Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            ]
        ],
        label_col="Label",
        benign_label="BENIGN",
        feature_alias={
            "flow_duration_ms": "Flow Duration",
            "out_pkts": "Total Fwd Packets",
            "in_pkts": "Total Backward Packets",
            "out_bytes": "Total Length of Fwd Packets",
            "in_bytes": "Total Length of Bwd Packets",
            "min_ip_pkt_len": "Min Packet Length",
            "max_ip_pkt_len": "Max Packet Length",
            "flow_iat_mean": "Flow IAT Mean",
            "flow_iat_std": "Flow IAT Std",
            "fwd_psh_flags": "Fwd PSH Flags",
            "bwd_psh_flags": "Bwd PSH Flags",
            "syn_flag_count": "SYN Flag Count",
            "ack_flag_count": "ACK Flag Count",
            "fin_flag_count": "FIN Flag Count",
            "rst_flag_count": "RST Flag Count",
            "down_up_ratio": "Down/Up Ratio",
            "init_win_bytes_fwd": "Init_Win_bytes_forward",
            "init_win_bytes_bwd": "Init_Win_bytes_backward",
        },
        drop_columns=("Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"),
        notes="Original CICFlowMeter CSVs; column headers include leading spaces in the raw release -- loader strips these.",
    ),
    "CICIoT2023": DatasetSpec(
        name="CICIoT2023",
        kind="directory",
        default_paths=[
            os.path.join(_paths.DRIVE_BASE, "CICIoT2023", "CSV"),
            os.path.join(_paths.DRIVE_BASE, "CICIoT2023"),
        ],
        label_col="label",  # derived per-file via folder_label_fn, not a literal column in most releases
        benign_label="Benign",
        feature_alias={
            "protocol": "Protocol Type",
            "flow_duration_ms": "Duration",
            "out_bytes": "Tot sum",
            "min_ip_pkt_len": "Min",
            "max_ip_pkt_len": "Max",
            "syn_flag_count": "syn_count",
            "ack_flag_count": "ack_count",
            "fin_flag_count": "fin_count",
            "rst_flag_count": "rst_count",
        },
        folder_label_fn="ciciot2023_label_from_filename",
        notes="Label comes from the CSV filename (e.g. 'DDoS-ICMP_Flood.csv'), not a label column -- see loaders.py.",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        spec = get_spec(sys.argv[2])
        print(f"name: {spec.name}\nkind: {spec.kind}\npaths: {spec.paths}")
        print(f"label_col: {spec.label_col}\nbenign_label: {spec.benign_label}")
        print(f"feature_alias ({len(spec.feature_alias)}): {spec.feature_alias}")
        print(f"drop_columns: {spec.drop_columns}")
    else:
        print("Registered datasets:")
        for n in DATASET_REGISTRY:
            print(f"  - {n}")
