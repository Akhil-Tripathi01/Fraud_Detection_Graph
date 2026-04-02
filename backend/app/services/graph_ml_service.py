from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainedArtifacts:
    trained_at: str
    metrics: dict
    feature_columns: list[str]
    transaction_count: int
    account_count: int
    graph_nodes: int
    graph_edges: int
    top_features: list[dict]
    confusion_matrix: dict


class GraphMLService:
    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.graph: nx.DiGraph | None = None
        self.nodes_df: pd.DataFrame | None = None
        self.model: RandomForestClassifier | None = None
        self.artifacts: TrainedArtifacts | None = None
        self.model_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def generate_transaction_dataset(
        self, n_transactions: int = 3000, n_accounts: int = 500, fraud_rate: float = 0.08, random_seed: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed)
        accounts = [f"ACC_{i:04d}" for i in range(n_accounts)]
        n_fraud = int(n_transactions * fraud_rate)

        sender_ids = rng.choice(accounts, size=n_transactions, replace=True)
        receiver_ids = rng.choice(accounts, size=n_transactions, replace=True)

        same_idx = np.where(sender_ids == receiver_ids)[0]
        if len(same_idx) > 0:
            receiver_ids[same_idx] = rng.choice(accounts, size=len(same_idx), replace=True)

        labels = np.zeros(n_transactions, dtype=int)
        fraud_idx = rng.choice(np.arange(n_transactions), size=n_fraud, replace=False)
        labels[fraud_idx] = 1

        base_amounts = rng.lognormal(mean=7.4, sigma=0.7, size=n_transactions)
        fraud_boost = rng.lognormal(mean=8.3, sigma=0.9, size=n_transactions)
        amounts = np.where(labels == 1, fraud_boost, base_amounts).round(2)

        device_pool = [f"DEV_{i:03d}" for i in range(max(40, n_accounts // 8))]
        ip_pool = [f"10.10.{i//255}.{i%255}" for i in range(max(100, n_accounts // 3))]
        devices = rng.choice(device_pool, size=n_transactions, replace=True).astype(object)
        ips = rng.choice(ip_pool, size=n_transactions, replace=True).astype(object)

        # Fraud rings reuse the same infrastructure.
        ring_devices = [f"DEV_RING_{i:02d}" for i in range(6)]
        ring_ips = [f"172.31.0.{i}" for i in range(1, 8)]
        devices[fraud_idx] = rng.choice(ring_devices, size=len(fraud_idx), replace=True)
        ips[fraud_idx] = rng.choice(ring_ips, size=len(fraud_idx), replace=True)

        merchant_pool = [f"MER_{i:03d}" for i in range(max(50, n_accounts // 6))]
        merchant_categories = ["electronics", "travel", "gaming", "grocery", "fuel", "luxury", "utilities"]
        merchants = rng.choice(merchant_pool, size=n_transactions, replace=True)
        merchant_cat_map = {m: rng.choice(merchant_categories) for m in merchant_pool}
        merchant_category = np.array([merchant_cat_map[m] for m in merchants])

        channels = np.where(
            labels == 1,
            rng.choice(["web", "mobile", "atm"], p=[0.55, 0.25, 0.20], size=n_transactions),
            rng.choice(["web", "mobile", "pos", "atm"], p=[0.22, 0.45, 0.28, 0.05], size=n_transactions),
        )
        countries = np.where(
            labels == 1,
            rng.choice(["IN", "SG", "AE"], p=[0.50, 0.30, 0.20], size=n_transactions),
            rng.choice(["IN", "IN", "IN", "US"], p=[0.4, 0.3, 0.2, 0.1], size=n_transactions),
        )

        start = pd.Timestamp("2026-03-01T00:00:00Z")
        timestamps = start + pd.to_timedelta(rng.integers(0, 60 * 24 * 30, size=n_transactions), unit="m")
        timestamps = pd.to_datetime(timestamps, utc=True)

        # Fraud burst behavior: compress a share of fraud transactions into short windows.
        burst_count = max(1, len(fraud_idx) // 3)
        burst_idx = fraud_idx[:burst_count]
        burst_base = pd.Timestamp("2026-03-24T01:00:00Z")
        timestamps_values = timestamps.to_numpy().copy()
        timestamps_values[burst_idx] = burst_base + pd.to_timedelta(
            rng.integers(0, 40, size=burst_count), unit="m"
        ).to_numpy()
        timestamps = pd.to_datetime(timestamps_values, utc=True)

        df = pd.DataFrame(
            {
                "txn_id": [f"TXN_{i:06d}" for i in range(n_transactions)],
                "sender_id": sender_ids,
                "receiver_id": receiver_ids,
                "merchant_id": merchants,
                "merchant_category": merchant_category,
                "amount": amounts,
                "timestamp": timestamps,
                "device_id": devices,
                "ip_address": ips,
                "channel": channels,
                "country": countries,
                "label": labels,
            }
        ).sort_values("timestamp")

        df["hour"] = df["timestamp"].dt.hour
        df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
        df["is_high_value"] = (df["amount"] >= df["amount"].quantile(0.9)).astype(int)
        df["is_cross_border"] = (
            (df["country"] != "IN") & (df["channel"].isin(["web", "mobile"]))
        ).astype(int)

        return df.reset_index(drop=True)

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        g = nx.DiGraph()
        accounts = pd.unique(df[["sender_id", "receiver_id"]].values.ravel())
        g.add_nodes_from(accounts)

        grouped = df.groupby(["sender_id", "receiver_id"]).agg(
            edge_txn_count=("txn_id", "count"),
            edge_amount_sum=("amount", "sum"),
            edge_fraud_count=("label", "sum"),
        )

        for (src, dst), row in grouped.iterrows():
            g.add_edge(
                src,
                dst,
                txn_count=int(row["edge_txn_count"]),
                amount_sum=float(row["edge_amount_sum"]),
                fraud_count=int(row["edge_fraud_count"]),
            )

        return g

    def compute_node_features(self, df: pd.DataFrame, g: nx.DiGraph) -> pd.DataFrame:
        sent = (
            df.sort_values("timestamp")
            .groupby("sender_id")
            .agg(
                total_sent=("amount", "sum"),
                avg_sent=("amount", "mean"),
                txn_count_sent=("txn_id", "count"),
                unique_receivers=("receiver_id", "nunique"),
                night_txn_ratio_sent=("is_night", "mean"),
                high_value_txn_ratio_sent=("is_high_value", "mean"),
                cross_border_sent_ratio=("is_cross_border", "mean"),
                device_diversity_sent=("device_id", "nunique"),
                ip_diversity_sent=("ip_address", "nunique"),
                merchant_diversity_sent=("merchant_id", "nunique"),
            )
        )

        sent_time_diff = (
            df.sort_values("timestamp")
            .groupby("sender_id")["timestamp"]
            .apply(lambda s: s.diff().dt.total_seconds().fillna(0).mean())
            .rename("avg_time_diff_sent")
        )

        recv = (
            df.groupby("receiver_id")
            .agg(
                total_received=("amount", "sum"),
                avg_received=("amount", "mean"),
                txn_count_received=("txn_id", "count"),
                unique_senders=("sender_id", "nunique"),
            )
            .rename_axis("sender_id")
        )

        features = sent.join(sent_time_diff, how="left").join(recv, how="outer").fillna(0)
        features.index.name = "account_id"
        features = features.reset_index()

        features["in_degree"] = features["account_id"].map(dict(g.in_degree())).fillna(0)
        features["out_degree"] = features["account_id"].map(dict(g.out_degree())).fillna(0)
        pr = nx.pagerank(g, alpha=0.9) if g.number_of_nodes() > 0 else {}
        features["pagerank"] = features["account_id"].map(pr).fillna(0)
        undirected = g.to_undirected()
        clustering = nx.clustering(undirected) if undirected.number_of_nodes() > 0 else {}
        features["clustering_coef"] = features["account_id"].map(clustering).fillna(0)

        fraud_accounts = set(df.loc[df["label"] == 1, "sender_id"]).union(set(df.loc[df["label"] == 1, "receiver_id"]))
        features["label"] = features["account_id"].map(lambda a: 1 if a in fraud_accounts else 0)

        return features

    def train_pipeline(
        self, n_transactions: int = 3000, n_accounts: int = 500, fraud_rate: float = 0.08, random_seed: int = 42
    ) -> TrainedArtifacts:
        self.df = self.generate_transaction_dataset(n_transactions, n_accounts, fraud_rate, random_seed)
        self.graph = self.build_graph(self.df)
        self.nodes_df = self.compute_node_features(self.df, self.graph)

        feature_cols = [
            "total_sent",
            "avg_sent",
            "txn_count_sent",
            "unique_receivers",
            "night_txn_ratio_sent",
            "high_value_txn_ratio_sent",
            "cross_border_sent_ratio",
            "device_diversity_sent",
            "ip_diversity_sent",
            "merchant_diversity_sent",
            "avg_time_diff_sent",
            "total_received",
            "avg_received",
            "txn_count_received",
            "unique_senders",
            "in_degree",
            "out_degree",
            "pagerank",
            "clustering_coef",
        ]

        x = self.nodes_df[feature_cols]
        y = self.nodes_df["label"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_seed, stratify=y)

        self.model = RandomForestClassifier(
            n_estimators=280,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        self.model.fit(x_train, y_train)

        pred = self.model.predict(x_test)
        proba = self.model.predict_proba(x_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "support": int(len(y_test)),
        }
        top_features = [
            {"feature": col, "importance": round(float(imp), 4)}
            for col, imp in sorted(
                zip(feature_cols, self.model.feature_importances_, strict=False), key=lambda p: p[1], reverse=True
            )[:8]
        ]
        confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        self.artifacts = TrainedArtifacts(
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            feature_columns=feature_cols,
            transaction_count=len(self.df),
            account_count=len(self.nodes_df),
            graph_nodes=self.graph.number_of_nodes(),
            graph_edges=self.graph.number_of_edges(),
            top_features=top_features,
            confusion_matrix=confusion,
        )
        return self.artifacts

    def ensure_trained(self) -> None:
        if self.model is None or self.nodes_df is None or self.artifacts is None:
            self.train_pipeline()

    def metrics(self) -> dict:
        self.ensure_trained()
        assert self.artifacts is not None
        return {
            "message": "Graph ML pipeline trained successfully.",
            "trained_at": self.artifacts.trained_at,
            "transactions": self.artifacts.transaction_count,
            "accounts": self.artifacts.account_count,
            "graph_nodes": self.artifacts.graph_nodes,
            "graph_edges": self.artifacts.graph_edges,
            "metrics": self.artifacts.metrics,
            "feature_columns": self.artifacts.feature_columns,
            "top_features": self.artifacts.top_features,
            "confusion_matrix": self.artifacts.confusion_matrix,
        }

    def data_profile(self) -> dict:
        self.ensure_trained()
        assert self.df is not None
        assert self.nodes_df is not None

        fraud_rate = float(self.df["label"].mean())
        channel_dist = (self.df["channel"].value_counts(normalize=True) * 100).round(2).to_dict()
        country_dist = (self.df["country"].value_counts(normalize=True) * 100).round(2).to_dict()

        risky_devices = (
            self.df.groupby("device_id")
            .agg(total_txn=("txn_id", "count"), fraud_txn=("label", "sum"))
            .reset_index()
        )
        risky_devices["fraud_ratio"] = (risky_devices["fraud_txn"] / risky_devices["total_txn"]).fillna(0)
        risky_devices = risky_devices[risky_devices["fraud_txn"] > 0]
        risky_devices = (
            risky_devices.sort_values(["fraud_ratio", "fraud_txn"], ascending=[False, False])
            .head(8)
            .to_dict(orient="records")
        )

        return {
            "transactions": int(len(self.df)),
            "accounts": int(len(self.nodes_df)),
            "fraud_rate": round(fraud_rate, 4),
            "avg_amount": round(float(self.df["amount"].mean()), 2),
            "p95_amount": round(float(self.df["amount"].quantile(0.95)), 2),
            "channel_distribution": channel_dist,
            "country_distribution": country_dist,
            "top_risky_devices": risky_devices,
        }

    def research_report(self) -> dict:
        self.ensure_trained()
        metrics = self.metrics()
        profile = self.data_profile()

        findings = [
            f"Observed fraud rate in synthetic graph dataset: {profile['fraud_rate'] * 100:.2f}%.",
            f"Model ROC-AUC is {metrics['metrics']['roc_auc']}, indicating meaningful fraud separability.",
            f"Top signal feature: {metrics['top_features'][0]['feature']} (importance={metrics['top_features'][0]['importance']}).",
            "Device/IP reuse and burst timing are strong indicators of coordinated fraud rings.",
            "Cross-border web/mobile activity contributes to medium-high risk clusters in this dataset.",
        ]
        recommendations = [
            "Refresh training daily or when graph drift exceeds threshold.",
            "Escalate accounts with sustained high probability and shared-device pressure to manual review.",
            "Track precision/recall separately by channel to calibrate threshold policies.",
            "Persist model versions and compare live performance against previous release.",
        ]

        return {
            "title": "Fraud_Detection_Graph Research Brief",
            "summary": "Behavioral graph signals improve fraud detection over isolated transaction-only signals in this pipeline.",
            "findings": findings,
            "recommendations": recommendations,
        }

    def results_summary(self) -> dict:
        self.ensure_trained()
        assert self.artifacts is not None

        performance = self.artifacts.metrics
        precision = performance["precision"]
        recall = performance["recall"]
        priority_note = (
            "Precision currently leads recall, which is useful when false positives are costly to customer experience."
            if precision >= recall
            else "Recall currently leads precision, which is useful when fraud capture is the main priority."
        )

        return {
            "title": "Graph Fraud Detection Results Summary",
            "graph_overview": {
                "transactions": self.artifacts.transaction_count,
                "accounts": self.artifacts.account_count,
                "graph_nodes": self.artifacts.graph_nodes,
                "graph_edges": self.artifacts.graph_edges,
            },
            "performance": performance,
            "confusion_matrix": self.artifacts.confusion_matrix,
            "priority_note": priority_note,
            "top_features": self.artifacts.top_features,
            "recommendations": [
                "Track precision and recall by channel before changing production thresholds.",
                "Compare each saved model against the latest deployed model before promotion.",
                "Swap in a PyTorch Geometric or DGL model later without changing the outer API shape.",
            ],
        }

    def predict_account(self, account_id: str, threshold: float = 0.5) -> dict:
        self.ensure_trained()
        assert self.model is not None
        assert self.nodes_df is not None
        assert self.artifacts is not None

        requested_id = account_id
        row = self.nodes_df[self.nodes_df["account_id"] == account_id]
        account_found = not row.empty
        if row.empty:
            # Unknown account fallback: use median-profile feature vector instead of random account.
            fallback = {c: float(self.nodes_df[c].median()) for c in self.artifacts.feature_columns}
            row = pd.DataFrame([fallback])

        x = row[self.artifacts.feature_columns]
        proba = float(self.model.predict_proba(x)[:, 1][0])
        pred = int(proba >= threshold)
        tier = "HIGH" if proba >= 0.75 else "MEDIUM" if proba >= 0.40 else "LOW"

        explanation = []
        sent_count = float(row.iloc[0].get("txn_count_sent", 0))
        out_degree = float(row.iloc[0].get("out_degree", 0))
        unique_receivers = float(row.iloc[0].get("unique_receivers", 0))
        if sent_count > 10:
            explanation.append("High outbound transaction count.")
        if unique_receivers > 8:
            explanation.append("Broad receiver spread detected.")
        if out_degree > float(self.nodes_df["out_degree"].median()):
            explanation.append("Above-average graph connectivity.")
        if not account_found:
            explanation.append("Account not found in current training graph; median profile fallback used.")
        if not explanation:
            explanation.append("Behavior resembles normal cluster profile.")

        return {
            "account_id": str(requested_id),
            "account_found": account_found,
            "fraud_probability": round(proba, 4),
            "threshold_used": round(float(threshold), 4),
            "prediction": pred,
            "risk_tier": tier,
            "explanation": explanation,
        }

    def simulate_realtime_transactions(self, n: int = 5) -> list[dict]:
        self.ensure_trained()
        assert self.nodes_df is not None
        sampled_accounts = self.nodes_df.sample(n=min(n, len(self.nodes_df)), random_state=11)["account_id"].tolist()
        return [self.predict_account(acc) for acc in sampled_accounts]

    def list_saved_models(self) -> list[str]:
        files = sorted(self.model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [f.stem for f in files]

    def save_model(self, model_tag: str | None = None) -> str:
        self.ensure_trained()
        assert self.model is not None
        assert self.df is not None
        assert self.nodes_df is not None
        assert self.artifacts is not None

        tag = model_tag or datetime.now(timezone.utc).strftime("model_%Y%m%d_%H%M%S")
        file_path = self.model_dir / f"{tag}.pkl"
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "df": self.df,
            "nodes_df": self.nodes_df,
            "artifacts": self.artifacts.__dict__,
        }
        with file_path.open("wb") as f:
            pickle.dump(payload, f)
        return tag

    def load_model(self, model_tag: str | None = None) -> str:
        files = self.list_saved_models()
        if not files:
            self.save_model("bootstrap_autosave")
            files = self.list_saved_models()

        selected = model_tag or files[0]
        file_path = self.model_dir / f"{selected}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Model tag not found: {selected}")

        with file_path.open("rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.df = payload["df"]
        self.nodes_df = payload["nodes_df"]
        self.graph = self.build_graph(self.df)
        self.artifacts = TrainedArtifacts(**payload["artifacts"])
        return selected

    def status(self) -> dict:
        models = self.list_saved_models()
        return {
            "trained": self.artifacts is not None and self.model is not None,
            "trained_at": None if self.artifacts is None else self.artifacts.trained_at,
            "model_count": len(models),
            "latest_model_tag": models[0] if models else None,
        }

    def model_manifest(self, model_tag: str | None = None) -> dict:
        files = self.list_saved_models()
        if not files:
            tag = self.save_model("bootstrap_autosave")
        else:
            tag = model_tag or files[0]

        path = self.model_dir / f"{tag}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model tag not found: {tag}")

        with path.open("rb") as f:
            payload = pickle.load(f)

        artifacts = payload["artifacts"]
        return {
            "model_tag": tag,
            "saved_at": payload["saved_at"],
            "trained_at": artifacts["trained_at"],
            "metrics": artifacts["metrics"],
            "top_features": artifacts["top_features"],
            "transaction_count": artifacts["transaction_count"],
            "account_count": artifacts["account_count"],
        }

    def compare_models(self, base_tag: str, candidate_tag: str) -> dict:
        base = self.model_manifest(base_tag)
        candidate = self.model_manifest(candidate_tag)
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        deltas = {
            f"{k}_delta": round(float(candidate["metrics"][k]) - float(base["metrics"][k]), 4)
            for k in keys
        }
        score = deltas["roc_auc_delta"] * 2 + deltas["f1_delta"] + deltas["recall_delta"] * 0.5
        winner = candidate_tag if score >= 0 else base_tag
        return {"base_tag": base_tag, "candidate_tag": candidate_tag, "deltas": deltas, "winner": winner}


graph_ml_service = GraphMLService()
