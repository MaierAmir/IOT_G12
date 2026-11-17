"""dashboard.py - Deliverable III live dashboard (Streamlit).

Reads the predictions table written by the Azure Function (read-only SAS token),
joins the true labels locally (replay_labels.csv, label = labels[sample_id % 2027]),
and shows: live occupancy (predicted vs true), running accuracy / macro-F1,
end-to-end delay split by stage, sensor values, and link gaps.

Run on the Pi:
    ~/dash-venv/bin/streamlit run ~/Wiktor/dashboard.py --server.port 8501
Open from the laptop (Tailscale):  http://100.113.149.37:8501

Secrets: the SAS token is read from ~/Wiktor/.dash_env (TABLE_SAS=...), never from code.
Set DASH_DEMO=1 to run with synthetic data (no Azure needed).
"""
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ACCOUNT_URL = "https://ee5127g12wbstore.table.core.windows.net"
TABLE = "predictions"
HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV = os.path.join(HERE, "replay_labels.csv")
ENV_FILE = os.path.expanduser("~/Wiktor/.dash_env")
N_REPLAY = 2027
DEMO = os.environ.get("DASH_DEMO") == "1"

STAGES = [  # (label, start column, end column)
    ("Pi -> IoT Hub", "gateway_ts", "enqueued_ts"),
    ("IoT Hub -> Function", "enqueued_ts", "func_start_ts"),
    ("Function prep + raw write", "func_start_ts", "endpoint_call_ts"),
    ("ML endpoint call", "endpoint_call_ts", "endpoint_back_ts"),
    ("Write prediction", "endpoint_back_ts", "stored_ts"),
]

st.set_page_config(page_title="EE5127 G12 occupancy", layout="wide")


# ---------- data access ---------------------------------------------------------
def read_sas():
    sas = os.environ.get("TABLE_SAS")
    if sas:
        return sas
    with open(ENV_FILE) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            if k == "TABLE_SAS":
                return v.strip().strip('"').strip("'")
    raise RuntimeError("TABLE_SAS not found in " + ENV_FILE)


@st.cache_resource
def table_client():
    from azure.core.credentials import AzureSasCredential
    from azure.data.tables import TableClient
    sas = read_sas().lstrip("?")
    return TableClient(endpoint=ACCOUNT_URL, table_name=TABLE,
                       credential=AzureSasCredential(sas))


@st.cache_data
def load_labels():
    lab = pd.read_csv(LABELS_CSV)
    return lab.set_index("sample_id")["label"].to_dict()


def demo_rows():
    """Synthetic rows that look like a real run (for testing the page offline)."""
    labels = load_labels()
    n = int((time.time() % 600) // 2) + 40          # grows while you watch
    rng = np.random.default_rng(1)
    rows, t = [], 1.79e9
    for sid in range(n):
        y = labels[sid % N_REPLAY]
        p = y if rng.random() > 0.03 else int(rng.integers(0, 4))
        if sid in (25, 26, 27):                     # simulated link gap
            continue
        g = t + sid * 30
        e = g + rng.normal(0.12, 0.02) + (0.6 if sid == 0 else 0)
        f = e + abs(rng.normal(0.03, 0.01))
        c = f + abs(rng.normal(0.04, 0.01))
        b = c + abs(rng.normal(0.12, 0.03))
        rows.append(dict(PartitionKey="demo", RowKey=f"{sid:06d}", prediction=p,
                         error="", gateway_ts=g, enqueued_ts=e, func_start_ts=f,
                         endpoint_call_ts=c, endpoint_back_ts=b, stored_ts=b + 0.03,
                         batch_size=1, S5_CO2=400 + 300 * (y > 0) + rng.normal(0, 10),
                         S1_Light=10 + 120 * (y > 0) + rng.normal(0, 5)))
    return rows


@st.cache_data(ttl=10)
def list_runs():
    if DEMO:
        return ["demo"]
    seen = {}
    for e in table_client().list_entities(select=["PartitionKey"]):
        seen[e["PartitionKey"]] = seen.get(e["PartitionKey"], 0) + 1
    return sorted(seen, reverse=True)


def load_run(run):
    if DEMO:
        rows = demo_rows()
    else:
        rows = list(table_client().query_entities(f"PartitionKey eq '{run}'"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["sample_id"] = df["RowKey"].astype(int)
    df = df.sort_values("sample_id").reset_index(drop=True)
    labels = load_labels()
    df["true"] = df["sample_id"].map(lambda s: labels[s % N_REPLAY])
    for name, a, b in STAGES:
        df[name] = (df[b] - df[a]) * 1000.0            # ms
    df["total_ms"] = (df["stored_ts"] - df["gateway_ts"]) * 1000.0
    df["time"] = pd.to_datetime(df["gateway_ts"], unit="s", utc=True) \
                   .dt.tz_convert("Europe/Dublin")
    return df


# ---------- metrics -------------------------------------------------------------
def macro_f1(y, p):
    classes = sorted(set(y) | set(p))
    f1s = []
    for c in classes:
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


# ---------- page ----------------------------------------------------------------
st.title("Room occupancy - live predictions")
st.caption("Feather Sense (BLE) -> Raspberry Pi gateway -> Azure IoT Hub -> "
           "Azure Function -> Azure ML endpoint -> Table Storage -> this page")

with st.sidebar:
    LATEST = "Latest run (auto)"
    choice = st.selectbox("Run (run_id from gateway.py)",
                          [LATEST] + list_runs(),
                          help="'Latest' switches to a new run by itself")
    refresh = st.slider("Refresh every (s)", 2, 60, 5)
    window = st.slider("Samples shown in charts", 20, N_REPLAY, 200, step=20)
    st.caption("True labels are joined here, on the Pi; they are never sent "
               "to the cloud.")


@st.fragment(run_every=refresh)
def live():
    if choice == LATEST:
        runs = list_runs()
        if not runs:
            st.info("No rows in the predictions table yet.")
            return
        run = runs[0]
    else:
        run = choice
    st.caption(f"Showing run **{run}**")
    df = load_run(run)
    if df.empty:
        st.info("Waiting for the first message of this run ...")
        return
    ok = df[df["prediction"] >= 0]
    y, p = ok["true"].to_numpy(), ok["prediction"].to_numpy()
    acc = float(np.mean(y == p)) if len(ok) else float("nan")
    f1 = macro_f1(y, p) if len(ok) else float("nan")
    last = df.iloc[-1]
    ids = df["sample_id"]
    gaps = int((ids.max() - ids.min() + 1) - ids.nunique())
    age = time.time() - last["stored_ts"]
    age_txt = (f"last update {age:.0f} s ago" if age < 120 else
               f"last update {age / 60:.0f} min ago" if age < 7200 else
               "run finished")

    c = st.columns(6)
    c[0].metric("Predicted occupancy", int(last["prediction"]),
                help="Number of people (0-3) for the latest sample")
    c[1].metric("True label", int(last["true"]))
    c[2].metric("Samples stored", len(df), f"{gaps} missing" if gaps else None,
                delta_color="inverse")
    c[3].metric("Accuracy so far", f"{acc:.3f}")
    c[4].metric("Macro-F1 so far", f"{f1:.3f}", help="Offline D2 test: 0.986")
    c[5].metric("Median delay (Pi -> stored)", f"{df['total_ms'].median():.0f} ms",
                age_txt, delta_color="off")

    errs = df[df["error"].astype(str) != ""]
    if len(errs):
        st.error(f"{len(errs)} endpoint errors, latest: {errs.iloc[-1]['error']}")

    view = df.tail(window)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Occupancy: predicted vs true")
        occ = view.melt(id_vars=["sample_id"], value_vars=["prediction", "true"],
                        var_name="series", value_name="people")
        st.altair_chart(
            alt.Chart(occ).mark_line(interpolate="step-after").encode(
                x=alt.X("sample_id:Q", title="sample_id"),
                y=alt.Y("people:Q", scale=alt.Scale(domain=[-0.2, 3.2]),
                        axis=alt.Axis(values=[0, 1, 2, 3]), title="people"),
                color=alt.Color("series:N", legend=alt.Legend(orient="top")),
                strokeDash=alt.condition(alt.datum.series == "true",
                                         alt.value([5, 3]), alt.value([1, 0])),
                tooltip=["sample_id", "series", "people"]).properties(height=260),
            width="stretch")

        st.subheader("Sensor values")
        sens = [s for s in ("S5_CO2", "S1_Light") if s in view]
        st.line_chart(view.set_index("sample_id")[sens], height=200)

    with right:
        st.subheader("Delay by stage (median, ms)")
        med = pd.DataFrame({"stage": [s[0] for s in STAGES],
                            "ms": [view[s[0]].median() for s in STAGES]})
        st.altair_chart(
            alt.Chart(med).mark_bar().encode(
                x=alt.X("ms:Q", title="median ms"),
                y=alt.Y("stage:N", sort=[s[0] for s in STAGES], title=None),
                tooltip=["stage", alt.Tooltip("ms:Q", format=".0f")])
            .properties(height=200), width="stretch")

        st.subheader("Total delay per sample (ms)")
        st.line_chart(view.set_index("sample_id")["total_ms"], height=180)

        st.subheader("Confusion matrix (rows = true)")
        cm = pd.crosstab(ok["true"], ok["prediction"], dropna=False)
        st.dataframe(cm, width="stretch")

    st.subheader("Latest messages")
    cols = ["sample_id", "time", "prediction", "true", "total_ms", "batch_size",
            "error"]
    st.dataframe(df[cols].tail(15).iloc[::-1], width="stretch",
                 hide_index=True)
    st.download_button("Download this run as CSV", df.to_csv(index=False),
                       file_name=f"run_{run}.csv", key=f"dl_{run}", mime="text/csv")


live()
