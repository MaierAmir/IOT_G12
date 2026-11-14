#export STORAGE_CONN="<>"
#streamlit run app.py

import os, time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from azure.data.tables import TableServiceClient
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000"}   # Okabe-Ito
N_ROWS = 2027
LABELS = os.environ.get("LABELS_CSV", os.path.join(os.path.dirname(__file__), "..", "..", "Dataset", "replay_labels.csv"))
STAGES = [("BLE hop (chunk spread)", "ble_s"), ("Gateway queue", "gw_queue_s"),
          ("Gateway -> IoT Hub", "uplink_s"), ("IoT Hub -> Function", "hub_fn_s"),
          ("AML endpoint", "endpoint_s")]

st.set_page_config(page_title="EE5127 occupancy - live", layout="wide")


@st.cache_resource
def service():
    return TableServiceClient.from_connection_string(os.environ["STORAGE_CONN"])


@st.cache_data
def labels():
    return pd.read_csv(LABELS).set_index("sample_id")["label"]


def list_runs():
    t = service().get_table_client("telemetry")
    return sorted({e["PartitionKey"] for e in t.list_entities(select=["PartitionKey"])})


def load(run):
    q = f"PartitionKey eq '{run}'"
    tel = pd.DataFrame(list(service().get_table_client("telemetry").query_entities(q)))
    prd = pd.DataFrame(list(service().get_table_client("predictions").query_entities(q)))
    if tel.empty:
        return tel
    df = tel.drop(columns=["PartitionKey", "RowKey", "Timestamp"], errors="ignore")
    if not prd.empty:
        df = df.merge(prd.drop(columns=["PartitionKey", "RowKey", "Timestamp", "sid"], errors="ignore")
                      .assign(sid=prd["sid"]), on="sid", how="left")
    df = df.sort_values("sid").reset_index(drop=True)
    df["label"] = (df["sid"] % N_ROWS).map(labels())           # join by record identity
    df["ok"] = df.get("pred", pd.Series(dtype=float)) == df["label"]
    df["ble_s"] = df["ble_ms"] / 1000
    df["gw_queue_s"] = df["t_tx"] - df["t_rx"]
    df["uplink_s"] = df["t_enq"] - df["t_tx"]
    df["hub_fn_s"] = df["t_fn"] - df["t_enq"]
    df["endpoint_s"] = df.get("t_score_end", np.nan) - df.get("t_score_start", np.nan)
    df["e2e_s"] = df.get("t_stored", np.nan) - df["t_rx"]
    df["t"] = pd.to_datetime(df["t_rx"], unit="s", utc=True)
    return df


def with_gaps(df, col, gap_s):
    """Insert a NaN row where the stream stopped, so the line breaks instead of bridging an outage."""
    out = df[["t", col]].copy()
    brk = out["t"].diff().dt.total_seconds() > gap_s
    if brk.any():
        fill = out[brk].copy(); fill["t"] = fill["t"] - pd.Timedelta(milliseconds=1); fill[col] = np.nan
        out = pd.concat([out, fill]).sort_values("t")
    return out


def stream_figure(df):
    gap = max(5 * df["t"].diff().dt.total_seconds().median(), 5.0) if len(df) > 2 else 60
    panels = [("S1_Light", "Light S1 (lux)", OI["orange"]), ("S5_CO2", "CO2 S5 (ppm)", OI["green"]),
              ("S1_Sound", "Sound S1", OI["sky"]), ("S6_PIR", "PIR S6 (0/1)", OI["purple"])]
    
    fig = make_subplots(rows=len(panels) + 1, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=[p[1] for p in panels] + ["Occupancy: true label (line) vs prediction (markers)"])
    
    for i, (c, _, colr) in enumerate(panels, 1):
        d = with_gaps(df, c, gap)
        fig.add_trace(go.Scatter(x=d["t"], y=d[c], mode="lines", line=dict(color=colr, width=1.3),
                                 connectgaps=False, showlegend=False), row=i, col=1)
        
    r = len(panels) + 1
    lab = with_gaps(df, "label", gap)
    fig.add_trace(go.Scatter(x=lab["t"], y=lab["label"], mode="lines", line=dict(color=OI["black"], width=1.5, shape="hv"),
                             name="true label", connectgaps=False), row=r, col=1)
    
    good, bad = df[df["ok"]], df[~df["ok"] & df["pred"].isin([0, 1, 2, 3])]
    err = df[df["pred"] == -1]

    fig.add_trace(go.Scatter(x=good["t"], y=good["pred"], mode="markers", name="prediction (correct)",
                             marker=dict(symbol="circle", size=5, color=OI["blue"])), row=r, col=1)
    
    fig.add_trace(go.Scatter(x=bad["t"], y=bad["pred"], mode="markers", name="prediction (wrong)",
                             marker=dict(symbol="x", size=10, color=OI["verm"], line=dict(width=2))), row=r, col=1)
    
    if len(err):
        fig.add_trace(go.Scatter(x=err["t"], y=[-0.5] * len(err), mode="markers", name="endpoint error",
                                 marker=dict(symbol="triangle-down", size=9, color=OI["black"])), row=r, col=1)
        
    fig.update_yaxes(tickvals=[0, 1, 2, 3], row=r, col=1)
    fig.update_layout(height=820, margin=dict(l=40, r=10, t=40, b=30), legend=dict(orientation="h", y=-0.04))
    fig.update_xaxes(title_text="gateway arrival time (UTC)", row=r, col=1)
    return fig


def confusion_figure(df):
    d = df[df["pred"].isin([0, 1, 2, 3])]
    cm = confusion_matrix(d["label"], d["pred"], labels=[0, 1, 2, 3])
    fig = go.Figure(go.Heatmap(z=cm, x=[0, 1, 2, 3], y=[0, 1, 2, 3], colorscale="Blues", showscale=False,
                               text=cm, texttemplate="%{text}", hovertemplate="true %{y}, predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(xaxis_title="predicted people", yaxis_title="true people", height=380,
                      yaxis=dict(autorange="reversed", dtick=1), xaxis=dict(dtick=1), margin=dict(t=10))
    return fig


def delay_table(df):
    rows = []
    for name, c in STAGES + [("End to end (gateway receive -> stored)", "e2e_s")]:
        s = df[c].dropna()
        if len(s):
            rows.append({"stage": name, "median (s)": s.median(), "p95 (s)": s.quantile(.95), "max (s)": s.max()})
    return pd.DataFrame(rows).set_index("stage").round(3)


def latest_tile(df):
    """Live view of newst record, what model says now answer key, sensors"""

    scored = df[df["pred"].notna()] if "pred" in df else df.iloc[0:0]
    last = scored.iloc[-1] if len(scored) else df.iloc[-1]   # newest rec that has pred
    pred, lab = last.get("pred", np.nan), last["label"]
    c = st.columns([3, 2, 2, 4])

    if pd.isna(pred):
        c[0].metric("Current occupancy (predicted)", "scoring...")

    elif int(pred) == -1:
        c[0].metric("Current occupancy (predicted)", "endpoint error")

    else:
        p = int(pred)
        c[0].metric("Current occupancy (predicted)", f"{p} {'person' if p == 1 else 'people'}")

    if pd.isna(lab):
        c[1].metric("True label", "-")

    else:
        mark = "" if pd.isna(pred) or int(pred) == -1 else (" (Correct)" if int(pred) == int(lab) else " (Wrong)")
        c[1].metric("True label", f"{int(lab)}{mark}")

    c[2].metric("Latest sample", f"#{int(last['sid'])}")
    c[2].caption(f"received {max(time.time() - last['t_rx'], 0):.0f} s ago")
    c[3].markdown(
        "**Latest sensor values**  \n"
        f"Light S1-S4: {last['S1_Light']:.0f} / {last['S2_Light']:.0f} / {last['S3_Light']:.0f} / "
        f"{last['S4_Light']:.0f} lux  \n"
        f"CO2 S5: {last['S5_CO2']:.0f} ppm (slope {last['S5_CO2_Slope']:.2f})  \n"
        f"Sound S1-S4: {last['S1_Sound']:.2f} / {last['S2_Sound']:.2f} / {last['S3_Sound']:.2f} / "
        f"{last['S4_Sound']:.2f}  \n"
        f"PIR S6 / S7: {last['S6_PIR']:.0f} / {last['S7_PIR']:.0f}")


def body(run, last_n):
    df = load(run)

    if df.empty:
        st.info("No rows for this run yet."); return
    
    if last_n:
        df = df.tail(last_n)

    latest_tile(df)
    st.divider()
    scored = df[df["pred"].isin([0, 1, 2, 3])]
    acc = scored["ok"].mean() if len(scored) else float("nan")
    base = (scored["label"] == 0).mean() if len(scored) else float("nan")
    f1 = f1_score(scored["label"], scored["pred"], average="macro") if len(scored) else float("nan")
    errs = int((df.get("status", pd.Series(dtype=str)).fillna("") .str.startswith("error")).sum())
    c = st.columns(6)

    c[0].metric("Raw records stored", len(df))
    c[1].metric("Predictions returned", len(scored))
    c[2].metric("Accuracy vs labels", f"{acc:.3f}")
    c[2].caption(f"always-empty baseline: {base:.3f}")
    c[3].metric("Macro-F1", f"{f1:.3f}")
    c[4].metric("Median end-to-end", f"{df['e2e_s'].median():.2f} s")
    c[5].metric("Endpoint errors", errs)

    st.caption("Model stamped on the predictions: " +
               (", ".join(sorted(df["model"].dropna().unique())) if "model" in df else "-"))
    
    t1, t2, t3 = st.tabs(["Live stream", "Predictions vs labels", "Delay"])

    with t1:
        st.plotly_chart(stream_figure(df), use_container_width=True)
        st.caption("Shared time axis. The replay is a sparse random 20 % of the recording (median 122 s apart "
                   "in the original data), so lines show replay order, not original time. A line breaks where "
                   "the stream stopped for longer than 5x the median interval (outage).")
        
    with t2:
        l, r = st.columns([1, 1])
        l.plotly_chart(confusion_figure(df), use_container_width=True)
        r.write("Latest wrong predictions")
        r.dataframe(df[~df["ok"] & df["pred"].notna()][["sid", "label", "pred", "model"]].tail(15), hide_index=True)

    with t3:
        st.dataframe(delay_table(df), use_container_width=True)
        long = df[[c for _, c in STAGES]].rename(columns={c: n for n, c in STAGES})
        fig = go.Figure([go.Box(y=long[n].dropna(), name=n, boxpoints=False) for n in long])
        fig.update_layout(yaxis_title="seconds", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gateway and cloud clocks are NTP-synchronised; offsets of tens of ms are possible. "
                   "Dashboard refresh interval adds up to one refresh period on top.")


st.title("Room occupancy - live pipeline")
runs = list_runs()
default = os.environ.get("RUN_ID")
run = st.sidebar.selectbox("Run", runs, index=runs.index(default) if default in runs else max(len(runs) - 1, 0)) if runs else None
refresh = st.sidebar.slider("Refresh every (s)", 1, 30, 3)
last_n = st.sidebar.number_input("Show last N records (0 = all)", 0, 5000, 0)

if run is None:
    st.info("No runs in storage yet - start the gateway.")
    
else:
    @st.fragment(run_every=refresh)
    def live():
        body(run, int(last_n))
        st.caption(f"updated {time.strftime('%H:%M:%S')}")
    live()
