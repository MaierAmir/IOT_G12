import json, os, time, urllib.request
from azure.data.tables import TableServiceClient, UpdateMode

FEATURES = ["S1_Light", "S2_Light", "S3_Light", "S4_Light",
            "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
            "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]          # = deploy_config.json order
INTS = {"S1_Light", "S2_Light", "S3_Light", "S4_Light", "S5_CO2", "S6_PIR", "S7_PIR"}


class Ctx:
    def __init__(self, telemetry, predictions, scorer, model_tag):
        self.telemetry, self.predictions, self.scorer, self.model_tag = telemetry, predictions, scorer, model_tag


def make_ctx_from_env():
    svc = TableServiceClient.from_connection_string(os.environ["STORAGE_CONN"])
    tel = svc.create_table_if_not_exists("telemetry")
    pred = svc.create_table_if_not_exists("predictions")
    scorer = make_scorer(os.environ["AML_SCORING_URI"], os.environ["AML_KEY"], os.environ.get("AML_DEPLOYMENT"))
    return Ctx(tel, pred, scorer, os.environ.get("MODEL_TAG", "ee5127-g12-occupancy:unknown"))


def make_scorer(uri, key, deployment=None, timeout=15):
    """Returns f(rows)->list[int]. rows = list of lists in FEATURES order (ints already cast)."""
    def scorer(rows):
        body = json.dumps({"input_data": {"columns": FEATURES, "data": rows}}).encode()
        req = urllib.request.Request(uri, body, {"Content-Type": "application/json",
                                                 "Authorization": "Bearer " + key})
        if deployment:
            req.add_header("azureml-model-deployment", deployment)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        if isinstance(out, dict):                                  # custom score.py wraps list
            out = out.get("predictions", out)
        return [int(x) for x in out]
    return scorer


def process(body: bytes, enq_epoch: float, ctx: Ctx):
    """Handle one IoT Hub message. enq_epoch = IoT Hub enqueued time (UTC epoch seconds)."""
    t_fn = time.time()
    msg = json.loads(body)
    run, sid = str(msg["run"]), int(msg["sid"])
    rk = f"{sid:08d}"
    # cloudside schma guard same casts signature expects
    row = [int(round(float(msg[f]))) if f in INTS else float(msg[f]) for f in FEATURES]

    t0 = time.time()
    ctx.telemetry.upsert_entity({
        "PartitionKey": run, "RowKey": rk, "sid": sid, "src": msg.get("src", "?"),
        "t_rx": float(msg["t_rx"]), "t_tx": float(msg.get("t_tx", 0)), "ble_ms": float(msg.get("ble_ms", 0)),
        "t_enq": enq_epoch, "t_fn": t_fn, **dict(zip(FEATURES, row))}, mode=UpdateMode.REPLACE)
    tel_write_ms = (time.time() - t0) * 1000

    t_s0 = time.time()
    try:
        pred, status = ctx.scorer([row])[0], "ok"
    except Exception as e:                                         # endpoint down/slow, keep the raw row
        pred, status = -1, ("error: " + repr(e))[:250]
    t_s1 = time.time()

    ctx.predictions.upsert_entity({
        "PartitionKey": run, "RowKey": rk, "sid": sid, "pred": int(pred), "status": status,
        "model": ctx.model_tag, "t_score_start": t_s0, "t_score_end": t_s1,
        "t_stored": time.time(), "tel_write_ms": tel_write_ms}, mode=UpdateMode.REPLACE)
    return pred, status
