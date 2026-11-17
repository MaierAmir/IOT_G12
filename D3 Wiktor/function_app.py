import json, os, time, logging, urllib.request
import azure.functions as func
from azure.data.tables import TableServiceClient

app = func.FunctionApp()

COLS = ["S1_Light", "S2_Light", "S3_Light", "S4_Light",
        "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
        "S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]

_svc = TableServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
raw_t = _svc.create_table_if_not_exists("rawdata")
pred_t = _svc.create_table_if_not_exists("predictions")


@app.event_hub_message_trigger(arg_name="events", event_hub_name="%EH_NAME%",
                               connection="EH_CONN", cardinality="many")
def score(events: list[func.EventHubEvent]):
    t0 = time.time()
    msgs = []
    for e in events:
        try:
            m = json.loads(e.get_body().decode("utf-8"))
            m["_enq"] = e.enqueued_time.timestamp() if e.enqueued_time else None
            msgs.append(m)
        except Exception as ex:
            logging.error("bad message: %s", ex)
    if not msgs:
        return

    # 1) raw data first, so nothing is lost if the endpoint fails
    for m in msgs:
        raw_t.upsert_entity({
            "PartitionKey": str(m["run_id"]), "RowKey": f"{int(m['sample_id']):06d}",
            "body": json.dumps({k: v for k, v in m.items() if k != "_enq"}),
            "enqueued_ts": m["_enq"] or 0.0, "func_start_ts": t0})

    # 2) one endpoint call for the whole batch (usually one message)
    body = json.dumps({"input_data": {
        "columns": COLS, "index": list(range(len(msgs))),
        "data": [[m[c] for c in COLS] for m in msgs]}}).encode()
    req = urllib.request.Request(os.environ["ENDPOINT_URL"], body, {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["ENDPOINT_KEY"]})
    t_call, err = time.time(), ""
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            preds = json.loads(r.read())
    except Exception as ex:
        preds, err = [None] * len(msgs), str(ex)[:200]
        logging.error("endpoint call failed: %s", ex)
    t_back = time.time()

    # 3) predictions (upsert on run_id + sample_id, so a re-sent message is not duplicated)
    for m, p in zip(msgs, preds):
        ent = {"PartitionKey": str(m["run_id"]), "RowKey": f"{int(m['sample_id']):06d}",
               "prediction": -1 if p is None else int(p), "error": err,
               "gateway_ts": float(m["gateway_ts"]), "enqueued_ts": m["_enq"] or 0.0,
               "func_start_ts": t0, "endpoint_call_ts": t_call, "endpoint_back_ts": t_back,
               "stored_ts": time.time(), "batch_size": len(msgs)}
        ent.update({c: float(m[c]) for c in COLS})
        pred_t.upsert_entity(ent)
