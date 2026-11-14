"""Azure Function (Python v2 model): IoT Hub built-in endpoint -> AML online endpoint -> Table Storage."""
import datetime as dt
import logging
import time
from typing import List

import azure.functions as func
import pipeline

app = func.FunctionApp()
_ctx = None


def ctx():
    global _ctx
    if _ctx is None:
        _ctx = pipeline.make_ctx_from_env()          # reads STORAGE_CONN, AML_SCORING_URI, AML_KEY, MODEL_TAG
    return _ctx


@app.function_name(name="ScoreTelemetry")
@app.event_hub_message_trigger(arg_name="events",
                               connection="IOTHUB_EH_CONN",          # Event Hub-compatible connection string
                               event_hub_name="%IOTHUB_EH_NAME%",    # Event Hub-compatible name
                               cardinality="many")
def score_telemetry(events: List[func.EventHubEvent]):
    c = ctx()
    for ev in events:
        enq = ev.enqueued_time
        if enq is None:
            enq_epoch = time.time()
        else:
            if enq.tzinfo is None:
                enq = enq.replace(tzinfo=dt.timezone.utc)
            enq_epoch = enq.timestamp()
        try:
            pred, status = pipeline.process(ev.get_body(), enq_epoch, c)
            logging.info("scored pred=%s status=%s", pred, status)
        except Exception:
            logging.exception("bad message skipped")
