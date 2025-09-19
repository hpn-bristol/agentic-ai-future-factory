import json
from langchain.schema import Document as LCDoc
from .pipeline_utils import pipe_key
import config


def purge_feedback_vectors(db):
    res = db._collection.get(where={"type": "feedback"}, include=["metadatas"])
    ids = res["ids"]
    if ids:
        db._collection.delete(ids=ids)
        print(f"🧹 removed {len(ids)} feedback vectors")
    else:
        print("🧹 no feedback vectors to purge")


def log_feedback(db, intent, pipeline_txt, label, reward):
    label_l = label.lower()
    nice_label = {"perfect": "perfect", "partial": "Partial",
                  "bad": "Bad"}.get(label_l, label.title())

    tail = {
        "perfect": "choose this one if you meet this intent again in the future",
        "partial": "works but can be better; try to remove the unnecessary modules next time",
        "bad": "never choose this combination again for this intent"
    }.get(label_l, "")

    outcome = f"{nice_label}, {tail}"
    steps = [ln.strip() for ln in pipeline_txt.splitlines() if ln.strip()]
    pipeline_one_line = ", ".join(
        steps[:-1]) + ", and " + steps[-1] if len(steps) > 1 else (steps[0] if steps else "")

    text = (f"To satisfy '{intent}', we tested the pipeline "
            f"'{pipeline_one_line}', and the result is {outcome}.")

    meta = {"type": "feedback", "label": label,
            "reward": reward, "intent": intent}
    db.add_documents([LCDoc(page_content=text, metadata=meta)])

    with open(config.RAG_FEEDBACK_PATH, "a") as f:
        f.write(text + "\n")

    if label_l == "bad":
        key = pipe_key(pipeline_txt)
        config.BLACKLIST.setdefault(intent, []).append(key)
        with open(config.BL_PATH, "w") as f:
            f.write(json.dumps(config.BLACKLIST, indent=2))
