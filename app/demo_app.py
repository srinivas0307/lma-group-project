"""Demo web app (extension #6, id `ui`): Gradio front end for eval/demo.py. Optional dependency: requirements-demo.txt.

    .venv/bin/pip install -r requirements-demo.txt
    .venv/bin/python app/demo_app.py                 # http://127.0.0.1:7860 ; Task and Replay tabs work on CPU
    .venv/bin/python app/demo_app.py --share         # temporary public link (used in notebooks/kaggle/demo_ui_kaggle.ipynb)

Tabs:
  Task    a task and a pool candidate: contract, code, intent verifier vs conventional checks, answer key
  Replay  a recorded agent / Stage 2 run: its steps, the final code re-checked
  Live    upload a CSV and type a request: parser -> generator -> sandbox -> verifier -> repair (needs a GPU; the
          tab says so when there is none)
Everything shown is the text eval/demo.py prints (app/demo_core.py captures it).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import demo_core as core  # noqa: E402

DEFAULTS = dict(parser=os.environ.get("IGT_DEMO_PARSER", "Qwen/Qwen2.5-7B-Instruct"),
                generator=os.environ.get("IGT_DEMO_GENERATOR", "Qwen/Qwen2.5-3B-Instruct"),
                agent=os.environ.get("IGT_DEMO_AGENT", "Qwen/Qwen2.5-7B-Instruct"),
                adapter=os.environ.get("IGT_DEMO_ADAPTER", ""))


def build_app():
    import gradio as gr
    with gr.Blocks(title="Intent-gated data transformation") as app:
        gr.Markdown("# Intent-gated data transformation: demo\nRequest → intent contract → code → sandbox → "
                    "intent verifier → repair. Task and Replay use recorded data (CPU); Live needs a GPU.")
        with gr.Tab("Task"):
            ids = core.task_ids()
            t_id = gr.Dropdown(ids, value=ids[0] if ids else None, label="task")
            t_cand = gr.Dropdown(choices=core.candidates(ids[0]) if ids else [], value=None,
                                 label="candidate (empty = the first runnable wrong one)")
            t_contract = gr.Dropdown(core.contract_sources(), value="gold", label="contract")
            t_btn = gr.Button("Run", variant="primary")
            t_out = gr.Textbox(label="output", lines=40)
            t_id.change(lambda i: gr.update(choices=core.candidates(i), value=None), t_id, t_cand)
            t_btn.click(lambda i, c, k: core.task(i, c or None, k)[0], [t_id, t_cand, t_contract], t_out)
        with gr.Tab("Replay"):
            folders = core.run_folders()
            r_folder = gr.Dropdown(folders, value=folders[0] if folders else None, label="run folder")
            r_id = gr.Dropdown(core.run_ids(folders[0]) if folders else [], label="scenario / task")
            r_btn = gr.Button("Replay", variant="primary")
            r_out = gr.Textbox(label="output", lines=40)
            r_folder.change(lambda f: gr.update(choices=core.run_ids(f), value=None), r_folder, r_id)
            r_btn.click(lambda f, i: core.replay(f, i)[0] if f and i else "pick a folder and a run", [r_folder, r_id], r_out)
        with gr.Tab("Live"):
            gpu = core.gpu_available()
            gr.Markdown("GPU found: models load on the first run (several minutes), then stay loaded." if gpu else
                        "**No GPU here: Live is disabled.** Run notebooks/kaggle/demo_ui_kaggle.ipynb on Kaggle.")
            l_file = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
            l_req = gr.Textbox(label="request", placeholder="e.g. Remove passengers with no recorded fare")
            with gr.Row():
                l_parser = gr.Textbox(DEFAULTS["parser"], label="parser model")
                l_adapter = gr.Textbox(DEFAULTS["adapter"], label="parser adapter (fine-tuned; optional)")
                l_gen = gr.Textbox(DEFAULTS["generator"], label="generator model")
                l_agent = gr.Textbox(DEFAULTS["agent"], label="agent model")
            with gr.Row():
                l_4bit = gr.Checkbox(True, label="4-bit")
                l_gc = gr.Checkbox(True, label="generator sees the contract")
                l_rep = gr.Checkbox(True, label="repair")
            l_btn = gr.Button("Run", variant="primary", interactive=gpu)
            l_out = gr.Textbox(label="output", lines=40)

            def _live(f, req, p, ad, g, a, q, gc, rep):
                if not f or not req.strip():
                    return "upload a CSV and type a request"
                return core.live(f, req, p, g, a, adapter=ad or None, load_in_4bit=q, generator_contract=gc,
                                 repair=rep)[0]
            l_btn.click(_live, [l_file, l_req, l_parser, l_adapter, l_gen, l_agent, l_4bit, l_gc, l_rep], l_out)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    build_app().queue().launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
