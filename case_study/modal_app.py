"""Run the case study on Modal GPUs, sharded across containers. Results land in the `memloss-results` volume.

    modal run modal_app.py::main --task A --n 2 --shards 1 --tag pilot-   # timing and accuracy check first
    modal run modal_app.py::main --task A --n 60 --shards 4
    modal run modal_app.py::main --task B --n 100 --shards 4 --max-new-tokens 24
    modal run modal_app.py::main --task A --n 60 --shards 2 --n-records 1400 --conditions full --qk agnostic --tag sweep-N1400-
    modal run modal_app.py::main --task A --n 60 --shards 4 --protect --tag v2-protect-   # item kept at the same budget
    modal run modal_app.py::keysim --model mistralai/Mistral-7B-Instruct-v0.3   # key similarity inside the reader
    modal run modal_app.py::ann --n 500                                            # Section 7: ANN vs exact search
    modal volume get memloss-results / results/

Library versions are pinned to the ones the harness was tested with locally."""

import modal

app = modal.App("memloss-case-study")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1", "transformers==5.2.0", "kvpress==0.5.5", "datasets==5.0.1", "accelerate",
                 "faiss-cpu==1.15.1")
    .add_local_python_source("memloss")
)
results = modal.Volume.from_name("memloss-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("memloss-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", timeout=4 * 3600, memory=32768,  # host RAM for loading 7B weights
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run_shard(task: str, n: int, shard: int, n_shards: int, max_new_tokens: int, model: str, tag: str,
              n_records: int, conditions: str, qk: str, protect: bool, sham: bool) -> str:
    import subprocess
    import sys

    out = f"/results/{tag}{task}-{model.split('/')[-1]}-shard{shard}.jsonl"
    subprocess.run([sys.executable, "-m", "memloss.run", "--task", task, "--n", str(n), "--shard", str(shard),
                    "--n-shards", str(n_shards), "--out", out, "--max-new-tokens", str(max_new_tokens),
                    "--model", model, "--n-records", str(n_records), "--conditions", conditions,
                    "--qk", qk] + (["--protect"] if protect else []) + (["--sham"] if sham else []), check=True)
    results.commit()
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(task: str = "A", n: int = 60, shards: int = 4, max_new_tokens: int = 16,
         model: str = "Qwen/Qwen2.5-7B-Instruct", tag: str = "", n_records: int = 200, conditions: str = "all",
         qk: str = "agnostic,aware", protect: bool = False, sham: bool = False):
    # `tag` prefixes output files, so a pilot never shares a file (and resume state) with the full run.
    args = [(task, n, s, shards, max_new_tokens, model, tag, n_records, conditions, qk, protect, sham) for s in range(shards)]
    for path in run_shard.starmap(args):
        print("done:", path)


@app.function(image=image, gpu="A100-80GB", timeout=2 * 3600, memory=32768,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def keysim_job(model: str, n: int) -> str:
    import subprocess
    import sys

    out = f"/results/keysim-{model.split('/')[-1]}.json"
    subprocess.run([sys.executable, "-m", "memloss.keysim", "--model", model, "--n", str(n), "--out", out], check=True)
    results.commit()
    hf_cache.commit()
    return out


@app.function(image=image, gpu="A100-80GB", timeout=4 * 3600, memory=32768,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def ann_job(n: int, configs: str = "", tag: str = "") -> str:
    import subprocess
    import sys

    out = f"/results/ann-{tag}n{n}.jsonl"
    subprocess.run([sys.executable, "-m", "memloss.ann", "--n", str(n), "--out", out] +
                   (["--configs", configs] if configs else []), check=True)
    results.commit()
    hf_cache.commit()
    return out


@app.local_entrypoint()
def keysim(model: str = "Qwen/Qwen2.5-7B-Instruct", n: int = 10):
    print("done:", keysim_job.remote(model, n))


@app.local_entrypoint()
def ann(n: int = 500, configs: str = "", tag: str = ""):
    print("done:", ann_job.remote(n, configs, tag))
