#!/usr/bin/env python3

from cohere_perf_compare_common import main


if __name__ == "__main__":
    raise SystemExit(main(gpu_label="nvidia-gpu", default_output_name="cohere-cpu-vs-nvidia-results.json"))
