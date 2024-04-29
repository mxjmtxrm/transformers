from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig


REPO_ID = "hf-transformers-bot/benchmark"
EXPERIMENT_NAME = "text_generation_with_static_cache_models"

# for list with static cache support
# https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+_setup_cache%28self&type=code
TEXT_GENERATION_WITH_STATIC_CACHE_MODELS_LIST = [
    #"google/gemma-2b",
    # "google/gemma-7b",
    # "huggyllama/llama-7b",
    "meta-llama/Llama-2-7b-hf",
    # "mistralai/Mistral-7B-v0.1",
]
INPUT_SHAPES = {
    "batch_size": 1,
    "sequence_length": 16,
}
GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "min_new_tokens": 128,
    "do_sample": False,
}
TORCH_COMPILE_CONFIG = {
    "backend": "inductor",
    "mode": "reduce-overhead",
    "fullgraph": True,
}
CACHE_IMPLEMENTATION = "static"


def benchmark_text_generation_with_static_cache_models():
    for model in TEXT_GENERATION_WITH_STATIC_CACHE_MODELS_LIST:
        launcher_config = ProcessConfig(start_method="spawn")  # isolated process
        benchmark_config = InferenceConfig(
            memory=True,
            latency=True,
            input_shapes=INPUT_SHAPES,
            generate_kwargs=GENERATE_KWARGS,
            iterations=2,
            duration=0,
        )
        backend_config = PyTorchConfig(
            model=model,
            device="cuda",
            device_ids="0",
            torch_dtype="float16",
            no_weights=True,
            torch_compile=True,
            torch_compile_config=TORCH_COMPILE_CONFIG,
            cache_implementation=CACHE_IMPLEMENTATION,
        )

        experiment_config = ExperimentConfig(
            experiment_name=EXPERIMENT_NAME,
            benchmark=benchmark_config,
            launcher=launcher_config,
            backend=backend_config,
        )

        benchmark_report = launch(experiment_config)

        # so far
        print(benchmark_report.decode.latency.values)
        print(benchmark_report.decode.latency.mean)

        experiment_config.push_to_hub(
            commit_message="Added experiment config",
            subfolder=f"{EXPERIMENT_NAME}/{model}",
            repo_id=REPO_ID,
            private=True,
        )
        benchmark_report.push_to_hub(
            commit_message="Added benchmark report",
            subfolder=f"{EXPERIMENT_NAME}/{model}",
            repo_id=REPO_ID,
            private=True,
        )


if __name__ == "__main__":
    benchmark_text_generation_with_static_cache_models()