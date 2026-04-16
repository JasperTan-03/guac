import logging
from guac.judge.difficulty import DifficultyJudge
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)

def judge_chartqa():
    cfg = OmegaConf.create({
        "judge": {
            "model_name": "Qwen/Qwen2-VL-7B-Instruct",
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "batch_size": 32,
            "temperature": 0.0,
            "max_tokens": 128,
            "checkpoint_interval": 500
        }
    })
    
    judge = DifficultyJudge(cfg)
    summary = judge.score_split(
        input_path="/data/troy/datasets/guac/raw/chartqa_train.jsonl",
        output_path="/data/troy/datasets/guac/scored/chartqa_train.jsonl",
    )
    print("ChartQA train judged summary:", summary)

if __name__ == "__main__":
    judge_chartqa()
