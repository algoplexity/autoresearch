"""
Small Colab helper for documenting recommended prepare.py usage.
"""

DEFAULT_NUM_SHARDS = 4

def recommended_prepare_command(num_shards: int = DEFAULT_NUM_SHARDS) -> str:
    return f"python prepare.py --num-shards {num_shards}"

if __name__ == "__main__":
    print(recommended_prepare_command())