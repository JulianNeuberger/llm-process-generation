import dataclasses
import typing


@dataclasses.dataclass
class Price:
    prompt: float
    completion: float


prices: typing.Dict[str, Price] = {
    "gpt-4-0125-preview": Price(prompt=0.01, completion=0.03),
    "gpt-4o-2024-05-13": Price(prompt=0.005, completion=0.015),
    "gpt-3.5-turbo-0125": Price(prompt=0.0005, completion=0.0015),
    "claude-3-opus-20240229": Price(prompt=0.015, completion=0.075),
    "claude-3-sonnet-20240229": Price(prompt=0.003, completion=0.015),
    "gpt-4-turbo-2024-04-09": Price(prompt=0.01, completion=0.03),
    "meta-llama/Meta-Llama-3-70B-Instruct": Price(prompt=0.00059, completion=0.00079),
    "deepinfra/airoboros-70b": Price(prompt=0.0007, completion=0.0009),
    "Qwen/Qwen1.5-72B-Chat": Price(prompt=0.00117, completion=0.00117),
}


def get_cost_for_tokens(
    model_name: str, num_input_tokens: int, num_output_tokens
) -> float:
    if model_name not in prices:
        print(
            f"Could not calculate price for {model_name}, prices not in lookup table."
        )
        return -1
    price = prices[model_name]
    return (
        num_input_tokens / 1000.0 * price.prompt
        + num_output_tokens / 1000.0 * price.completion
    )
