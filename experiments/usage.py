import dataclasses
import typing


@dataclasses.dataclass
class Price:
    prompt: float
    completion: float


prices: typing.Dict[str, Price] = {
    "gpt-4-0125-preview": Price(prompt=0.01, completion=0.03),
    "gpt-3.5-turbo-0125": Price(prompt=0.0005, completion=0.0015),
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
