import dataclasses
import typing


@dataclasses.dataclass
class PromptResult:
    prompts: typing.List[str]
    steps: typing.List[typing.List[str]]
    formatter_args: typing.List[typing.Dict[str, typing.Any]]
    formatters: typing.List[str]
    input_tokens: int
    output_tokens: int
    total_costs: float
    answers: typing.List[str]
    original_id: str

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dic: typing.Dict, meta_dic: typing.Dict):
        prompts = dic.get("prompts", None)
        if prompts is None:
            prompts = [dic["prompt"]]

        answers = dic.get("answers", None)
        if answers is None:
            answers = [dic["answer"]]

        formatters = dic.get("formatters", None)
        if formatters is None:
            formatters = [meta_dic["formatter"]]

        steps = dic.get("steps", None)
        if steps is None:
            steps = [meta_dic["steps"]]

        formatter_args = dic.get("formatter_args", None)
        if formatter_args is None:
            formatter_args = [{}] * len(answers)

        return PromptResult(
            prompts=prompts,
            input_tokens=dic["input_tokens"],
            output_tokens=dic["output_tokens"],
            total_costs=dic["total_costs"],
            answers=answers,
            original_id=dic["original_id"],
            formatters=formatters,
            formatter_args=formatter_args,
            steps=steps,
        )

    def __add__(self, other):
        if not isinstance(other, PromptResult):
            raise ValueError()
        if self.original_id != other.original_id:
            print("WARNING: Adding PromptResults run on different documents!")
        return PromptResult(
            prompts=self.prompts + other.prompts,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_costs=self.total_costs + other.total_costs,
            answers=self.answers + other.answers,
            original_id=self.original_id,
            steps=self.steps + other.steps,
            formatters=self.formatters + other.formatters,
            formatter_args=self.formatter_args + other.formatter_args,
        )


@dataclasses.dataclass
class RunMeta:
    num_shots: int
    model: str
    temperature: float

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dic: typing.Dict):
        chat_open_ai_default_temperature = 0.7
        return RunMeta(
            num_shots=dic["num_shots"],
            model=dic["model"],
            temperature=dic.get("temperature", chat_open_ai_default_temperature),
        )


@dataclasses.dataclass
class ExperimentResult:
    meta: RunMeta
    results: typing.List[PromptResult]

    def to_dict(self):
        return {
            "meta": self.meta.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(dic: typing.Dict):
        return ExperimentResult(
            meta=RunMeta.from_dict(dic["meta"]),
            results=[PromptResult.from_dict(r, dic["meta"]) for r in dic["results"]],
        )
