from enum import Enum

from .prompter import KG_QA_MULTI_HOP_REASONING_DATA


def get_all_type() -> list:
    return [SubgraphType.MULTI_HOP_REASONING, SubgraphType.QUANTITATIVE_COMPARISON,
            SubgraphType.SET_OPERATION, SubgraphType.ALL]


class SubgraphType(Enum):
    MULTI_HOP_REASONING = 1
    QUANTITATIVE_COMPARISON = 2
    SET_OPERATION = 3
    ALL = 4

    def __str__(self):
        return self.name.lower()

    def get_prompt(self):
        if self == SubgraphType.MULTI_HOP_REASONING:
            return KG_QA_MULTI_HOP_REASONING_DATA
        elif self == SubgraphType.QUANTITATIVE_COMPARISON:
            return "quantitative comparison"
        elif self == SubgraphType.SET_OPERATION:
            return "set operation"
        elif self == SubgraphType.ALL:
            return "all"
        else:
            raise ValueError("Invalid type")
