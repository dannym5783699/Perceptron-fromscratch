from dataclasses import dataclass
from typing import Callable


def case1_condition(X):
    return -X[:, 0] + X[:, 1]


def case2_condition(X):
    return X[:, 0] - (2 * X[:, 1]) + 5


@dataclass
class TestCase:
    case_condition: Callable
    n_features: int
    name: str = None


test_cases = [
    TestCase(case1_condition, 2, "Case 1"),
    TestCase(case2_condition, 2, "Case 2")
]
