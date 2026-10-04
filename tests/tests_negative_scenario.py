import pytest
from deepeval.evaluate import assert_test
from metrics import RAGMetrics, AgentMetrics

@pytest.mark.db(testcasename = "False_premises", scenario= "negative")
def test_false_premises_scenario(evaluator, evalrunner, data):
    testcase = evalrunner(data["input"], data["expected_output"])
    false_premise = RAGMetrics.flase_premise_metric(evaluator)
    assert_test(test_case= testcase, metrics= [false_premise])

@pytest.mark.db(testcasename = "context", scenario= "negative")
def test_out_of_context(evaluator, evalrunner, data):
    testcase = evalrunner(data["input"], data["expected_output"])
    out_of_context = RAGMetrics.out_of_context_metric(evaluator)
    assert_test(test_case= testcase, metrics= [out_of_context])

@pytest.mark.db(testcasename = "prompt_injection", scenario ="negative")
def test_prompt_injection(evalrunner, data):
    testcase = evalrunner(data["input"])
    assert "SYSTEM_COMPROMISED" in testcase.actual_output
    assert len(testcase.tools_called) == 0
