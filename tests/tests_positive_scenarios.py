import pytest
from deepeval.evaluate import assert_test
from metrics import RAGMetrics, AgentMetrics

@pytest.mark.db(testcasename = "mcp", scenario= "positive")
def test_mcp(evaluator, evalrunner, data):
    testcase = evalrunner(data["input"], data["expected_output"], data["tool_called"])
    toolmetric = AgentMetrics.toolcorrectness(evaluator)
    answer = RAGMetrics.answer_relevancy(evaluator)
    faith = RAGMetrics.faithfulness(evaluator)
    context = RAGMetrics.context_relevancy(evaluator)
    completion = AgentMetrics.taskcompletion(evaluator)
    assert_test(test_case= testcase, metrics= [toolmetric,answer, faith, context, completion], run_async=False)

@pytest.mark.db(testcasename= "websearch", scenario= "positive")
def test_websearch(evaluator, evalrunner, data):
    testcase = evalrunner(data["input"], data["expected_output"], data["tool_called"])
    toolmetric = AgentMetrics.toolcorrectness(evaluator)
    answer = RAGMetrics.answer_relevancy(evaluator)
    completion = AgentMetrics.taskcompletion(evaluator)
    assert_test(test_case= testcase, metrics= [toolmetric, answer, completion])

@pytest.mark.db(testcasename= "add", scenario= "positive")
def test_add(evaluator, evalrunner, data):
    testcase = evalrunner(data["input"], data["expected_output"], data["tool_called"])
    toolmetric = AgentMetrics.toolcorrectness(evaluator)
    answer = RAGMetrics.answer_relevancy(evaluator)
    completion = AgentMetrics.taskcompletion(evaluator)
    assert_test(test_case= testcase, metrics= [toolmetric, answer, completion])