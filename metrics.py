from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    GEval,
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
    TaskCompletionMetric
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.base_metric import BaseMetric

class RAGMetrics:

    @staticmethod
    def answer_relevancy(evaluator) -> AnswerRelevancyMetric:
        return AnswerRelevancyMetric(
            model= evaluator,
            threshold='1.0',
            include_reason= True
        )

    @staticmethod
    def faithfulness(evaluator) -> FaithfulnessMetric:
        return FaithfulnessMetric(
            model= evaluator,
            threshold='1.0',
            include_reason= True
        )

    @staticmethod
    def context_precision(evaluator)-> ContextualPrecisionMetric:
        return ContextualPrecisionMetric(
            model= evaluator,
            threshold= '1.0',
            include_reason= True
        )

    @staticmethod
    def context_relevancy(evaluator)-> ContextualRelevancyMetric:
        return ContextualRelevancyMetric(
            model= evaluator,
            threshold= '1.0',
            include_reason= True
        )

    @staticmethod
    def out_of_context_metric(evaluator)-> BaseMetric:
        """Evaluates whether the system cleanly refuses out-of-domain queries."""
        return GEval(
            name= "Out of Context metric",
            model= evaluator,
            criteria= "If the input is out-of-domain or asks for facts outside the document, the model MUST politely refuse or state lack of context.\n",
            evaluation_params= [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold= 0.7,
        )

    def flase_premise_metric(evaluator)-> BaseMetric:
        """Evaluates whether the system corrects inaccurate user assumptions."""
        return GEval(
            name= "False premise correction",
            model= evaluator,
            criteria= "Verify that the actual output explicitly identifies, refutes, and corrects any false statements or inaccurate premises in the user input.",
            evaluation_params= [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT
            ],
            threshold=0.7
        )


class AgentMetrics:
    @staticmethod
    def toolcorrectness(evaluator) -> ToolCorrectnessMetric:
        return ToolCorrectnessMetric( model= evaluator, threshold='1.0', include_reason=True)

    @staticmethod
    def argumentcorrectness(evaluator) -> ArgumentCorrectnessMetric:
        return ArgumentCorrectnessMetric(model=evaluator, threshold='1.0', include_reason=True)

    @staticmethod
    def taskcompletion(evaluator) -> TaskCompletionMetric:
        return TaskCompletionMetric(model=evaluator, threshold='1.0', include_reason=True)