import os
import pytest
from dotenv import load_dotenv
from deepeval.evaluate import AsyncConfig
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, ToolCall
from Agents.agenticrag import query_AI_agents
from Data.dataloader import csvloader

load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")

@pytest.fixture(scope= "session")
def config():
    asy = AsyncConfig(run_async= False, throttle_value= 12, max_concurrent= 1)
    return asy

@pytest.fixture(scope= "session")
def evaluator():
    evaluator = GeminiModel(
        model = "gemini-3.5-flash-lite",
        api_key= apikey,
        temperature= 0.9
    )
    return evaluator

@pytest.fixture()
def evalrunner():
    def _ask(prompt: str, expected_output: str = None, tool_called : str = None) -> LLMTestCase:
        response, tools, context = query_AI_agents(prompt)
        return LLMTestCase(
            input= prompt,
            expected_output= expected_output,
            actual_output= response,
            expected_tools= [ToolCall(name=tool_called)] if tool_called else None,
            tools_called= [ToolCall(name= tools)] if tool_called else None,
            retrieval_context= [context] if context else None
        )
    return _ask

def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("db")
    if marker:
        testcasename = marker.kwargs["testcasename"].lower()
        scenario = marker.kwargs['scenario']
        data = csvloader(testcasename= testcasename, scenario= scenario)
        metafunc.parametrize("data", data)