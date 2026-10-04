import json

def csvloader(testcasename: str, scenario: str):
    path = "/home/zero/Documents/PythonPrograms/AgenticRAG/Data/" + scenario + ".json"
    with open(path, 'r') as f:
        docs = json.load(f)
        data = [doc for doc in docs if doc["testcasename"] == testcasename]
    return data
