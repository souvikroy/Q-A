from langchain_openai import ChatOpenAI

def is_clause(text: str):
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    messages = [
        {"role": "system", "content": "Classify the question as either 'Clause' or 'Not Clause'."},
    ]
    messages.append({"role": "user", "content": text})
    response = llm.invoke(messages)
    return response.content == "Clause"