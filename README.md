## CrewAI integration
```
https://ollama.com/download/
ollama run llama3.2
ollama pull mistral
ollama run mistral

pip install chainlit crewai langchain langchain_community langchain_ollama
```

``` mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD
    UserInput["❓ User Question"]
    Rephraser["🔄 Rephraser Agent"]
    Answerer["📘 Answer Agent"]
    FactChecker["🔍 Fact Check Agent"]
    Summarizer["📝 Summary Agent"]
    FinalResponse["✅ Final Response"]

    UserInput --> Rephraser
    Rephraser --> Answerer
    Answerer --> FactChecker
    FactChecker --> Summarizer
    Summarizer --> FinalResponse
```
