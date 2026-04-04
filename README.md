# LangGraph Workflow Examples

A collection of Jupyter notebooks demonstrating progressively complex LangGraph workflow patterns — from single-node LLM calls to parallel multi-evaluator pipelines.

## Notebooks

| Notebook | What it does |
|---|---|
| `bmi_workflow.ipynb` | Sequential graph: calculates BMI, then labels the category |
| `simple_llm_workflow.ipynb` | Single-node graph: routes a question through an OpenAI LLM |
| `basic_chatbot.ipynb` | Stateful chatbot with message history using `add_messages` |
| `batsman_workflow.ipynb` | Parallel graph: computes cricket batting stats (SR, BPB, boundary %) concurrently |
| `essay_workflow.ipynb` | Parallel + fan-in: evaluates an essay on 3 dimensions using GPT-4o-mini with structured output, then aggregates into a summary score |

## Requirements

```bash
pip install -r requirements.txt
```

You also need an OpenAI API key. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Running

```bash
jupyter notebook
```

Open any notebook and run all cells from top to bottom.

## Patterns Covered

**Sequential pipeline** (`bmi_workflow`): nodes run one after the other, each reading and writing to a shared state dict.

```
START → calculate_bmi → label_bmi → END
```

**Parallel fan-out / fan-in** (`batsman_workflow`, `essay_workflow`): multiple nodes run concurrently from `START`, then converge at a single aggregation node.

```
START ──→ evaluate_language ──→
       ├→ evaluate_analysis ──→ final_evaluation → END
       └→ evaluate_thought  ──→
```

**Stateful chat** (`basic_chatbot`): uses `Annotated[list[BaseMessage], add_messages]` to accumulate conversation history across turns.

**Structured output** (`essay_workflow`): uses `model.with_structured_output(EvaluationSchema)` to get typed `feedback` and `score` fields from the LLM.

## Project Structure

```
.
├── bmi_workflow.ipynb          # Sequential: BMI calculator
├── simple_llm_workflow.ipynb   # Single-node: LLM Q&A
├── basic_chatbot.ipynb         # Stateful chatbot
├── batsman_workflow.ipynb      # Parallel: cricket stats
├── essay_workflow.ipynb        # Parallel + structured output: essay grader
├── requirements.txt
└── README.md
```

## Key Dependencies

- [LangGraph](https://github.com/langchain-ai/langgraph) — graph-based workflow orchestration
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/llms/openai) — OpenAI model wrappers
- [Pydantic](https://docs.pydantic.dev/) — structured output schemas (`essay_workflow`)
- [python-dotenv](https://pypi.org/project/python-dotenv/) — `.env` file loading
