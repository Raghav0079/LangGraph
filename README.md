# LangGraph Workflow Examples

A collection of Jupyter notebooks demonstrating LangGraph workflow patterns — from single-node LLM calls to parallel multi-evaluator pipelines with structured output. Each notebook is self-contained and runnable.

---

## Requirements

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

## Running

```bash
jupyter notebook
```

Open any notebook and run all cells top to bottom. MNIST (for SWIN) and OpenAI credentials load automatically if configured.

---

## Notebooks

### 1. `bmi_workflow.ipynb` — Sequential Pipeline

Calculates BMI from weight and height, then classifies it into a category. The simplest graph pattern: a straight chain of nodes.

**State:**
```python
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float        # written by calculate_bmi
    category: str     # written by label_bmi
```

**Graph:**
```
START → calculate_bmi → label_bmi → END
```

**Example:**
```python
workflow.invoke({'weight_kg': 80, 'height_m': 1.73})
# → {'bmi': 26.73, 'category': 'Overweight', ...}
```

**Category thresholds:**

| BMI | Category |
|---|---|
| < 18.5 | Underweight |
| 18.5 – 24.9 | Normal |
| 25 – 29.9 | Overweight |
| ≥ 30 | Obese |

---

### 2. `simple_llm_workflow.ipynb` — Single-Node LLM Call

Routes a plain text question through an OpenAI model and stores the answer in state. Shows the minimal viable LangGraph + LLM setup.

**State:**
```python
class LLMState(TypedDict):
    question: str
    answer: str
```

**Graph:**
```
START → llm_qa → END
```

**Example:**
```python
workflow.invoke({'question': 'How far is the moon from Earth?'})
# → {'answer': 'The average distance is about 384,400 km (238,855 miles).'}
```

**Model used:** `ChatOpenAI()` (defaults to `gpt-3.5-turbo`)

---

### 3. `basic_chatbot.ipynb` — Stateful Conversation

A multi-turn chatbot that accumulates message history using LangGraph's `add_messages` reducer. Each call appends to the message list rather than overwriting it.

**State:**
```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**Graph:**
```
START → chat_node → END
```

**Example:**
```python
from langchain_core.messages import HumanMessage

state = {'messages': [HumanMessage(content='What is the capital of India?')]}
chatbot.invoke(state)['messages'][-1].content
# → 'New Delhi'
```

The `add_messages` annotation means you can call `invoke` repeatedly with new `HumanMessage`s and the full conversation context is preserved.

**Model used:** `ChatOpenAI()` (defaults to `gpt-3.5-turbo`)

---

### 4. `batsman_workflow.ipynb` — Parallel Fan-Out

Computes three cricket batting statistics concurrently from the same input state, then combines them into a summary string. Demonstrates fan-out from `START` to multiple parallel nodes.

**State:**
```python
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    sr: float              # Strike Rate
    bpb: float             # Balls per Boundary
    boundary_percent: float
    summary: str
```

**Graph:**
```
              ┌→ calculate_sr ──────────┐
START ────────┼→ calculate_bpb ─────────┼→ summary → END
              └→ calculate_boundary_% ──┘
```

**Formulas:**

| Stat | Formula |
|---|---|
| Strike Rate | `(runs / balls) × 100` |
| Balls per Boundary | `balls / (fours + sixes)` |
| Boundary % | `((fours×4 + sixes×6) / runs) × 100` |

**Example:**
```python
workflow.invoke({'runs': 100, 'balls': 50, 'fours': 6, 'sixes': 4})
# → {'sr': 200.0, 'bpb': 5.0, 'boundary_percent': 48.0, 'summary': '...'}
```

No LLM required — pure Python computation nodes.

---

### 5. `essay_workflow.ipynb` — Parallel Evaluation + Structured Output

Evaluates a UPSC-style essay on three independent dimensions in parallel, then fans in to aggregate the scores and write a combined summary. Uses Pydantic structured output to extract typed `feedback` and `score` fields from the model.

**State:**
```python
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]  # accumulates across parallel nodes
    avg_score: float
```

**Graph:**
```
              ┌→ evaluate_language ──────┐
START ────────┼→ evaluate_analysis ──────┼→ final_evaluation → END
              └→ evaluate_thought  ──────┘
```

**Structured output schema:**
```python
class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score: int    = Field(description='Score out of 10', ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)
```

The `individual_scores` field uses `Annotated[list[int], operator.add]` so each parallel node's `[score]` list is appended rather than overwritten. The fan-in node averages them.

**Example:**
```python
result = workflow.invoke({'essay': my_essay_string})
print(result['avg_score'])        # e.g. 7.33
print(result['overall_feedback']) # synthesized summary
```

**Model used:** `ChatOpenAI(model='gpt-4o-mini')`

---

## LangGraph Concepts Index

| Concept | Where it appears |
|---|---|
| Sequential edges (`add_edge`) | All notebooks |
| Parallel fan-out from `START` | `batsman_workflow`, `essay_workflow` |
| Fan-in to single node | `batsman_workflow`, `essay_workflow` |
| `add_messages` reducer | `basic_chatbot` |
| `operator.add` list reducer | `essay_workflow` |
| `with_structured_output` | `essay_workflow` |
| `.env` / `load_dotenv` | `simple_llm_workflow`, `essay_workflow` |

---

## Project Structure

```
.
├── bmi_workflow.ipynb          # Sequential: BMI calculator (no LLM)
├── simple_llm_workflow.ipynb   # Single-node: LLM Q&A
├── basic_chatbot.ipynb         # Stateful chatbot with message history
├── batsman_workflow.ipynb      # Parallel: cricket batting stats (no LLM)
├── essay_workflow.ipynb        # Parallel + structured output: essay grader
├── requirements.txt
└── README.md
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Graph-based workflow orchestration |
| `langchain-openai` | OpenAI model wrappers |
| `langchain-core` | `BaseMessage`, `HumanMessage`, reducers |
| `pydantic` | Structured output schemas (`essay_workflow`) |
| `python-dotenv` | `.env` API key loading |
| `openai` | Underlying OpenAI SDK |
