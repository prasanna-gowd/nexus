# NEXUS Phase 5: Multi-Agent Orchestration & Socratic Pedagogy Implementation Plan

## 1. Overview & Scope
Phase 5 introduces the autonomous multi-agent orchestration layer, transforming NEXUS from a passive intelligence pipeline into a closed-loop pedagogical ecosystem. Using **LangGraph**, this phase implements a `StateGraph` state machine to manage the student's learning session continuously.

The system orchestrates three specialized agents as nodes:
1. **Roadmap Agent Node:** Connects to Phase 4 endpoints to fetch and recalibrate the active trajectory (DAG).
2. **Socrates Agent Node:** Fetches real-world vector contexts and guides the student via Socratic dialogue[cite: 4].
3. **Cognitive Profiler Node:** Evaluates student responses, performs dynamic knowledge tracing, computes a mastery score, and conditionally routes the flow[cite: 4].

---

## 2. Directory Structure & Dependencies

Add these files to the repository structure:
```text
nexus-market-pipeline/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py                 # Graph state schema (TypedDict)
│   │   ├── prompts.py               # System prompts for agents
│   │   ├── nodes.py                 # Core agent logic and LLM invocations
│   │   └── orchestrator.py          # LangGraph StateGraph assembly and routing

Dependencies:
Ensure you have the required agent framework packages installed:
pip install langgraph langchain-openai langchain-core

3. Component Specifications
3.1 Shared Learner State (src/agents/state.py)
This defines the memory structure passed between agent nodes during execution, keeping track of what the student knows and what needs to be taught[cite: 4].

# src/agents/state.py
from typing import TypedDict, List, Dict, Any

class LearnerState(TypedDict):
    student_id: str
    target_role: str
    verified_skills: List[str]
    active_trajectory: List[str]  # Ordered DAG sequence
    current_skill: str            # Skill currently being taught (s_k)
    chat_history: List[Dict[str, str]]
    mastery_score: float          # 0.0 to 1.0 confidence


3.2 Agent System Prompts (src/agents/prompts.py)
System prompts explicitly restrict the agents to Socratic discovery and structured knowledge tracing, ensuring they do not give direct answers[cite: 4].

# src/agents/prompts.py

SOCRATES_PROMPT = """You are the NEXUS Socrates Agent, a master technical tutor.
Your goal is to teach the skill: '{current_skill}'.
Here is real-world context for how this skill is used in industry:
{real_world_context}

RULES:
1. NEVER give the student the direct answer or copy-paste code upfront.
2. Ask one guiding question at a time.
3. Ground your scenario in the real-world context provided.
4. If the student struggles, provide a scaffolded hint, but make them do the work.
"""

PROFILER_PROMPT = """You are the NEXUS Cognitive Profiler. 
Your job is to evaluate the student's latest response regarding the concept: '{current_skill}'.
Assess their conceptual depth, correctness, and problem-solving approach.
Return ONLY a JSON object with a single key "mastery_score" containing a float between 0.0 and 1.0.
A score >= 0.8 indicates they have mastered the current step and can move on.
"""

3.3 Node Functions (src/agents/nodes.py)
These are the core execution blocks that invoke the LLM and the tools built in Phase 4.

# src/agents/nodes.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agents.state import LearnerState
from src.agents.prompts import SOCRATES_PROMPT, PROFILER_PROMPT
from src.agents.nexus_tools import analyze_student_skill_gap, fetch_real_world_context

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def roadmap_agent_node(state: LearnerState) -> dict:
    """Invokes Phase 4 API to align learning path with live market DAG."""
    # Only fetch the roadmap if we don't have an active trajectory
    if not state.get("active_trajectory"):
        gap_analysis = analyze_student_skill_gap(
            target_role=state["target_role"],
            student_skills=state["verified_skills"]
        )
        trajectory = [step["skill_name"] for step in gap_analysis["ordered_trajectory"]]
        current = trajectory[0] if trajectory else ""
        return {"active_trajectory": trajectory, "current_skill": current}
    
    # If we already have a trajectory, advance to the next skill
    trajectory = state["active_trajectory"]
    current = trajectory[0] if trajectory else ""
    return {"current_skill": current}

def socrates_agent_node(state: LearnerState) -> dict:
    """Executes pedagogical dialogue grounded in real-world API contexts."""
    skill = state.get("current_skill")
    if not skill:
        return {"chat_history": state.get("chat_history", []) + [{"role": "agent", "content": "You have mastered all required skills!"}]}

    # Retrieve job context to ground the teaching
    try:
        context_data = fetch_real_world_context(skill)
        context_str = "\n".join([c["snippet"] for c in context_data.get("contexts", [])])
    except Exception:
        context_str = "Standard industry implementation."

    prompt = SOCRATES_PROMPT.format(current_skill=skill, real_world_context=context_str)
    
    messages = [SystemMessage(content=prompt)]
    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    response = llm.invoke(messages)
    
    new_history = state.get("chat_history", []) + [{"role": "agent", "content": response.content}]
    return {"chat_history": new_history}

def cognitive_profiler_node(state: LearnerState) -> dict:
    """Evaluates student response against task and performs Knowledge Tracing."""
    skill = state.get("current_skill")
    history = state.get("chat_history", [])
    
    if not history or history[-1]["role"] != "user":
        return {"mastery_score": 0.0}

    student_msg = history[-1]["content"]
    agent_msg = history[-2]["content"] if len(history) > 1 else ""

    eval_prompt = PROFILER_PROMPT.format(current_skill=skill)
    messages = [
        SystemMessage(content=eval_prompt),
        HumanMessage(content=f"Agent asked: {agent_msg}\nStudent answered: {student_msg}")
    ]
    
    response = llm.invoke(messages)
    try:
        # Extract the JSON mastery score safely
        score_data = json.loads(response.content.strip("```json\n"))
        score = float(score_data.get("mastery_score", 0.0))
    except Exception:
        score = 0.0
        
    updates = {"mastery_score": score}
    
    # If the student demonstrates competence, promote the skill
    if score >= 0.8:
        updates["verified_skills"] = state["verified_skills"] + [skill]
        updates["active_trajectory"] = state["active_trajectory"][1:]
        
    return updates


    3.4 StateGraph Assembly & Routing (src/agents/orchestrator.py)
This file compiles the agents into a cyclic graph, using conditional edges to control the flow between testing, tutoring, and advancing[cite: 4].


# src/agents/orchestrator.py
from langgraph.graph import StateGraph, END
from src.agents.state import LearnerState
from src.agents.nodes import roadmap_agent_node, socrates_agent_node, cognitive_profiler_node

def mastery_router(state: LearnerState) -> str:
    """Routes execution based on the mastery score threshold."""
    if not state.get("current_skill"):
        return END
    
    if state.get("mastery_score", 0.0) >= 0.8:
        # Mastery achieved: Go back to the Roadmap to fetch the next concept
        return "Roadmap_Agent"
    else:
        # Need more tutoring: Route back to Socrates
        return "Socrates_Agent"

# 1. Initialize StateGraph
workflow = StateGraph(LearnerState)

# 2. Add Nodes
workflow.add_node("Roadmap_Agent", roadmap_agent_node)
workflow.add_node("Socrates_Agent", socrates_agent_node)
workflow.add_node("Cognitive_Profiler", cognitive_profiler_node)

# 3. Define Fixed Edges
workflow.set_entry_point("Roadmap_Agent")
workflow.add_edge("Roadmap_Agent", "Socrates_Agent")
# After Socrates asks a question, wait for user input (handled by API/UI interruption), 
# then run the Profiler to evaluate the response.
workflow.add_edge("Socrates_Agent", "Cognitive_Profiler")

# 4. Add Conditional Routing
workflow.add_conditional_edges(
    "Cognitive_Profiler",
    mastery_router,
    {
        "Roadmap_Agent": "Roadmap_Agent",
        "Socrates_Agent": "Socrates_Agent",
        END: END
    }
)

# 5. Compile the graph
nexus_app = workflow.compile()

# src/agents/orchestrator.py
from langgraph.graph import StateGraph, END
from src.agents.state import LearnerState
from src.agents.nodes import roadmap_agent_node, socrates_agent_node, cognitive_profiler_node

def mastery_router(state: LearnerState) -> str:
    """Routes execution based on the mastery score threshold."""
    if not state.get("current_skill"):
        return END
    
    if state.get("mastery_score", 0.0) >= 0.8:
        # Mastery achieved: Go back to the Roadmap to fetch the next concept
        return "Roadmap_Agent"
    else:
        # Need more tutoring: Route back to Socrates
        return "Socrates_Agent"

# 1. Initialize StateGraph
workflow = StateGraph(LearnerState)

# 2. Add Nodes
workflow.add_node("Roadmap_Agent", roadmap_agent_node)
workflow.add_node("Socrates_Agent", socrates_agent_node)
workflow.add_node("Cognitive_Profiler", cognitive_profiler_node)

# 3. Define Fixed Edges
workflow.set_entry_point("Roadmap_Agent")
workflow.add_edge("Roadmap_Agent", "Socrates_Agent")
# After Socrates asks a question, wait for user input (handled by API/UI interruption), 
# then run the Profiler to evaluate the response.
workflow.add_edge("Socrates_Agent", "Cognitive_Profiler")

# 4. Add Conditional Routing
workflow.add_conditional_edges(
    "Cognitive_Profiler",
    mastery_router,
    {
        "Roadmap_Agent": "Roadmap_Agent",
        "Socrates_Agent": "Socrates_Agent",
        END: END
    }
)

# 5. Compile the graph
nexus_app = workflow.compile()

4. Execution & Validation Plan
To verify the architecture, the Antigravity agent should create a quick local test script (test_flow.py) to simulate a multi-turn interaction.

# test_flow.py
from src.agents.orchestrator import nexus_app

initial_state = {
    "student_id": "STU-001",
    "target_role": "Machine Learning Engineer",
    "verified_skills": ["Python"],
    "active_trajectory": [],
    "current_skill": "",
    "chat_history": [],
    "mastery_score": 0.0
}

# Step 1: Run the first pass (Roadmap computes DAG -> Socrates asks first question)
output = nexus_app.invoke(initial_state)
print("Socrates Agent:", output["chat_history"][-1]["content"])

# Step 2: Simulate a perfect student response
output["chat_history"].append({
    "role": "user", 
    "content": "I would use a directed acyclic graph to ensure we resolve dependencies in topological order."
})

# Step 3: Continue the workflow (Profiler evaluates -> Roadmap advances skill)
output = nexus_app.invoke(output)
print("Mastery Score:", output["mastery_score"])
print("Next Active Skill:", output["current_skill"])