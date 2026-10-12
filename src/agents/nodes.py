"""
Agent Node Implementations for NEXUS Multi-Agent StateGraph.

Features:
- Roadmap Agent: Prerequisite DAG alignment & skill trajectory scheduling.
- Socrates Agent: Context-grounded inquiry & scaffolding hints.
- Cognitive Profiler: Knowledge Tracing, rubric evaluation & mastery thresholding.
- Dual LLM Engine: ChatOpenAI (when API key present) + intelligent local simulator.
"""
import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.agents.state import LearnerState
from src.agents.prompts import SOCRATES_PROMPT, PROFILER_PROMPT
from src.agents.nexus_tools import analyze_student_skill_gap, fetch_real_world_context

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MASTERY_THRESHOLD = float(os.getenv("MASTERY_THRESHOLD", "0.8"))


def get_llm():
    """Initializes ChatOpenAI if credentials exist; otherwise returns None."""
    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOpenAI: {e}")
    return None


# --- 1. Roadmap Agent Node ---
def roadmap_agent_node(state: LearnerState) -> Dict[str, Any]:
    """
    Analyzes skill gaps, establishes the active learning trajectory,
    and nominates the next unmastered prerequisite skill.
    """
    logger.info("Executing Roadmap Agent Node...")
    verified = list(state.get("verified_skills", []))
    target_role = state.get("target_role", "Software Engineer")
    trajectory = list(state.get("active_trajectory", []))

    # If active trajectory is empty, generate from Skill-Gap Engine
    if not trajectory:
        gap_analysis = analyze_student_skill_gap(
            target_role=target_role,
            student_skills=verified
        )
        trajectory = [step["skill_name"] for step in gap_analysis.get("ordered_trajectory", [])]

    # Filter out any skills that are already verified
    verified_lower = {s.lower() for s in verified}
    remaining_trajectory = [s for s in trajectory if s.lower() not in verified_lower]

    if not remaining_trajectory:
        logger.info("Student has completed all skills in trajectory!")
        return {
            "active_trajectory": [],
            "current_skill": "",
            "is_completed": True,
            "pending_response": False,
        }

    next_skill = remaining_trajectory[0]
    logger.info(f"Roadmap Agent selected next skill: {next_skill}")

    return {
        "active_trajectory": remaining_trajectory,
        "current_skill": next_skill,
        "is_completed": False,
        "attempt_count": 0,
        "mastery_score": 0.0,
    }


# --- 2. Socrates Agent Node ---
def socrates_agent_node(state: LearnerState) -> Dict[str, Any]:
    """
    Conducts interactive Socratic dialogue grounded in real-world market context.
    Provides scaffolded hints during remediation attempts.
    """
    logger.info("Executing Socrates Agent Node...")
    if state.get("is_completed"):
        completion_msg = {
            "role": "agent",
            "content": f"🎉 Outstanding achievement! You have mastered all key prerequisite skills for the {state.get('target_role')} roadmap. You are ready for industry technical interviews!"
        }
        history = list(state.get("chat_history", [])) + [completion_msg]
        return {
            "chat_history": history,
            "pending_response": False,
        }

    skill = state.get("current_skill", "Software Engineering")
    attempt = state.get("attempt_count", 0)
    prev_score = state.get("mastery_score", 0.0)

    # 1. Retrieve real-world job context
    try:
        context_data = fetch_real_world_context(skill)
        contexts = context_data.get("contexts", [])
        context_str = "\n".join([f"- [{c.get('company', 'Industry')}]: {c.get('snippet', '')}" for c in contexts])
    except Exception as e:
        logger.warning(f"Context retrieval error: {e}")
        context_str = f"Production engineering standards for {skill} in modern distributed systems."

    # 2. Invoke LLM or Fallback Simulator
    llm = get_llm()
    agent_text = ""

    if llm is not None:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            prompt = SOCRATES_PROMPT.format(current_skill=skill, real_world_context=context_str)
            messages = [SystemMessage(content=prompt)]
            for msg in state.get("chat_history", []):
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            if attempt > 0 and prev_score < MASTERY_THRESHOLD:
                messages.append(HumanMessage(content=f"[SYSTEM NOTE: Student scored {prev_score:.2f}. Provide a supportive scaffolding hint.]"))

            response = llm.invoke(messages)
            agent_text = response.content.strip()
        except Exception as e:
            logger.error(f"OpenAI invocation failed: {e}")

    if not agent_text:
        # Intelligent Local Pedagogical Simulator
        agent_text = _simulate_socratic_response(skill, attempt, prev_score, context_str)

    history = list(state.get("chat_history", [])) + [{"role": "agent", "content": agent_text}]

    return {
        "chat_history": history,
        "pending_response": True, # Awaiting human response
    }


def _simulate_socratic_response(skill: str, attempt: int, prev_score: float, context: str) -> str:
    """Deterministic, high-quality Socratic questions grounded in domain skills."""
    skill_lower = skill.lower()

    if attempt == 0:
        # Initial inquiry
        questions = {
            "python": f"In backend and data pipelines, Python is widely adopted. Imagine your service needs to process 10,000 requests asynchronously without blocking. Between multiprocessing, threading, and asyncio, which architectural approach would you select for I/O-bound tasks, and why?",
            "docker": f"Looking at production hiring requirements for {skill}, containerization isolates dependencies across environments. When designing a production Dockerfile, why is multi-stage building critical, and what security benefits does running as a non-root user provide?",
            "sql": f"In production database systems, relational queries must maintain high throughput. If an essential analytical query runs slowly across 5 million rows, what tools would you use to diagnose the bottleneck before deciding to add an index?",
            "react": f"In enterprise frontends, component reactivity is central. When state changes frequently in child components, how do you prevent unnecessary re-renders across the component tree without compromising data freshness?",
            "kubernetes": f"In cloud-native deployments, Kubernetes manages pod lifecycles. What is the fundamental difference between a Liveness Probe and a Readiness Probe, and what happens if a readiness probe fails?",
            "pytorch": f"In production machine learning pipelines, tensors flow through backpropagation. Why do we explicitly call `optimizer.zero_grad()` before computing gradients in a training loop?",
        }
        return questions.get(
            skill_lower,
            f"Regarding production application of **{skill}**: When architecting a scalable solution in industry, what is the primary technical challenge you must address, and what trade-offs guide your design decision?"
        )
    else:
        # Remediation attempt - provide scaffolded hint
        hints = {
            "python": f"💡 **Scaffolding Hint for {skill}:** Think about Python's Global Interpreter Lock (GIL). For tasks that spend most of their time waiting for network responses (I/O-bound), how does an event loop with non-blocking sockets differ from creating heavy OS processes?",
            "docker": f"💡 **Scaffolding Hint for {skill}:** Consider how Docker caches image layers. If you compile your assets in an initial builder stage and only copy the compiled binary into an Alpine or Distroless image, what happens to the final image size and attack surface?",
            "sql": f"💡 **Scaffolding Hint for {skill}:** Consider the SQL `EXPLAIN ANALYZE` execution plan. What is the operational difference between a Sequential Scan (Seq Scan) and an Index Scan when searching for specific records?",
            "react": f"💡 **Scaffolding Hint for {skill}:** Recall React's memoization primitives (`useMemo`, `useCallback`, `React.memo`). How does reference equality of callback functions trigger or prevent reconciliation in child components?",
            "kubernetes": f"💡 **Scaffolding Hint for {skill}:** Focus on traffic routing versus container restarts. Which probe determines whether the Kubernetes Service sends incoming HTTP traffic to the pod, and which probe restarts the container when it is deadlocked?",
        }
        return hints.get(
            skill_lower,
            f"💡 **Scaffolding Hint for {skill}:** Break the problem down into fundamentals. Consider how data flows through the system and what specific mechanism prevents failure under load."
        )


# --- 3. Cognitive Profiler Node ---
def cognitive_profiler_node(state: LearnerState) -> Dict[str, Any]:
    """
    Evaluates student's technical answer against conceptual criteria,
    computes mastery score (0.0 - 1.0), and updates verified skills.
    """
    logger.info("Executing Cognitive Profiler Node...")
    history = state.get("chat_history", [])
    skill = state.get("current_skill", "")

    if not history or history[-1]["role"] != "user":
        logger.info("No student response present to evaluate.")
        return {"pending_response": False}

    student_response = history[-1]["content"]
    agent_question = ""
    for msg in reversed(history[:-1]):
        if msg["role"] == "agent":
            agent_question = msg["content"]
            break

    # 1. Score response via LLM or Rubric Simulator
    llm = get_llm()
    score = 0.0
    reasoning = ""

    if llm is not None:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            prompt = PROFILER_PROMPT.format(
                current_skill=skill,
                agent_question=agent_question,
                student_response=student_response
            )
            eval_res = llm.invoke([SystemMessage(content=prompt)])
            cleaned_text = eval_res.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned_text)
            score = float(data.get("mastery_score", 0.0))
            reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"LLM profiler evaluation parsing failed: {e}")
            score, reasoning = _evaluate_student_rubric(skill, student_response)
    else:
        score, reasoning = _evaluate_student_rubric(skill, student_response)

    score = max(0.0, min(1.0, round(score, 2)))
    logger.info(f"Cognitive Profiler assigned mastery score: {score} for {skill} ({reasoning})")

    updates: Dict[str, Any] = {
        "mastery_score": score,
        "pending_response": False,
    }

    verified = list(state.get("verified_skills", []))
    trajectory = list(state.get("active_trajectory", []))
    attempts = state.get("attempt_count", 0)

    if score >= MASTERY_THRESHOLD:
        # Mastery achieved
        if skill and skill not in verified:
            verified.append(skill)
        if trajectory and trajectory[0].lower() == skill.lower():
            trajectory = trajectory[1:]

        updates["verified_skills"] = verified
        updates["active_trajectory"] = trajectory
        updates["attempt_count"] = 0

        # Positive acknowledgment
        history_update = list(history) + [{
            "role": "agent",
            "content": f"✅ **Mastery Verified ({int(score*100)}%)**: {reasoning}\nYou have demonstrated competence in **{skill}**!"
        }]
        updates["chat_history"] = history_update
    else:
        # Remediation required
        updates["attempt_count"] = attempts + 1
        history_update = list(history) + [{
            "role": "agent",
            "content": f"🔄 **Knowledge Tracing ({int(score*100)}%)**: {reasoning}\nLet's examine this concept a little closer."
        }]
        updates["chat_history"] = history_update

    return updates


def _evaluate_student_rubric(skill: str, response: str) -> (float, str):
    """
    Intelligent conceptual evaluation rubric for offline/test mode.
    Examines reasoning depth, keywords in context, and structural coherence.
    """
    resp_clean = response.lower().strip()
    words = resp_clean.split()

    if len(words) < 3:
        return 0.1, "Response was too brief to assess technical understanding."

    # Generic negative or confusion indicators
    confusion = ["i don't know", "idk", "no idea", "not sure", "what do you mean"]
    if any(c in resp_clean for c in confusion):
        return 0.2, "Student expressed uncertainty regarding the concept."

    # Domain-specific conceptual indicator sets (must exhibit conceptual reasoning)
    skill_indicators = {
        "python": {
            "advanced": ["asyncio", "event loop", "non-blocking", "gil", "i/o bound", "coroutines", "await"],
            "moderate": ["threading", "multiprocessing", "asynchronous", "parallel"],
        },
        "numpy": {
            "advanced": ["vectorization", "broadcasting", "c-contiguous", "memory layout", "strides", "simd", "ndarray"],
            "moderate": ["array", "matrix", "vector", "dimension", "performance"],
        },
        "docker": {
            "advanced": ["multi-stage", "attack surface", "layer", "caching", "distroless", "alpine", "non-root"],
            "moderate": ["container", "image", "dependencies", "build"],
        },
        "sql": {
            "advanced": ["explain", "explain analyze", "execution plan", "seq scan", "index scan", "b-tree", "query planner"],
            "moderate": ["index", "performance", "slow", "table", "where"],
        },
        "postgresql": {
            "advanced": ["explain", "explain analyze", "execution plan", "seq scan", "index scan", "b-tree", "query planner"],
            "moderate": ["index", "performance", "slow", "table", "where"],
        },
        "react": {
            "advanced": ["usecallback", "usememo", "react.memo", "reconciliation", "referential equality", "virtual dom"],
            "moderate": ["props", "state", "render", "component"],
        },
        "kubernetes": {
            "advanced": ["readiness", "liveness", "traffic", "endpoints", "restart", "deadlock", "kubelet"],
            "moderate": ["pod", "cluster", "service", "health check"],
        },
    }

    domain = skill_indicators.get(skill.lower())
    if domain:
        adv_hits = sum(1 for kw in domain["advanced"] if kw in resp_clean)
        mod_hits = sum(1 for kw in domain["moderate"] if kw in resp_clean)

        if adv_hits >= 2 or (adv_hits >= 1 and mod_hits >= 2):
            return 0.90, "Demonstrated strong grasp of underlying mechanics, trade-offs, and practical design."
        elif adv_hits == 1 or mod_hits >= 2:
            return 0.70, "Identified valid techniques but explanation lacks detail on core execution trade-offs."
        elif mod_hits == 1:
            return 0.50, "Basic familiarity shown, but requires deeper conceptual elaboration."
        else:
            return 0.35, "Answer touches on general concepts but does not clearly address the technical question."

    # Generic fallback analysis: length, depth, reasoning terms
    reasoning_terms = [
        "because", "therefore", "in order to", "prevents", "ensures", "trade-off",
        "optimizes", "optimize", "optimizing", "isolates", "reduces", "reduce", "avoids", "avoid"
    ]
    reasoning_count = sum(1 for term in reasoning_terms if term in resp_clean)

    if len(words) >= 12 and reasoning_count >= 1:
        return 0.85, "Clear problem-solving approach and reasoned justification."
    elif len(words) >= 8:
        return 0.65, "Plausible response, but deeper explanation of mechanisms is warranted."
    else:
        return 0.40, "Response lacks sufficient technical explanation."
