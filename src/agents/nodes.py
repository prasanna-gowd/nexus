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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MASTERY_THRESHOLD = float(os.getenv("MASTERY_THRESHOLD", "0.8"))


def get_llm():
    """Initializes LLM: tries Gemini first, then OpenAI, then returns None for local fallback."""
    if GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
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
        "question_index": 0,
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
    q_idx = state.get("question_index", 0)
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
            messages = [
                SystemMessage(content=prompt),
                SystemMessage(content=f"IMPORTANT: The student is working on skill '{skill}' (Question #{q_idx + 1}, Attempt #{attempt + 1}). Do not repeat earlier questions. Ask a fresh, non-trivial Socratic inquiry focused on real architectural trade-offs, concurrency, debugging, or optimization.")
            ]
            for msg in state.get("chat_history", []):
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            if attempt > 0 and prev_score < MASTERY_THRESHOLD:
                messages.append(HumanMessage(content=f"[SYSTEM NOTE: Student scored {prev_score:.2f}. Provide a supportive scaffolding hint before the next guiding question.]"))

            response = llm.invoke(messages)
            agent_text = response.content.strip()
        except Exception as e:
            logger.error(f"OpenAI invocation failed: {e}")

    if not agent_text:
        # Intelligent Local Pedagogical Simulator with Dynamic Question Banks
        agent_text = _simulate_socratic_response(skill, attempt, prev_score, context_str, q_idx)

    history = list(state.get("chat_history", [])) + [{"role": "agent", "content": agent_text}]

    return {
        "chat_history": history,
        "pending_response": True, # Awaiting human response
    }


# --- Dynamic Progressive Question Banks for Technical Skills ---
SKILL_QUESTION_BANKS = {
    "python": [
        "In backend and data pipelines, Python is widely adopted. Imagine your service needs to process 10,000 incoming network requests concurrently without blocking. Between multiprocessing, threading, and `asyncio`, which architectural approach would you select for I/O-bound tasks, and why?",
        "When designing high-throughput Python microservices, memory consumption can balloon due to object lifecycle patterns. How does Python's memory management (reference counting plus cyclic Garbage Collection) operate, and how would you diagnose memory growth under sustained production load?",
        "In enterprise Python applications, robust error handling and resource cleanup are paramount. Why is utilizing custom context managers (`with` statement using `__enter__` and `__exit__`) preferable to manual try/finally blocks for managing shared resources like database connection pools?",
        "In distributed Python architectures, CPU-bound processing struggles under Python's Global Interpreter Lock (GIL). When would you choose background workers (like Celery with Redis) versus in-process C-extensions (like NumPy/Cython) to bypass this limitation?"
    ],
    "docker": [
        "Looking at production hiring requirements for Docker, containerization isolates dependencies across environments. When designing a production Dockerfile, why is multi-stage building critical, and what security benefits does running as a non-root user provide?",
        "In microservice CI/CD pipelines, Docker build times directly impact deployment velocity. How does Docker's layer caching mechanism operate, and how should you order instructions (`COPY`, `RUN pip install`, `COPY .`) to maximize cache hits across frequent code changes?",
        "During a containerized service crash, inspecting container state without disrupting data is essential. What is the fundamental operational difference between `docker stop` (SIGTERM) and `docker kill` (SIGKILL), and how do you ensure graceful connection draining?",
        "In multi-container production topologies, how do Docker bridge networks and container DNS resolution facilitate secure service-to-service communication without exposing internal ports to the host machine?"
    ],
    "sql": [
        "In production database systems, relational queries must maintain high throughput. If an essential analytical query runs slowly across 5 million rows, what tools would you use to diagnose the bottleneck before deciding to add an index?",
        "When handling concurrent financial or e-commerce transactions, race conditions can cause data anomalies. What are the four ACID isolation levels, and what specific anomaly does 'Repeatable Read' or 'Serializable' prevent?",
        "When table sizes grow into tens of millions of records, standard B-tree indexes may consume substantial memory. When would you use a Partial Index, Compound Index, or Table Partitioning, and what trade-offs exist for write throughput?",
        "In database design, normalization reduces redundancy but can cause heavy JOIN operations. Under what specific high-scale read patterns would you deliberately choose denormalization or materialized views?"
    ],
    "postgresql": [
        "In production PostgreSQL architectures, WAL (Write-Ahead Logging) ensures durability. How does WAL enable point-in-time recovery (PITR) and streaming replication across primary and replica nodes?",
        "When executing high-dimensional similarity searches using pgvector, what are the architectural trade-offs between an IVFFlat index and an HNSW index regarding build time, memory usage, and recall accuracy?",
        "In PostgreSQL connection management, why does opening hundreds of direct database connections degrade performance, and how does connection pooling (such as PgBouncer) protect server CPU and memory?",
        "When tuning autovacuum in PostgreSQL, why is vacuuming essential for MVCC (Multi-Version Concurrency Control), and what consequences arise if dead tuples are not reclaimed promptly?"
    ],
    "react": [
        "In enterprise frontends, component reactivity is central. When state changes frequently in child components, how do you prevent unnecessary re-renders across the component tree without compromising data freshness?",
        "When orchestrating asynchronous data fetching and side effects in React, why should data fetching ideally be managed via caching libraries (like React Query / SWR) rather than raw `useEffect`, and what race conditions does this avoid?",
        "In large-scale React architectures, state management can become fragmented. When would you choose React Context vs. an external store (Zustand/Redux), and how does Context affect component rendering re-evaluation?",
        "Explain React's Fiber reconciliation architecture. What was the limitation of the legacy Stack reconciler, and how does Fiber enable interruptible, prioritized rendering for responsive user interfaces?"
    ],
    "kubernetes": [
        "In cloud-native deployments, Kubernetes manages pod lifecycles. What is the fundamental difference between a Liveness Probe and a Readiness Probe, and what happens if a readiness probe fails?",
        "When deploying a stateful service (like Kafka or PostgreSQL) on Kubernetes, why is a StatefulSet required instead of a standard Deployment, and how does persistent volume binding differ?",
        "In Kubernetes cluster security, how do NetworkPolicies and RBAC (Role-Based Access Control) enforce least-privilege communication between microservices across different namespaces?",
        "When configuring horizontal pod autoscaling (HPA), what metrics (CPU, memory, custom Prometheus metrics) provide the most reliable scaling triggers, and how do you prevent thrashing (rapid scale up/down oscillations)?"
    ],
    "machine learning": [
        "In production machine learning pipelines, why is model evaluation on cross-validation data critical, and how do you distinguish between high variance (overfitting) and high bias (underfitting)?",
        "When deploying ML models to production inference services, what is Data Drift and Concept Drift, and what statistical tests or monitoring metrics would you use to detect when a model needs retraining?",
        "When dealing with severe class imbalance in classification datasets, why is raw accuracy a deceptive metric, and what evaluation metrics (ROC-AUC, PR-AUC, F1-Score) and sampling techniques would you employ?",
        "In distributed ML training, what is the fundamental difference between Data Parallelism (DDP) and Model Parallelism, and what network communication bottlenecks arise during gradient synchronization?"
    ],
    "pytorch": [
        "In production machine learning pipelines, tensors flow through backpropagation. Why do we explicitly call `optimizer.zero_grad()` before computing gradients in a training loop?",
        "When training large deep learning models, GPU memory saturation is a common hurdle. How does Mixed Precision Training (`torch.cuda.amp.autocast`) and Gradient Accumulation allow training larger batch sizes on constrained hardware?",
        "When preparing PyTorch models for low-latency production serving, why is converting from standard eager Python execution to TorchScript or ONNX beneficial?",
        "In PyTorch custom Dataset and DataLoader implementations, why is `num_workers > 0` and `pin_memory=True` important for preventing CPU data loading bottlenecks from starving GPU compute?"
    ],
    "aws": [
        "When designing a highly available, fault-tolerant web application on AWS, how do you architect across multiple Availability Zones (AZs) using VPCs, subnets, and Application Load Balancers?",
        "In serverless cloud architectures using AWS Lambda and API Gateway, what causes cold starts, and what architectural strategies (provisioned concurrency, lightweight runtimes, connection reuse) minimize invocation latency?",
        "In AWS cloud security, what is the principle of least privilege in IAM (Identity and Access Management) roles and policies, and why should IAM roles with temporary credentials be used instead of long-lived access keys?",
        "When architecting scalable data storage on AWS, what criteria guide your selection between S3, EBS, EFS, and DynamoDB for specific throughput, latency, and consistency requirements?"
    ],
    "fastapi": [
        "In modern asynchronous Python web APIs, FastAPI utilizes Python type hints and Pydantic. How does FastAPI's asynchronous event loop handle concurrent `async def` endpoints versus blocking synchronous `def` endpoints?",
        "When building enterprise FastAPI applications, how does FastAPI's Dependency Injection (`Depends`) system facilitate clean authentication, database session management, and test mocking?",
        "In high-traffic REST APIs, how do you handle background processing for long-running jobs in FastAPI using `BackgroundTasks` vs external Celery workers without delaying the HTTP client response?",
        "When implementing security in FastAPI, how do OAuth2 with Password (and Bearer JWT tokens) integrate with Pydantic request models to secure sensitive endpoints?"
    ],
    "pandas": [
        "When processing analytical datasets with Pandas, why is vectorized computation using NumPy ufuncs orders of magnitude faster than iterating with `.iterrows()`, and how does memory layout impact speed?",
        "In large data transformations, Pandas DataFrames can exhaust RAM. What memory optimization techniques (downcasting datatypes, `category` dtype for low-cardinality strings, chunksize in `read_csv`) do you apply before resorting to distributed frameworks?",
        "When performing complex multi-table joins in Pandas (`merge`), what is the difference between inner, outer, and left joins, and how do you handle key collision or Cartesian product explosions?",
        "In time series analysis using Pandas, how do you handle missing timestamps, resampling (downsampling vs upsampling), and rolling window aggregations without lookahead bias?"
    ],
    "numpy": [
        "In numerical computing, how does NumPy achieve high performance using C-contiguous arrays and SIMD vectorization compared to native Python lists?",
        "Explain NumPy array broadcasting rules. What conditions must two array dimensions satisfy to be compatible for element-wise arithmetic operations without copying memory?",
        "In high-performance linear algebra, what is the difference between an in-place operation (`arr += 1`) and creating a new array view versus a deep copy (`arr.copy()`), and how do memory strides affect caching?",
        "When indexing multidimensional NumPy arrays, what distinguishes basic slicing (which returns memory views) from advanced fancy indexing (which allocates new memory arrays)?"
    ],
    "devops": [
        "In modern CI/CD automation, what is the difference between Continuous Delivery and Continuous Deployment, and what automated gates (unit tests, integration tests, security scanners) ensure zero-defect deployments?",
        "When releasing critical services to production, how do Canary Deployments and Blue/Green Deployments differ in routing traffic and enabling instantaneous zero-downtime rollback?",
        "In Infrastructure as Code (IaC) with Terraform, what is State Drift, and how do remote state locks (like S3 + DynamoDB) prevent race conditions in team environments?",
        "In site reliability engineering (SRE), what are SLOs (Service Level Objectives), SLAs, and Error Budgets, and how does an exhausted error budget guide engineering priority between feature delivery and stability?"
    ],
    "git": [
        "In collaborative software development, what is the fundamental difference between `git merge` and `git rebase`, and in what scenarios is rebasing preferred for maintaining clean commit histories?",
        "During a critical production bug investigation, how would you use `git bisect` to locate the exact commit that introduced a regression across hundreds of historical commits?",
        "When resolving merge conflicts in Git, what are the roles of HEAD, the working directory, and the index (staging area), and how does `git checkout --ours` vs `--theirs` function during a rebase?",
        "When a bad commit has already been pushed to a protected shared branch, why is `git revert` strongly preferred over `git reset --hard` and force pushing?"
    ],
    "typescript": [
        "In modern web applications, how does TypeScript's structural type system differ from nominal type systems in languages like Java or C++?",
        "When modeling complex application state, why are Discriminated Unions and exhaustive type checking (using `never`) safer than broad `any` or `unknown` types?",
        "In TypeScript generics, how do conditional types (`T extends U ? X : Y`) and the `infer` keyword enable powerful type transformations in API client libraries?",
        "What is the difference between TypeScript `interface` and `type` alias, and when should one be favored over the other in library design?"
    ],
    "javascript": [
        "In asynchronous JavaScript, explain the Event Loop, Call Stack, Microtask Queue (Promises), and Macrotask Queue (`setTimeout`). In what order are tasks executed?",
        "In JavaScript memory management, how do closures maintain access to outer lexical scope, and how can careless closure retention cause memory leaks in long-running browser sessions?",
        "What is Prototype Inheritance in JavaScript? How does property lookup travel up the prototype chain (`__proto__`), and how did ES6 `class` syntax modernize this?",
        "In browser environments, what is Event Bubbling versus Event Capturing, and how does Event Delegation optimize memory when handling clicks across hundreds of dynamic list items?"
    ],
    "node.js": [
        "In high-concurrency Node.js architectures, how does the libuv threadpool complement the single-threaded V8 event loop for blocking I/O and cryptographic operations?",
        "When handling high-volume HTTP streams in Node.js, what is 'backpressure', and how does piping streams (`readable.pipe(writable)`) prevent buffer overflow and memory exhaustion?",
        "In production Node.js microservices, how do Worker Threads differ from the Node.js Cluster module for scaling CPU-intensive processing across multiple CPU cores?",
        "When designing resilient Node.js services, how do you handle `uncaughtException` and `unhandledRejection` events gracefully without leaving socket connections in indeterminate states?"
    ],
    "rest api": [
        "In enterprise REST API design, what is the architectural difference between idempotent HTTP methods (PUT, DELETE, GET) and non-idempotent methods (POST), and how do idempotency keys prevent duplicate payments or orders?",
        "When versioning high-scale public REST APIs, what are the trade-offs between URI path versioning (`/v1/users`), query parameters, and custom request headers (`Accept: application/vnd.company.v1+json`)?",
        "In high-traffic REST services, how do HTTP caching headers (`Cache-Control`, `ETag`, `If-None-Match`) minimize server load and decrease mobile client latency?",
        "When paginating through datasets with millions of rows in a REST endpoint, why is Keyset/Cursor-based pagination vastly superior to offset-based (`OFFSET 100000 LIMIT 50`) pagination?"
    ],
    "html/css": [
        "In modern web application architecture, how does the browser Critical Rendering Path work (DOM -> CSSOM -> Render Tree -> Layout -> Paint), and how do non-blocking CSS/script techniques optimize First Contentful Paint (FCP)?",
        "When building responsive, accessible components, how does CSS Grid compare with Flexbox for multi-dimensional layout orchestration, and why should semantic HTML5 elements be used instead of generic `<div>` tags for accessibility (ARIA)?",
        "In large-scale CSS architecture (BEM, CSS Modules, or Tailwind), how do you prevent specificity collisions and cascading side effects across micro-frontends?",
        "In high-performance web animations, why should transforms and opacity (`transform: translate3d()`, `opacity`) be preferred over layout properties (`left`, `top`, `height`) to avoid CPU-bound reflows?"
    ]
}


def _simulate_socratic_response(skill: str, attempt: int, prev_score: float, context: str, q_idx: int = 0) -> str:
    """
    Dynamic, non-fixed Socratic questions and scaffolded hints.
    Rotates through progressive technical inquiries grounded in production requirements.
    """
    skill_lower = skill.lower()
    questions = SKILL_QUESTION_BANKS.get(skill_lower)

    if attempt == 0:
        # Initial inquiry: select progressive question by index
        if questions:
            selected_q = questions[q_idx % len(questions)]
            return f"**[Socratic Inquiry — {skill} (Question #{q_idx + 1})]**\n\n{selected_q}"
        else:
            # Dynamic template variations for emerging/unlisted skills
            templates = [
                f"Regarding production application of **{skill}**: When architecting a scalable service in industry, what is the core architectural principle of {skill} that prevents system bottlenecks, and what trade-offs guide your design decision?",
                f"Let's explore error resilience and debugging in **{skill}**: In a high-traffic production outage where latency spikes by 500%, what diagnostic telemetry or debugging tools specific to {skill} would you inspect first, and how would you isolate the root cause?",
                f"In high-throughput environments utilizing **{skill}**: What are the most common performance anti-patterns engineering teams encounter with {skill}, and how do you optimize memory consumption or execution time when processing massive workloads?",
                f"Considering modern enterprise architectures requiring **{skill}**: How do you securely integrate {skill} with upstream services and downstream databases while ensuring strict contract adherence and zero-downtime deployment?"
            ]
            selected_q = templates[q_idx % len(templates)]
            return f"**[Socratic Inquiry — {skill} (Question #{q_idx + 1})]**\n\n{selected_q}"
    else:
        # Remediation attempt - provide scaffolded hint and targeted follow-up question
        hints = {
            "python": f"💡 **Scaffolding Hint for {skill}:** Think about Python's Global Interpreter Lock (GIL). For tasks that spend most of their time waiting for network responses (I/O-bound), how does an event loop with non-blocking sockets differ from creating heavy OS processes? Consider `asyncio`'s cooperative multitasking.",
            "docker": f"💡 **Scaffolding Hint for {skill}:** Consider how Docker caches image layers. If you compile your assets in an initial builder stage and only copy the compiled binary into an Alpine or Distroless image, what happens to the final image size and attack surface?",
            "sql": f"💡 **Scaffolding Hint for {skill}:** Consider the SQL `EXPLAIN ANALYZE` execution plan. What is the operational difference between a Sequential Scan (Seq Scan) and an Index Scan when searching for specific records in a large table?",
            "postgresql": f"💡 **Scaffolding Hint for {skill}:** Think about MVCC and the Write-Ahead Log. Why does PostgreSQL write changes to WAL before updating data pages on disk, and how does this protect transactional integrity?",
            "react": f"💡 **Scaffolding Hint for {skill}:** Recall React's memoization primitives (`useMemo`, `useCallback`, `React.memo`). How does reference equality of callback functions trigger or prevent reconciliation in child components?",
            "kubernetes": f"💡 **Scaffolding Hint for {skill}:** Focus on traffic routing versus container restarts. Which probe determines whether the Kubernetes Service sends incoming HTTP traffic to the pod, and which probe restarts the container when it is deadlocked?",
            "machine learning": f"💡 **Scaffolding Hint for {skill}:** Think about the bias-variance trade-off. If your training loss is very low but validation loss is high, what causes this divergence, and how does regularization (L1/L2, dropout) constrain model complexity?",
            "pytorch": f"💡 **Scaffolding Hint for {skill}:** Recall that PyTorch accumulates gradients by default. If you don't zero the gradients before calling `loss.backward()`, what happens to the parameter updates in subsequent iterations?",
            "aws": f"💡 **Scaffolding Hint for {skill}:** Consider redundancy across Availability Zones. Why does placing EC2 instances behind an Application Load Balancer across at least two AZs protect availability during a data-center outage?",
            "fastapi": f"💡 **Scaffolding Hint for {skill}:** Notice that `async def` runs directly on the asyncio event loop thread, while synchronous `def` routes run on an external thread pool. What happens if you execute a blocking `time.sleep()` inside an `async def` endpoint?",
            "pandas": f"💡 **Scaffolding Hint for {skill}:** Remember that Python loops overhead each operation in interpreter bytecode, whereas NumPy/Pandas vectorization runs in compiled C with preallocated memory blocks.",
            "numpy": f"💡 **Scaffolding Hint for {skill}:** Focus on memory layout: a C-contiguous array stores adjacent row elements sequentially in memory, enabling CPU cache-line prefetching and SIMD vector instructions.",
            "devops": f"💡 **Scaffolding Hint for {skill}:** Contrast automated staging versus automated production deployment. In Continuous Delivery, human approval gates the final production push; in Continuous Deployment, every passing pipeline automatically releases to production.",
            "git": f"💡 **Scaffolding Hint for {skill}:** Consider Git's commit graph. `git merge` creates a merge commit preserving parallel branch history, whereas `git rebase` replays your commits linearly onto the tip of the base branch.",
            "typescript": f"💡 **Scaffolding Hint for {skill}:** In a structural type system (duck typing), two types are compatible if they share the same property shapes, regardless of their declared class or interface names.",
            "javascript": f"💡 **Scaffolding Hint for {skill}:** Microtasks (Promises, `queueMicrotask`) have higher execution priority than Macrotasks (`setTimeout`, `setInterval`). Microtasks run immediately after the current call stack clears, before the browser renders.",
            "node.js": f"💡 **Scaffolding Hint for {skill}:** Remember that the libuv threadpool manages blocking I/O (like file system calls and DNS lookups) asynchronously so the single-threaded V8 event loop is never blocked.",
            "rest api": f"💡 **Scaffolding Hint for {skill}:** Focus on idempotency: calling an idempotent operation multiple times produces the identical side-effect on the server as making a single call.",
            "html/css": f"💡 **Scaffolding Hint for {skill}:** Trace the Critical Rendering Path: the browser constructs DOM and CSSOM trees into a Render Tree before computing Layout geometry and Painting pixels."
        }
        hint_text = hints.get(
            skill_lower,
            f"💡 **Scaffolding Hint for {skill}:** Break the problem down into fundamentals. Consider how data flows through the system and what specific mechanism prevents failure under load."
        )
        return f"{hint_text}\n\n**Follow-up Question:** Given this principle, how would you structure your solution in production code to address the requirement?"


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
        updates["question_index"] = 0

        # Positive acknowledgment
        history_update = list(history) + [{
            "role": "agent",
            "content": f"✅ **Mastery Verified ({int(score*100)}%)**: {reasoning}\nYou have demonstrated competence in **{skill}**!"
        }]
        updates["chat_history"] = history_update
    else:
        # Remediation required
        updates["attempt_count"] = attempts + 1
        updates["question_index"] = state.get("question_index", 0) + 1
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
        "javascript": {
            "advanced": ["event loop", "microtask", "macrotask", "closure", "prototype", "lexical scope", "delegation"],
            "moderate": ["callback", "promise", "async", "event", "scope"],
        },
        "typescript": {
            "advanced": ["structural", "discriminated union", "generics", "conditional type", "infer", "type guard"],
            "moderate": ["interface", "type", "types", "compiler", "safety"],
        },
        "git": {
            "advanced": ["rebase", "bisect", "cherry-pick", "fast-forward", "merge commit", "reflog", "staging area"],
            "moderate": ["commit", "branch", "merge", "push", "pull", "checkout"],
        },
        "node.js": {
            "advanced": ["libuv", "threadpool", "backpressure", "stream", "worker thread", "cluster", "event loop"],
            "moderate": ["async", "npm", "server", "express", "process"],
        },
        "rest api": {
            "advanced": ["idempotent", "idempotency", "cursor", "keyset", "etag", "cache-control", "status code"],
            "moderate": ["endpoint", "http", "get", "post", "header", "json"],
        },
        "html/css": {
            "advanced": ["critical rendering path", "cssom", "reflow", "repaint", "grid", "flexbox", "semantic"],
            "moderate": ["dom", "layout", "responsive", "style", "selector"],
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
