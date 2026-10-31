# Related work

Companion to [design.md](design.md) and [findings.md](findings.md). Each entry was checked against
the paper's own page (arXiv abstract, proceedings or lab page) on the date given. Entries are
grouped by what they offer this project.

---

## E9 · Language models that improve the planner's model (checked 2026-10-29)

E9's question: instead of planning, can a small model add *structure* to the PDDL model
(mutexes, landmarks, macros, preferences) that code then verifies? The closest work splits into
four groups.

### 1. LLMs propose search knowledge; a sound procedure filters it

- **LM-Landmarks** (Bukkapatnam and Karuturi, ICML 2026 workshop LM4Plan). An LM proposes
  candidate landmark facts from the problem's language description. Each candidate is *formally
  verified against the PDDL model* by delete relaxation before Fast Downward uses it, so a
  hallucinated landmark cannot reach the planner. On 8 IPC domains, coverage rises 7.5 points
  over FF and 3.1 over LAMA, and search nodes fall by 41% and 37%.
  This is E9's condition (D) for landmarks, on IPC domains rather than an embodied multi-robot
  domain. https://icml.cc/virtual/2026/68028
- **Classical Planning with LLM-Generated Heuristics** (Corrêa, Pereira and Seipp, NeurIPS
  2025). The LLM writes domain-specific heuristic functions in Python. They are scored on
  training tasks inside greedy best-first search, and the best is kept. They solve more unseen
  tasks than domain-independent heuristics.
  The knowledge only guides the search, so soundness never depends on the LLM. This is the
  template for condition (E). https://arxiv.org/abs/2503.18809
- **Embedding LLMs into Planning Graphs** (Zhuo, Chen and Pan, arXiv 2403.00783, 2024). The LLM
  prunes and orders candidate action sets inside Graphplan's expansion, and mutexes are still
  computed by the planner. The LLM shapes search order, not soundness.
  https://arxiv.org/abs/2403.00783
- **Goal-ordering challenges** (Hirsch, Uziel and Anaby-Tavor, NeurIPS 2024 workshop). LLMs are
  tested on whether an ordering between two goals holds (necessary, reasonable, optimal). They
  do worst on *reasonable* orderings, the kind of ordering knowledge an LM would propose in E9.
  A warning that ordering proposals need checking.
  https://mlanthology.org/neuripsw/2024/hirsch2024neuripsw-evaluating

### 2. LLMs build or refine the planning model from feedback

- **LLM-built world models** (Guan, Valmeekam, Sreedharan and Kambhampati, NeurIPS 2023). The LLM
  writes the PDDL domain, a sound planner plans, and validator or human feedback corrects the
  domain. This is the "LLM edits the model, not the plan" idea, with a human in the loop.
  https://arxiv.org/abs/2305.14909
- **Environment-interaction PDDL refinement** (Mahdavi, Aoki, Tang and Cao, NeurIPS 2024).
  Several candidate problem files are generated, and the domain is refined against an
  *Exploration Walk* score computed from environment interaction, with no human. 66% solve rate
  versus 29% for GPT-4 planning directly. Closest to verifying proposals against the simulator.
  https://arxiv.org/abs/2407.12979
- **PSALM** (Zhu, Singh, Jia and Thomason, arXiv 2406.02791, 2024). The LLM proposes action
  preconditions and effects, which are kept or dropped by executing plans and observing
  outcomes. E9 forbids new actions and effects; PSALM shows why verification against execution
  is needed when they are proposed. https://arxiv.org/abs/2406.02791

### 3. LLMs translate constraints into formal specifications

- **CaStL** (Guo, Kingston and Kavraki, ICRA 2025). Goal conditions, action orderings and action
  blocking are pulled out of natural language in stages and turned into planner constraints for
  long-horizon task and motion planning. The constraints come from the *user's* words, not from
  the scene. https://kavrakilab.org/publications/guo2025-castl.html

### 4. Inter-agent interference in classical multi-agent planning

- **A Single-Agent Approach to Multiagent Planning** (Crosby, Jonsson and Rovatsos, ECAI 2014).
  Concurrency constraints are attached to *subsets of objects*: how many agents may act on an
  object at once. A multi-agent problem with such constraints is compiled into a single-agent
  one. This is the formal model for E9's receptacle occupancy, where a receptacle is an object
  that at most one robot may work at.
  https://ebooks.iospress.nl/pdf/doi/10.3233/978-1-61499-419-0-237
- **MA-PDDL** (Kovacs, 2012) adds multi-agent concurrency to PDDL. It is cited by Crosby et al.;
  E9 does not need its full syntax.

### What is new in E9

- No checked work has a *small* model propose **inter-robot resource constraints** (occupancy,
  narrow passages) for an *embodied* multi-robot planner, with each proposal verified against
  the simulator.
- The closest cases verify landmarks formally (LM-Landmarks) or refine domains from environment
  interaction (Mahdavi et al.), both for single agents.
- E9 runs on top of E8's result: planning stays with the planner (F25–F26), and only its model
  is improved.

### Not yet checked (found in search, pending)

- *Exploiting Macro-Actions in Learning GPT-Based General Planning Policies* (2025, PlanGPT with
  macros). Authors and venue not confirmed.
- An ICAPS 2025 demo that turns natural-language constraints into PDDL3 trajectory constraints.
  The PDF did not extract.
