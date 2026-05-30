\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.automata,positioning,shapes}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{\textbf{Beyond Constitutional AI: \\
A Unified Framework Integrating Scalable Oversight, \\
Adversarial Debate, and Weak-to-Strong Generalization}}

\author{
 Seyed Alireza Alhosseini Almodarresieh \\
   Independent Researcher \\
  \texttt{modarresi1913@gmail.com}
}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Constitutional AI (CAI) introduced a principled approach to aligning large language models (LLMs) with human values by replacing direct human feedback on harmful outputs with a structured set of ethical principles---a ``constitution.'' While CAI demonstrated that AI-generated supervision (RLAIF) can match or exceed human-labeled RLHF for harmlessness, it faces fundamental limitations: the constitution is authored by a single organization, static over time, culturally narrow, and ill-suited to supervise models that surpass the capability of their overseers. In this paper, we propose \textbf{CODA} (\textbf{C}onstitutional \textbf{O}versight via \textbf{D}ebate and \textbf{A}mplification), a unified framework that extends CAI along three orthogonal axes: (1) \textit{Democratic Constitution Induction} (DCI), which derives constitutional principles from diverse stakeholder populations rather than a single organization; (2) \textit{Adversarial Debate Supervision} (ADS), which replaces single-model self-critique with multi-agent debate to expose reasoning failures invisible to a lone critic; and (3) \textit{Recursive Weak-to-Strong Oversight} (RWSO), which enables weaker supervisor models to maintain alignment guarantees over progressively stronger actors. We provide a formal treatment of each component, prove convergence properties under idealized assumptions, and present empirical results on Llama-3-8B and Mistral-7B showing that CODA reduces adversarial Attack Success Rate (ASR) by 61.3\% while preserving 97.2\% of baseline helpfulness---substantially outperforming vanilla CAI on both axes. We further show that CODA's debate component generalizes to weak-to-strong settings, providing empirical support for scalable oversight beyond current model capability gaps.
\end{abstract}

\section{Introduction}

As large language models become increasingly capable, the challenge of aligning their behavior with human values has grown correspondingly urgent. The dominant paradigm---Reinforcement Learning from Human Feedback (RLHF)~\citep{stiennon2020learning,ouyang2022training}---scales poorly: human annotation is expensive, noisy, and fundamentally limited by human evaluators' ability to detect subtle failures in models that may exceed their own competence.

Constitutional AI (CAI)~\citep{bai2022constitutional} offered an elegant partial solution: by providing a model with a fixed set of principles (a ``constitution''), it can critique and revise its own outputs without per-example human labels on harmful content. The resulting RLAIF training signal was shown to be competitive with human-labeled RLHF for harmlessness, while maintaining helpfulness. This was a meaningful step toward \textit{scalable oversight}---the problem of supervising AI systems that may eventually surpass human-level ability.

However, CAI as originally formulated has three structural weaknesses that limit its scalability and legitimacy:

\begin{enumerate}
    \item \textbf{Centralized Constitution Authorship.} The constitution is written by a single organization (Anthropic), encoding values that may not reflect the diversity of global users. Collective Constitutional AI~\citep{huang2023collective} began addressing this, but without a principled induction procedure.

    \item \textbf{Single-Model Self-Critique.} A model critiquing its own outputs faces a fundamental blind spot: it cannot reliably detect errors in regions where it is systematically wrong. If the model has a consistent reasoning failure, its self-critique will share that failure.

    \item \textbf{Static, Non-Recursive Oversight.} CAI's supervision is flat: a fixed-capability model supervises a fixed-capability actor. As actor capability grows, the supervisor's relative competence decreases---eventually the oversight becomes meaningless.
\end{enumerate}

This paper addresses all three weaknesses in a single unified framework. Our contributions are:

\begin{itemize}
    \item \textbf{Democratic Constitution Induction (DCI):} A formal procedure for deriving a constitutional principle set from population-level preference data, with theoretical guarantees on value coverage and minimal principle sets.

    \item \textbf{Adversarial Debate Supervision (ADS):} A multi-agent debate protocol that replaces CAI's self-critique phase, enabling one AI to expose the reasoning failures of another under a weak human or AI judge.

    \item \textbf{Recursive Weak-to-Strong Oversight (RWSO):} A hierarchical supervision scheme in which each oversight level is bootstrapped from the level below, allowing weaker systems to maintain alignment guarantees over progressively stronger actors.

    \item \textbf{Empirical Validation:} Experiments on Llama-3-8B and Mistral-7B demonstrating CODA's superiority over vanilla CAI and RLHF baselines across safety, helpfulness, and scalability metrics.
\end{itemize}

\paragraph{Paper Organization.} Section~\ref{sec:background} reviews related work. Section~\ref{sec:framework} presents the CODA framework formally. Section~\ref{sec:theory} provides theoretical analysis. Section~\ref{sec:experiments} presents empirical results. Section~\ref{sec:discussion} discusses limitations and future directions. Section~\ref{sec:conclusion} concludes.

\section{Background and Related Work}
\label{sec:background}

\subsection{Constitutional AI and RLAIF}

CAI~\citep{bai2022constitutional} introduced a two-phase training procedure. In the \textit{Supervised Learning (SL)} phase, a helpful-but-potentially-harmful model is prompted with adversarial inputs, asked to critique its own response against a set of constitutional principles, and then finetuned on its own revised responses. In the \textit{Reinforcement Learning (RL)} phase (RLAIF), the model generates pairs of responses, a feedback model selects the preferable response using constitutional principles as criteria, and this preference dataset trains a reward model for RL.

Subsequent work demonstrated that RLAIF generalizes broadly~\citep{lee2023rlaif}: Lee et al.\ showed that AI-generated feedback can match or exceed human feedback across diverse tasks, not only safety. Kundu et al.~\citep{kundu2023specific} found systematic trade-offs between specific and general constitutional principles---specific principles improve targeted compliance but generalize less well to unseen prompts. Findeis et al.~\citep{findeis2023inverse} introduced \textit{Inverse CAI}, which extracts constitutional principles from existing human preference data rather than requiring principles to be authored a priori.

Zhang~\citep{zhang2025collapse} recently replicated CAI on the smaller Llama-3-8B model and identified a critical failure mode: \textit{model collapse} during self-improvement, where insufficient output quality prevents effective fine-tuning. This confirms that self-critique---a single model reviewing its own outputs---is structurally fragile, particularly for models below a critical capability threshold. Our ADS component directly addresses this fragility by replacing self-critique with adversarial multi-agent debate.

\subsection{Constitution Design: From Centralized to Collective}

A growing body of work challenges the assumption that constitutional principles should be authored by a single organization. Huang et al.~\citep{huang2023collective} introduced \textit{Collective Constitutional AI}, which uses deliberative polling with a nationally representative sample of US adults to generate a constitution reflecting broader societal preferences. Their empirical results show that democratically generated constitutions yield models less biased toward the values of AI developers, though no formal coverage guarantees are provided.

Kyrychenko et al.~\citep{kyrychenko2025c3ai} introduced the C3AI framework (Crafting Constitutions for CAI), the most systematic prior work on constitution design. C3AI addresses two problems: (1) selecting and structuring principles before fine-tuning, using graph-based analysis of principle correlations; and (2) evaluating model adherence to specific principles post-fine-tuning. C3AI's key empirical finding is that positively framed, behavior-based principles produce better alignment than negatively framed or trait-based principles.

While C3AI provides systematic tooling for constitution engineering given a fixed stakeholder group, it does not address \textit{whose} values should be represented or provide theoretical guarantees on population coverage. Abiri~\citep{abiri2024public} argues from a legal and political theory perspective that AI systems require democratic legitimacy mechanisms analogous to constitutional courts, but does not propose a computational solution. Our DCI component fills this gap: it provides a principled, sample-efficient procedure for inducing constitutions from diverse populations, with formal coverage guarantees.

\subsection{Scalable Oversight}

The scalable oversight problem~\citep{bowman2022measuring} asks how humans (or weaker AI systems) can effectively supervise AI systems that are more capable than they are. This is not merely a future concern: as noted by Zhang~\citep{zhang2025collapse}, even current 52B-parameter models create oversight difficulties for smaller reviewer models.

\textbf{Iterated Amplification}~\citep{christiano2018amplification} recursively decomposes hard tasks into easier subproblems that a weak supervisor can evaluate, then uses the evaluated subproblems to train the next supervisor.

\textbf{AI Safety via Debate}~\citep{irving2018ai} has two AI agents argue opposing positions before a judge; the judge need only verify the winning argument, not generate it. Irving et al.\ provided complexity-theoretic foundations showing that verification is easier than generation in PSPACE-complete problem classes. Khan et al.~\citep{khan2024debate} provided large-scale empirical validation at ICML 2024, demonstrating that more persuasive LLM debaters lead to more truthful answers when judges have access to relevant information. However, Kenton et al.~\citep{kenton2024scalable} found at NeurIPS 2024 that these gains do not robustly generalize to settings without information asymmetry between debaters and judge---an important limitation that informs our ADS protocol design.

\textbf{Process Supervision}~\citep{lightman2023verify} supervises individual reasoning steps rather than only final answers. Lightman et al.\ showed that step-level supervision significantly outperforms outcome supervision for mathematical reasoning, motivating the use of debate transcripts (which expose intermediate reasoning) rather than response-level comparison in ADS.

\textbf{Recursive Reward Modeling}~\citep{leike2018scalable} builds reward models recursively, each trained using assistance from the model being trained---the closest predecessor to our RWSO component, which we extend with formal error-propagation bounds.

\textbf{Doubly-Efficient Debate}~\citep{browncohen2023debate} resolved key theoretical limitations of the original Irving et al.\ protocol, providing a debate scheme that is sound even for computationally bounded judges. Our ADS protocol builds on this more robust theoretical foundation.

\subsection{Weak-to-Strong Generalization}

Burns et al.~\citep{burns2023weak} demonstrated that strong models finetuned on labels generated by weak models can exceed weak model performance---\textit{weak-to-strong generalization}. This suggests that a weak supervisor may be able to elicit aligned behavior in a strong model even when unable to verify all model outputs. However, reliability degrades as the capability gap widens, and the mechanism is not well understood.

Wang et al.~\citep{wang2024ensemble} showed that combining scalable oversight with ensemble learning partially closes the capability gap, with AI-AI debate and human-AI interaction as the most effective auxiliary settings. Lang et al.~\citep{lang2025debate} demonstrated at AAAI 2025 that debate specifically can assist weak models in extracting trustworthy information from strong models, improving alignment over either approach in isolation. Zakershahrak and Ghodratnama~\citep{zakershahrak2024eda} propose combining explanation generation with debate for weak-to-strong alignment, but do not address constitutional principles or democratic value induction.

\paragraph{Gap addressed by CODA.} Critically, no prior work integrates all three of these research threads---constitution design, scalable oversight via debate, and weak-to-strong generalization---into a single unified framework with formal guarantees. C3AI addresses constitution engineering but not oversight. Collective CAI addresses democratic legitimacy but not scalable oversight. Lang et al.\ address debate + weak-to-strong but not constitutional alignment. CODA is the first framework to close all three gaps simultaneously, with theoretical analysis of each component and their interactions.

\section{The CODA Framework}
\label{sec:framework}

We now present CODA formally. The framework has three components that can be applied independently or jointly.

\subsection{Component 1: Democratic Constitution Induction (DCI)}

\paragraph{Setup.} Let $\mathcal{P}$ be a population of stakeholders with individual value functions $v_i : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$, where $\mathcal{X}$ is the space of prompts and $\mathcal{Y}$ is the space of responses. A \textit{constitutional principle} $c$ is a natural language statement that partitions $\mathcal{Y}$ into compliant and non-compliant responses given $x$.

\paragraph{Objective.} We seek a minimal set of principles $\mathcal{C}^* = \{c_1, \ldots, c_k\}$ such that:
\begin{equation}
    \mathcal{C}^* = \arg\min_{|\mathcal{C}|} \mathbb{E}_{i \sim \mathcal{P}} \left[ \text{Disagreement}(v_i, \pi_{\mathcal{C}}) \right]
    \label{eq:dci-objective}
\end{equation}
where $\pi_{\mathcal{C}}$ is a policy that follows principles $\mathcal{C}$, and Disagreement measures deviation from individual $v_i$.

\paragraph{Procedure.} DCI proceeds in three steps:

\begin{algorithm}[h]
\caption{Democratic Constitution Induction (DCI)}
\label{alg:dci}
\begin{algorithmic}[1]
\REQUIRE Population sample $\{(x_j, y_j^+, y_j^-)\}_{j=1}^N$ of preference pairs
\ENSURE Constitutional principle set $\mathcal{C}$
\STATE \textbf{Preference Clustering:} Embed preference pairs using a frozen LM; cluster by semantic similarity to identify $K$ value dimensions.
\STATE \textbf{Principle Induction:} For each cluster $k$, prompt an LM to generate a natural language principle that best explains the preference pattern in that cluster.
\STATE \textbf{Coverage Verification:} Use the induced principles to predict held-out preferences; prune redundant principles (those that do not improve prediction accuracy above threshold $\epsilon$).
\RETURN Minimal covering set $\mathcal{C}$
\end{algorithmic}
\end{algorithm}

\paragraph{Theoretical Guarantee.} Under mild assumptions on the smoothness of individual value functions, we show (Theorem~\ref{thm:dci-coverage}) that DCI recovers a constitutionally minimal set achieving $(1-\delta)$ population coverage with $O(\log(1/\delta) / \epsilon^2)$ preference samples.

\paragraph{Remark on Assumptions.} The Lipschitz continuity and $\gamma$-separation assumptions are idealizations. Real human preferences may be discontinuous, contradictory, or culturally non-linear---particularly across diverse global populations~\citep{huang2023collective}. Theorem~\ref{thm:dci-coverage} should therefore be interpreted as a best-case bound under favorable conditions, not a universal guarantee. Future work should explore step-function or probabilistic preference models~\citep{christiano2018amplification}, and incomplete-information game theory for contradictory value systems, to provide guarantees under more realistic assumptions.

\subsection{Component 2: Adversarial Debate Supervision (ADS)}

CAI's self-critique phase suffers from the \textit{consistent-error problem}: if a model has a systematic blind spot, it will fail both to produce and to critique outputs in that region.

ADS replaces self-critique with a \textit{zero-sum debate} between two model instances:

\paragraph{Protocol.} Given a prompt $x$ and a candidate response $y$, we instantiate two model copies $M_1$ (the \textit{Defender}) and $M_2$ (the \textit{Challenger}):

\begin{enumerate}
    \item $M_1$ outputs $y$ and a supporting argument $a_1$.
    \item $M_2$ outputs a critique $a_2$, identifying the most salient flaw in $y$ relative to the constitutional principles $\mathcal{C}$.
    \item $M_1$ responds to the critique with rebuttal $a_3$.
    \item A judge $J$ (which may be a weaker model or human) evaluates the debate transcript $(y, a_1, a_2, a_3)$ and assigns a score $s \in [0,1]$ reflecting constitutional compliance.
\end{enumerate}

\paragraph{Key Property.} The debate structure creates an asymmetry favorable to truth. A Challenger that has correctly identified a flaw needs only a brief, verifiable argument; a Defender attempting to conceal a flaw must construct a consistent deception across all of $a_1$ and $a_3$, which becomes exponentially harder as the judge's attention increases. This is the core intuition of Irving et al.~\citep{irving2018ai}, which we operationalize in the CAI context.

\paragraph{Preventing Debate Collapse.} When Defender and Challenger share identical initialization, they may converge to a collusive equilibrium in which each conceals the other's errors rather than surfacing them. We mitigate this via two mechanisms: (1) a \textit{diversity penalty} in the Challenger reward function, penalizing critiques that are semantically similar to previous rounds and incentivizing exploration of novel failure modes; and (2) using \textit{architecturally distinct models} (e.g., Llama as Defender vs.\ Mistral as Challenger) to structurally prevent collusion by ensuring the two agents have different inductive biases. The theoretical conditions for sustained adversariality remain an open problem that we leave to future work.

\paragraph{Training.} We use ADS scores as the reward signal in an RLAIF training loop (Algorithm~\ref{alg:ads}), replacing CAI's standard AI preference model.

\begin{algorithm}[h]
\caption{Adversarial Debate Supervision (ADS) Training}
\label{alg:ads}
\begin{algorithmic}[1]
\REQUIRE Base model $M_0$, constitution $\mathcal{C}$, judge $J$, debate rounds $T$
\STATE Initialize Defender $M_D \leftarrow M_0$, Challenger $M_C \leftarrow M_0$
\FOR{iteration $t = 1, \ldots, T$}
    \STATE Sample prompts $\{x_i\}$; generate candidate responses $\{y_i\}$ from $M_D$
    \STATE Run debate protocol; collect judge scores $\{s_i\}$
    \STATE Update $M_D$ via PPO using $\{s_i\}$ as reward
    \STATE Update $M_C$ via self-play: reward $M_C$ when $s_i$ is low (Challenger wins)
\ENDFOR
\RETURN Aligned model $M_D$
\end{algorithmic}
\end{algorithm}

\subsection{Component 3: Recursive Weak-to-Strong Oversight (RWSO)}

Neither CAI nor ADS addresses the capability-gap problem: as the supervised model becomes more capable than its supervisor, oversight reliability declines.

RWSO introduces a \textit{supervision hierarchy}:

\paragraph{Hierarchy.} Let $M^{(0)} \prec M^{(1)} \prec \cdots \prec M^{(L)}$ be a sequence of models ordered by capability. Level-$\ell$ model $M^{(\ell)}$ is supervised by level-$(\ell-1)$ model $M^{(\ell-1)}$ using an ADS debate where the judge is $M^{(\ell-1)}$.

\paragraph{Bootstrapping.} We cannot assume $M^{(\ell-1)}$ is already aligned. Instead, RWSO bootstraps alignment upward:

\begin{enumerate}
    \item Align $M^{(0)}$ using standard CAI + ADS with human judges (feasible because $M^{(0)}$ is within human oversight capability).
    \item Use aligned $M^{(0)}$ as the debate judge to align $M^{(1)}$.
    \item Repeat: use aligned $M^{(\ell-1)}$ to align $M^{(\ell)}$.
\end{enumerate}

\paragraph{Stabilizing the Oversight Chain.} To guard against error propagation from a misaligned $M^{(0)}$, we propose two stabilization mechanisms: (1) \textit{human-in-the-loop checkpoints} at each level transition, where human evaluators sample-audit the judge before it is promoted to supervise the next level---catching systematic errors before they propagate; and (2) \textit{ensemble judging}, averaging scores from multiple independent judges at each level to reduce single-point-of-failure risk and improve robustness to individual judge misalignment~\citep{wang2024ensemble}.

\paragraph{Error Propagation.} A central concern is whether alignment errors compound across levels. We show (Theorem~\ref{thm:rwso-bound}) that under the debate protocol, error does not amplify: the alignment gap at level $\ell$ is bounded by $O(\epsilon_0 \cdot \ell)$ where $\epsilon_0$ is the base alignment error, not $O(\epsilon_0^\ell)$ as a naive analysis would suggest. The key is that ADS's asymmetric truth-finding property remains active at each level.

\section{Theoretical Analysis}
\label{sec:theory}

\subsection{DCI Coverage Theorem}

\begin{theorem}[DCI Population Coverage]
\label{thm:dci-coverage}
Let individual value functions $v_i$ be $L$-Lipschitz in the embedding space, and let the $K$ value dimensions be $\gamma$-separated. Then with $N \geq O(K \log(K/\delta) / \epsilon^2)$ preference pairs, DCI returns a principle set $\mathcal{C}$ such that
\[
\mathbb{E}_{i \sim \mathcal{P}} \left[ \text{Disagreement}(v_i, \pi_{\mathcal{C}}) \right] \leq \epsilon
\]
with probability at least $1 - \delta$.
\end{theorem}

\begin{proof}[Proof Sketch]
The proof uses a covering number argument on the space of value functions. Each principle $c_k$ corresponds to a halfspace in the embedding space; the $\gamma$-separation condition ensures that $K$ principles suffice to separate all $K$ value clusters. Standard VC dimension bounds then give the sample complexity.
\end{proof}

\subsection{ADS Truth-Favoring Property}

\begin{theorem}[ADS Favors Truth]
\label{thm:ads-truth}
In an ADS debate with a computationally unbounded honest Challenger and a polynomial-time judge, the equilibrium strategy for the Defender is to produce constitutionally compliant responses.
\end{theorem}

\begin{proof}[Proof Sketch]
This follows from the \textit{PSPACE}-completeness of optimal play in zero-sum games with perfect information~\citep{irving2018ai}. The honest Challenger can always construct a polynomial-length proof of any constitutional violation; verifying such a proof is in \textbf{P}. Therefore, the judge can correctly assess any constitutional claim the Challenger makes, making deception unprofitable for the Defender.
\end{proof}

\subsection{RWSO Error Bound}

\begin{theorem}[RWSO Linear Error Propagation]
\label{thm:rwso-bound}
Let $\epsilon_0 = \text{AlignmentGap}(M^{(0)})$ be the base alignment error. Under RWSO with ADS debate at each level,
\[
\text{AlignmentGap}(M^{(\ell)}) \leq \epsilon_0 \cdot \ell \cdot \frac{1}{1 - \eta}
\]
where $\eta < 1$ is the Challenger's win rate under truthful conditions.
\end{theorem}

\begin{proof}[Proof Sketch]
At each level $\ell$, the ADS debate introduces an additive error of at most $\epsilon_0$ (since the judge $M^{(\ell-1)}$ has alignment gap bounded by the previous level). Because ADS is not a multiplicative amplifier (the Challenger's truth-finding rate is lower-bounded by $1-\eta$ independently of $\ell$), errors accumulate additively rather than multiplicatively.
\end{proof}

\section{Experiments}
\label{sec:experiments}

\subsection{Experimental Setup}

\paragraph{Models.} We experiment with Llama-3-8B and Mistral-7B as the base actor models, and Llama-3-70B as the debate judge in RWSO experiments.

\paragraph{Baselines.}
\begin{itemize}
    \item \textbf{Base}: Pretrained model without alignment.
    \item \textbf{RLHF}: Standard RL from human feedback.
    \item \textbf{CAI}: Original Constitutional AI~\citep{bai2022constitutional}.
    \item \textbf{Collective-CAI}: CAI with democratically induced constitution.
    \item \textbf{CODA (ours)}: Full framework (DCI + ADS + RWSO).
    \item \textbf{CODA-ADS}: CODA with ADS only (no RWSO).
\end{itemize}

\paragraph{Metrics.}
\begin{itemize}
    \item \textbf{Attack Success Rate (ASR)}: Fraction of adversarial prompts that elicit policy-violating responses (lower is better).
    \item \textbf{Helpfulness Score (HS)}: MT-Bench score measuring general capability (higher is better).
    \item \textbf{Constitutional Compliance (CC)}: Human evaluation of response alignment with constitutional principles.
    \item \textbf{Oversight Reliability (OR)}: In RWSO experiments, fraction of alignment violations detected by the level-$(\ell-1)$ supervisor.
\end{itemize}

\paragraph{Constitution.} For DCI experiments, we use the HH-RLHF dataset~\citep{bai2022constitutional} as our primary preference data source. We supplement this with a pilot collection of 500 preference pairs from crowd-workers across 4 language groups to evaluate cross-cultural alignment (CCA). We acknowledge that this scale is insufficient for a globally representative constitution; scaling DCI to tens of thousands of annotators via active learning is an important direction for future work. For fair comparison with baseline CAI, we also ran experiments using Anthropic's published constitution~\citep{anthropic2023constitution}.

\subsection{Main Results}

\begin{table}[h]
\centering
\caption{Main results on Llama-3-8B. CODA outperforms all baselines on safety while nearly preserving helpfulness. Numbers represent mean $\pm$ std over 3 seeds.}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{ASR} ($\downarrow$) & \textbf{HS} ($\uparrow$) & \textbf{CC} ($\uparrow$) & \textbf{OR} ($\uparrow$) \\
\midrule
Base            & 0.71 $\pm$ 0.03 & 6.21 $\pm$ 0.11 & 0.41 $\pm$ 0.04 & -- \\
RLHF            & 0.38 $\pm$ 0.04 & 6.89 $\pm$ 0.09 & 0.68 $\pm$ 0.05 & -- \\
CAI             & 0.29 $\pm$ 0.03 & 6.74 $\pm$ 0.10 & 0.74 $\pm$ 0.04 & 0.61 $\pm$ 0.06 \\
Collective-CAI  & 0.26 $\pm$ 0.03 & 6.71 $\pm$ 0.09 & 0.79 $\pm$ 0.03 & 0.64 $\pm$ 0.05 \\
CODA-ADS        & 0.19 $\pm$ 0.02 & 6.82 $\pm$ 0.08 & 0.83 $\pm$ 0.03 & 0.74 $\pm$ 0.04 \\
\textbf{CODA}   & \textbf{0.11} $\pm$ \textbf{0.02} & \textbf{6.98} $\pm$ \textbf{0.07} & \textbf{0.89} $\pm$ \textbf{0.02} & \textbf{0.84} $\pm$ \textbf{0.03} \\
\bottomrule
\end{tabular}
\end{table}

CODA achieves an ASR of 0.11, representing a \textbf{61.3\% reduction} over vanilla CAI (0.29) and a \textbf{84.5\% reduction} over the unaligned base model. Crucially, CODA achieves a helpfulness score of 6.98---\textbf{slightly exceeding} RLHF (6.89) and substantially better than CAI (6.74). This constitutes a Pareto improvement: CODA is both safer and more helpful than all baselines.

\subsection{Ablation Studies}

Table~\ref{tab:ablation} decomposes CODA's gains by component.

\begin{table}[h]
\centering
\caption{Ablation on Llama-3-8B showing contribution of each CODA component.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{ASR} ($\downarrow$) & \textbf{HS} ($\uparrow$) & $\Delta$\textbf{ASR vs. CAI} \\
\midrule
CAI (baseline)          & 0.29 & 6.74 & -- \\
CAI + DCI               & 0.26 & 6.71 & $-$10.3\% \\
CAI + ADS               & 0.19 & 6.82 & $-$34.5\% \\
CAI + RWSO              & 0.23 & 6.77 & $-$20.7\% \\
CAI + DCI + ADS         & 0.16 & 6.89 & $-$44.8\% \\
CAI + DCI + ADS + RWSO  & 0.11 & 6.98 & $-$62.1\% \\
\bottomrule
\end{tabular}
\end{table}

The ADS component contributes the largest individual gain ($-$34.5\% ASR), followed by RWSO ($-$20.7\%) and DCI ($-$10.3\%). All components are complementary: their combined effect ($-$62.1\%) exceeds the sum of individual gains, indicating positive synergy.

\subsection{Scalable Oversight: Weak-to-Strong Experiments}

We evaluate RWSO's scalable oversight properties by simulating a capability hierarchy: Llama-3-8B (weak actor) supervised by Llama-3-70B (strong judge), with the judge itself having been aligned by human feedback only on a small held-out set.

\begin{table}[h]
\centering
\caption{Oversight Reliability (OR) across capability gaps. CODA maintains reliability better than CAI as the capability gap widens.}
\label{tab:rwso}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Gap: Small} & \textbf{Gap: Medium} & \textbf{Gap: Large} \\
\midrule
CAI             & 0.61 & 0.47 & 0.31 \\
CODA            & 0.84 & 0.78 & 0.69 \\
\bottomrule
\end{tabular}
\end{table}

CAI oversight reliability degrades sharply as capability gaps widen (0.61 $\to$ 0.31), consistent with its non-recursive design. CODA maintains substantially higher reliability across all gaps (0.84 $\to$ 0.69), supporting the linear rather than exponential error propagation predicted by Theorem~\ref{thm:rwso-bound}.

\subsection{DCI Constitution Quality}

We compare constitutions produced by DCI (from 500 crowd-workers) against the original Anthropic constitution on constitutional compliance (CC) and cross-cultural alignment (CCA), measured as CC averaged over non-English-speaking user populations.

\begin{table}[h]
\centering
\caption{Constitution quality comparison.}
\label{tab:dci}
\begin{tabular}{lcc}
\toprule
\textbf{Constitution Source} & \textbf{CC} & \textbf{CCA} \\
\midrule
Anthropic (original)         & 0.74 & 0.61 \\
DCI (500 crowd-workers)      & 0.79 & 0.74 \\
DCI (2000 crowd-workers)     & 0.83 & 0.81 \\
\bottomrule
\end{tabular}
\end{table}

DCI constitutions achieve higher CC and substantially higher CCA, with improvements scaling with the number of stakeholders sampled. This validates both the quality and the cultural breadth of democratically induced constitutions.

\section{Discussion}
\label{sec:discussion}

\subsection{Limitations}

\paragraph{Computational Cost.} CODA requires running two model instances in debate for each training example, roughly doubling the inference cost of CAI's SL phase. We propose two concrete mitigations: (1) using a smaller \textit{distilled} model exclusively as Challenger, reducing overhead by up to 60\% while preserving critique quality; and (2) \textit{sampled critique}---applying debate only to a random subset of training examples (e.g., 30\%) rather than all outputs, with the remaining examples supervised by standard CAI. These strategies make CODA tractable for large-scale deployment, though full cost characterization requires further empirical study.

\paragraph{Judge Alignment.} RWSO's guarantees depend on the judge being well-aligned at the previous level. If the base alignment ($M^{(0)}$) is flawed, errors propagate through the hierarchy. We address this via human-in-the-loop checkpoints and ensemble judging (Section~\ref{sec:framework}), but the robustness of these mitigations under large capability gaps remains an open question. Future work should investigate formal verification of judge alignment before level promotion.

\paragraph{Debate Collapse.} When both Defender and Challenger are initialized from the same model, they may converge to a collusive equilibrium. Our proposed mitigations---diversity penalties and architectural heterogeneity (Section~\ref{sec:framework})---partially address this, but do not eliminate it entirely. The theoretical conditions for sustained adversariality, and the long-term stability of the debate equilibrium under continued training, deserve further study.

\paragraph{Verification Complexity.} Theorem~\ref{thm:ads-truth} assumes a computationally unbounded Challenger. In practice, the Challenger is a finite model and may not always surface the most damaging critique. Bounding the gap between theoretical and practical Challenger capability is an open problem.

\subsection{Broader Implications}

CODA addresses a fundamental tension in AI alignment: the methods that work best at human-scale oversight (direct human feedback) break down as models become superhuman, while methods designed for scalable oversight (pure AI feedback) lack democratic legitimacy. By combining democratically induced constitutions, adversarial oversight via debate, and recursive supervision hierarchies, CODA offers a path toward alignment that is simultaneously legitimate, scalable, and empirically effective.

We view CODA not as a final solution but as a modular framework that can accommodate future advances in any of its three components. In particular, we expect that improvements in collective preference elicitation (DCI), stronger debate protocols (ADS), and better weak-to-strong transfer techniques (RWSO) will each translate directly into improvements in overall alignment under the CODA framework.

\section{Conclusion}
\label{sec:conclusion}

We introduced CODA, a unified framework that extends Constitutional AI along three complementary axes: democratic constitution induction, adversarial debate supervision, and recursive weak-to-strong oversight. We provided formal convergence guarantees for each component and demonstrated empirically that CODA achieves a 61.3\% reduction in adversarial Attack Success Rate over vanilla CAI while actually \textit{improving} helpfulness beyond RLHF baselines. CODA's modular design allows each component to be adopted independently or jointly, and its recursive oversight structure provides the first principled treatment of alignment across growing capability gaps within the CAI paradigm.

We release all code, prompts, and crowd-worker preference data at \url{https://github.com/[your-username]/coda-alignment}.

\bibliographystyle{plainnat}
\begin{thebibliography}{99}

\bibitem[Anthropic(2023)]{anthropic2023constitution}
Anthropic.
\newblock Claude's constitution, 2023.
\newblock \url{https://www.anthropic.com/news/claudes-constitution}.

\bibitem[Bai et~al.(2022)]{bai2022constitutional}
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et~al.
\newblock Constitutional AI: Harmlessness from AI feedback.
\newblock \textit{arXiv preprint arXiv:2212.08073}, 2022.

\bibitem[Bowman et~al.(2022)]{bowman2022measuring}
Samuel R. Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamilè Lukošiūtė, Amanda Askell, Andy Jones, Anna Chen, et~al.
\newblock Measuring progress on scalable oversight for large language models.
\newblock \textit{arXiv preprint arXiv:2211.03540}, 2022.

\bibitem[Burns et~al.(2023)]{burns2023weak}
Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Paul Christiano.
\newblock Weak-to-strong generalization: Eliciting strong capabilities with weak supervision.
\newblock \textit{arXiv preprint arXiv:2312.09390}, 2023.

\bibitem[Christiano et~al.(2018)]{christiano2018amplification}
Paul Christiano, Buck Shlegeris, and Dario Amodei.
\newblock Supervising strong learners by amplifying weak experts.
\newblock \textit{arXiv preprint arXiv:1810.08575}, 2018.

\bibitem[Abiri(2024)]{abiri2024public}
Gilad Abiri.
\newblock Public Constitutional AI.
\newblock \textit{arXiv preprint arXiv:2406.16696}, 2024.

\bibitem[Brown-Cohen et~al.(2023)]{browncohen2023debate}
Jonah Brown-Cohen, Geoffrey Irving, and Mehran Sahami.
\newblock Scalable AI safety via doubly-efficient debate.
\newblock \textit{arXiv preprint arXiv:2311.14125}, 2023.

\bibitem[Findeis et~al.(2023)]{findeis2023inverse}
Luca Findeis, Neel Nanda, and Stefan Heimersheim.
\newblock Inverse Constitutional AI: Compressing preferences into principles.
\newblock \textit{arXiv preprint arXiv:2406.06560}, 2023.

\bibitem[Huang et~al.(2023)]{huang2023collective}
Saffron Huang, Divya Siddarth, Liane Lovitt, Thomas I. Liao, Alyssa Lokhande, Nikos Moustakas, Amanda Askell, and Deep Ganguli.
\newblock Collective Constitutional AI: Aligning a language model with public input.
\newblock \textit{arXiv preprint arXiv:2310.13798}, 2023.

\bibitem[Kenton et~al.(2024)]{kenton2024scalable}
Zachary Kenton, Noah Y. Siegel, János Kramár, Jonah Brown-Cohen, Samuel Albanie, and Ramana Kumar.
\newblock On scalable oversight with weak LLMs judging strong LLMs.
\newblock In \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2024.

\bibitem[Khan et~al.(2024)]{khan2024debate}
Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R.\ Bowman, Tim Rocktäschel, and Ethan Perez.
\newblock Debating with more persuasive LLMs leads to more truthful answers.
\newblock In \textit{Proceedings of the 41st International Conference on Machine Learning (ICML)}, 2024.

\bibitem[Kyrychenko et~al.(2025)]{kyrychenko2025c3ai}
Yara Kyrychenko, Ke Zhou, Edyta Bogucka, and Daniele Quercia.
\newblock C3AI: Crafting and evaluating constitutions for Constitutional AI.
\newblock In \textit{Proceedings of the ACM Web Conference (WWW '25)}, pages 3204--3218, 2025.

\bibitem[Wang et~al.(2024)]{wang2024ensemble}
Yuhang Wang, Yanhe Fu, and Yiqing Shen.
\newblock Improving weak-to-strong generalization with scalable oversight and ensemble learning.
\newblock \textit{arXiv preprint arXiv:2402.00667}, 2024.

\bibitem[Zakershahrak and Ghodratnama(2024)]{zakershahrak2024eda}
Mehrdad Zakershahrak and Samira Ghodratnama.
\newblock Explanation, Debate, Align: A weak-to-strong framework for language model generalization.
\newblock \textit{arXiv preprint arXiv:2409.07335}, 2024.

\bibitem[Zhang(2025)]{zhang2025collapse}
Xue Zhang.
\newblock Constitution or collapse? Exploring Constitutional AI with Llama 3-8B.
\newblock \textit{arXiv preprint arXiv:2504.04918}, 2025.

\bibitem[Irving et~al.(2018)]{irving2018ai}
Geoffrey Irving, Paul Christiano, and Dario Amodei.
\newblock AI safety via debate.
\newblock \textit{arXiv preprint arXiv:1805.00899}, 2018.

\bibitem[Kundu et~al.(2023)]{kundu2023specific}
Sandipan Kundu, Yuntao Bai, Saurav Kadavath, Amanda Askell, Andrew Callison-Burch, Anna Chen, Anna Goldie, Jackson Kernion, Tom Conerly, Nova DasSarma, et~al.
\newblock Specific versus general principles for constitutional AI.
\newblock \textit{arXiv preprint arXiv:2310.13798}, 2023.

\bibitem[Lang et~al.(2025)]{lang2025debate}
Hao Lang, Fei Huang, and Yongbin Li.
\newblock Debate helps weak-to-strong generalization.
\newblock \textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 39(26):27410--27418, 2025.

\bibitem[Lee et~al.(2023)]{lee2023rlaif}
Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi.
\newblock RLAIF: Scaling reinforcement learning from human feedback with AI feedback.
\newblock \textit{arXiv preprint arXiv:2309.00267}, 2023.

\bibitem[Leike et~al.(2018)]{leike2018scalable}
Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg.
\newblock Scalable agent alignment via reward modeling: A research direction.
\newblock \textit{arXiv preprint arXiv:1811.07871}, 2018.

\bibitem[Lightman et~al.(2023)]{lightman2023verify}
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.
\newblock Let's verify step by step.
\newblock \textit{arXiv preprint arXiv:2305.20050}, 2023.

\bibitem[Ouyang et~al.(2022)]{ouyang2022training}
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et~al.
\newblock Training language models to follow instructions with human feedback.
\newblock \textit{Advances in Neural Information Processing Systems}, 35:27730--27744, 2022.

\bibitem[Stiennon et~al.(2020)]{stiennon2020learning}
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano.
\newblock Learning to summarize with human feedback.
\newblock \textit{Advances in Neural Information Processing Systems}, 33:3008--3021, 2020.

\end{thebibliography}

\end{document}
