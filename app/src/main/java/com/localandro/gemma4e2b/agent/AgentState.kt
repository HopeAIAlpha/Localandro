package com.localandro.gemma4e2b.agent

/**
 * Finite-State-Machine states for the agentic orchestration loop.
 *
 * Flow: IDLE → PLANNING → EXECUTING → OBSERVING → REFINING → IDLE
 * Error from any active state transitions to ERROR, which can recover to IDLE.
 */
enum class AgentPhase {
    /** Waiting for user input. */
    IDLE,
    /** Model is analysing the user request and deciding which tools to invoke. */
    PLANNING,
    /** A tool call has been parsed; the tool is being executed. */
    EXECUTING,
    /** Tool execution finished; result is being fed back into context. */
    OBSERVING,
    /** Model is refining its answer using the tool result. */
    REFINING,
    /** An unrecoverable error occurred. */
    ERROR
}

/**
 * Immutable snapshot of the agent loop at a point in time.
 *
 * @param phase       Current FSM phase.
 * @param planSummary Short natural-language summary of the plan the model proposed.
 * @param activeTool  Name of the tool currently executing, if any.
 * @param iterations  Number of Plan→Execute→Observe cycles completed so far for
 *                    the current user request (capped by [ActionOrchestrator.MAX_ITERATIONS]).
 * @param errorMessage Human-readable error description, if [phase] is [AgentPhase.ERROR].
 */
data class AgentState(
    val phase: AgentPhase = AgentPhase.IDLE,
    val planSummary: String? = null,
    val activeTool: String? = null,
    val iterations: Int = 0,
    val errorMessage: String? = null
)
