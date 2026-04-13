package com.localandro.gemma4e2b.memory

/**
 * A single fact stored in the agent's long-term memory.
 *
 * @param id         Auto-generated primary key.
 * @param key        Short topic tag (e.g. "user_name", "device_model").
 * @param value      The factual content.
 * @param source     Origin of the fact (e.g. "user_input", "tool:device_info").
 * @param createdAt  Epoch millis when the fact was first stored.
 * @param accessedAt Epoch millis of the last time this fact was read.
 * @param accessCount Number of times this fact has been included in context.
 */
data class MemoryEntry(
    val id: Long = 0,
    val key: String,
    val value: String,
    val source: String = "agent",
    val createdAt: Long = System.currentTimeMillis(),
    val accessedAt: Long = System.currentTimeMillis(),
    val accessCount: Int = 0
)
