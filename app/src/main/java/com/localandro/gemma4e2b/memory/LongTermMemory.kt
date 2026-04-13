package com.localandro.gemma4e2b.memory

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

/**
 * SQLite-backed long-term memory that persists facts across sessions.
 *
 * The agent uses this to remember user preferences, device characteristics,
 * and other factual knowledge that should survive app restarts and remain
 * available offline.
 *
 * Thread-safe via SQLite's built-in locking.
 */
class LongTermMemory(context: Context) : SQLiteOpenHelper(
    context, DATABASE_NAME, null, DATABASE_VERSION
) {

    companion object {
        private const val TAG = "LongTermMemory"
        private const val DATABASE_NAME = "agent_memory.db"
        private const val DATABASE_VERSION = 1
        private const val TABLE = "facts"

        private const val COL_ID = "id"
        private const val COL_KEY = "fact_key"
        private const val COL_VALUE = "fact_value"
        private const val COL_SOURCE = "source"
        private const val COL_CREATED = "created_at"
        private const val COL_ACCESSED = "accessed_at"
        private const val COL_ACCESS_COUNT = "access_count"

        /** Maximum number of facts to keep. Oldest/least-used are evicted. */
        private const val MAX_FACTS = 500
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE $TABLE (
                $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_KEY TEXT NOT NULL,
                $COL_VALUE TEXT NOT NULL,
                $COL_SOURCE TEXT NOT NULL DEFAULT 'agent',
                $COL_CREATED INTEGER NOT NULL,
                $COL_ACCESSED INTEGER NOT NULL,
                $COL_ACCESS_COUNT INTEGER NOT NULL DEFAULT 0,
                UNIQUE($COL_KEY) ON CONFLICT REPLACE
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_facts_key ON $TABLE ($COL_KEY)")
        db.execSQL("CREATE INDEX idx_facts_accessed ON $TABLE ($COL_ACCESSED)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE")
        onCreate(db)
    }

    // ── Write operations ────────────────────────────────────────────

    /**
     * Stores or updates a fact. If a fact with the same [key] exists,
     * it is replaced (UPSERT via `UNIQUE ON CONFLICT REPLACE`).
     */
    fun storeFact(key: String, value: String, source: String = "agent") {
        val now = System.currentTimeMillis()
        val cv = ContentValues().apply {
            put(COL_KEY, key)
            put(COL_VALUE, value)
            put(COL_SOURCE, source)
            put(COL_CREATED, now)
            put(COL_ACCESSED, now)
            put(COL_ACCESS_COUNT, 0)
        }
        writableDatabase.insertWithOnConflict(TABLE, null, cv, SQLiteDatabase.CONFLICT_REPLACE)
        evictIfNeeded()
        Log.d(TAG, "Stored fact: $key = $value")
    }

    /** Deletes the fact with the given [key], if present. */
    fun deleteFact(key: String) {
        writableDatabase.delete(TABLE, "$COL_KEY = ?", arrayOf(key))
    }

    // ── Read operations ─────────────────────────────────────────────

    /**
     * Retrieves a single fact by [key], or `null` if not found.
     * Updates [MemoryEntry.accessedAt] and [MemoryEntry.accessCount] on read.
     */
    fun getFact(key: String): MemoryEntry? {
        val db = readableDatabase
        val cursor = db.query(
            TABLE, null, "$COL_KEY = ?", arrayOf(key),
            null, null, null, "1"
        )
        return cursor.use {
            if (it.moveToFirst()) {
                val entry = cursorToEntry(it)
                touchFact(entry.id)
                entry
            } else null
        }
    }

    /**
     * Returns all stored facts, ordered by most-recently-accessed first.
     */
    fun allFacts(): List<MemoryEntry> {
        val db = readableDatabase
        val cursor = db.query(
            TABLE, null, null, null,
            null, null, "$COL_ACCESSED DESC"
        )
        return cursor.use {
            val entries = mutableListOf<MemoryEntry>()
            while (it.moveToNext()) {
                entries.add(cursorToEntry(it))
            }
            entries
        }
    }

    /**
     * Searches facts whose key or value contain [query] (case-insensitive).
     */
    fun searchFacts(query: String): List<MemoryEntry> {
        val db = readableDatabase
        val like = "%$query%"
        val cursor = db.query(
            TABLE, null,
            "$COL_KEY LIKE ? OR $COL_VALUE LIKE ?", arrayOf(like, like),
            null, null, "$COL_ACCESSED DESC", "50"
        )
        return cursor.use {
            val entries = mutableListOf<MemoryEntry>()
            while (it.moveToNext()) {
                entries.add(cursorToEntry(it))
            }
            entries
        }
    }

    /**
     * Builds a summary of all stored facts formatted for injection into
     * the system prompt's memory section.
     */
    fun buildMemoryContext(): String {
        val facts = allFacts()
        if (facts.isEmpty()) return ""
        return buildString {
            appendLine("Known facts about the user and device:")
            facts.forEach { fact ->
                appendLine("- ${fact.key}: ${fact.value}")
            }
        }
    }

    // ── Internal ────────────────────────────────────────────────────

    private fun touchFact(id: Long) {
        writableDatabase.execSQL(
            "UPDATE $TABLE SET $COL_ACCESSED = ?, $COL_ACCESS_COUNT = $COL_ACCESS_COUNT + 1 WHERE $COL_ID = ?",
            arrayOf(System.currentTimeMillis(), id)
        )
    }

    private fun evictIfNeeded() {
        val count = readableDatabase.rawQuery("SELECT COUNT(*) FROM $TABLE", null).use {
            it.moveToFirst(); it.getInt(0)
        }
        if (count > MAX_FACTS) {
            val excess = count - MAX_FACTS
            writableDatabase.execSQL(
                "DELETE FROM $TABLE WHERE $COL_ID IN " +
                "(SELECT $COL_ID FROM $TABLE ORDER BY $COL_ACCESSED ASC LIMIT ?)",
                arrayOf(excess)
            )
            Log.d(TAG, "Evicted $excess oldest facts")
        }
    }

    private fun cursorToEntry(cursor: android.database.Cursor): MemoryEntry {
        return MemoryEntry(
            id = cursor.getLong(cursor.getColumnIndexOrThrow(COL_ID)),
            key = cursor.getString(cursor.getColumnIndexOrThrow(COL_KEY)),
            value = cursor.getString(cursor.getColumnIndexOrThrow(COL_VALUE)),
            source = cursor.getString(cursor.getColumnIndexOrThrow(COL_SOURCE)),
            createdAt = cursor.getLong(cursor.getColumnIndexOrThrow(COL_CREATED)),
            accessedAt = cursor.getLong(cursor.getColumnIndexOrThrow(COL_ACCESSED)),
            accessCount = cursor.getInt(cursor.getColumnIndexOrThrow(COL_ACCESS_COUNT))
        )
    }
}
