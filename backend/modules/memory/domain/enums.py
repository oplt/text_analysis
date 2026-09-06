from enum import StrEnum


class MemoryLevel(StrEnum):
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    PROJECT = "project"
    EPISODIC = "episodic"


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    SKILL_LEVEL = "skill_level"
    INSTRUCTION = "instruction"
    WARNING = "warning"
    TASK_STATE = "task_state"
    PROJECT_CONTEXT = "project_context"
    DECISION = "decision"
    EVENT = "event"


class MemorySource(StrEnum):
    CHAT = "chat"
    DOCUMENT = "document"
    FILE = "file"
    TOOL = "tool"
    MANUAL = "manual"
    SYSTEM = "system"
    REPO = "repo"
    API = "api"


class MemoryPrivacy(StrEnum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    DO_NOT_STORE = "do_not_store"


class MemoryOperation(StrEnum):
    ADD = "add"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    FORGET = "forget"


class AgentScope(StrEnum):
    GLOBAL_AGENT = "global_agent"
    USER_AGENT = "user_agent"
