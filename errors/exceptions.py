class HarnessException(Exception):
    status_code: int = 500
    detail: str = "An internal harness error occurred."

    def __init__(self, detail: str | None = None):
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class UnsupportedProviderError(HarnessException):
    status_code = 400

    def __init__(self, provider: str):
        super().__init__(f"Unsupported LLM provider: {provider}")


class UnsupportedOperationError(HarnessException):
    status_code = 400

    def __init__(self, operation: str):
        super().__init__(f"Unsupported operation: {operation}")


class ToolExecutionError(HarnessException):
    status_code = 400

    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


class GuardrailViolationError(HarnessException):
    status_code = 422

    def __init__(self, reason: str):
        super().__init__(f"Content blocked by safety guardrails: {reason}")


class CollectionNotAllowedError(HarnessException):
    status_code = 403

    def __init__(self, collection: str, allowed: list[str]):
        super().__init__(f"Collection '{collection}' is not in the allowed list: {allowed}")


class NoDbContextError(HarnessException):
    status_code = 500

    def __init__(self):
        super().__init__("No database context set — call set_db_context() before executing DB tools")


class OAuthEmailError(HarnessException):
    status_code = 400

    def __init__(self, provider: str):
        super().__init__(f"Could not retrieve email from {provider}")


class InvalidCredentialError(HarnessException):
    status_code = 400

    def __init__(self, mechanism: str = "credential"):
        super().__init__(f"Invalid or expired {mechanism}")


class NotInitializedError(HarnessException):
    status_code = 500

    def __init__(self, name: str):
        super().__init__(f"{name} not initialized — call its init function first")
