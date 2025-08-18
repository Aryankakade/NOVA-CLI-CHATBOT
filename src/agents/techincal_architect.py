# src/agents/technical_architect.py
import asyncio
from typing import Dict, Any, Optional


class TechnicalArchitect:
    """
    Dedicated system-design / architecture expert.
    Returns a dict (mirrors the other agents) so the orchestrator can
    log, format or enrich the reply without hitting placeholders.
    """

    def __init__(self) -> None:
        print("🏗️  Technical Architect agent initialised!")

    # ---------- PUBLIC ENTRY POINT ----------
    async def provide_architecture_guidance(
        self,
        user_query: str,
        context: Optional[str] = "",
    ) -> Dict[str, Any]:
        q = user_query.lower()

        if "microservice" in q:
            return await self._microservices_blueprint(user_query)
        if "monolith" in q:
            return await self._monolith_blueprint(user_query)
        if "event" in q and "driven" in q:
            return await self._event_driven_blueprint(user_query)
        if "serverless" in q:
            return await self._serverless_blueprint(user_query)

        # generic fall-back
        return await self._generic_architecture_advice(user_query)

    # ---------- BLUEPRINT HELPERS ----------
    async def _microservices_blueprint(self, q: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": (
                "### Microservices Architecture\n"
                "• API-gateway front-door\n"
                "• Service discovery & health checks\n"
                "• Async messaging (Kafka / RabbitMQ)\n"
                "• DB-per-service + eventual consistency\n"
                "• Centralised observability (Prometheus + Jaeger)\n"
                "\nNext steps ⇒ define bounded contexts & automate CI/CD."
            ),
            "agent_type": "technical_architect",
            "confidence": 0.95,
        }

    async def _monolith_blueprint(self, q: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": (
                "### Well-Structured Monolith\n"
                "• Layered or Hexagonal design\n"
                "• Clear module boundaries, enforce via CI\n"
                "• Horizontal replicas behind LB\n"
                "• Robust regression tests + feature flags\n"
                "\nGood for small teams or early-stage products."
            ),
            "agent_type": "technical_architect",
            "confidence": 0.9,
        }

    async def _event_driven_blueprint(self, q: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": (
                "### Event-Driven Architecture\n"
                "• Immutable events on a broker (Kafka/Pulsar)\n"
                "• Schema registry for versioning\n"
                "• Idempotent consumers + DLQ\n"
                "• Enables real-time analytics & loose coupling."
            ),
            "agent_type": "technical_architect",
            "confidence": 0.9,
        }

    async def _serverless_blueprint(self, q: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": (
                "### Serverless\n"
                "• Functions triggered by HTTP/events\n"
                "• API-Gateway + IAM for auth\n"
                "• Use DynamoDB/S3 for persistence\n"
                "• Observability: CloudWatch, X-Ray\n"
                "\nIdeal for spiky or unpredictable workloads."
            ),
            "agent_type": "technical_architect",
            "confidence": 0.88,
        }

    async def _generic_architecture_advice(self, q: str) -> Dict[str, Any]:
        return {
            "success": True,
            "content": (
                "### Architecture Consultation\n"
                "Start with business & NFRs ➜ pick the **simplest** pattern "
                "that meets them.  Sketch a context diagram, define SLIs/SLOs, "
                "choose tech stacks your team masters, and instrument from day-1."
            ),
            "agent_type": "technical_architect",
            "confidence": 0.8,
        }
