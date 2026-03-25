"""Community detection (Leiden) + LLM summary generation.

Workflow:
1. Export entities + relationships from Neo4j into a NetworkX graph
2. Run Leiden community detection (via graspologic / leidenalg)
3. For each community, generate a title + summary via LLM
4. Embed summaries and store in PostgreSQL communities table

This is an offline batch process, triggered via API or CLI — not during
real-time ingest.
"""

import json
import logging
from typing import Any

import networkx as nx
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.neo4j_service import neo4j_service
from services.embedding_service import embed

logger = logging.getLogger(__name__)

# ── Graph export ────────────────────────────────────────────────────────────


def _export_graph_from_neo4j() -> nx.Graph:
    """Export all entities and relationships from Neo4j as a NetworkX graph."""
    G = nx.Graph()

    # Nodes
    nodes = neo4j_service.run_cypher(
        "MATCH (n) WHERE n.name IS NOT NULL "
        "RETURN n.name AS name, labels(n)[0] AS label, "
        "n.document_id AS doc_id",
        {},
    )
    for row in nodes:
        G.add_node(row["name"], label=row.get("label", ""), doc_id=row.get("doc_id"))

    # Edges
    edges = neo4j_service.run_cypher(
        "MATCH (a)-[r]->(b) "
        "WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
        "RETURN a.name AS source, b.name AS target, type(r) AS rel_type",
        {},
    )
    for row in edges:
        G.add_edge(row["source"], row["target"], rel_type=row.get("rel_type", ""))

    logger.info("Exported graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ── Community detection ─────────────────────────────────────────────────────


def _detect_communities(G: nx.Graph) -> list[set[str]]:
    """Run Leiden community detection on the graph.

    Falls back to connected components if leidenalg is not installed.
    Returns list of sets of node names.
    """
    if G.number_of_nodes() == 0:
        return []

    try:
        from graspologic.partition import hierarchical_leiden

        community_map = hierarchical_leiden(G, max_cluster_size=30, random_seed=42)
        # community_map is list of HierarchicalCluster(node, cluster, level, ...)
        clusters: dict[int, set[str]] = {}
        for item in community_map:
            cid = item.cluster
            clusters.setdefault(cid, set()).add(item.node)
        communities = list(clusters.values())
        logger.info("Leiden detected %d communities", len(communities))
    except ImportError:
        logger.warning("graspologic not installed — falling back to connected components")
        communities = [set(c) for c in nx.connected_components(G) if len(c) >= 2]
        logger.info("Connected components: %d communities", len(communities))

    # Filter out singleton communities
    return [c for c in communities if len(c) >= 2]


# ── LLM summary generation ─────────────────────────────────────────────────

_COMMUNITY_SUMMARY_PROMPT = (
    "You are an expert analyst for a Vietnamese mining chemical procurement system.\n"
    "Given a community of related entities and their relationships from a knowledge graph,\n"
    "generate a concise summary.\n\n"
    "Return JSON with:\n"
    '{"title": "Short Vietnamese title (max 10 words)",\n'
    ' "summary": "1-2 sentence Vietnamese summary of the community theme",\n'
    ' "full_content": "Detailed Vietnamese paragraph describing all entities, '
    'relationships, and their significance"}\n\n'
    "Focus on: what business domain this community represents, key entities,\n"
    "and what questions this community can answer."
)


def _build_community_context(
    G: nx.Graph, members: set[str]
) -> str:
    """Build a text description of a community for LLM summarization."""
    lines = ["ENTITIES:"]
    for node in sorted(members):
        data = G.nodes.get(node, {})
        lines.append(f"  - {node} (label: {data.get('label', '?')})")

    lines.append("\nRELATIONSHIPS:")
    for u, v, data in G.edges(data=True):
        if u in members and v in members:
            lines.append(f"  - {u} --[{data.get('rel_type', '?')}]--> {v}")

    return "\n".join(lines)


def _generate_summary(community_context: str) -> dict[str, str]:
    """Call LLM to generate community title, summary, and full_content."""
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _COMMUNITY_SUMMARY_PROMPT},
                {"role": "user", "content": community_context},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        logger.warning("Community summary generation failed: %s", exc)
        return {}


# ── Main orchestrator ───────────────────────────────────────────────────────


def build_communities(db: Session) -> dict[str, Any]:
    """Full pipeline: export → detect → summarize → store.

    Args:
        db: SQLAlchemy session (caller manages commit).

    Returns:
        Stats dict: {communities_created, entities_mapped, ...}
    """
    if not neo4j_service.available:
        logger.warning("Neo4j unavailable — cannot build communities")
        return {"error": "Neo4j unavailable"}

    # 1. Export graph
    G = _export_graph_from_neo4j()
    if G.number_of_nodes() < 2:
        return {"error": "Graph too small", "nodes": G.number_of_nodes()}

    # 2. Detect communities
    communities = _detect_communities(G)
    if not communities:
        return {"error": "No communities detected"}

    # 3. Clear old community data
    db.execute(text("DELETE FROM community_entities"))
    db.execute(text("DELETE FROM communities"))

    # 4. For each community: summarize + embed + store
    total_entities_mapped = 0
    for i, members in enumerate(communities):
        context = _build_community_context(G, members)

        # LLM summary
        result = _generate_summary(context)
        title = result.get("title", f"Community {i + 1}")
        summary = result.get("summary", "")
        full_content = result.get("full_content", "")

        # Count internal edges
        internal_edges = sum(
            1 for u, v in G.edges() if u in members and v in members
        )

        # Embed summary for global search vector similarity
        summary_vec = None
        embed_text = f"{title}. {summary}"
        try:
            vecs = embed([embed_text])
            if vecs:
                summary_vec = vecs[0]
        except Exception as exc:
            logger.warning("Failed to embed community %d summary: %s", i, exc)

        # Insert community
        row = db.execute(
            text("""
                INSERT INTO communities
                    (level, title, summary, full_content, embedding,
                     entity_count, relationship_count, rank)
                VALUES
                    (0, :title, :summary, :full_content,
                     CAST(:embedding AS vector),
                     :entity_count, :rel_count, :rank)
                RETURNING id
            """),
            {
                "title": title,
                "summary": summary,
                "full_content": full_content,
                "embedding": str(summary_vec) if summary_vec else None,
                "entity_count": len(members),
                "rel_count": internal_edges,
                "rank": len(members) + internal_edges,  # simple rank heuristic
            },
        )
        community_id = row.fetchone()[0]

        # Insert community-entity membership
        for node in members:
            data = G.nodes.get(node, {})
            db.execute(
                text("""
                    INSERT INTO community_entities (community_id, entity_name, entity_label)
                    VALUES (:cid, :name, :label)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "cid": community_id,
                    "name": node,
                    "label": data.get("label", "Unknown"),
                },
            )
            total_entities_mapped += 1

    db.commit()

    stats = {
        "communities_created": len(communities),
        "entities_mapped": total_entities_mapped,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
    }
    logger.info("Community build complete: %s", stats)
    return stats
