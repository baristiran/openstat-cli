"""Network analysis commands: network descriptives, centrality, community detection."""

from __future__ import annotations

from openstat.commands.base import command
from openstat.session import Session


def _require_nx():
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise ImportError("Network analysis requires networkx. Install with: pip install networkx")


@command("network", usage="network <subcommand> ...")
def cmd_network(session: Session, args: str) -> str:
    """Network analysis using NetworkX.

    Subcommands:
      network build   from <source_col> to <target_col> [weight=<col>]
      network describe
      network centrality  [--degree|--betweenness|--closeness|--eigenvector]
      network community   [--louvain|--greedy]
      network plot        [--layout=spring|circular|kamada]

    Examples:
      network build from sender to receiver
      network build from from_node to to_node weight=strength
      network describe
      network centrality --degree
      network community --greedy
      network plot
    """
    from openstat.commands.base import CommandArgs
    ca = CommandArgs(args)
    if not ca.positional:
        return cmd_network.__doc__ or "Usage: network <subcommand>"

    subcmd = ca.positional[0].lower()

    if subcmd == "build":
        return _build_network(session, args)
    elif subcmd == "describe":
        return _describe_network(session)
    elif subcmd == "centrality":
        return _centrality(session, args)
    elif subcmd == "community":
        return _community(session, args)
    elif subcmd == "plot":
        return _plot_network(session, args)
    else:
        return (
            f"Unknown subcommand: {subcmd}\n"
            "Available: build, describe, centrality, community, plot"
        )


def _build_network(session: Session, args: str) -> str:
    nx = _require_nx()
    import re

    df = session.require_data()

    m_from = re.search(r"from\s+(\w+)", args)
    m_to = re.search(r"to\s+(\w+)", args)
    if not m_from or not m_to:
        return "Usage: network build from <source_col> to <target_col> [weight=<col>]"

    src_col = m_from.group(1)
    tgt_col = m_to.group(1)

    m_w = re.search(r"weight[= ](\w+)", args)
    weight_col = m_w.group(1) if m_w else None

    for c in [src_col, tgt_col] + ([weight_col] if weight_col else []):
        if c not in df.columns:
            return f"Column not found: {c}"

    sub = df.select([c for c in [src_col, tgt_col, weight_col] if c]).drop_nulls()

    G = nx.DiGraph() if "--directed" in args else nx.Graph()
    for row in sub.iter_rows():
        src, tgt = str(row[0]), str(row[1])
        w = float(row[2]) if weight_col else 1.0
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += w
        else:
            G.add_edge(src, tgt, weight=w)

    session._network = G
    session._network_weight_col = weight_col

    directed_str = "directed" if G.is_directed() else "undirected"
    return (
        f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({directed_str})\n"
        f"Source: '{src_col}' → Target: '{tgt_col}'"
        + (f"  Weight: '{weight_col}'" if weight_col else "")
        + "\nUse 'network describe', 'network centrality', 'network community', 'network plot'"
    )


def _describe_network(session: Session) -> str:
    nx = _require_nx()
    G = getattr(session, "_network", None)
    if G is None:
        return "No network built. Use 'network build from <src> to <tgt>' first."

    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    is_connected = nx.is_connected(G.to_undirected()) if G.is_directed() else nx.is_connected(G)

    lines = [
        f"Nodes: {n}",
        f"Edges: {e}",
        f"Density: {density:.4f}",
        f"Connected: {is_connected}",
    ]

    if not G.is_directed():
        if n > 0 and e > 0:
            try:
                avg_clust = nx.average_clustering(G)
                lines.append(f"Avg Clustering: {avg_clust:.4f}")
            except Exception:
                pass
            try:
                if nx.is_connected(G):
                    avg_path = nx.average_shortest_path_length(G)
                    lines.append(f"Avg Path Length: {avg_path:.4f}")
                    diam = nx.diameter(G)
                    lines.append(f"Diameter: {diam}")
            except Exception:
                pass

    # Degree distribution summary
    import numpy as np
    degrees = [d for _, d in G.degree()]
    if degrees:
        lines += [
            "",
            f"Degree — Min: {min(degrees)}  Mean: {np.mean(degrees):.2f}  Max: {max(degrees)}",
        ]

    # Top 5 nodes by degree
    top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    if top:
        lines.append("")
        lines.append("Top nodes by degree:")
        for node, deg in top:
            lines.append(f"  {str(node):<20} degree = {deg}")

    return "\n" + "=" * 50 + "\nNetwork Descriptives\n" + "=" * 50 + "\n" + "\n".join(lines) + "\n" + "=" * 50


def _centrality(session: Session, args: str) -> str:
    nx = _require_nx()
    G = getattr(session, "_network", None)
    if G is None:
        return "No network built. Use 'network build' first."

    if "--betweenness" in args:
        label, scores = "Betweenness Centrality", nx.betweenness_centrality(G)
    elif "--closeness" in args:
        label, scores = "Closeness Centrality", nx.closeness_centrality(G)
    elif "--eigenvector" in args:
        try:
            label, scores = "Eigenvector Centrality", nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            label, scores = "Degree Centrality (fallback)", nx.degree_centrality(G)
    else:
        label, scores = "Degree Centrality", nx.degree_centrality(G)

    top_n = 20
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    lines = [f"Top {min(top_n, len(sorted_nodes))} nodes by {label}:"]
    lines.append(f"  {'Node':<25} {'Score':>10}")
    lines.append("  " + "-" * 37)
    for node, score in sorted_nodes:
        lines.append(f"  {str(node):<25} {score:>10.4f}")

    return "\n" + "=" * 50 + f"\n{label}\n" + "=" * 50 + "\n" + "\n".join(lines) + "\n" + "=" * 50


def _community(session: Session, args: str) -> str:
    nx = _require_nx()
    G = getattr(session, "_network", None)
    if G is None:
        return "No network built. Use 'network build' first."

    G_undir = G.to_undirected() if G.is_directed() else G

    try:
        if "--louvain" in args:
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G_undir)
                communities = {}
                for node, comm_id in partition.items():
                    communities.setdefault(comm_id, []).append(node)
                method = "Louvain"
            except ImportError:
                return "Louvain requires python-louvain. Install with: pip install python-louvain"
        else:
            # Greedy modularity (built into networkx)
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G_undir))
            communities = {i: list(c) for i, c in enumerate(comms)}
            method = "Greedy Modularity"

        n_comm = len(communities)
        modularity = None
        try:
            from networkx.algorithms.community.quality import modularity as nx_mod
            modularity = nx_mod(G_undir, [set(v) for v in communities.values()])
        except Exception:
            pass

        lines = [
            f"Method: {method}",
            f"Communities found: {n_comm}",
        ]
        if modularity is not None:
            lines.append(f"Modularity: {modularity:.4f}")
        lines.append("")

        for cid, members in sorted(communities.items(), key=lambda x: -len(x[1])):
            sample = ", ".join(str(m) for m in members[:5])
            if len(members) > 5:
                sample += f", ... (+{len(members)-5})"
            lines.append(f"  Community {cid+1:>3}: {len(members):>4} nodes  [{sample}]")

        return "\n" + "=" * 55 + f"\nCommunity Detection ({method})\n" + "=" * 55 + "\n" + "\n".join(lines) + "\n" + "=" * 55

    except Exception as exc:
        return f"community detection error: {exc}"


def _plot_network(session: Session, args: str) -> str:
    nx = _require_nx()
    G = getattr(session, "_network", None)
    if G is None:
        return "No network built. Use 'network build' first."

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from openstat.plots.plotter import _unique_path

    # Layout
    if "--circular" in args:
        pos = nx.circular_layout(G)
        layout_name = "circular"
    elif "--kamada" in args:
        pos = nx.kamada_kawai_layout(G)
        layout_name = "kamada-kawai"
    else:
        pos = nx.spring_layout(G, seed=42)
        layout_name = "spring"

    n = G.number_of_nodes()
    fig_size = min(max(6, n * 0.3), 16)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    node_size = max(50, min(500, 2000 // (n + 1)))
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_size=node_size,
        node_color="#4C72B0",
        edge_color="#AAAAAA",
        font_size=max(6, min(10, 120 // (n + 1))),
        arrows=G.is_directed(),
        width=0.8,
        alpha=0.9,
    )
    ax.set_title(f"Network ({n} nodes, {G.number_of_edges()} edges) — {layout_name} layout")
    ax.axis("off")
    fig.tight_layout()

    session.output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(session.output_dir, "network_plot")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    session.plot_paths.append(str(path))
    return f"Network plot saved: {path}"
