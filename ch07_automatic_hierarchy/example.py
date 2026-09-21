"""
Chapter 7 -- Automatic Hierarchy
===================================

Compass: hierarchical_segment / segment_awareness handling a small
segment via fallback, plus the drilldown vs. exclusive distinction.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import hierarchical_segment, segment_sizes


def make_support_ticket_data(seed=41):
    """
    A customer support team, three levels deep: department, team,
    agent. Most agents have handled plenty of tickets. A handful are
    recently hired and have handled only a few each -- not enough,
    individually, to say anything reliable about their own resolution
    times. One team (Upsell, in Sales) is brand new outright.
    """
    rng = np.random.default_rng(seed)
    rows = []

    established = {
        ("Billing", "Invoicing", "Agent_1"): 45,
        ("Billing", "Invoicing", "Agent_2"): 38,
        ("Billing", "Refunds", "Agent_3"): 52,
        ("Technical", "Network", "Agent_4"): 60,
        ("Technical", "Software", "Agent_5"): 55,
        ("Sales", "Renewals", "Agent_6"): 70,
        ("Sales", "Renewals", "Agent_7"): 65,
    }
    # 5 new Invoicing agents: too few individually (5 each), but
    # pooled together (25) clears min_size at the TEAM level
    new_invoicing_agents = {("Billing", "Invoicing", f"Agent_{i}"): 5 for i in range(10, 15)}
    # New Network agents (4 each) and a brand-new Support team (6
    # each): neither pool clears min_size on its own at the team
    # level, but pooled together at the DEPARTMENT level, they do
    new_network_agents = {("Technical", "Network", f"Agent_{i}"): 4 for i in range(20, 22)}
    new_support_team = {("Technical", "Support", f"Agent_{i}"): 6 for i in range(30, 33)}
    # A genuinely tiny, brand-new team with no one else left
    # unassigned in its department to pool with -- falls all the way
    # to the population-wide catch-all
    tiny_upsell_team = {("Sales", "Upsell", f"Agent_{i}"): 4 for i in range(40, 42)}

    all_groups = {**established, **new_invoicing_agents, **new_network_agents,
                  **new_support_team, **tiny_upsell_team}

    for (dept, team, agent), n in all_groups.items():
        resolution_time = rng.normal(24, 8, n).clip(1, None)
        satisfaction = 90 - 0.8 * resolution_time + rng.normal(0, 5, n)
        for rt, sat in zip(resolution_time, satisfaction):
            rows.append({"department": dept, "team": team, "agent": agent,
                          "resolution_hours": rt, "satisfaction": sat})

    return pd.DataFrame(rows)


def main():
    df = make_support_ticket_data()
    print(f"Total tickets: {len(df)}\n")

    # --- Fallback in action ---
    hierarchy = [["department", "team", "agent"], ["department", "team"], ["department"]]
    segmented = hierarchical_segment(df, hierarchy, min_size=20)

    check = segmented.groupby(["department", "team", "agent"], observed=True).agg(
        n=("agent", "size"), segment_id=("segment_id", "first"), segment_level=("segment_level", "first"),
    )
    print("=== Where each agent's tickets ended up ===")
    print(check.to_string())

    level_counts = segmented.groupby("segment_level", observed=True).size()
    print("\n=== Tickets per fallback level ===")
    level_names = {0: "Agent (most specific)", 1: "Team", 2: "Department", 3: "ALL (least specific)"}
    for level, n in level_counts.items():
        print(f"  Level {level} ({level_names[level]}): {n} tickets")

    print(
        "\nNotice what happened to the fallback groups. The five new"
        "\nInvoicing agents, 5 tickets each, don't clear min_size=20 on"
        "\ntheir own -- but pooled together, 25 tickets does, so all"
        "\nfive get assigned to the shared 'Billing_Invoicing' segment"
        "\nat the TEAM level. The new Network and Support agents don't"
        "\nclear min_size even pooled within their own teams (8 and 18"
        "\ntickets respectively) -- but pooled together at the"
        "\nDEPARTMENT level, 26 tickets clears it. The two brand-new"
        "\nUpsell agents have no one else left unassigned in Sales to"
        "\npool with at all, so their 8 tickets fall all the way to the"
        "\npopulation-wide 'ALL' segment."
    )

    # --- Chart: level distribution ---
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [level_names[lv] for lv in sorted(level_counts.index)]
    values = [level_counts[lv] for lv in sorted(level_counts.index)]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]
    ax.bar(labels, values, color=colors[:len(values)])
    ax.set_ylabel("Number of tickets")
    ax.set_title("Every ticket gets a home, at whichever level has enough data")
    for i, v in enumerate(values):
        ax.text(i, v + 3, str(v), ha="center")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig("fallback_levels.png", dpi=150)
    print("\nChart saved to fallback_levels.png")

    # --- Exclusive vs. drilldown ---
    print("\n\n=== Exclusive vs. drilldown, concretely ===")
    exclusive_sizes = segment_sizes(segmented)
    print(f"Exclusive: {len(exclusive_sizes)} segments, {exclusive_sizes.sum()} tickets total "
          f"(every ticket counted exactly once)")

    drilldown_total = 0
    drilldown_segments = 0
    for cols in hierarchy[:-1]:  # skip the "ALL" catch-all level for this comparison
        sizes = df.groupby(cols, observed=True).size()
        qualifying = sizes[sizes >= 20]
        drilldown_total += qualifying.sum()
        drilldown_segments += len(qualifying)
    print(f"Drilldown (levels 0-1 only): {drilldown_segments} qualifying groups, "
          f"{drilldown_total} ticket-appearances (the same ticket counted once per level it qualifies at)")

    print(
        "\nAn agent like Agent_1 (45 tickets) qualifies on their own at"
        "\nthe agent level AND their team, Billing_Invoicing, qualifies"
        "\ntoo once you add in the other Invoicing agents. In exclusive"
        "\nmode, Agent_1's tickets are counted once, at the most"
        "\nspecific level available. In drilldown mode, the same 45"
        "\ntickets are counted again as part of the broader team-level"
        "\npicture -- deliberate double-counting, useful when you want"
        "\nto see every granularity at once rather than committing each"
        "\nticket to a single 'best' level."
    )


if __name__ == "__main__":
    main()
