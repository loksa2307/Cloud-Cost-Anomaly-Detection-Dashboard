def recommend_row(row, service_type_map, thresholds):
    service = row['Service']
    cost = row['Cost']
    avg_util = row.get('AvgCPU', None)           # optional
    growth_pct = row.get('GrowthPct', 0)         # e.g., recent % increase
    suggestion = []
    priority = "Low"

    # High-cost baseline
    if cost >= thresholds['high_cost']:
        suggestion.append(f"High spend for {service}: evaluate Reserved Instances/Savings Plans.")
        priority = "High"

    # Low utilization for compute
    if service_type_map.get(service) == 'compute' and avg_util is not None:
        if avg_util < thresholds['cpu_util_low']:
            suggestion.append(f"{service}: CPU avg {avg_util:.0f}% — consider downsizing or rightsizing.")
            priority = "High"

    # Sudden growth
    if growth_pct and growth_pct > thresholds['spike_pct']:
        suggestion.append(f"Spike detected: {growth_pct:.0f}% increase — investigate jobs, deployments, or scale events.")
        if priority != "High": priority = "Medium"

    # Default
    if not suggestion:
        suggestion.append("No immediate action. Monitor for trends.")

    return {"recommendation": " / ".join(suggestion), "priority": priority}
