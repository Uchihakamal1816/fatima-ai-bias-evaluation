#!/usr/bin/env python3
"""
Generate polished, publication-ready figures for the Fatima Fellowship evaluation.
Produces 4 figures in assets/:
1. prr_by_domain.png: Grouped bar chart with clean offsets and no overlapping labels.
2. stance_breakdown.png: Stacked horizontal distribution of model stances with ample title/legend padding.
3. metrics_comparison.png: Grouped bar comparison of PRR, Truthfulness, and Sycophancy.
4. failure_taxonomy_flow.png: Clean process flow diagram with complete margins.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# General style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

summary = data['summary']
domains_data = summary['domains']
overall = summary['overall']

# -------------------------------------------------------------
# FIGURE 1: Premise Resistance Rate across Domains and Framing Conditions
# -------------------------------------------------------------
def plot_domain_comparison():
    domains = [
        'Healthcare &\nFolk Medicine',
        'Historical\nRevisionism',
        'Pseudoscience &\nSuperstition',
        'Socio-Cultural\nPrejudices'
    ]
    raw_keys = [
        'Healthcare & Folk Medicine',
        'Historical Revisionism',
        'Pseudoscience & Superstition',
        'Socio-Cultural Prejudices'
    ]
    
    neutral_prr = [domains_data[k]['neutral_prompt']['premise_resistance_rate'] for k in raw_keys]
    biased_prr = [domains_data[k]['biased_leading_prompt']['premise_resistance_rate'] for k in raw_keys]
    authority_prr = [domains_data[k]['authority_pressure_prompt']['premise_resistance_rate'] for k in raw_keys]
    
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=300)
    
    rects1 = ax.bar(x - width, neutral_prr, width, label='Neutral Baseline ($C_{neutral}$)', color='#2b5c8f', edgecolor='black', linewidth=0.8, zorder=3)
    rects2 = ax.bar(x, biased_prr, width, label='Biased Leading ($C_{biased}$)', color='#d9534f', edgecolor='black', linewidth=0.8, zorder=3)
    rects3 = ax.bar(x + width, authority_prr, width, label='Authority Pressure ($C_{authority}$)', color='#f0ad4e', edgecolor='black', linewidth=0.8, zorder=3)
    
    ax.set_ylabel('Premise Resistance Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Premise Resistance Rate Across Knowledge Domains Under Adversarial Framing\n(Qwen2.5-1.5B-Instruct)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 125)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.6, zorder=1)
    
    def autolabel(rects, is_biased_historical=False):
        for rect in rects:
            height = rect.get_height()
            if height == 0 and is_biased_historical:
                continue
            ax.annotate(f'{height:.0f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2, is_biased_historical=True)
    autolabel(rects3)
    
    # Clean arrow pointing to 0% collapse
    ax.annotate('100% Collapse\n(0% Resistance)',
                xy=(x[1], 1), xytext=(x[1], 32),
                arrowprops=dict(arrowstyle="->", color='#c9302c', lw=2.2),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#c9302c',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee', edgecolor='#c9302c', alpha=0.95))

    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=10.5)
    plt.tight_layout()
    plt.savefig('assets/prr_by_domain.png', dpi=300)
    plt.close()
    print("Generated assets/prr_by_domain.png")

# -------------------------------------------------------------
# FIGURE 2: Stance Distribution (Resisted, Hedged, Capitulated, Refusal)
# -------------------------------------------------------------
def plot_stance_distribution():
    conditions = ['Neutral Baseline', 'Biased Leading', 'Authority Pressure']
    cond_keys = ['neutral_prompt', 'biased_leading_prompt', 'authority_pressure_prompt']
    
    resisted = [overall[k]['resisted_count'] / 18 * 100 for k in cond_keys]
    hedged = [overall[k]['hedged_capitulation_count'] / 18 * 100 for k in cond_keys]
    capitulated = [overall[k]['capitulated_count'] / 18 * 100 for k in cond_keys]
    refusal = [overall[k]['refusal_count'] / 18 * 100 for k in cond_keys]
    
    y = np.arange(len(conditions))
    height = 0.52
    
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    
    p1 = ax.barh(y, resisted, height, label='Resisted (Factual & Objective)', color='#2e7d32', edgecolor='black', linewidth=0.8, zorder=3)
    p2 = ax.barh(y, hedged, height, left=resisted, label='Hedged Capitulation (Token Caveat + Sycophancy)', color='#f57c00', edgecolor='black', linewidth=0.8, zorder=3)
    left_cap = [r + h for r, h in zip(resisted, hedged)]
    p3 = ax.barh(y, capitulated, height, left=left_cap, label='Full Capitulation (Direct Unfiltered Sycophancy)', color='#d32f2f', edgecolor='black', linewidth=0.8, zorder=3)
    left_ref = [r + h + c for r, h, c in zip(resisted, hedged, capitulated)]
    p4 = ax.barh(y, refusal, height, left=left_ref, label='Guardrail Refusal (Evasive Canned Filter)', color='#7b1fa2', edgecolor='black', linewidth=0.8, zorder=3)
    
    ax.set_xlabel('Percentage of Responses (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Behavioral Stance Distribution by Prompt Framing\n(18 Scenarios, 54 Total Evaluations on Qwen2.5-1.5B-Instruct)', fontsize=13.5, fontweight='bold', pad=22)
    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontweight='bold', fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    for idx in range(len(conditions)):
        if resisted[idx] > 8:
            ax.text(resisted[idx] / 2, y[idx], f"{resisted[idx]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        if hedged[idx] > 5:
            ax.text(resisted[idx] + hedged[idx] / 2, y[idx], f"{hedged[idx]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        if capitulated[idx] > 5:
            ax.text(left_cap[idx] + capitulated[idx] / 2, y[idx], f"{capitulated[idx]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        if refusal[idx] > 5:
            ax.text(left_ref[idx] + refusal[idx] / 2, y[idx], f"{refusal[idx]:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            
    # Clean non-overlapping legend placement
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig('assets/stance_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/stance_breakdown.png")

# -------------------------------------------------------------
# FIGURE 3: Core Metric Comparison: Truthfulness vs Sycophancy vs Resistance
# -------------------------------------------------------------
def plot_metrics_comparison():
    conditions = ['Neutral Baseline ($C_{neutral}$)', 'Biased Leading ($C_{biased}$)', 'Authority Pressure ($C_{authority}$)']
    cond_keys = ['neutral_prompt', 'biased_leading_prompt', 'authority_pressure_prompt']
    
    prr = [overall[k]['premise_resistance_rate'] for k in cond_keys]
    truthfulness = [overall[k]['avg_truthfulness_score'] * 100 for k in cond_keys]
    sycophancy = [overall[k]['avg_sycophancy_score'] * 100 for k in cond_keys]
    
    x = np.arange(len(conditions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    
    r1 = ax.bar(x - width, prr, width, label='Premise Resistance Rate (%)', color='#1976d2', edgecolor='black', linewidth=0.8, zorder=3)
    r2 = ax.bar(x, truthfulness, width, label='Truthfulness Score (x100)', color='#388e3c', edgecolor='black', linewidth=0.8, zorder=3)
    r3 = ax.bar(x + width, sycophancy, width, label='Sycophancy Index (x100)', color='#e53935', edgecolor='black', linewidth=0.8, zorder=3)
    
    ax.set_ylabel('Score / Percentage (0 - 100)', fontsize=12, fontweight='bold')
    ax.set_title('Epistemic Integrity Degradation Across Framing Conditions\n(Qwen2.5-1.5B-Instruct)', fontsize=13.5, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontweight='bold', fontsize=10.5)
    ax.set_ylim(0, 120)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')
                        
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=10.5)
    plt.tight_layout()
    plt.savefig('assets/metrics_comparison.png', dpi=300)
    plt.close()
    print("Generated assets/metrics_comparison.png")

# -------------------------------------------------------------
# FIGURE 4: Failure Taxonomy & Behavioral Routing Diagram
# -------------------------------------------------------------
def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    # Title
    ax.text(6.25, 7.1, "Taxonomy of Epistemic Collapse Under Biased Prompting", ha="center", va="center", fontsize=15, fontweight="bold")
    
    # Input prompt box
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.8), 3.2, 2.2, boxstyle="round,pad=0.25", facecolor="#e1f5fe", edgecolor="#0288d1", linewidth=2.2))
    ax.text(2.1, 5.4, "User Input Prompt", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#01579b")
    ax.text(2.1, 4.4, "Contains Cultural, Folk, or\nHistorical Presupposition\n(e.g., 'Turmeric cures 3rd degree burns')", ha="center", va="center", fontsize=9.5, color="#0277bd")
    
    # Model Processing
    ax.add_patch(patches.FancyBboxPatch((4.5, 3.8), 3.2, 2.2, boxstyle="round,pad=0.25", facecolor="#ede7f6", edgecolor="#512da8", linewidth=2.2))
    ax.text(6.1, 5.4, "Frontier LLM Processing", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#311b92")
    ax.text(6.1, 4.4, "Instruction-Following Bias\n+ Lack of Premise Verification\n(RLHF/DPO Sycophancy)", ha="center", va="center", fontsize=9.5, color="#4527a0")
    
    # Arrow Input -> Model
    ax.annotate('', xy=(4.4, 4.9), xytext=(3.8, 4.9), arrowprops=dict(arrowstyle="->", lw=2.5, color="#37474f"))
    
    # 4 Output Branches (adjusted vertical positions to avoid clipping)
    # 1. Resisted (Factual)
    ax.add_patch(patches.FancyBboxPatch((8.7, 5.4), 3.4, 1.25, boxstyle="round,pad=0.2", facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=2))
    ax.text(10.4, 6.2, "1. Epistemic Resistance (50.0%)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1b5e20")
    ax.text(10.4, 5.7, "Rejects false premise;\nUpholds empirical consensus.", ha="center", va="center", fontsize=8.5, color="#2e7d32")
    
    # 2. Hedged Capitulation
    ax.add_patch(patches.FancyBboxPatch((8.7, 3.8), 3.4, 1.25, boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ef6c00", linewidth=2))
    ax.text(10.4, 4.6, "2. Hedged Capitulation (16.7%)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#e65100")
    ax.text(10.4, 4.1, "Token caveat + generates\npseudo-scientific rationales.", ha="center", va="center", fontsize=8.5, color="#ef6c00")
    
    # 3. Full Capitulation
    ax.add_patch(patches.FancyBboxPatch((8.7, 2.2), 3.4, 1.25, boxstyle="round,pad=0.2", facecolor="#ffebee", edgecolor="#c62828", linewidth=2))
    ax.text(10.4, 3.0, "3. Full Capitulation (27.8%)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#b71c1c")
    ax.text(10.4, 2.5, "Direct premise adoption;\nFabricates biology / eugenics.", ha="center", va="center", fontsize=8.5, color="#c62828")

    # 4. Refusal
    ax.add_patch(patches.FancyBboxPatch((8.7, 0.6), 3.4, 1.25, boxstyle="round,pad=0.2", facecolor="#f3e5f5", edgecolor="#7b1fa2", linewidth=2))
    ax.text(10.4, 1.4, "4. Guardrail Refusal (5.6%)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#4a148c")
    ax.text(10.4, 0.9, "Canned safety refusal without\neducating or explaining.", ha="center", va="center", fontsize=8.5, color="#7b1fa2")

    # Arrows to outputs
    ax.annotate('', xy=(8.6, 6.0), xytext=(7.8, 5.1), arrowprops=dict(arrowstyle="->", lw=2, color="#2e7d32"))
    ax.annotate('', xy=(8.6, 4.4), xytext=(7.8, 4.9), arrowprops=dict(arrowstyle="->", lw=2, color="#ef6c00"))
    ax.annotate('', xy=(8.6, 2.8), xytext=(7.8, 4.7), arrowprops=dict(arrowstyle="->", lw=2, color="#c62828"))
    ax.annotate('', xy=(8.6, 1.2), xytext=(7.8, 4.5), arrowprops=dict(arrowstyle="->", lw=2, color="#7b1fa2"))
    
    plt.tight_layout()
    plt.savefig('assets/failure_taxonomy_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated assets/failure_taxonomy_flow.png")

if __name__ == '__main__':
    plot_domain_comparison()
    plot_stance_distribution()
    plot_metrics_comparison()
    plot_architecture_diagram()
    print("All enhanced charts generated successfully!")
