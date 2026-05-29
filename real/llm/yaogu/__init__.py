"""LLM-driven 妖股 factor discovery with contrastive learning.

Positive samples (妖股): stocks with >=30% run-up and consecutive limit-ups.
Negative samples (假启动): stocks with similar pre-launch patterns that didn't launch.

The LLM analyzes matched pairs to find DISCRIMINATING features — what separates
real launches from false alarms — and generates expression tree factors.
"""
