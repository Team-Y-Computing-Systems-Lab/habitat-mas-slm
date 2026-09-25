"""Simulation host: one habitat-lab env, a pool of robots, served over HTTP/JSON.

The host runs in the `habitat` conda env (Python 3.9). Everything else (MCP
servers, planner, SLM) talks to it through `client.SimClient`, which only uses
the standard library, so it works from any Python version or machine (e.g. a
Jetson running the planner while the sim runs on the workstation).
"""
