# Dashboard UI check

Drives the dashboard page in jsdom (no real browser) against a running server
loaded with the Agent Fuel study's 660 runs.

    AGENT_FUEL_REPO=path/to/agent-fuel python tests/ui/make_research_logs.py /tmp/research_logs
    agent-stoploss dashboard /tmp/research_logs --port 8741 --no-browser &
    cd tests/ui && npm install && node ui_test.js
