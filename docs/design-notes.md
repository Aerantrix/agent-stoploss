# Design notes: why the defaults are what they are

The defaults below were checked by replaying the 660 recorded runs of the Agent
Fuel study (see the README). The aim throughout: stop the runs that are going
nowhere, and break as few healthy runs as possible. Where those two pull apart,
the default leans towards not breaking work, and replay shows you the price.

## Shadow by default

A stop is a product decision: the user gets a partial answer or none. Nobody
should take that decision from a library's defaults. Shadow mode records the
verdicts so `agent-stoploss replay` can show what enforce would have done on
your own traffic before you switch.

## Loop detector: threshold 3

The second identical call gets a warning, the third ends the run. Two is too
few: one retry after a transient error is reasonable behaviour, and stopping
it would break healthy runs. More than three only adds spend: in the study 126
of the 129 runs that made the same call three times failed anyway.

"Identical" is decided on fingerprints, not on raw text:

- **calls** are compared after normalising the arguments (case, surrounding
  spaces, key order);
- **results** are compared after dropping the fields that echo the arguments
  back, the volatile fields (`timestamp`, `request_id`, `trace_id` by default),
  UUIDs and timestamps. Bare dates are kept: a report for another day is a
  different result. Without this, an agent rewording its query and getting the
  same empty answer each time would look like progress;
- **errors** count towards the call fingerprint only; a failed call's text is
  left to the error classifier and the circuit breaker.

## Error classifier: terminal and transient statuses

400, 401, 403, 404, 405, 410 and 422 mean the same request cannot succeed
however often it is sent, so the run ends at the first one. There is nothing
for a warning to achieve. 408, 425, 429 and the 5xx gateway errors are
transient and left to the circuit breaker. 404 is the one to think about: if a
resource your agent polls may appear later, move 404 to the transient list.

An error with no HTTP status (a refused connection, a timeout) is neither: the
circuit breaker judges it by frequency. Frameworks that pass tool errors on as
text, such as the Claude Agent SDK, have the status read from explicit forms
only ("HTTP 401", "status: 503", "404 Not Found"); a bare number in a message
is too often a count or a port.

## Circuit breaker: 4 failures in the last 5 calls, 60 s cooldown

The breaker is for a dependency that is down, not for one bad answer. A window
of five with four failures lets a flapping service (one failure in two calls)
through, and stops a dead one by its fifth call. After the cooldown one probe
call is let through; a success closes the breaker.

This is the default with a known cost, and the least tested one. In the study
the breaker was the first to fire in only three runs, and all three were
healthy: in the flapping-service task the local 7B model, at temperature 1, hit
four failures in five calls. The study's failing runs were caught earlier by
the other mechanisms; the breaker is there for what the study did not cover,
an agent that varies every call to a dependency that is down. Raising the bar
to 5 of 5 does not save those three runs: they repeat a call and the loop
detector stops them later. If a tool of yours flaps by design, exempt it under
`tools:`.

## Run budget: off

A fixed token ceiling cannot tell a stuck agent from a long honest task. In the
study a 50,000-token budget cut a legitimate forty-domain task at step 29 of 41.
The mechanisms above stop the stuck runs without a ceiling, so the budget is
off by default. Replay suggests a value: 1.5 times your most expensive run
labelled ok, rounded up to a thousand tokens. Use it as a backstop, not as the
main control.

## minimum_steps: 3

Replay needs a fair baseline for "wasted". Any run spends something before
anyone, human or library, can tell that a tool is broken. The spend up to the
end of step 3 is counted as that minimum; only the spend above it is called
excess, and savings are reported against it. That keeps the replay from
claiming credit for tokens no control could have saved.

## Notices, not silent stops

In enforce mode the agent is told, in a short user-turn notice, why it is about
to be stopped, one step before a loop or failing-tool stop. An agent that heeds
it can finish on its own with a partial answer, which is usually better than a
hard stop. The notice changes the agent's behaviour, and replay assumes the run
went on as recorded, so replay cannot say how often agents heed it.

## Outcomes: labelled or marked as a guess

Whether a stop saved money or broke good work depends on whether the run was
going to succeed. agent-stoploss cannot know that, so `run.outcome()` lets you
say. Unlabelled runs get a heuristic (finished without an exception: ok) and
the journal marks the label as a heuristic, so the replay report can say how
much of its verdict rests on guesses.
