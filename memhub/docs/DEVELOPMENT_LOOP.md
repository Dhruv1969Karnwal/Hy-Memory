# Development acceptance loop

A step counts as finished only after it has passed through these states:

`Ready → In progress → Verify → Done`

If any check fails:

`Verify → Rework → Verify`

## Before starting

1. Read `AGENTS.md`, the project rules, and the implementation plan.
2. Confirm this step's inputs, artifacts, edge cases, and hard metrics.
3. Write the failing test or Gate first and confirm that it catches the target regression.

## Implementation and verification

1. Implement the smallest possible scope.
2. Run the affected tests.
3. Run `pnpm check`.
4. Run this step's end-to-end flow with a temporary data directory.
5. Save the commands, exit codes, sample counts, and failure reasons.

## Definition of done

- Code, tests, and documentation agree.
- The automated tests are all green.
- The Gate really turns red once the target regression is injected.
- The end-to-end flow needs no manual database or configuration fix to pass.
- Known issues are triaged and enter the plan.

If any one of these is unmet, the step cannot be marked Done.
Pre-existing failures and this round's failures are recorded separately; assertions are never weakened to buy green.
