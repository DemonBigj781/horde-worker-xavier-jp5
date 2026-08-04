# Xavier TUI and Resource Reporting Design

## Goal

Restore a usable Textual dashboard to the 10.1.2 JetPack 5 worker, then use
one lightweight resource sampler for both the dashboard and periodic console
status output.

## TUI

- Keep the normal headless `run_worker` path unchanged unless `--tui` is requested.
- Add a `horde-worker` command that selects TUI mode.
- Run Textual and the existing asynchronous process-manager loop in the same
  main-thread event loop so signal handling and graceful shutdown remain valid.
- Show worker identity, process states, queue depth, session counters, and
  recent logs.
- Treat TUI failure as an explicit startup failure rather than starting a
  second worker accidentally.

## Resources

- Sample Linux memory, swap, load average, and proportional process-tree memory
  from `/proc`.
- Read Jetson GPU clock, temperature, and fan state from sysfs when those files
  exist.
- Never invoke `tegrastats`, `sudo`, or a persistent helper process from the worker.
- Use the same immutable snapshot in the TUI and the 30-second console status
  report.
- Degrade cleanly on non-Linux and non-Jetson systems.

## Slow Worker Tuning

- Preserve the Xavier-specific 60-second R2 upload timeout independently.
- Keep `queue_size: 0`, disabled post-process overlap, and the explicit preload
  timeout.
- Disable `extra_slow_worker` after the TUI/resource deployment is verified
  idle-safe.

## Validation

- Unit-test CLI routing, TUI composition, snapshots, formatting, and timeout
  behavior.
- Run the focused Python suite on Xavier with one CPU thread.
- Start the TUI in Textual test mode before touching the live worker.
- Deploy under maintenance, wait for zero popped jobs, back up changed files,
  restart, and verify a successful submitted generation with zero faults.
