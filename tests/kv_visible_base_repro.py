#!/usr/bin/env python3
"""Reproduces the disk KV cache hole behind a thinking-visible checkpoint.

A chat client that declares no tools replays its history without reasoning,
while the server keeps each reply's sampled reasoning in the KV.  The server
bridges the two with a "thinking-visible" checkpoint keyed by the visible
text.  This script drives the sequence in which the stores made on top of
such a checkpoint matter, and reads the server log to say what each request
loaded and how much it had to prefill:

  1. turn 1: system + user, a full reply (checkpoint C1)
     - an unrelated request evicts C1 to disk (--no-interloper skips it)
  2. turn 2 in three requests, each extending the last: user -> A1,
     + user-role result R1 -> A2, + user-role result R2 -> A3
  3. the client rewrites its history: R1 is cut to 75% plus a marker
  4. a pre-heat with the rewritten history, max_tokens 1, either
       a: ending at assistant A3, or
       b: with a placeholder user message " " after A3
  5. the next real turn: rewritten history + a new user message

Cases: control (no rewrite, no pre-heat), control-a / control-b (no rewrite,
pre-heat a / b), a and b (rewrite, pre-heat a / b).  Every case starts from a
system prompt with its own nonce, so no case reads another's disk entries.

All text is synthetic.  Needs only the Python standard library.

  ./ds4-server --model M.gguf --ctx 196608 --port 8011 \\
      --kv-disk-dir /tmp/empty-dir --kv-disk-space-mb 16384 \\
      --kv-cache-cold-max-tokens 196608 --kv-cache-cold-anchor last-user \\
      > /tmp/server.log 2>&1 &
  python3 tests/kv_visible_base_repro.py --port 8011 --log /tmp/server.log
"""
import argparse
import http.client
import json
import os
import random
import re
import sys
import time

CASES = ("control", "control-a", "control-b", "a", "b")
MARKER = "\n[... result truncated ...]"
STUB_USER = {"role": "user", "content": " "}

PLACES = ["Skerry Point", "Gannet Rock", "Thrift Head", "Marram Sound", "Kelp Narrows",
          "Osprey Bank", "Tern Shoal", "Heather Ness", "Cormorant Ledge", "Sandwort Bay"]
THINGS = ["the fog signal", "the lamp rotation motor", "the tide gauge", "the barometer array",
          "the relief boat winch", "the rainwater cistern", "the radio beacon", "the storm shutters",
          "the oil store", "the landing derrick", "the anemometer mast", "the signal flag locker"]
ACTIONS = ["is inspected", "is logged", "is recalibrated", "is cleaned with fresh water",
           "is checked against the reference card", "is reported to the district office",
           "is covered and lashed down", "is tested for a full minute"]
WHEN = ["at first light", "before every watch change", "after any gale above force eight",
        "on the first calm day of the month", "whenever the visibility drops below two cables",
        "at the turn of each spring tide", "once the relief boat has cleared the sound",
        "as soon as the evening lamp is lit"]
WHY = ["so that the next keeper finds it exactly as recorded",
       "because salt spray shortens its working life",
       "since the district inspector reads these entries first",
       "to keep the week's readings comparable with last year's",
       "as the handbook's earlier editions were found wanting on this point",
       "so a single keeper can manage the station alone if need be"]


def handbook(rng, articles):
    lines = []
    for n in range(1, articles + 1):
        place = rng.choice(PLACES)
        lines.append(
            f"Article {n}. At {place}, {rng.choice(THINGS)} {rng.choice(ACTIONS)} "
            f"{rng.choice(WHEN)}, {rng.choice(WHY)}. The entry carries the keeper's initials, "
            f"the reading {rng.randint(100, 999)}.{rng.randint(0, 9)} and the reference "
            f"{place.split()[0].upper()}-{rng.randint(1000, 9999)}.")
    return "\n".join(lines)


def readings(rng, title, rows):
    lines = [title, "day | hour | station | level_cm | wind_kn | pressure_hpa | note"]
    for n in range(rows):
        lines.append(
            f"{1 + n // 4} | {(n % 4) * 6:02d}:00 | {rng.choice(PLACES)} | {rng.randint(40, 460)} | "
            f"{rng.randint(0, 48)} | {rng.randint(968, 1036)} | {rng.choice(THINGS)[4:]} "
            f"{rng.choice(['nominal', 'drifting', 'rechecked', 'salt-crusted', 'steady'])}")
    return "\n".join(lines)


class Server:
    def __init__(self, host, port, log_path, model, effort):
        self.host, self.port, self.log_path = host, port, log_path
        self.model, self.effort = model, effort

    def log_size(self):
        return os.path.getsize(self.log_path)

    def log_since(self, offset, until=None, timeout=10.0):
        """The log lines written after `offset`; waits for a line matching `until`."""
        deadline = time.monotonic() + timeout
        while True:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                lines = f.read().decode("utf-8", "replace").splitlines()
            if until is None or any(until in line for line in lines) or time.monotonic() > deadline:
                return lines
            time.sleep(0.1)

    def chat(self, messages, max_tokens):
        """One streamed request as the client sends it: no tools, the effort at top level
        (what the OpenAI client's extra_body does), usage asked for."""
        body = {"model": self.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, "reasoning_effort": self.effort,
                "max_tokens": max_tokens}
        offset = self.log_size()
        started = time.monotonic()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=900)
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(500)!r}")
        content, reasoning, finish, usage = [], [], None, None
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish = choice.get("finish_reason") or finish
        conn.close()
        lines = self.log_since(offset, until=" finish=")
        return Reply("".join(content), "".join(reasoning), finish, usage or {},
                     time.monotonic() - started, lines)


class Reply:
    def __init__(self, content, reasoning, finish, usage, seconds, log_lines):
        self.content, self.reasoning, self.finish = content, reasoning, finish
        self.usage, self.seconds = usage, seconds
        self.log = [l for l in log_lines if " chunk " not in l and " chunk=" not in l]
        text = "\n".join(self.log)
        span = re.search(r"chat ctx=(\d+)\.\.(\d+):(\d+)\S* .*prompt start", text)
        self.cached, self.prompt, self.prefilled = (int(x) for x in span.groups()) if span else (None,) * 3
        hit = re.search(r"kv cache hit text tokens=(\d+) text=(\d+) .*key=(\S+)", text)
        if hit:
            self.source = f"disk {hit.group(3)} @{hit.group(1)}"
        elif "thinking live continuation" in text:
            self.source = "live thinking-visible"
        elif self.cached:
            self.source = "live"
        else:
            self.source = "none"
        self.stores = re.findall(r"kv cache stored tokens=(\d+) .*?reason=(\S+) key=(\S+)", text)
        self.usage_prompt = self.usage.get("prompt_tokens")
        self.usage_cached = (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens")

    def row(self):
        """As the client replays it: the content verbatim, the reasoning beside it."""
        row = {"role": "assistant", "content": self.content}
        if self.reasoning:
            row["reasoning_content"] = self.reasoning
        return row

    def stores_text(self):
        return ", ".join(f"{t} {r}/{k}" for t, r, k in self.stores) or "-"


def show(step, reply, verbose):
    print(f"  {step:<14} prompt={reply.prompt} cached={reply.cached} prefilled={reply.prefilled} "
          f"source=[{reply.source}] usage={reply.usage_prompt}/{reply.usage_cached} "
          f"finish={reply.finish} stores=[{reply.stores_text()}] {reply.seconds:.1f}s")
    if verbose:
        for line in reply.log:
            print(f"      | {line}")


def run_case(server, case, args):
    rng = random.Random(4711)
    nonce = f"{case}-{os.getpid()}-{int(time.time())}"
    system = (f"Station handbook, working copy {nonce}.\n"
              "You are the duty assistant of a lighthouse district. Answer briefly. When asked to "
              "request data, reply with exactly the one line you are told to write.\n\n"
              + handbook(rng, args.articles))
    r1 = ("Result of fetch_readings(week=12). Next, reply with exactly one line: "
          "fetch_calibration(sensor=3)\n" + readings(rng, "readings week 12", args.result_rows))
    r2 = (readings(rng, "Result of fetch_calibration(sensor=3)", args.result_rows) +
          "\nNow answer the original question in two short sentences.")
    u3 = ("Thanks. One more thing, answered in one short sentence: which of the stations in "
          "those readings would you send the relief boat to first?")
    print(f"\n=== case {case} ===")

    def turn(step, messages, max_tokens=args.max_tokens, must_stop=True):
        reply = server.chat(messages, max_tokens)
        show(step, reply, args.verbose or step.startswith(("4", "5")))
        if must_stop and reply.finish != "stop":
            raise RuntimeError(f"{step}: finish={reply.finish}, the scenario needs finish=stop")
        return reply

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "In one short sentence: what is this handbook for?"}]
    messages.append(turn("1 turn-1", messages).row())
    if not args.no_interloper:
        turn("  interloper", [{"role": "system", "content": f"Unrelated session {nonce}."},
                              {"role": "user", "content": "Say the word ready."}])
    messages.append({"role": "user", "content": "I need a tide summary for week 12. First request the raw "
                     "readings by replying with exactly one line: fetch_readings(week=12)"})
    messages.append(turn("2 round-1", messages).row())
    r1_index = len(messages)
    messages.append({"role": "user", "content": r1})
    messages.append(turn("2 round-2", messages).row())
    messages.append({"role": "user", "content": r2})
    a3 = turn("2 round-3", messages)
    messages.append(a3.row())

    if not case.startswith("control"):
        cut = r1[:len(r1) * 3 // 4] + MARKER
        messages[r1_index] = {"role": "user", "content": cut}
        print(f"  3 rewrite      R1 {len(r1)} -> {len(cut)} chars")
    pre_heat = None
    shape = case.split("-")[-1]
    if shape in ("a", "b"):
        heat = messages + ([STUB_USER] if shape == "b" else [])
        pre_heat = turn(f"4 pre-heat-{shape}", heat, max_tokens=1, must_stop=False)
    # Only the prefill of step 5 is measured; its reply may run into the cap.
    final = turn("5 next-turn", messages + [{"role": "user", "content": u3}],
                 max_tokens=args.final_max_tokens, must_stop=False)
    return {"case": case, "pre_heat": pre_heat, "final": final, "own": len(u3)}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True, help="the server's log file")
    parser.add_argument("--cases", default=",".join(CASES), help=f"comma-separated, of: {', '.join(CASES)}")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--effort", default="low", help="reasoning_effort; it must not be none")
    parser.add_argument("--max-tokens", type=int, default=3000, help="cap of a real turn's reply")
    parser.add_argument("--final-max-tokens", type=int, default=600, help="cap of the reply of step 5")
    parser.add_argument("--articles", type=int, default=150, help="size of the system prompt")
    parser.add_argument("--result-rows", type=int, default=64, help="size of a command result")
    parser.add_argument("--no-interloper", action="store_true",
                        help="leave C1 live instead of evicting it to disk after turn 1")
    parser.add_argument("--verbose", action="store_true", help="print the log lines of every request")
    args = parser.parse_args()
    server = Server(args.host, args.port, args.log, args.model, args.effort)
    results = [run_case(server, case, args) for case in args.cases.split(",")]

    # What step 5 must prefill at best: its own user message.  The control case measures
    # it (a live continuation prefills exactly the new tail); without one, estimate.
    control = next((r for r in results if r["case"] == "control"), None)
    own = control["final"].prefilled if control else results[0]["own"] // 4 + 6
    own_note = "measured by the control case" if control else "estimated from its length"
    print(f"\nstep 5's own new user message: {own} tokens ({own_note})")
    print(f"\n{'case':<10} | {'step 4 pre-heat: prompt cached prefilled source':<58} | step 5 next turn: prompt cached prefilled source")
    for r in results:
        heat, final = r["pre_heat"], r["final"]
        heat_text = f"{heat.prompt} {heat.cached} {heat.prefilled} [{heat.source}]" if heat else "-"
        print(f"{r['case']:<10} | {heat_text:<58} | {final.prompt} {final.cached} {final.prefilled} [{final.source}]")
    print()
    bad = 0
    for r in results:
        final = r["final"]
        extra = final.prefilled - own
        if extra <= 16:
            print(f"VERDICT {r['case']}: OK - step 5 prefilled {final.prefilled} tokens, its own message is {own}")
        else:
            bad += 1
            print(f"VERDICT {r['case']}: RE-PREFILL - step 5 prefilled {final.prefilled} tokens, "
                  f"{extra} more than its own message ({own}); served by [{final.source}]")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
