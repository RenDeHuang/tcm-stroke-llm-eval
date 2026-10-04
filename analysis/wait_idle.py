# -*- coding: utf-8 -*-
"""等待某个模型的主任务 runner 全部退出后再继续。

用途：API 配额为**按模型**限速，若在主任务尚未跑完时叠加消融任务，
会挤占同一模型的额度并制造可避免的 429。本脚本轮询进程表，
直到不再存在针对该模型、且任务名不是 ABL 的 runner.py 进程为止。

注意：只依赖进程命令行做匹配，不使用任何近似或启发式推断。
"""
import subprocess, sys, time

RUNNER = "scripts/tcm_stroke_eval/runner.py"


def busy(model: str):
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        if RUNNER not in line:
            continue
        if f"--models {model} " not in line:
            continue
        if "--tasks ABL" in line or "--tasks=ABL" in line:
            continue
        if "wait_idle.py" in line:
            continue
        hits.append(line.strip())
    return hits


def main():
    model, poll = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120
    while True:
        hits = busy(model)
        if not hits:
            print(f"[wait_idle] {model}: 无其他 runner 占用，放行 @ {time.strftime('%F %T')}", flush=True)
            return
        print(f"[wait_idle] {model}: 仍有 {len(hits)} 个主任务 runner 在跑，{poll}s 后复查 @ {time.strftime('%F %T')}", flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    main()
