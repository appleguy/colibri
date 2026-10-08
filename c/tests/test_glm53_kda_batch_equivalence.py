"""Verify experimental batched KDA projections against scalar GLM53."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
def run(binary:Path,fixture:Path,flag:str,ids:list[int],extra_tokens:int):
    env={**os.environ,"GLM53_KDA_BATCH_PROJ":flag,"GLM53_BITS":"32"}
    args=[str(binary),"--model",str(fixture),"--ids",",".join(map(str,ids)),
          "--greedy",str(extra_tokens),"--logits"]
    start=time.monotonic()
    proc=subprocess.run(args,env=env,stdin=subprocess.DEVNULL,
       capture_output=True,text=True,timeout=180)
    duration=time.monotonic()-start
    if proc.returncode:
        raise RuntimeError(f"{flag} engine exited {proc.returncode}: {proc.stderr[-1000:]}")
    if flag=="1" and "[GLM53_KDA_BATCH]" not in proc.stderr:
        raise AssertionError("batched KDA was never exercised by fixture")
    lines={line.split()[0]:line.split()[1:]
           for line in proc.stdout.splitlines() if line.strip()}
    return {"flag":flag,"teacher":[int(v) for v in lines["teacher_forcing"]],
            "greedy":[int(v) for v in lines["greedy"]],
            "logits":[float(v) for v in lines["last_logits"]],
            "wall_s":round(duration,4)}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--binary",required=True,type=Path)
    parser.add_argument("--fixture",required=True,type=Path)
    parser.add_argument("--max-logit-abs",type=float,default=2e-5)
    args=parser.parse_args()
    reference=json.loads((args.fixture/"ref.json").read_text())
    ids=reference["prompt_ids"]
    expected=len(reference["greedy_new_ids"])
    baseline=run(args.binary,args.fixture,"0",ids,expected)
    batched=run(args.binary,args.fixture,"1",ids,expected)
    if baseline["teacher"]!=batched["teacher"]:raise AssertionError("teacher forcing diverged")
    if baseline["greedy"]!=batched["greedy"]:raise AssertionError("greedy token sequence diverged")
    a,b=baseline["logits"],batched["logits"]
    if len(a)!=len(b):raise AssertionError("logit shape changed")
    max_error=max((abs(x-y) for x,y in zip(a,b)),default=0)
    if max_error>args.max_logit_abs:
        raise AssertionError(f"logit max diff {max_error} > {args.max_logit_abs}")
    if baseline["teacher"]!=reference["teacher_forcing_ids"]:
        raise AssertionError("scalar baseline disagrees with oracle")
    if batched["greedy"]!=reference["greedy_new_ids"]:
        raise AssertionError("batched inference disagrees with oracle")
    print(json.dumps({"passed":True,"max_logit_abs":max_error,
       "scalar_wall_s":baseline["wall_s"],"batched_wall_s":batched["wall_s"],
       "speedup":round(baseline["wall_s"]/batched["wall_s"],3)
            if batched["wall_s"]>0 else None,
       "scope":"tiny model, startup included, not a production speed claim"},indent=2))
if __name__=="__main__":main()
