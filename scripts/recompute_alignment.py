#!/usr/bin/env python3
"""Independently rescore saved source-order predictions; never call model APIs.

The frozen 0.2.4 predictions are already in the submitted source-order index
space. This script verifies identical inputs/candidates before reusing them.
Historical timeout exclusions are explicit, with full-N diagnostics alongside.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "rebuttal/05_anonymous_code_release/RxnOptBench_anonymous_review_upload_0.2.4"
NEW = ROOT / "camera_ready_submission/alignment_work/dataset_candidate"
ARTIFACT = OLD
DATASET = NEW / "data/original_json"
TASKS = {
    "mc_single": "multiple_choice_single_varying.json",
    "mc_all": "multiple_choice_all_varying.json",
    "fib_all": "fill_in_blank_all_blank.json",
    "mc_single_reference": "multiple_choice_single_varying_reference.json",
    "mc_all_reference": "multiple_choice_all_varying_reference.json",
    "fib_all_reference": "fill_in_blank_all_blank_reference.json",
    "mc_single_reference_control": "multiple_choice_single_varying_reference_control.json",
    "mc_all_reference_control": "multiple_choice_all_varying_reference_control.json",
}
MODELS = [
    ("chemistry", "ChemLLM"), ("chemistry", "ChemDFM"), ("chemistry", "ChemVLM"),
    ("open_source", "Intern-S1-Pro"), ("open_source", "Qwen3.5-397B-A17B"),
    ("open_source", "Kimi-K2.6"), ("open_source", "GLM-5.1"),
    ("open_source", "MiniMax-M2.7"), ("open_source", "DeepSeek-V4-Pro"),
    ("proprietary", "GPT-5.5"), ("proprietary", "Gemini-3.1-Pro"),
    ("proprietary", "Claude-Opus-4.7"),
]
KEYS = ["reagents", "solvents", "catalysts", "reaction_temperature", "reaction_time", "light_condition", "atmosphere", "ligand"]
TIER_NAMES = {"chemistry": "Chemistry LLMs", "open_source": "Open Source LLMs", "proprietary": "Proprietary LLMs"}

def load(path):
    return json.loads(path.read_text())

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def tsv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

def sha(path):
    path=path.resolve()
    label=None
    for prefix,base in (("artifact",ARTIFACT),("dataset",DATASET),("baseline",OLD),("output",OUT)):
        try:
            label=prefix+"/"+path.relative_to(base.resolve()).as_posix()
            break
        except ValueError:
            pass
    if label is None:
        label="script/"+path.name
    return {"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}

def configure(argv=None):
    """Resolve released inputs through CLI; no private paths are required."""
    global ARTIFACT, DATASET, OLD, OUT, NEW
    candidates=[Path.cwd(),Path(__file__).resolve().parent,*Path(__file__).resolve().parents]
    detected=next((p for p in candidates if (p/"results/predictions").is_dir() and (p/"data/original_json").is_dir()),None)
    default_artifact=detected or (ARTIFACT if (ARTIFACT/"results/predictions").is_dir() else None)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path,default=default_artifact,
                        help="Downloaded data artifact root containing results/predictions and historical/submitted_v0.2.4")
    parser.add_argument("--dataset-root",type=Path,
                        help="Directory containing the eight active native JSON files; default ARTIFACT/data/original_json")
    parser.add_argument("--baseline-root",type=Path,
                        help="Historical 0.2.4 root with data/original_json and results/table2_metrics.tsv")
    parser.add_argument("--output","--output-dir",dest="output",type=Path,default=OUT,
                        help="Directory for generated TSV/JSON/TeX reports")
    args=parser.parse_args(argv)
    if args.artifact_root is None:
        parser.error("Supply --artifact-root pointing to the downloaded RxnOptBench data artifact")
    ARTIFACT=args.artifact_root.resolve()
    local_default=(ARTIFACT==OLD.resolve() and DATASET.is_dir())
    DATASET=(args.dataset_root or (DATASET if local_default else ARTIFACT/"data/original_json")).resolve()
    if args.baseline_root:
        OLD=args.baseline_root.resolve()
    elif (ARTIFACT/"historical/submitted_v0.2.4/data/original_json").is_dir():
        OLD=(ARTIFACT/"historical/submitted_v0.2.4").resolve()
    elif not (OLD/"data/original_json").is_dir():
        parser.error("Supply --baseline-root or retain artifact historical/submitted_v0.2.4")
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    for task,filename in TASKS.items():
        for path in (DATASET/filename,OLD/"data/original_json"/filename):
            if not path.is_file(): parser.error("Missing required native dataset: "+path.name)
    if not (ARTIFACT/"results/predictions").is_dir():
        parser.error("artifact root must contain results/predictions")
    return args

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def no_dr_utility(s):
    y = float(s["yield"])
    if s.get("ee") is not None:
        f = (100 + min(abs(float(s["ee"])), 100)) / 200
    elif s.get("er") is not None:
        ratio = float(s["er"])
        f = 1 if math.isinf(ratio) else ratio / (ratio + 1)
    else:
        f = 1
    return y * f

def rels(vals):
    m = max(vals)
    return [v / m if m else 1.0 for v in vals]

def maxset(vals):
    m = max(vals)
    return {i for i, v in enumerate(vals) if math.isclose(v, m, rel_tol=0, abs_tol=1e-9)}

def selkey(sel, keys):
    return tuple(sel.get(k) for k in keys)

def question_values(q, fib=False):
    if fib:
        rows = q["meta"]["legal_combinations"]
        return ([r["score"] for r in rows], [r["effective_score"] for r in rows], [r["relative_score"] for r in rows])
    return (q["meta"]["scores"], q["meta"]["effective_scores"], q["meta"]["option_relative_scores"])

def observation(q, pred, fib=False):
    if not fib:
        valid = isinstance(pred, int) and not isinstance(pred, bool) and 0 <= pred < len(q["options"])
        return {"id": q["id"], "valid": valid, "hit": valid, "exact": bool(valid and pred in q["answer"]), "relative": q["meta"]["option_relative_scores"][pred] if valid else 0.0}
    keys = q["query_keys"]
    valid = isinstance(pred, dict) and all(isinstance(pred.get(k), int) and not isinstance(pred.get(k), bool) and 0 <= pred[k] < len(q["input"]["candidate_values"][k]) for k in keys)
    lookup = {selkey(r["selection"], keys): r for r in q["meta"]["legal_combinations"]}
    row = lookup.get(selkey(pred, keys)) if valid else None
    relative = row["relative_score"] if row else None
    return {"id": q["id"], "valid": valid, "hit": row is not None, "exact": bool(row and relative == 1), "relative": relative}

def aggregate(observations, fib=False):
    n = len(observations)
    valid = sum(o["valid"] for o in observations)
    hits = sum(o["hit"] for o in observations)
    correct = sum(o["exact"] for o in observations)
    out = {"n": n, "valid": valid, "invalid": n-valid, "exact_count": correct, "accuracy": correct/n if n else 0}
    if fib:
        out.update({"hit_count": hits, "unobserved_valid": valid-hits, "hit": hits/n if n else 0,
                    "strict": correct/n if n else 0,
                    "score": sum(o["relative"] for o in observations if o["hit"])/hits if hits else 0,
                    "score_denominator": hits})
    else:
        out.update({"score": sum(o["relative"] for o in observations)/n if n else 0, "score_denominator": n})
    return out

def metrics_for(dataset, predictions, fib=False, excluded=()):
    obs = [observation(q, predictions.get(q["id"]), fib) for q in dataset if q["id"] not in excluded]
    return aggregate(obs, fib), obs

def headline(tasks):
    return {"MC_single_Acc_pct": 100*tasks["mc_single"]["accuracy"],
            "MC_single_Score": tasks["mc_single"]["score"],
            "MC_all_Acc_pct": 100*tasks["mc_all"]["accuracy"],
            "MC_all_Score": tasks["mc_all"]["score"],
            "FIB_Hit_pct": 100*tasks["fib_all"]["hit"],
            "FIB_Strict_Acc_pct": 100*tasks["fib_all"]["strict"],
            "FIB_Score": tasks["fib_all"]["score"]}

def audit_datasets(old, new):
    report = {}
    errors = []
    for task in TASKS:
        fib = task.startswith("fib")
        oldby = {q["id"]: q for q in old[task]}
        newby = {q["id"]: q for q in new[task]}
        assert len(newby) == len(new[task]) == len(oldby)
        assert list(oldby) == list(newby)
        changed_answers, changed_relative, changed_effective = [], [], []
        dr_questions = []
        violations = []
        for sid, q in newby.items():
            o = oldby[sid]
            unchanged = ("id", "question_type", "target_key", "varying_keys", "input", "options", "query_keys")
            for key in unchanged:
                if q.get(key) != o.get(key):
                    errors.append({"task": task, "id": sid, "mutation": key})
            scores, effective, relative = question_values(q, fib)
            olds, oldeff, oldrel = question_values(o, fib)
            preserved_raw = ([r.get("raw_observed_score", r["score"]) for r in q["meta"]["legal_combinations"]]
                             if fib else q["meta"].get("raw_observed_scores", scores))
            if preserved_raw != olds:
                errors.append({"task": task, "id": sid, "mutation": "raw_scores_not_preserved"})
            expected = [no_dr_utility(s) for s in scores]
            expected_relative = rels(expected)
            if any(abs(a-b) > 2e-5 for a,b in zip(effective, expected)):
                violations.append({"id": sid, "field": "effective"})
            if any(abs(a-b) > 1e-6 for a,b in zip(relative, expected_relative)):
                violations.append({"id": sid, "field": "relative"})
            if max(relative) != 1 or any(not 0 <= r <= 1 for r in relative):
                violations.append({"id": sid, "field": "relative_bounds_or_no_unit_gold"})
            if fib:
                keys = q["query_keys"]
                rows = q["meta"]["legal_combinations"]
                expected_best = {selkey(rows[i]["selection"], keys) for i in maxset(expected)}
                answer_set = {selkey(a, keys) for a in q["answer"]}
                one_set = {selkey(r["selection"], keys) for r in rows if r["relative_score"] == 1}
                if len({selkey(r["selection"], keys) for r in rows}) != len(rows):
                    violations.append({"id": sid, "field": "duplicate_selection"})
                if [r["selection"] for r in rows] != [r["selection"] for r in o["meta"]["legal_combinations"]]:
                    errors.append({"task": task, "id": sid, "mutation": "legal_support"})
            else:
                expected_best = maxset(expected)
                answer_set = set(q["answer"])
                one_set = {i for i,r in enumerate(relative) if r == 1}
            if answer_set != expected_best or answer_set != one_set:
                violations.append({"id": sid, "field": "answer_unit_score_tie_set", "answer": sorted(answer_set), "truebest": sorted(expected_best), "unit_relative": sorted(one_set)})
            if q["answer"] != o["answer"]:
                changed_answers.append(sid)
            if relative != oldrel:
                changed_relative.append(sid)
            if effective != oldeff:
                changed_effective.append(sid)
            if any(s.get("dr") is not None or s.get("rr") is not None for s in preserved_raw):
                dr_questions.append(sid)
        # Self consistency: a supplied true gold must hit full score on every q.
        gold = {q["id"]: q["answer"][0] for q in new[task]}
        gold_results, _ = metrics_for(new[task], gold, fib)
        if gold_results["accuracy"] != 1 or gold_results["score"] != 1:
            violations.append({"field": "all_gold_full_score", "metrics": gold_results})
        report[task] = {"n":len(new[task]), "has_dr_rr":len(dr_questions),
                        "changed_answer_sets":len(changed_answers), "changed_answer_ids":changed_answers,
                        "changed_relative_scores":len(changed_relative), "changed_relative_ids":changed_relative,
                        "changed_effective_scores":len(changed_effective), "changed_effective_ids":changed_effective,
                        "gold_results":gold_results, "score_policy_violations":violations}
        errors.extend({"task": task, **v} for v in violations)
    save("dataset_independent_audit.json", {"errors": errors, "tasks": report})
    return report, errors

def uniform(dataset, fib=False):
    if not fib:
        return {"n":len(dataset), "accuracy":mean(len(q["answer"])/len(q["options"]) for q in dataset), "score":mean(mean(q["meta"]["option_relative_scores"]) for q in dataset)}
    h, strict, sums, means = [], [], [], []
    for q in dataset:
        total = math.prod(len(q["input"]["candidate_values"][k]) for k in q["query_keys"])
        rows=q["meta"]["legal_combinations"]
        h.append(len(rows)/total)
        strict.append(sum(r["relative_score"] == 1 for r in rows)/total)
        sums.append(sum(r["relative_score"] for r in rows)/total)
        means.append(mean(r["relative_score"] for r in rows))
    # Exact E[sum(X_i R_i)/sum X_i | sum X_i>0] for independent tuple
    # sampling, using Poisson-binomial prefix/suffix polynomials. Numerical
    # quadrature of integral_0^1 prod_j(1-p_j+p_j*x) gives E[1/(1+H_-i)].
    # Composite Simpson cross-check against degree-exact Gauss-Legendre below.
    intervals = 20000
    accum = 0.0
    for t in range(intervals+1):
        x=t/intervals
        terms=[1-p+p*x for p in h]
        logprod=sum(math.log(a) for a in terms if a>0)
        if any(a==0 for a in terms):
            prod=0.0
            numerator=sum(s*math.prod(terms[:i]+terms[i+1:]) for i,s in enumerate(sums))
        else:
            prod=math.exp(logprod)
            numerator=prod*sum(s/a for s,a in zip(sums,terms))
        coeff=1 if t in (0,intervals) else (4 if t%2 else 2)
        accum+=coeff*numerator
    expected_simpson=accum/(3*intervals)
    import numpy as np
    nodes, weights=np.polynomial.legendre.leggauss((len(dataset)+1)//2)
    gauss=0.0
    for node,weight in zip(nodes,weights):
        x=(node+1)/2
        terms=np.asarray([1-p+p*x for p in h])
        prod=float(np.prod(terms))
        gauss+=weight*prod*sum(s/a for s,a in zip(sums,terms))/2
    expected=gauss
    assert abs(expected-expected_simpson)<1e-9, (expected, expected_simpson)
    p_nonzero=1-math.prod(1-p for p in h)
    return {"n":len(dataset), "hit":mean(h), "strict":mean(strict),
            "historical_macro_legal_score":mean(means),
            "pooled_ratio_of_expectations_score":sum(sums)/sum(h),
            "score":expected/p_nonzero,
            "conditional_nonzero_hit_probability":p_nonzero,
            "score_simpson_crosscheck":expected_simpson/p_nonzero,
            "quadrature_absolute_difference":abs(expected-expected_simpson),
            "score_expectation_method":"Finite-batch conditional E[Score|at least one Hit], uniform independent per-key tuples, one per question. Integral of product_j(1-p_j+p_j*t) * sum_i(p_i*mean_R_i/(1-p_i+p_i*t)) from0to1, divided P(H>0); polynomial degree N-1, Gauss-Legendre ceil(N/2) nodes, crosschecked Simpson20000 to <1e-9."}

def perkey(dataset, predictions, fib=False):
    counts=Counter(); correct=Counter()
    for q in dataset:
        pred=predictions.get(q["id"])
        if fib:
            for k in q["query_keys"]:
                counts[k]+=1
                correct[k]+=int(isinstance(pred,dict) and any(pred.get(k)==a[k] for a in q["answer"]))
        else:
            k=q["target_key"]; counts[k]+=1
            correct[k]+=int(isinstance(pred,int) and not isinstance(pred,bool) and pred in q["answer"])
    out={k:{"n":counts[k],"correct":correct[k],"accuracy_pct":100*correct[k]/counts[k]} for k in counts}
    for label,keys in (("top8",KEYS),("all",list(counts))):
        n=sum(counts[k] for k in keys); c=sum(correct[k] for k in keys)
        out[label]={"n":n,"correct":c,"accuracy_pct":100*c/n}
    return out

def paired(dataset_ref, dataset_control, pred_ref, pred_control):
    controls={q["id"].removesuffix("_ref_control"):q for q in dataset_control}
    obsr=[]; obsc=[]; semantic_fail=[]; order_changes=[]; context_changes=[]
    for q in dataset_ref:
        base=q["id"].removesuffix("_ref")
        c=controls[base]
        # Different option ordering is legitimate; compare chemical option-key
        # to score and target sets, never raw candidate indices across twins.
        rm={canonical(opt):(q["meta"]["option_relative_scores"][i], i in q["answer"]) for i,opt in enumerate(q["options"])}
        cm={canonical(opt):(c["meta"]["option_relative_scores"][i], i in c["answer"]) for i,opt in enumerate(c["options"])}
        if rm != cm:
            semantic_fail.append(base)
        if q["options"] != c["options"]:
            order_changes.append(base)
        context={k:v for k,v in q["input"].items() if k!="reference_conditions"}
        if context != c["input"]:
            context_changes.append(base)
        obsr.append(observation(q,pred_ref.get(q["id"])))
        obsc.append(observation(c,pred_control.get(c["id"])))
    r=aggregate(obsr); c=aggregate(obsc)
    both=sum(a["exact"] and b["exact"] for a,b in zip(obsr,obsc))
    refonly=sum(a["exact"] and not b["exact"] for a,b in zip(obsr,obsc))
    ctrlonly=sum(not a["exact"] and b["exact"] for a,b in zip(obsr,obsc))
    return {"n":len(obsr),"reference":r,"control":c,
            "delta_accuracy_pp":100*(r["accuracy"]-c["accuracy"]),"delta_score":r["score"]-c["score"],
            "both_correct":both,"reference_only_correct":refonly,"control_only_correct":ctrlonly,
            "both_wrong":len(obsr)-both-refonly-ctrlonly,"semantic_pair_failures":semantic_fail,
            "ordered_options_differ_count":len(order_changes),"ordered_options_differ_ids":order_changes,
            "nonreference_context_differ_count":len(context_changes),"nonreference_context_differ_ids":context_changes,
            "design_boundary":"Candidate sets, chemical-option-to-utility mappings, and nonreference input contexts are paired. Current archived file option ordering differs for some twins, so this check does not establish prompt identity except the reference block or freedom from position confounds; no contemporaneous provider request bytes were verified."}

def main():
    configure()
    old={t:load(OLD/"data/original_json"/f) for t,f in TASKS.items()}
    new={t:load(DATASET/f) for t,f in TASKS.items()}
    manifest=[sha(Path(__file__).resolve())]+[sha(p/f) for p in (OLD/"data/original_json",DATASET) for f in TASKS.values()]
    audit,errors=audit_datasets(old,new)
    raw={}; tables={v:[] for v in ("old","target_policy","full_N")}; perkeys={}; pairout={}; exclusions=[]
    for tier,model in MODELS:
        d=ARTIFACT/"results/predictions"/tier/model
        predictions={task:load(d/(task+".json")) for task in TASKS}
        methods={task:load(d/(task+"_parse_methods.json")) for task in TASKS}
        # Every task may contain a historical no-response placeholder. Preserve
        # archive files but never interpret such indices as actual responses.
        cleaned_predictions={task:{sid:(None if methods[task].get(sid)=="api_timeout_skip" else pred)
                                  for sid,pred in predictions[task].items()} for task in TASKS}
        manifest.extend(sha(d/(task+suffix+".json")) for task in TASKS for suffix in ("","_parse_methods"))
        results={"old":{},"target_policy":{},"full_N":{}}
        for task in TASKS:
            assert set(predictions[task]) == {q["id"] for q in new[task]}
            excluded={sid for sid,m in methods[task].items() if model=="DeepSeek-V4-Pro" and task in ("mc_all","fib_all") and m=="api_timeout_skip"}
            all_timeouts={sid for sid,method in methods[task].items() if method=="api_timeout_skip"}
            if all_timeouts:
                exclusions.append({"model":model,"task":task,"count":len(all_timeouts),"ids":sorted(all_timeouts),
                                   "historical_headline_drop":bool(excluded),"policy":"drop only for main-table historical exclusions; otherwise full-N missing response contributes zero"})
            for label,data,exc in (("old",old[task],excluded),("target_policy",new[task],excluded),("full_N",new[task],set())):
                # Timeout records have no raw model response; archived parsed
                # indices are placeholders, never successful model outputs.
                effective_predictions = cleaned_predictions[task]
                results[label][task],_=metrics_for(data,effective_predictions,task.startswith("fib"),exc)
        raw[model]={"tier":tier,**results}
        for label in tables:
            tables[label].append({"Tier":TIER_NAMES[tier],"Model":model,**headline(results[label])})
        perkeys[model]={task:perkey(new[task],
                    {sid: (None if methods[task].get(sid)=="api_timeout_skip" else pred) for sid,pred in predictions[task].items()},
                    task.startswith("fib")) for task in ("mc_single","fib_all")}
        pairout[model]={base:paired(new[base+"_reference"],new[base+"_reference_control"],cleaned_predictions[base+"_reference"],cleaned_predictions[base+"_reference_control"]) for base in ("mc_single","mc_all")}
        pairout[model]["fib_all_unpaired_diagnostic"]={"reference":results["target_policy"]["fib_all_reference"],"no_reference_all":results["target_policy"]["fib_all"],"note":"No dedicated FIB reference_control subset exists. Full-set no-reference and reference cohorts differ; comparison is descriptive and not a matched treatment-effect estimate."}
    save("all_metrics.json",raw)
    save("per_key_accuracy.json",perkeys)
    save("paired_reference_control.json",pairout)
    save("timeout_exclusions.json",exclusions)
    for label,rows in tables.items():
        tsv("table2_"+label+".tsv",rows)
    ur={t:uniform(new[t],t.startswith("fib")) for t in TASKS}
    old_ur={t:uniform(old[t],t.startswith("fib")) for t in ("mc_single","mc_all","fib_all")}
    save("uniform_random.json",{"target_policy":ur,"old":old_ur})
    random_tasks={t:{"accuracy":u.get("accuracy",u.get("strict")),**u} for t,u in ur.items()}
    randomrow={"Tier":"Baseline","Model":"Uniform random",**headline(random_tasks)}
    tsv("table2_target_policy_with_random.tsv",[randomrow]+tables["target_policy"])
    # No exceptions disguise discrepancies between prediction-level computation
    # and frozen printed cells. Report every original-table difference.
    with (OLD/"results/table2_metrics.tsv").open() as f:
        printed={r["Model"]:r for r in csv.DictReader(f,delimiter="\t")}
    diffs=[]
    for i,(_,model) in enumerate(MODELS):
        for col in headline(raw[model]["old"]):
            fmt=".1f" if col.endswith("pct") else ".3f"
            o=tables["old"][i][col]; n=tables["target_policy"][i][col]
            if format(o,fmt)!=printed[model][col] or format(n,fmt)!=printed[model][col]:
                diffs.append({"model":model,"metric":col,"printed":printed[model][col],"old_recomputed":format(o,fmt),"target_policy":format(n,fmt),"new_minus_old":n-o})
    save("metric_change_manifest.json",{"printed_vs_old_and_new":diffs,"dataset_audit":audit,
         "policy":"All unverified dr/rr axes excluded; no raw scores or predictions edited; original timeout drop policy kept for Table 2 and full-N also reported.",
         "errors":errors})
    lines=[]
    for row in [randomrow]+tables["target_policy"]:
        vals=[row[k] for k in headline(raw["ChemLLM"]["old"])]
        lines.append(row["Model"]+" & "+" & ".join(f"{v:.1f}\\%" if k.endswith("pct") else f"{v:.3f}" for k,v in zip(headline(raw["ChemLLM"]["old"]), vals))+r" \\")
    (OUT/"table2_target_policy_rows.tex").write_text("\n".join(lines)+"\n")
    rows=[]
    for _,model in MODELS:
        for task in ("mc_single","fib_all"):
            rows.append({"model":model,"task":task,**{k:perkeys[model][task][k]["accuracy_pct"] for k in KEYS},"top8_overall":perkeys[model][task]["top8"]["accuracy_pct"],"all_keys_overall":perkeys[model][task]["all"]["accuracy_pct"]})
    tsv("per_key_accuracy.tsv",rows)
    save("input_manifest.json",manifest)
    print(json.dumps({"errors":len(errors),"all_models":len(raw),"tasks_per_model":len(TASKS),"changed_answer_sets":{t:a["changed_answer_sets"] for t,a in audit.items()},"table2":tables["target_policy"],"random":randomrow,"timeout_exclusions":[{k:v for k,v in e.items() if k!="ids"} for e in exclusions]},indent=2))
    if errors:
        raise SystemExit("Dataset invariant errors; inspect dataset_independent_audit.json")

if __name__ == "__main__":
    main()
