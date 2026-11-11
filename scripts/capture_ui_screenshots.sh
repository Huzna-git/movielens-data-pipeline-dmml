#!/usr/bin/env bash
# Capture the Prefect and MLflow UI screenshots used in the final report.
# Needs: `make prefect-server` (http://127.0.0.1:4200) and `make mlflow-ui` (http://127.0.0.1:5000)
# running, one completed flow run (PREFECT_API_URL=http://127.0.0.1:4200/api make pipeline), Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PREFECT=${PREFECT_API_URL:-http://127.0.0.1:4200/api}
MLFLOW=${MLFLOW_URL:-http://127.0.0.1:5000}

FLOW_RUN=$(curl -s -X POST "$PREFECT/flow_runs/filter" -H 'Content-Type: application/json' \
           -d '{"sort":"START_TIME_DESC","limit":1}' | $PY -c 'import sys,json; print(json.load(sys.stdin)[0]["id"])')
EXP=$(curl -s "$MLFLOW/api/2.0/mlflow/experiments/get-by-name?experiment_name=recommendation-pipeline" \
      | $PY -c 'import sys,json; print(json.load(sys.stdin)["experiment"]["experiment_id"])')
RUN=$(curl -s -X POST "$MLFLOW/api/2.0/mlflow/runs/search" -H 'Content-Type: application/json' \
      -d "{\"experiment_ids\":[\"$EXP\"],\"max_results\":1,\"filter\":\"attributes.run_name = 'implicit_als'\",\"order_by\":[\"attributes.start_time DESC\"]}" \
      | $PY -c 'import sys,json; print(json.load(sys.stdin)["runs"][0]["info"]["run_id"])')
echo "flow run $FLOW_RUN | mlflow experiment $EXP | implicit_als run $RUN"

# JavaScript run inside the page before the capture: MLflow 3 opens in its GenAI view, so switch to
# "Model training", close the assistant side panel, expand the metric columns.
CLICK_MT='[...document.querySelectorAll("button,[role=radio],[role=tab],label,span")].find(e => e.textContent.trim() === "Model training")?.click()'
CLOSE_X='(()=>{const bs=[...document.querySelectorAll("button")].filter(b=>{const r=b.getBoundingClientRect();return r.top>10&&r.top<70&&r.left>window.innerWidth-200;}).sort((a,b)=>b.getBoundingClientRect().left-a.getBoundingClientRect().left); bs[0]?.click(); return bs.length;})()'
MORE='(()=>{const b=[...document.querySelectorAll("button,div[role=button],span")].find(e=>e.textContent.trim().startsWith("Show more")); if(b){b.click(); return "clicked";} return "none";})()'
TICK='(()=>{let n=0; ["ndcg_at_10","precision_at_10","recall_at_10","coverage","diversity","rmse","mae"].forEach(t=>{const e=[...document.querySelectorAll("label,span,div")].find(x=>x.children.length===0&&x.textContent.trim()===t); if(e){e.dispatchEvent(new MouseEvent("click",{bubbles:true})); n++;}}); return n;})()'
ESC='(()=>{document.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true})); const t=[...document.querySelectorAll("h2,h3,span")].find(e=>e.textContent.trim()==="Runs"); t && t.dispatchEvent(new MouseEvent("click",{bubbles:true})); return 1;})()'
CHART='(()=>{const e=document.elementFromPoint(296,120); if(!e) return "none"; e.dispatchEvent(new MouseEvent("click",{bubbles:true})); return e.tagName;})()'

$PY scripts/capture_ui.py 11_prefect_flow_run_ui "http://127.0.0.1:4200/runs/flow-run/$FLOW_RUN" --wait 8 --width 1440 --height 900
$PY scripts/capture_screenshots.py url 12_prefect_runs_list "http://127.0.0.1:4200/runs" 1440 900 12000
$PY scripts/capture_screenshots.py url 13_prefect_dashboard "http://127.0.0.1:4200/dashboard" 1440 900 12000
$PY scripts/capture_ui.py 14_mlflow_experiment_runs "$MLFLOW/#/experiments/$EXP" --wait 6 --js "$CLICK_MT" --js "$CLOSE_X" --js "$MORE" --js "$TICK" --js "$ESC" --width 1800 --height 900
$PY scripts/capture_ui.py 15_mlflow_run_implicit_als "$MLFLOW/#/experiments/$EXP" --wait 5 --js "$CLICK_MT" \
    --js "location.hash = '#/experiments/$EXP/runs/$RUN'" --js "$CLOSE_X" --width 1600 --height 1150
$PY scripts/capture_ui.py 16_mlflow_experiment_chart "$MLFLOW/#/experiments/$EXP" --wait 6 --js "$CLICK_MT" --js "$CLOSE_X" \
    --js "$CHART" --width 1600 --height 1000
