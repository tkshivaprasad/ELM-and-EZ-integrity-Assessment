"""Build ONE self-contained HTML page for a doctor to check your ELM/EZ annotations.

Run:
    python make_review_page.py                      (verification: doctor checks your lines and grades)
    python make_review_page.py --blind              (independent grading: doctor sees only the
                                                     original B-scans and gives their own grades;
                                                     use this to report inter-grader agreement)
    python make_review_page.py --classes AMD DME    (only some disease classes)
    python make_review_page.py --shuffle            (random order instead of grouped by class)

Output: outputs/review/annotation_review.html (or annotation_grading_blind.html).
Send that single file to the doctor. They open it in Chrome or Edge, review, and
click "Download results (CSV)" to send their answers back. Their progress is saved
in their browser, so they can close it and continue later on the same computer.
"""
import argparse
import base64
import json
import random
from datetime import date

import cv2
import numpy as np

import config as C
from utils import load_table, read_gray, read_mask, column_coverage

MAX_W = 1000      # images are downscaled to this width to keep the file small


def layer_rows(mask, c, scale):
    """Centre row of layer c in each column (None where absent), scaled to the display size."""
    px = mask == c
    n = px.sum(axis=0)
    rows = (px * np.arange(mask.shape[0])[:, None]).sum(axis=0)
    out = np.where(n > 0, rows / np.maximum(n, 1), np.nan)
    W = mask.shape[1]
    new_w = max(1, int(round(W * scale)))
    xs = np.linspace(0, W - 1, new_w)
    res = []
    for x in xs:
        v = out[int(round(x))]
        res.append(None if np.isnan(v) else round(float(v) * scale, 1))
    return res


def encode(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", action="store_true", help="doctor grades independently; no lines or grades shown")
    ap.add_argument("--classes", nargs="*", help="only these disease classes, e.g. AMD DME")
    ap.add_argument("--shuffle", action="store_true", help="random order (always on with --blind)")
    ap.add_argument("--title", default="ELM and EZ annotation review")
    args = ap.parse_args()

    df = load_table()
    if args.classes:
        df = df[df["disease_class"].isin(args.classes)]
    items = []
    for r in df.to_dict("records"):
        img, mask = read_gray(r["image_path"]), read_mask(r["mask_path"])
        H, W = img.shape
        s = min(1.0, MAX_W / W)
        small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
        it = dict(id=r["filename"], cls=str(r.get("disease_class", "")), w=small.shape[1], h=small.shape[0],
                  img=encode(small))
        if not args.blind:
            it.update(elm=str(r.get("elm_grade", "") or ""), ez=str(r.get("ez_grade", "") or ""),
                      covElm=round(column_coverage(mask, 1), 1), covEz=round(column_coverage(mask, 2), 1),
                      lines={"elm": layer_rows(mask, 1, s), "ez": layer_rows(mask, 2, s)})
        items.append(it)
    order = {"Normal": 0, "Drusen": 1, "DME": 2, "AMD": 3}
    if args.shuffle or args.blind:    # blind grading: random order, so disease groups are not visible
        random.Random(42).shuffle(items)
    else:
        items.sort(key=lambda d: (order.get(d["cls"], 9), d["id"]))

    meta = dict(title=args.title, blind=args.blind, created=date.today().isoformat(), n=len(items),
                key=f"octReview.{'blind' if args.blind else 'verify'}.{len(items)}.{date.today().isoformat()}")
    html = TEMPLATE.replace("__META__", json.dumps(meta)).replace("__DATA__", json.dumps(items, separators=(",", ":")))
    html = html.replace("__TITLE__", args.title)
    out = C.OUTPUT_DIR / "review"
    out.mkdir(parents=True, exist_ok=True)
    f = out / ("annotation_grading_blind.html" if args.blind else "annotation_review.html")
    f.write_text(html, encoding="utf-8")
    print(f"Saved {f} ({len(items)} images, {f.stat().st_size / 1e6:.1f} MB). Send this one file to the doctor.")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#dde1e6;--text:#1d232b;--muted:#5d6774;--accent:#2457c5;
--ok:#1f8a5b;--okbg:#e5f4ec;--bad:#c2410c;--badbg:#fdeee5;--elm:#e0389f;--ez:#16a3cf;
font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;font-size:15px;line-height:1.45;color:var(--text)}
*{box-sizing:border-box}body{margin:0;background:var(--bg)}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0 8px 0 0}
.who{display:flex;gap:6px;align-items:center;color:var(--muted)}
.who input{border:1px solid var(--line);border-radius:6px;padding:6px 8px;width:170px;font:inherit}
.prog{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:8px}
.bar{width:140px;height:7px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok);width:0}
button{font:inherit;cursor:pointer}
.btn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 12px}
.btn:hover{border-color:#9aa3ad}.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.filters{display:flex;gap:4px}.filters button{border:1px solid var(--line);background:#fff;border-radius:16px;padding:4px 12px;color:var(--muted)}
.filters button[aria-pressed=true]{background:var(--text);border-color:var(--text);color:#fff}
main{max-width:1500px;margin:0 auto;padding:18px 20px 60px}
.intro{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin-bottom:18px}
.intro h2{font-size:16px;margin:0 0 6px}.intro p,.intro li{margin:4px 0;color:var(--muted)}
.key{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-weight:600}
.key i{width:22px;height:3px;border-radius:2px;display:inline-block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:16px;overflow:hidden}
.card.done{border-color:#b9dcc9}.card.flag{border-color:#f2b796}
.chead{display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.num{color:var(--muted);font-variant-numeric:tabular-nums}.fname{font-weight:600}
.tag{font-size:12px;padding:1px 9px;border-radius:10px;background:#eef1f5;color:var(--muted)}
.status{margin-left:auto;font-size:13px;font-weight:600}.status.ok{color:var(--ok)}.status.bad{color:var(--bad)}.status.todo{color:var(--muted)}
.pics{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px 16px;background:#11151a}
.pics figure{margin:0}.pics figcaption{color:#c9d1db;font-size:12.5px;margin-bottom:4px}
.pics img,.pics canvas{width:100%;height:auto;display:block;border-radius:4px;cursor:zoom-in;background:#000}
.pics.single{grid-template-columns:minmax(0,760px);justify-content:center}
.ctrl{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px 22px;padding:12px 16px}
.q{display:flex;flex-direction:column;gap:5px}.q>span{font-size:13px;color:var(--muted)}.q b{color:var(--text)}
.seg{display:flex;flex-wrap:wrap;gap:6px}
.seg button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 11px;font-size:14px}
.seg button[aria-pressed=true].good{background:var(--okbg);border-color:var(--ok);color:var(--ok);font-weight:600}
.seg button[aria-pressed=true].bad{background:var(--badbg);border-color:var(--bad);color:var(--bad);font-weight:600}
.seg button[aria-pressed=true].pick{background:#e8eefc;border-color:var(--accent);color:var(--accent);font-weight:600}
.sub{display:none;margin-top:4px}.sub.show{display:flex}
textarea{font:inherit;border:1px solid var(--line);border-radius:6px;padding:6px 8px;min-height:38px;resize:vertical;width:100%}
.full{grid-column:1/-1}
.modal{position:fixed;inset:0;background:rgba(10,12,16,.97);z-index:20;display:none;flex-direction:column;padding:16px}
.modal.show{display:flex}.mbar{display:flex;gap:10px;align-items:center;color:#fff;margin-bottom:10px;flex-wrap:wrap}
.mbar .btn{background:#2a313a;color:#fff;border-color:#47515d}
.mpics{flex:1;display:grid;grid-template-columns:1fr;grid-template-rows:1fr 1fr;gap:10px;min-height:0}.mpics.single{grid-template-rows:1fr}
.mpics div{overflow:auto;background:#000;border-radius:6px}.mpics img,.mpics canvas{display:block;transform-origin:0 0}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:#1d232b;color:#fff;padding:9px 16px;border-radius:8px;opacity:0;transition:opacity .2s;z-index:30}
.toast.show{opacity:1}
.empty{color:var(--muted);text-align:center;padding:40px}
@media (max-width:800px){.pics{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <label class="who">Reviewer <input id="rev" placeholder="Name / initials" autocomplete="off"></label>
  <div class="filters" role="group" aria-label="Show">
    <button data-f="all" aria-pressed="true">All</button>
    <button data-f="todo" aria-pressed="false">To review</button>
    <button data-f="flag" aria-pressed="false" id="fflag">Needs correction</button>
  </div>
  <button class="btn" id="tl" hidden>Hide lines</button>
  <div class="prog"><span id="pt">0 / 0 reviewed</span><span class="bar"><i id="pb"></i></span></div>
  <button class="btn primary" id="dl">Download results (CSV)</button>
</header>
<main>
  <section class="intro" id="intro"></section>
  <div id="list"></div>
</main>
<div class="modal" id="modal">
  <div class="mbar"><b id="mt"></b><button class="btn" id="mzo">−</button><button class="btn" id="mzi">+</button>
  <button class="btn" id="mlines" hidden>Hide lines</button><span style="flex:1"></span><button class="btn" id="mclose">Close (Esc)</button></div>
  <div class="mpics" id="mp"></div>
</div>
<div class="toast" id="toast"></div>
<script>
const META=__META__;
const DATA=__DATA__;
const GR=["Intact","Disrupted","Absent"];
const $=s=>document.querySelector(s);
let R={},filter="all",showLines=true;
try{const s=JSON.parse(localStorage.getItem(META.key)||"{}");R=s.r||{};if(s.rev)$("#rev").value=s.rev}catch(e){}
function save(){try{localStorage.setItem(META.key,JSON.stringify({r:R,rev:$("#rev").value}))}catch(e){}}
let tt;function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("show");clearTimeout(tt);tt=setTimeout(()=>t.classList.remove("show"),2600)}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* ---------- intro ---------- */
$("#intro").innerHTML = META.blind ?
`<h2>How to grade</h2><p>For each OCT B-scan, please give your own grade for the external limiting membrane (ELM) and the ellipsoid zone (EZ). No annotations, diagnoses or file names are shown, and the order is random, so your grading is independent.</p>
<ul><li><b>Intact</b>: continuous across the scan. <b>Disrupted</b>: visible but with breaks or irregular loss. <b>Absent</b>: not visible where it should be.</li>
<li>Use <b>Ungradable</b> if image quality does not allow a judgement. Click an image to enlarge it.</li>
<li>Your progress is saved in this browser. When finished, click <b>Download results (CSV)</b> and send the file back.</li></ul>` :
`<h2>How to review</h2><p>Each B-scan is shown twice: the <b>original</b> on the left and the <b>annotation</b> on the right.</p>
<p><span class="key"><i style="background:var(--elm)"></i>ELM (external limiting membrane)</span><span class="key"><i style="background:var(--ez)"></i>EZ (ellipsoid zone)</span></p>
<ul><li>Gaps in a line mean the grader judged the layer to be disrupted there.</li>
<li>Please check whether each line follows the correct layer, and whether you agree with the integrity grade. If you disagree, choose the grade you would give.</li>
<li>Click an image to enlarge both views; use + / − to zoom. Your progress is saved in this browser. When finished, click <b>Download results (CSV)</b> and send the file back.</li></ul>`;
if(!META.blind){$("#tl").hidden=false;$("#mlines").hidden=false}else{$("#fflag").textContent="Ungradable"}

/* ---------- drawing ---------- */
const imgs={};
function getImg(d){if(!imgs[d.id]){const i=new Image();i.src=d.img;imgs[d.id]=i}return imgs[d.id]}
function draw(cv,d,scale=1){
  const im=getImg(d);
  const go=()=>{cv.width=d.w*scale;cv.height=d.h*scale;const x=cv.getContext("2d");
    x.drawImage(im,0,0,cv.width,cv.height);
    if(!showLines||!d.lines)return;
    x.lineWidth=Math.max(2,d.w/300)*scale;x.lineJoin="round";x.lineCap="round";
    for(const [k,c] of [["elm","#e0389f"],["ez","#16a3cf"]]){
      const L=d.lines[k];x.strokeStyle=c;x.beginPath();let on=false;
      for(let i=0;i<L.length;i++){const v=L[i];if(v===null){on=false;continue}
        const px=(i+.5)*scale,py=v*scale;if(!on){x.moveTo(px,py);on=true}else x.lineTo(px,py)}
      x.stroke()}};
  if(im.complete&&im.naturalWidth)go();else im.onload=go;
}

/* ---------- review state ---------- */
function rec(id){return R[id]||(R[id]={})}
function isDone(d){const r=R[d.id];if(!r)return false;
  if(META.blind)return r.ungradable||(r.elmG&&r.ezG);
  const g=k=>r[k+"G"]==="agree"||(r[k+"G"]==="disagree"&&r[k+"New"]);
  return r.elmT&&r.ezT&&g("elm")&&g("ez")}
function isFlag(d){const r=R[d.id];if(!r)return false;
  if(META.blind)return !!r.ungradable;
  return r.elmT==="fix"||r.ezT==="fix"||r.elmG==="disagree"||r.ezG==="disagree"}

/* ---------- cards ---------- */
function seg(name,opts,cur){return `<div class="seg" data-k="${name}">${opts.map(([v,l,cls])=>
  `<button data-v="${v}" class="${cls}" aria-pressed="${cur===v}">${l}</button>`).join("")}</div>`}
function controls(d){const r=R[d.id]||{};
  if(META.blind){return `
   <div class="q"><span>ELM grade</span>${seg("elmG",GR.map(g=>[g,g,"pick"]),r.elmG)}</div>
   <div class="q"><span>EZ grade</span>${seg("ezG",GR.map(g=>[g,g,"pick"]),r.ezG)}</div>
   <div class="q"><span>Image quality</span>${seg("ungradable",[[true,"Ungradable","bad"]],r.ungradable)}</div>
   <div class="q full"><span>Comment (optional)</span><textarea data-k="note">${esc(r.note||"")}</textarea></div>`}
  const gq=(k,lab,g)=>`<div class="q"><span>${lab} grade: <b>${esc(g||"not graded")}</b></span>
     ${seg(k+"G",[["agree","Agree","good"],["disagree","Disagree","bad"]],r[k+"G"])}
     <div class="sub ${r[k+"G"]==="disagree"?"show":""}" data-sub="${k}">${seg(k+"New",GR.filter(x=>x!==g).map(x=>[x,"Should be "+x,"pick"]),r[k+"New"])}</div></div>`;
  return `
   <div class="q"><span>Is the <b style="color:var(--elm)">ELM line</b> on the correct layer?</span>${seg("elmT",[["ok","Correct","good"],["fix","Needs correction","bad"]],r.elmT)}</div>
   <div class="q"><span>Is the <b style="color:var(--ez)">EZ line</b> on the correct layer?</span>${seg("ezT",[["ok","Correct","good"],["fix","Needs correction","bad"]],r.ezT)}</div>
   ${gq("elm","ELM",d.elm)}${gq("ez","EZ",d.ez)}
   <div class="q full"><span>Comment (optional), e.g. where the line is wrong</span><textarea data-k="note">${esc(r.note||"")}</textarea></div>`}
function status(d){return isDone(d)?(isFlag(d)?["bad",META.blind?"Ungradable":"Reviewed: needs correction"]:["ok","Reviewed"]):["todo","Not reviewed"]}

const list=$("#list");const cards={};
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const c=e.target;
  if(!c.dataset.drawn){const d=DATA[+c.dataset.i];const im=c.querySelector("img");im.src=d.img;
   const cv=c.querySelector("canvas");if(cv)draw(cv,d);c.dataset.drawn=1}}}),{rootMargin:"600px"});
DATA.forEach((d,i)=>{
  const c=document.createElement("article");c.className="card";c.dataset.i=i;
  const cov=META.blind?"":`<span class="tag">ELM coverage ${d.covElm}%</span><span class="tag">EZ coverage ${d.covEz}%</span>`;
  const label=META.blind?`Case ${i+1}`:esc(d.id), clsTag=META.blind?"":`<span class="tag">${esc(d.cls)}</span>`;
  c.innerHTML=`<div class="chead"><span class="num">${i+1} / ${DATA.length}</span><span class="fname">${label}</span>
    ${clsTag}${cov}<span class="status"></span></div>
    <div class="pics ${META.blind?"single":""}"><figure><figcaption>Original</figcaption><img alt="Original B-scan ${label}" width="${d.w}" height="${d.h}"></figure>
    ${META.blind?"":`<figure><figcaption>Annotation (pink = ELM, blue = EZ)</figcaption><canvas width="${d.w}" height="${d.h}"></canvas></figure>`}</div>
    <div class="ctrl">${controls(d)}</div>`;
  c.querySelector(".pics").addEventListener("click",()=>openModal(i));
  c.querySelector(".ctrl").addEventListener("click",e=>{const b=e.target.closest(".seg button");if(!b)return;
    const k=b.parentElement.dataset.k;let v=b.dataset.v;if(v==="true")v=true;const r=rec(d.id);
    r[k]=(r[k]===v)?undefined:v;
    if(k.endsWith("G")&&r[k]!=="disagree"&&!META.blind)delete r[k.slice(0,-1)+"New"];
    r.at=new Date().toISOString();save();refreshCard(i)});
  c.querySelector("textarea").addEventListener("input",e=>{const r=rec(d.id);r.note=e.target.value;r.at=new Date().toISOString();save()});
  list.appendChild(c);cards[i]=c;io.observe(c);refreshCard(i,true)});

function refreshCard(i,first){const d=DATA[i],c=cards[i],r=R[d.id]||{};
  c.querySelectorAll(".seg").forEach(s=>{const k=s.dataset.k;s.querySelectorAll("button").forEach(b=>{
    let v=b.dataset.v;if(v==="true")v=true;b.setAttribute("aria-pressed",String(r[k]===v))})});
  c.querySelectorAll(".sub").forEach(s=>s.classList.toggle("show",r[s.dataset.sub+"G"]==="disagree"));
  const [cls,txt]=status(d);const st=c.querySelector(".status");st.className="status "+cls;st.textContent=txt;
  c.classList.toggle("done",isDone(d)&&!isFlag(d));c.classList.toggle("flag",isDone(d)&&isFlag(d));
  if(!first){applyFilter();progress()}}
function applyFilter(){DATA.forEach((d,i)=>{const show=filter==="all"||(filter==="todo"&&!isDone(d))||(filter==="flag"&&isFlag(d));
  cards[i].style.display=show?"":"none"});
  let e=$("#emptymsg");const any=DATA.some((d,i)=>cards[i].style.display!=="none");
  if(!any&&!e){e=document.createElement("p");e.id="emptymsg";e.className="empty";e.textContent="No images in this view.";list.appendChild(e)}
  if(any&&e)e.remove()}
function progress(){const n=DATA.filter(isDone).length;$("#pt").textContent=`${n} / ${DATA.length} reviewed`;$("#pb").style.width=(100*n/DATA.length)+"%"}
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{filter=b.dataset.f;
  document.querySelectorAll(".filters button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));applyFilter()});
$("#rev").addEventListener("input",save);
$("#tl").onclick=()=>{showLines=!showLines;$("#tl").textContent=showLines?"Hide lines":"Show lines";
  $("#mlines").textContent=$("#tl").textContent;
  Object.values(cards).forEach(c=>{if(c.dataset.drawn){const cv=c.querySelector("canvas");if(cv)draw(cv,DATA[+c.dataset.i])}});
  if(mi>=0)renderModal()};

/* ---------- enlarge ---------- */
let mi=-1,mz=1;
function openModal(i){mi=i;mz=1;$("#modal").classList.add("show");renderModal()}
function renderModal(){const d=DATA[mi];$("#mt").textContent=META.blind?`Case ${mi+1}`:`${d.id} · ${d.cls}`;const p=$("#mp");p.className="mpics"+(META.blind?" single":"");
  const box=p.getBoundingClientRect();const rows=META.blind?1:2;
  const fitH=(box.height-10*(rows-1))/rows-4, fitW=box.width-4;
  const s=Math.max(.3,Math.min(fitW/d.w,fitH/d.h)||1)*mz;
  p.innerHTML=`<div><img src="${d.img}" style="width:${d.w*s}px" alt="Original" title="Original"></div>`+(META.blind?"":`<div><canvas></canvas></div>`);
  if(!META.blind)draw(p.querySelector("canvas"),d,s)}
$("#mclose").onclick=()=>{$("#modal").classList.remove("show");mi=-1};
$("#mzi").onclick=()=>{mz*=1.3;renderModal()};$("#mzo").onclick=()=>{mz/=1.3;renderModal()};
$("#mlines").onclick=()=>$("#tl").click();
document.addEventListener("keydown",e=>{if(mi<0)return;if(e.key==="Escape")$("#mclose").click();
  if(e.key==="+"||e.key==="=")$("#mzi").click();if(e.key==="-")$("#mzo").click();
  if(e.key==="ArrowRight"&&mi<DATA.length-1){mi++;mz=1;renderModal()}if(e.key==="ArrowLeft"&&mi>0){mi--;mz=1;renderModal()}});

/* ---------- export ---------- */
const cell=v=>{v=v==null?"":String(v);return /[",\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v};
$("#dl").onclick=()=>{const rev=$("#rev").value.trim();
  if(!rev){toast("Please enter your name or initials first.");$("#rev").focus();return}
  let rows;
  if(META.blind){rows=[["filename","disease_class","reviewer_elm_grade","reviewer_ez_grade","ungradable","comment","reviewer","reviewed_at"]];
    DATA.forEach(d=>{const r=R[d.id]||{};rows.push([d.id,d.cls,r.elmG||"",r.ezG||"",r.ungradable?"Yes":"No",r.note||"",rev,r.at||""])})}
  else{rows=[["filename","disease_class","grader_elm_grade","grader_ez_grade","elm_line","ez_line","elm_grade_review","reviewer_elm_grade","ez_grade_review","reviewer_ez_grade","comment","reviewer","reviewed_at"]];
    const fin=(r,k,g)=>r[k+"G"]==="agree"?g:(r[k+"New"]||"");
    DATA.forEach(d=>{const r=R[d.id]||{};rows.push([d.id,d.cls,d.elm,d.ez,
      {ok:"Correct",fix:"Needs correction"}[r.elmT]||"",{ok:"Correct",fix:"Needs correction"}[r.ezT]||"",
      {agree:"Agree",disagree:"Disagree"}[r.elmG]||"",fin(r,"elm",d.elm),{agree:"Agree",disagree:"Disagree"}[r.ezG]||"",fin(r,"ez",d.ez),
      r.note||"",rev,r.at||""])})}
  const csv="﻿"+rows.map(r=>r.map(cell).join(",")).join("\r\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
  a.download=`${META.blind?"independent_grading":"annotation_review"}_${rev.replace(/\W+/g,"_")}_${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(a);a.click();a.remove();
  const n=DATA.filter(isDone).length;toast(n<DATA.length?`Saved. ${DATA.length-n} images are not reviewed yet.`:"Saved. Thank you!")};
applyFilter();progress();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
