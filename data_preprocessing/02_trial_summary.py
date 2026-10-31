"""Step 2 — one-row-per-trial gaze sequences, plots and pricing-endpoint QC.

Read data/raw_use and MATLAB-decoded exports in outputs/data_preprocessing.
Gaze runs are AOI dwell episodes, NOT physiological velocity-defined fixations.
Times are relative to the first Tobii sample (not synchronized with GetSecs).
No trials are excluded automatically; endpoint gaze CANNOT prove mouse clicks.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc, Circle
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
POSITION = {1:"MRD",2:"MDR",3:"RDM",4:"RMD",5:"DMR",6:"DRM"}
XYZ = {1:"MRD",2:"MDR",3:"RMD",4:"RDM",5:"DMR",6:"DRM"}
NUM = {"M":1,"R":2,"D":3}
NAME = {1:"money",2:"rating",3:"distance"}
COLORS = {-1:"#999999",0:"#d49a37",1:"#3468af",2:"#c74c50",3:"#23976a",4:"#9d46ad"}
FIELDS = ("choice","rt","t0","t1","w1","w3","o1m","o1r","o1d","o2m","o2r","o2d")


def numerical(v):
    try:
        a=float(v)
        return a if math.isfinite(a) else np.nan
    except (TypeError,ValueError):
        return np.nan


def jarray(v):
    return json.dumps([None if not np.isfinite(x) else round(float(x),5) for x in v],separators=(",",":"))


def geometry(xc,yc):
    L=950.
    centers=((xc,yc-math.cos(math.pi/6)*L/2),
             (xc-L/2,yc+math.tan(math.pi/6)*L/2),
             (xc+L/2,yc+math.tan(math.pi/6)*L/2))
    return centers,400.,400.*356./787.


def arc_geometry(xc,yc):
    # main_price.m: L_arc=400, img_arc is 1800x900, up_arc=10
    return xc,yc+90.,100.,200.


def arc_area(x,y,xc,yc):
    cx,cy,inner,outer=arc_geometry(xc,yc)
    r=math.hypot(x-cx,y-cy)
    return inner-20<=r<=outer+20 and y<=cy+20


def endpoint_side(x,y,xc,yc,tolerance):
    if not (np.isfinite(x) and np.isfinite(y)):return 0
    cx,cy,inner,outer=arc_geometry(xc,yc)
    nearest=(float("inf"),0)
    for side in (-1,1):
        v=np.array((side*math.sin(math.radians(72)),-math.cos(math.radians(72))))
        p=np.array((x-cx,y-cy))
        r=np.clip(np.dot(p,v),inner,outer)
        dist=float(np.linalg.norm(p-r*v))
        if dist<nearest[0]:nearest=(dist,side)
    return nearest[1] if nearest[0]<=tolerance else 0


def classify(x,y,xc,yc,mode,pad,pricing):
    if not (np.isfinite(x) and np.isfinite(y) and 0<=x<=1 and 0<=y<=1):
        return -1
    width,height=2*xc,2*yc
    xx,yy=x*width,y*height
    if pricing and arc_area(xx,yy,xc,yc):return 4  # arc takes priority over bottom-left/right
    if mode=="legacy":
        if y<.35 and .3<x<.7:return 1
        if .5<y<1 and x<.5:return 2
        if .5<y<1 and x>.5:return 3
        if .35<y<.6 and .3<x<.7:return 0
        return -1
    centers,rw,rh=geometry(xc,yc)
    for i,(cx,cy) in enumerate(centers,1):
        if abs(xx-cx)<=rw/2+pad and abs(yy-cy)<=rh/2+pad:return i
    if abs(xx-xc)<130 and abs(yy-yc)<95:return 0
    return -1


def make_visits(gaze,codes,max_gap):
    if gaze.empty:return []
    times=gaze.rel_ms.to_numpy(float)
    durations=gaze.duration_ms.to_numpy(float)
    xs=gaze.selected_x_norm.to_numpy(float)
    ys=gaze.selected_y_norm.to_numpy(float)
    runs=[];start=0
    for end in range(1,len(times)+1):
        if end<len(times) and codes[end]==codes[end-1] and 0<=times[end]-times[end-1]<=max_gap:
            continue
        sl=slice(start,end)
        xx=xs[sl];yy=ys[sl]
        runs.append(dict(code=int(codes[start]),start_ms=float(times[start]),
              end_ms=float(times[end-1]+durations[end-1]),
              duration_ms=float(np.sum(durations[sl])),
              mean_x=float(np.nanmean(xx)) if np.isfinite(xx).any() else np.nan,
              mean_y=float(np.nanmean(yy)) if np.isfinite(yy).any() else np.nan,
              n_samples=end-start))
        start=end
    return runs


def subset_visits(runs,exclude):
    result=[]
    for v in runs:
        if v["code"] in exclude:continue
        if result and result[-1]["code"]==v["code"]:
            a=result[-1];n0=a["n_samples"];n1=v["n_samples"]
            for key in ("mean_x","mean_y"):
                p,q=a[key],v[key]
                if np.isfinite(q):
                    a[key]=((p*n0+q*n1)/(n0+n1)) if np.isfinite(p) else q
            a["n_samples"]+=n1;a["duration_ms"]+=v["duration_ms"]
            a["end_ms"]=v["end_ms"]
        else:result.append(v.copy())
    return result


def save_sequence(row,prefix,runs,pos,xyz):
    attrs=[NUM[pos[r["code"]-1]] if r["code"] in (1,2,3) else 0 for r in runs]
    axes=["XYZ"[xyz.index(pos[r["code"]-1])] if r["code"] in (1,2,3) else "" for r in runs]
    row.update({
        prefix+"_pos":json.dumps([r["code"] for r in runs]),
        prefix+"_attribute_code":json.dumps(attrs),
        prefix+"_attribute_label":json.dumps([NAME.get(a,"") for a in attrs]),
        prefix+"_xyz":json.dumps(axes),
        prefix+"_duration_ms":jarray([r["duration_ms"] for r in runs]),
        prefix+"_start_ms":jarray([r["start_ms"] for r in runs]),
        prefix+"_end_ms":jarray([r["end_ms"] for r in runs]),
        prefix+"_mean_x_norm":jarray([r["mean_x"] for r in runs]),
        prefix+"_mean_y_norm":jarray([r["mean_y"] for r in runs]),
        prefix+"_n_episodes":len(runs)
    })


def endpoint_qc(gaze,xc,yc,args):
    names=("endpoint_start_ms","endpoint_end_ms","endpoint_start_timestamp_us",
           "endpoint_end_timestamp_us","endpoint_duration_ms",
           "endpoint_span_ms","endpoint_gap_before_end_ms")
    result={k:np.nan for k in names}
    result.update(endpoint_candidate_flag=0,endpoint_side="",endpoint_pattern="none",
                  endpoint_n_visits=0,endpoint_n_valid_visits=0,
                  endpoint_n_samples=0,endpoint_visits_json="[]")
    if gaze.empty:return result
    t=gaze.rel_ms.to_numpy(float)
    duration=gaze.duration_ms.to_numpy(float)
    ts=gaze.system_timestamp_us.to_numpy(float)
    sides=gaze.endpoint_side.to_numpy(int)
    final=t[-1]+duration[-1]
    ix=np.flatnonzero((t>=final-args.endpoint_window_ms)&(sides!=0))
    if len(ix)==0:return result
    starts=[];a=int(ix[0]);previous=a
    for index in ix[1:]:
        index=int(index)
        if index!=previous+1 or sides[index]!=sides[previous] or t[index]-t[previous]>100:
            starts.append((a,previous,int(sides[previous])))
            a=index
        previous=index
    starts.append((a,previous,int(sides[previous])))
    visits=[]
    for a,b,side in starts:
        visits.append({
            "side":"right" if side==1 else "left",
            "start_ms":round(t[a],4),"end_ms":round(t[b]+duration[b],4),
            "start_timestamp_us":int(ts[a]),"end_timestamp_us":int(ts[b]),
            "duration_ms":round(float(duration[a:b+1].sum()),4),
            "n_samples":b-a+1,
            "ms_before_last_gaze":round(float(final-t[b]-duration[b]),4)})
    result["endpoint_visits_json"]=json.dumps(visits,separators=(",",":"))
    result["endpoint_n_visits"]=len(visits)
    good=[v for v in visits if v["duration_ms"]>=args.endpoint_min_visit_ms
          and v["n_samples"]>=args.endpoint_min_samples]
    result["endpoint_n_valid_visits"]=len(good)
    if not good:return result
    last=good[-1];side=last["side"];same=[v for v in good if v["side"]==side]
    recent=last["ms_before_last_gaze"]<=args.endpoint_max_end_gap_ms
    repeat=len(same)>=2
    sustained=last["duration_ms"]>=args.endpoint_sustained_ms
    flag=bool(recent and (repeat or sustained))
    result.update({
        "endpoint_candidate_flag":int(flag),
        "endpoint_side":side,
        "endpoint_pattern":"repeated_visits" if flag and repeat else ("sustained" if flag else "short_or_early"),
        "endpoint_start_ms":same[0]["start_ms"],"endpoint_end_ms":last["end_ms"],
        "endpoint_start_timestamp_us":same[0]["start_timestamp_us"],
        "endpoint_end_timestamp_us":last["end_timestamp_us"],
        "endpoint_duration_ms":round(sum(v["duration_ms"] for v in same),4),
        "endpoint_span_ms":round(last["end_ms"]-same[0]["start_ms"],4),
        "endpoint_gap_before_end_ms":last["ms_before_last_gaze"],
        "endpoint_n_samples":sum(v["n_samples"] for v in same)
    })
    return result


def one_trial(meta,behavior,raw,D,args):
    xc=numerical(meta.screen_center_x_px);yc=numerical(meta.screen_center_y_px)
    pos=POSITION[int(meta.cond_pos)]; xyz=XYZ[int(meta.xyz_map)]
    pricing=meta.response_type=="pricing"
    trial=int(behavior["trial"])
    row=dict(subject_id=str(meta.subject_id),session_order=int(meta.session_order),
             source_file=meta.source_file,response_type=meta.response_type,
             trial=trial,node1=int(meta.node1_importance_code),
             node2=int(meta.node2_correlation_code),
             importance=meta.importance,correlation=meta.correlation,
             xyz_map=int(meta.xyz_map),cond_pos=int(meta.cond_pos),
             xyz_to_attribute=xyz,pos_to_attribute=pos,aoi_mode=args.aoi_mode)
    row.update({("rt_seconds" if col=="rt" else col):numerical(behavior[col]) if col in behavior else np.nan for col in FIELDS})
    # PTB_v4/main_price.m stores preference on a signed -100..+100 scale.
    # Use the explicitly declared source scale, never infer it from each trial.
    response=row["choice"]
    score100=(100.0*response/args.price_scale_max) if pricing and np.isfinite(response) else np.nan
    absolute_score100=abs(score100) if np.isfinite(score100) else np.nan
    eligible=(pricing and np.isfinite(absolute_score100)
              and absolute_score100>args.endpoint_min_abs_price_100
              and absolute_score100<=100.0)
    row["price_response_100"]=score100
    row["price_abs_100"]=absolute_score100
    row["endpoint_detection_eligible"]=int(eligible)
    row["endpoint_detection_status"]=(
        "not_pricing" if not pricing else
        "missing_preference" if not np.isfinite(absolute_score100) else
        "invalid_preference_over_100" if absolute_score100>100 else
        "eligible" if eligible else "skipped_abs_preference_at_most_threshold"
    )
    for i,letter in enumerate("XYZ"):
        row["D_"+letter]=float(D[trial-1,i]) if 1<=trial<=len(D) else np.nan
    if raw is None:
        gaze=pd.DataFrame(columns=("system_timestamp_us","selected_x_norm","selected_y_norm","eye_fallback_used"))
    else:
        gaze=raw.copy()
    if not gaze.empty:
        for col in ("system_timestamp_us","selected_x_norm","selected_y_norm","eye_fallback_used"):
            gaze[col]=pd.to_numeric(gaze[col],errors="coerce")
        gaze=gaze[np.isfinite(gaze.system_timestamp_us)].sort_values("system_timestamp_us").drop_duplicates("system_timestamp_us").reset_index(drop=True)
    if not gaze.empty:
        ts=gaze.system_timestamp_us.to_numpy(float)
        t=(ts-ts[0])/1000.
        differences=np.diff(ts)/1000.
        good=differences[(differences>0)&(differences<=args.max_gap_ms)]
        last=float(np.median(good)) if len(good) else 0.
        duration=np.r_[np.clip(differences,0,args.max_gap_ms),last]
        gaze["rel_ms"]=t;gaze["duration_ms"]=duration
        gaze["aoi_code"]=[classify(x,y,xc,yc,args.aoi_mode,args.aoi_padding_px,pricing)
                          for x,y in zip(gaze.selected_x_norm,gaze.selected_y_norm)]
        if eligible:
            gaze["endpoint_side"]=[endpoint_side(x*2*xc,y*2*yc,xc,yc,args.endpoint_radius_px)
                        for x,y in zip(gaze.selected_x_norm,gaze.selected_y_norm)]
        else:gaze["endpoint_side"]=0
    else:
        gaze["rel_ms"]=pd.Series(dtype=float);gaze["duration_ms"]=pd.Series(dtype=float)
        gaze["aoi_code"]=pd.Series(dtype=int);gaze["endpoint_side"]=pd.Series(dtype=int)
    codes=gaze.aoi_code.to_numpy(int)
    smooth=codes.copy()
    if args.smooth_one_frame and len(smooth)>2:
        for i in range(1,len(codes)-1):
            if codes[i-1]==codes[i+1]:smooth[i]=codes[i-1]
    raw_runs=make_visits(gaze,smooth,args.max_gap_ms)
    pure1=subset_visits(raw_runs,{-1})
    pure2=subset_visits(raw_runs,{-1,0,4})
    for prefix,vs in (("arr2",raw_runs),("pure1",pure1),("pure2",pure2)):
        save_sequence(row,prefix,vs,pos,xyz)
    row.update(n_gaze_samples=len(gaze),n_attribute_visits=len(pure2),
               n_attribute_samples=int(np.sum((codes>=1)&(codes<=3))),
               n_valid_gaze_samples=int(np.sum(codes!=-1)),
               n_arc_samples=int(np.sum(codes==4)),
               n_fallback_eye_samples=int(gaze.eye_fallback_used.sum()) if len(gaze) else 0,
               gaze_recorded_span_ms=float(gaze.rel_ms.iloc[-1]) if len(gaze) else np.nan,
               gaze_covered_duration_ms=float(gaze.duration_ms.sum()) if len(gaze) else 0,
               gaze_valid_fraction=float(np.mean(codes!=-1)) if len(codes) else np.nan)
    row["first_attribute"]=NAME[NUM[pos[pure2[0]["code"]-1]]] if pure2 else ""
    row["first_xyz"]="XYZ"[xyz.index(pos[pure2[0]["code"]-1])] if pure2 else ""
    for attr in "MRD":
        row["du_"+NAME[NUM[attr]]+"_ms"]=sum(v["duration_ms"] for v in raw_runs
                           if v["code"] in (1,2,3) and pos[v["code"]-1]==attr)
    for letter in "XYZ":
        row["du_"+letter+"_ms"]=sum(v["duration_ms"] for v in raw_runs
                   if v["code"] in (1,2,3) and "XYZ"[xyz.index(pos[v["code"]-1])]==letter)
    for code,name in ((0,"center"),(4,"arc"),(-1,"other")):
        row["du_"+name+"_ms"]=sum(v["duration_ms"] for v in raw_runs if v["code"]==code)
    trans=Counter(tuple(sorted((a["code"],b["code"]))) for a,b in zip(pure2,pure2[1:]))
    row["n_attribute_transitions"]=sum(trans.values())
    for i in (1,2,3):
        for j in range(i+1,4):
            row["trans_"+NAME[NUM[pos[i-1]]]+"_"+NAME[NUM[pos[j-1]]]]=trans.get((i,j),0)
    # Endpoint screening is run ONLY for pricing responses with |score100| > threshold.
    row.update(endpoint_qc(gaze if eligible else pd.DataFrame(),xc,yc,args))
    if not eligible:
        row["endpoint_pattern"]="not_screened"
    row["endpoint_response_abs"]=abs(row["choice"]) if np.isfinite(row["choice"]) else np.nan
    row["endpoint_response_near_100"]=int(eligible)
    row["endpoint_interpretation"]="near_end_gaze_only_not_mouse_clicks"
    return row,gaze,raw_runs


def background(ax,meta):
    xc=float(meta.screen_center_x_px);yc=float(meta.screen_center_y_px)
    pos=POSITION[int(meta.cond_pos)];xyz=XYZ[int(meta.xyz_map)]
    centers,rw,rh=geometry(xc,yc)
    for (cx,cy),letter in zip(centers,pos):
        col=COLORS[NUM[letter]]
        ax.add_patch(Rectangle((cx-rw/2,cy-rh/2),rw,rh,fill=False,lw=.8,ec=col))
        ax.text(cx,cy,NAME[NUM[letter]]+" / "+"XYZ"[xyz.index(letter)],
                fontsize=5.5,ha="center",va="center",color=col)
    ax.plot(xc,yc,"k+",ms=3,mew=.7)
    if meta.response_type=="pricing":
        cx,cy,rmin,rmax=arc_geometry(xc,yc)
        for rad in (rmin,rmax):
            ax.add_patch(Arc((cx,cy),2*rad,2*rad,theta1=198,theta2=342,
                     lw=.65,linestyle="--",ec="#888888"))
        for side in (-1,1):
            v=np.array((side*math.sin(math.radians(72)),-math.cos(math.radians(72))))
            a=np.array((cx,cy))+rmin*v;b=np.array((cx,cy))+rmax*v
            ax.plot((a[0],b[0]),(a[1],b[1]),color="#aa3030",lw=.8)
            mid=(a+b)/2
            ax.text(mid[0],mid[1]+12,"-100" if side<0 else "+100",
                    ha="center",color="#aa3030",fontsize=5)
    ax.set_xlim(0,2*xc);ax.set_ylim(2*yc,0)
    ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])


def draw(ax,meta,row,gaze,runs,detail=False):
    background(ax,meta)
    if not gaze.empty:
        x=gaze.selected_x_norm.to_numpy(float)*2*float(meta.screen_center_x_px)
        y=gaze.selected_y_norm.to_numpy(float)*2*float(meta.screen_center_y_px)
        code=gaze.aoi_code.to_numpy(int)
        for c in (-1,0,1,2,3,4):
            mask=(code==c)&np.isfinite(x)&np.isfinite(y)
            if mask.any():
                ax.scatter(x[mask],y[mask],s=5 if detail else 1,
                           c=COLORS[c],alpha=.7,linewidths=0,rasterized=True)
        if meta.response_type=="pricing":
            near=gaze.endpoint_side.to_numpy(int)!=0
            if near.any():ax.scatter(x[near],y[near],s=12 if detail else 3,
                       facecolor="none",edgecolor="#a000b6",linewidth=.55)
    if detail:
        vs=[v for v in runs if v["code"] in (1,2,3) and np.isfinite(v["mean_x"]) and np.isfinite(v["mean_y"])]
        xc=float(meta.screen_center_x_px);yc=float(meta.screen_center_y_px)
        for i,v in enumerate(vs):
            x=v["mean_x"]*2*xc;y=v["mean_y"]*2*yc
            ax.add_patch(Circle((x,y),max(6,min(35,v["duration_ms"]*.035)),
                                fill=False,linewidth=1,ec=COLORS[v["code"]]))
            ax.text(x+4,y-6,str(i+1),fontsize=7)
        for a,b in zip(vs,vs[1:]):
            ax.annotate("",xy=(b["mean_x"]*2*xc,b["mean_y"]*2*yc),
               xytext=(a["mean_x"]*2*xc,a["mean_y"]*2*yc),
               arrowprops=dict(arrowstyle="->",lw=.65,color="#555",alpha=.6))
    resp=numerical(row["choice"])
    suffix=" [ENDPOINT?]" if row["endpoint_candidate_flag"] else ""
    text=f"T{row['trial']} {len(gaze)} samples | resp={resp:g}{suffix}" if np.isfinite(resp) else f"T{row['trial']} {len(gaze)} samples{suffix}"
    ax.set_title(text,fontsize=7 if detail else 5.4,pad=1.5,
                 color="#a000b6" if row["endpoint_candidate_flag"] else "black")


def overview(meta,items,args,out):
    cols=args.plot_columns;rows=math.ceil(len(items)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(2.7*cols,2.25*rows),squeeze=False)
    for ax,(row,gaze,runs) in zip(axes.flat,items):
        draw(ax,meta,row,gaze,runs)
    for ax in axes.flat[len(items):]:ax.axis("off")
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker="o",linestyle="",markersize=4,
                    color=COLORS[c],label=LABEL) for c,LABEL in
                    [(-1,"other"),(0,"center"),(1,"money"),(2,"rating"),(3,"distance"),(4,"arc")]]
    handles.append(Line2D([],[],marker="o",markerfacecolor="none",
                    linestyle="",color="#a000b6",label="near ±100 endpoint"))
    fig.legend(handles=handles,loc="upper center",ncol=7,
               fontsize=8,frameon=False,bbox_to_anchor=(.5,.978))
    fig.suptitle(f"Subject {meta.subject_id} | session {meta.session_order} | {meta.response_type} | node1={meta.node1_importance_code}, node2={meta.node2_correlation_code} | AOI={args.aoi_mode}",fontsize=11,y=.999)
    fig.subplots_adjust(left=.015,right=.99,bottom=.02,top=.925,wspace=.13,hspace=.32)
    out.mkdir(parents=True,exist_ok=True)
    fn=out/"overview.png"
    fig.savefig(fn,dpi=args.plot_dpi);plt.close(fig)
    print("  saved",fn)


def detail_plot(meta,row,gaze,runs,args,out):
    fig,ax=plt.subplots(figsize=(11,7))
    draw(ax,meta,row,gaze,runs,detail=True)
    text=f"endpoint={row['endpoint_pattern']} / {row['endpoint_side']} | visits={row['endpoint_n_visits']} | start={row['endpoint_start_ms']} ms, end={row['endpoint_end_ms']} ms, gap={row['endpoint_gap_before_end_ms']} ms"
    fig.text(.035,.04,text,fontsize=10)
    fig.tight_layout(rect=(0,.075,1,1))
    out.mkdir(parents=True,exist_ok=True)
    name=out/f"trial_{int(row['trial']):02d}.png"
    fig.savefig(name,dpi=args.plot_dpi);plt.close(fig)


def migrate_flat_figures(plots, sessions):
    """Preserve existing flat PNGs within a per-session legacy subfolder."""
    import re
    mapping={(str(row.subject_id),int(row.session_order)):str(row.response_type)
             for _,row in sessions.iterrows()}
    pattern=re.compile(
        r"^subject_(.+)_session_(\d+)_(?:(choice|pricing)_all_trials|trial_(\d+))\.png$"
    )
    count=0
    for old in plots.glob("*.png"):
        match=pattern.fullmatch(old.name)
        if not match:
            continue
        subject,order,task,_=match.groups()
        task=task or mapping.get((subject,int(order)))
        if not task:
            continue
        destination=(plots/f"subject_{subject}"/f"session_{int(order):02d}_{task}"/
                     "legacy"/old.name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            print(f"[WARNING] Flat image not migrated (destination exists): {old}")
            continue
        old.rename(destination)
        count+=1
    if count:
        print(f"Reorganized {count} previous flat PNGs into per-session legacy folders.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"data"/"raw_use")
    p.add_argument("--session-summary",type=Path,default=ROOT/"outputs"/"data_preprocessing"/"session_summary.csv")
    p.add_argument("--exports",type=Path,default=ROOT/"outputs"/"data_preprocessing"/"matlab_exports")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs"/"data_preprocessing")
    p.add_argument("--subject",type=str)
    p.add_argument("--max-sessions",type=int,default=0)
    p.add_argument("--aoi-mode",choices=("rectangle","legacy"),default="rectangle")
    p.add_argument("--aoi-padding-px",type=float,default=35)
    p.add_argument("--max-gap-ms",type=float,default=75)
    p.add_argument("--smooth-one-frame",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--endpoint-window-ms",type=float,default=1200)
    p.add_argument("--price-scale-max",type=float,default=100,
                   help="Positive maximum absolute preference in input (PTB_v4: 100).")
    p.add_argument("--endpoint-min-abs-price-100",type=float,default=90,
                   help="Only screen pricing trials with absolute preference on 100-point scale STRICTLY greater than this.")
    p.add_argument("--endpoint-radius-px",type=float,default=60)
    p.add_argument("--endpoint-min-visit-ms",type=float,default=80)
    p.add_argument("--endpoint-min-samples",type=int,default=5)
    p.add_argument("--endpoint-sustained-ms",type=float,default=300)
    p.add_argument("--endpoint-max-end-gap-ms",type=float,default=250)
    p.add_argument("--plot-columns",type=int,default=8)
    p.add_argument("--plot-dpi",type=int,default=130)
    p.add_argument("--details",choices=("flagged","all","none"),default="flagged")
    p.add_argument("--no-overviews",action="store_true")
    args=p.parse_args()
    if not np.isfinite(args.price_scale_max) or args.price_scale_max<=0:
        p.error("--price-scale-max must be positive.")
    if not 0<=args.endpoint_min_abs_price_100<=100:
        p.error("--endpoint-min-abs-price-100 must be between 0 and 100.")
    out=args.output_dir.resolve()
    if not out.is_relative_to((ROOT/"outputs").resolve()):
        p.error("All generated files must be under repository outputs/")
    if not args.session_summary.is_file():p.error("Run step 1 first")
    sessions=pd.read_csv(args.session_summary,dtype={"subject_id":str},keep_default_na=False)
    all_sessions=sessions.copy()
    if args.subject:sessions=sessions[sessions.subject_id==args.subject]
    sessions=sessions.sort_values(by=["subject_id","session_order"])
    if args.max_sessions:sessions=sessions.head(args.max_sessions)
    if sessions.empty:p.error("No sessions to process")
    out.mkdir(parents=True,exist_ok=True)
    plots=out/"gaze_qc";plots.mkdir(exist_ok=True)
    migrate_flat_figures(plots,all_sessions)
    rows=[]
    session_qc_rows=[]
    for _,meta in sessions.iterrows():
        stem=Path(meta.source_file).stem
        behavior_path=args.exports/(stem+"_behavior.csv")
        gaze_path=args.exports/(stem+"_gaze.csv")
        raw_mat=args.input/Path(meta.source_file)
        for path in (behavior_path,gaze_path,raw_mat):
            if not path.is_file():p.error(f"Missing {path}; run MATLAB exporter first")
        behavior=pd.read_csv(behavior_path)
        gaze=pd.read_csv(gaze_path)
        D=np.asarray(loadmat(raw_mat,variable_names=["infos"],squeeze_me=True,struct_as_record=False)["infos"].D,float)
        grouped={int(k):v for k,v in gaze.groupby("trial",sort=False)} if not gaze.empty else {}
        items=[]
        for _,b in behavior.iterrows():
            row,g,runs=one_trial(meta,b,grouped.get(int(b["trial"])),D,args)
            rows.append(row);items.append((row,g,runs))
        # Record trial eligibility separately from an actual endpoint anomaly.
        trial_rows=[item[0] for item in items]
        screened=[int(r["trial"]) for r in trial_rows if r["endpoint_detection_eligible"]]
        flagged=[int(r["trial"]) for r in trial_rows if r["endpoint_candidate_flag"]]
        statuses=Counter(r["endpoint_detection_status"] for r in trial_rows)
        session_qc_rows.append({
            "subject_id":str(meta.subject_id),
            "session_order":int(meta.session_order),
            "source_file":str(meta.source_file),
            "response_type":str(meta.response_type),
            "node1":int(meta.node1_importance_code),
            "node2":int(meta.node2_correlation_code),
            "n_trials":len(trial_rows),
            "endpoint_min_abs_price_100":args.endpoint_min_abs_price_100,
            "n_endpoint_detection_triggered":len(screened),
            "endpoint_detection_trial_indices":json.dumps(screened),
            "n_endpoint_candidates":len(flagged),
            "endpoint_candidate_trial_indices":json.dumps(flagged),
            "n_triggered_with_no_gaze":sum(
                r["endpoint_detection_eligible"] and r["n_gaze_samples"]==0
                for r in trial_rows),
            "n_skipped_abs_preference_at_most_threshold":
                statuses["skipped_abs_preference_at_most_threshold"],
            "n_skipped_missing_preference":statuses["missing_preference"],
            "n_skipped_invalid_preference":statuses["invalid_preference_over_100"],
            "n_not_pricing":statuses["not_pricing"],
        })
        print(
            f"Subject {meta.subject_id} session {meta.session_order} "
            f"({meta.response_type}): {len(items)} trials; "
            f"endpoint detection triggered={len(screened)}, "
            f"suspected endpoints={len(flagged)}"
        )
        session_plots=(plots/f"subject_{meta.subject_id}"/
                       f"session_{int(meta.session_order):02d}_{meta.response_type}")
        if not args.no_overviews:
            overview(meta,items,args,session_plots)
        if args.details!="none":
            for row,g,runs in items:
                if args.details=="all" or row["endpoint_candidate_flag"]:
                    detail_plot(meta,row,g,runs,args,session_plots/"trials")
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"trial_summary.csv",index=False,encoding="utf-8-sig")
    # One row per processed session, including choice sessions with 0 triggers.
    session_qc=pd.DataFrame(session_qc_rows)
    session_qc.to_csv(
        out/"session_endpoint_qc.csv",index=False,encoding="utf-8-sig"
    )
    cols=["subject_id","session_order","trial","choice","rt_seconds",
          "price_response_100","price_abs_100","endpoint_detection_eligible",
          "endpoint_detection_status","endpoint_candidate_flag","endpoint_response_near_100",
          "endpoint_pattern","endpoint_side","endpoint_start_ms","endpoint_end_ms",
          "endpoint_start_timestamp_us","endpoint_end_timestamp_us",
          "endpoint_duration_ms","endpoint_span_ms","endpoint_gap_before_end_ms",
          "endpoint_n_samples","endpoint_n_visits","endpoint_n_valid_visits",
          "endpoint_visits_json"]
    summary.loc[summary.response_type=="pricing",cols].to_csv(
           out/"pricing_endpoint_qc.csv",index=False,encoding="utf-8-sig")
    print(f"Done: {len(summary)} trial rows -> {out/'trial_summary.csv'}")
    print(
        f"Endpoint QC: {len(session_qc)} sessions, "
        f"{int(session_qc['n_endpoint_detection_triggered'].sum())} trials screened, "
        f"{int(session_qc['n_endpoint_candidates'].sum())} suspected; "
        f"saved -> {out/'session_endpoint_qc.csv'}"
    )


if __name__=="__main__":
    main()
