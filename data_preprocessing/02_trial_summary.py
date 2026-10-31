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
        if pricing:
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
    row.update(endpoint_qc(gaze,xc,yc,args) if pricing else endpoint_qc(pd.DataFrame(),xc,yc,args))
    row["endpoint_response_abs"]=abs(row["choice"]) if np.isfinite(row["choice"]) else np.nan
    row["endpoint_response_near_100"]=int(pricing and np.isfinite(row["choice"]) and abs(row["choice"])>=95)
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
    fn=out/f"subject_{meta.subject_id}_session_{int(meta.session_order):02d}_{meta.response_type}_all_trials.png"
    fig.savefig(fn,dpi=args.plot_dpi);plt.close(fig)
    print("  saved",fn)


def detail_plot(meta,row,gaze,runs,args,out):
    fig,ax=plt.subplots(figsize=(11,7))
    draw(ax,meta,row,gaze,runs,detail=True)
    text=f"endpoint={row['endpoint_pattern']} / {row['endpoint_side']} | visits={row['endpoint_n_visits']} | start={row['endpoint_start_ms']} ms, end={row['endpoint_end_ms']} ms, gap={row['endpoint_gap_before_end_ms']} ms"
    fig.text(.035,.04,text,fontsize=10)
    fig.tight_layout(rect=(0,.075,1,1))
    name=out/f"subject_{meta.subject_id}_session_{int(meta.session_order):02d}_trial_{int(row['trial']):02d}.png"
    fig.savefig(name,dpi=args.plot_dpi);plt.close(fig)


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
    out=args.output_dir.resolve()
    if not out.is_relative_to((ROOT/"outputs").resolve()):
        p.error("All generated files must be under repository outputs/")
    if not args.session_summary.is_file():p.error("Run step 1 first")
    sessions=pd.read_csv(args.session_summary,dtype={"subject_id":str},keep_default_na=False)
    if args.subject:sessions=sessions[sessions.subject_id==args.subject]
    sessions=sessions.sort_values(by=["subject_id","session_order"])
    if args.max_sessions:sessions=sessions.head(args.max_sessions)
    if sessions.empty:p.error("No sessions to process")
    out.mkdir(parents=True,exist_ok=True)
    plots=out/"gaze_qc";plots.mkdir(exist_ok=True)
    rows=[]
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
        print(f"Subject {meta.subject_id} session {meta.session_order}: {len(items)} trials, {sum(x[0]['endpoint_candidate_flag'] for x in items)} suspected endpoints")
        if not args.no_overviews:overview(meta,items,args,plots)
        if args.details!="none":
            for row,g,runs in items:
                if args.details=="all" or row["endpoint_candidate_flag"]:
                    detail_plot(meta,row,g,runs,args,plots)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"trial_summary.csv",index=False,encoding="utf-8-sig")
    cols=["subject_id","session_order","trial","choice","rt_seconds",
          "endpoint_candidate_flag","endpoint_response_near_100",
          "endpoint_pattern","endpoint_side","endpoint_start_ms","endpoint_end_ms",
          "endpoint_start_timestamp_us","endpoint_end_timestamp_us",
          "endpoint_duration_ms","endpoint_span_ms","endpoint_gap_before_end_ms",
          "endpoint_n_samples","endpoint_n_visits","endpoint_n_valid_visits",
          "endpoint_visits_json"]
    summary.loc[summary.response_type=="pricing",cols].to_csv(
           out/"pricing_endpoint_qc.csv",index=False,encoding="utf-8-sig")
    print(f"Done: {len(summary)} trial rows -> {out/'trial_summary.csv'}")


if __name__=="__main__":
    main()
