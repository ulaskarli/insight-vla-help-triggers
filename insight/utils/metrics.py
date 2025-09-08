#!/usr/bin/env python3
import numpy as np

def confusion(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    acc = (tp+tn)/max(1,(tp+tn+fp+fn))
    prec= tp/max(1,(tp+fp)) if (tp+fp)>0 else 0.0
    rec = tp/max(1,(tp+fn)) if (tp+fn)>0 else 0.0
    f1  = (2*prec*rec)/max(1e-8,(prec+rec))
    return tp,fp,tn,fn,acc,prec,rec,f1

def agg_macro(per_fold):
    return {
        "accuracy": float(np.mean([m["accuracy"] for m in per_fold])),
        "precision": float(np.mean([m["precision"] for m in per_fold])),
        "recall": float(np.mean([m["recall"] for m in per_fold])),
        "f1": float(np.mean([m["f1"] for m in per_fold])),
    }

def agg_micro(per_fold):
    TP = sum(m["tp"] for m in per_fold); FP = sum(m["fp"] for m in per_fold)
    TN = sum(m["tn"] for m in per_fold); FN = sum(m["fn"] for m in per_fold)
    precision = TP/max(1,(TP+FP)); recall = TP/max(1,(TP+FN))
    f1 = (2*precision*recall)/max(1e-8,(precision+recall))
    return {
        "tp": TP, "fp": FP, "tn": TN, "fn": FN,
        "accuracy": (TP+TN)/max(1,(TP+TN+FP+FN)),
        "precision": precision, "recall": recall, "f1": f1
    }