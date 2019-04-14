import pandas as pd
import numpy as np

def clean_ensamble(df, columns):
    fncmap = {"Size":clean_sizepref,"Current Ver":clean_ver ,"Reviews":int}
    for mapped in fncmap.keys():
        if mapped in columns:
            df[mapped] = df[mapped].apply(fncmap[mapped])
    return df.dropna()

def clean_sizepref(x):
    sep = list(x)
    pref = sep.pop()
    consts = {"M":1, "k":0.001}
    if pref in ["M","k"]:
        fin = consts[pref]*float("".join(sep))
    else:
        fin = 0
    return fin

def clean_ver(x):
    ret = np.nan
    if not isinstance(x, float):
        comps = list(x) if x != ret else ["V"]
        if comps[0].isdigit():
            trunc = []
            decSep = True
            iterN = 0
            end = False
            while iterN<len(comps) and not end:
                cand = comps[iterN]
                if cand.isdigit(): 
                    trunc.append(cand)
                else:
                    if decSep:
                        trunc.append(".")
                        decSep = False
                    else:
                        if not cand in [".", ","]:
                            end = True
                iterN+=1
            ret = float("".join(trunc))
    return ret
