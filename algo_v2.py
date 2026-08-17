"""
Deriving and testing a certificate that actually works.

Notation. Dense image embedding e_D(x), unit norm. Class prototypes t_k, unit
norm, fixed (the text side is not compressed). Compressed embedding e_C(x), and
the displacement delta(x) = e_C(x) - e_D(x).

PROPOSITION 1 (exact, no inequality).
For any two classes a, b the similarity gap changes exactly by the projection of
the displacement onto the difference of the two prototypes:
    <e_C, t_a - t_b>  =  <e_D, t_a - t_b>  +  <delta, t_a - t_b>
So the dense prediction k1 is preserved if and only if
    G(x) := max_{j != k1} <delta(x), t_j - t_k1>   <   m_D(x)
where m_D is the dense margin. Nothing is approximated here.

PROPOSITION 2 (the bound used in the proposal, and why it is weak).
    G(x) <= ||delta(x)|| * max_j ||t_j - t_k1||  <=  2 ||delta(x)||
Two losses happen at once. The prototype difference is bounded by 2 but for
near-orthogonal prototypes it is about sqrt(2). And Cauchy-Schwarz assumes
delta points exactly along the discriminative direction, which it does not.

PROPOSITION 3 (the replacement).
G(x) is computable without labels, at compression time, by whoever performs the
compression. Take q = the (1-alpha) empirical quantile of G over an unlabelled
probe set and ship q with the model. At deployment the site certifies a
prediction when its own compressed margin exceeds q. If probe and deployment
inputs are exchangeable, the probability that a certified prediction differs
from the dense model's is at most alpha, up to the usual finite-sample
correction of conformal prediction.

This script measures whether Proposition 3 holds in practice, and by how much
it beats Proposition 2.
"""
import numpy as np, torch
from common import build_world, get_model, prune, N_CLASS

SPARSITIES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA = 0.05
SEEDS = [0, 1, 2]


def analyse(seed):
    world = build_world(seed)
    dense = get_model(world, seed, verbose=False)
    protos, x, y = world["protos"], world["x_te"], world["y_te"]

    n = len(x)
    g = torch.Generator().manual_seed(1234 + seed)
    perm = torch.randperm(n, generator=g)
    probe_idx, test_idx = perm[:n // 2], perm[n // 2:]     # probe is UNLABELLED

    with torch.no_grad():
        eD = dense(x)
        simsD = eD @ protos.t()
        k1 = simsD.argmax(1)

    out = []
    for s in SPARSITIES:
        comp = prune(dense, s)
        with torch.no_grad():
            eC = comp(x)
            simsC = eC @ protos.t()
            kC = simsC.argmax(1)
            delta = eC - eD

            # G(x): worst projection of the displacement onto a discriminative direction
            diff = protos.unsqueeze(0) - protos[k1].unsqueeze(1)      # N x K x d
            proj = torch.einsum("nd,nkd->nk", delta, diff)            # N x K
            proj.scatter_(1, k1.unsqueeze(1), -1e9)                   # exclude j = k1
            G = proj.max(1).values

            eps95 = delta[probe_idx].norm(dim=1).quantile(0.95).item()
            q = G[probe_idx].quantile(1 - ALPHA).item()               # label-free

            mCtop = simsC.topk(2, dim=1).values
            marginC_all = mCtop[:, 0] - mCtop[:, 1]
            marginC = marginC_all[test_idx]

            # v3: per-class quantile, grouped by the COMPRESSED model's own
            # prediction on the probe set, so still no labels anywhere
            qc = torch.zeros(N_CLASS)
            for c in range(N_CLASS):
                sel = probe_idx[(kC[probe_idx] == c)]
                qc[c] = G[sel].quantile(1 - ALPHA) if len(sel) >= 10 else torch.tensor(float("inf"))
            thr_v3 = qc[kC[test_idx]]

        agree = (kC == k1)[test_idx]
        correct = (kC == y)[test_idx]

        res = {"sparsity": s, "eps95": eps95, "q": q,
               "overall_agree": agree.float().mean().item()}
        for name, thr in (("v1", 2 * eps95), ("v2", q), ("v3", thr_v3)):
            cert = marginC > thr
            frac = cert.float().mean().item()
            res[f"{name}_certified"] = frac
            res[f"{name}_agree_given_cert"] = agree[cert].float().mean().item() if cert.any() else float("nan")
            res[f"{name}_correct_given_cert"] = correct[cert].float().mean().item() if cert.any() else float("nan")
            if name == "v3":
                yt = y[test_idx]
                head = yt < 4
                tail = yt >= N_CLASS - 4
                res["v3_cert_head"] = cert[head].float().mean().item()
                res["v3_cert_tail"] = cert[tail].float().mean().item()
        out.append(res)
    return out, world, marginC, None


if __name__ == "__main__":
    print(f"target: at most {ALPHA:.0%} of certified predictions may differ from the dense model\n")
    agg = {}
    for seed in SEEDS:
        rows, _, _, _ = analyse(seed)
        print(f"seed {seed}")
        print("  spars  v1cert v1agr | v2cert v2agr | v3cert v3agr  head  tail")
        for r in rows:
            print(f"   {r['sparsity']:.2f}  {r['v1_certified']:.3f} {r['v1_agree_given_cert']:.3f} |"
                  f" {r['v2_certified']:.3f} {r['v2_agree_given_cert']:.3f} |"
                  f" {r['v3_certified']:.3f} {r['v3_agree_given_cert']:.3f}"
                  f"  {r['v3_cert_head']:.3f} {r['v3_cert_tail']:.3f}")
            agg.setdefault(r["sparsity"], []).append(r)
        print()

    import warnings; warnings.filterwarnings("ignore")
    print("mean over seeds (cert = fraction of predictions certified, agr = of those, fraction matching the dense model)")
    print("  spars  v1cert v1agr | v2cert v2agr | v3cert v3agr | v3 head  v3 tail")
    for s in SPARSITIES:
        rs = agg[s]
        m = lambda k: np.nanmean([r[k] for r in rs])
        print(f"   {s:.2f}  {m('v1_certified'):.3f} {m('v1_agree_given_cert'):.3f} |"
              f" {m('v2_certified'):.3f} {m('v2_agree_given_cert'):.3f} |"
              f" {m('v3_certified'):.3f} {m('v3_agree_given_cert'):.3f} |"
              f"  {m('v3_cert_head'):.3f}   {m('v3_cert_tail'):.3f}")
