# MMSA baselines on SJTU π cluster

- Login: ssh <user>@pilogin.hpc.sjtu.edu.cn (SJTU Wi-Fi or VPN, not eduroam)
- Env: module load miniconda3/24.3.0; conda env with Python 3.11
- Install: torch==2.1.2 first, then MMSA, then pin:
  numpy<2, pandas<2.3, scipy<1.14, scikit-learn<1.6, transformers==4.36.2
  (full list in requirements-hpc.txt)
- Mirrors: pypi.tuna.tsinghua.edu.cn, mirror.sjtu.edu.cn (conda-forge)
- Data: MMSA feature files in ~/mmsa_project/data/ (not in repo)
- Jobs: partition dgx2, --qos=normal, max 6 CPUs per GPU, 1 job at a timew
