
Repository navigation
Code
Issues
Pull requests
Actions
Projects
Wiki
Security and quality
1
 (1)
Insights
Settings
Important update
On April 24 we'll start using GitHub Copilot interaction data for AI model training unless you opt out. Review this update and manage your preferences in your GitHub account settings.
Run Rotation Model
Run Rotation Model #13
All jobs
Run details
run-model
failed 1 minute ago in 4m 9s
Search logs
1s
1s
0s
23s
Collecting pandas
  Downloading pandas-2.3.3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (91 kB)
Collecting requests
  Downloading requests-2.33.1-py3-none-any.whl.metadata (4.8 kB)
Collecting numpy
  Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
Collecting yfinance
  Downloading yfinance-1.2.1-py2.py3-none-any.whl.metadata (6.1 kB)
Collecting scikit-learn
  Downloading scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting python-dateutil>=2.8.2 (from pandas)
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas)
  Downloading pytz-2026.1.post1-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas)
  Downloading tzdata-2026.1-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting charset_normalizer<4,>=2 (from requests)
  Downloading charset_normalizer-3.4.7-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
Collecting idna<4,>=2.5 (from requests)
  Downloading idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.26 (from requests)
  Downloading urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2023.5.7 (from requests)
  Downloading certifi-2026.2.25-py3-none-any.whl.metadata (2.5 kB)
Collecting multitasking>=0.0.7 (from yfinance)
  Downloading multitasking-0.0.12.tar.gz (19 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting platformdirs>=2.0.0 (from yfinance)
  Downloading platformdirs-4.9.4-py3-none-any.whl.metadata (4.7 kB)
Collecting frozendict>=2.3.4 (from yfinance)
  Downloading frozendict-2.4.7-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (23 kB)
Collecting peewee>=3.16.2 (from yfinance)
  Downloading peewee-4.0.4-py3-none-any.whl.metadata (8.6 kB)
Collecting beautifulsoup4>=4.11.1 (from yfinance)
  Downloading beautifulsoup4-4.14.3-py3-none-any.whl.metadata (3.8 kB)
Collecting curl_cffi>=0.15 (from yfinance)
  Downloading curl_cffi-0.15.0-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (18 kB)
Collecting protobuf>=3.19.0 (from yfinance)
  Downloading protobuf-7.34.1-cp310-abi3-manylinux2014_x86_64.whl.metadata (595 bytes)
Collecting websockets>=13.0 (from yfinance)
  Downloading websockets-16.0-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (6.8 kB)
Collecting scipy>=1.8.0 (from scikit-learn)
  Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting joblib>=1.2.0 (from scikit-learn)
  Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn)
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting soupsieve>=1.6.1 (from beautifulsoup4>=4.11.1->yfinance)
  Downloading soupsieve-2.8.3-py3-none-any.whl.metadata (4.6 kB)
Collecting typing-extensions>=4.0.0 (from beautifulsoup4>=4.11.1->yfinance)
  Downloading typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting cffi>=2.0.0 (from curl_cffi>=0.15->yfinance)
  Downloading cffi-2.0.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.6 kB)
Collecting rich (from curl_cffi>=0.15->yfinance)
  Downloading rich-14.3.3-py3-none-any.whl.metadata (18 kB)
Collecting pycparser (from cffi>=2.0.0->curl_cffi>=0.15->yfinance)
  Downloading pycparser-3.0-py3-none-any.whl.metadata (8.2 kB)
Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas)
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting markdown-it-py>=2.2.0 (from rich->curl_cffi>=0.15->yfinance)
  Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->curl_cffi>=0.15->yfinance)
  Downloading pygments-2.20.0-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->curl_cffi>=0.15->yfinance)
  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Downloading pandas-2.3.3-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (12.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.8/12.8 MB 17.5 MB/s  0:00:00
Downloading requests-2.33.1-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.7-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (216 kB)
Downloading idna-3.11-py3-none-any.whl (71 kB)
Downloading urllib3-2.6.3-py3-none-any.whl (131 kB)
Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.8/16.8 MB 58.9 MB/s  0:00:00
Downloading yfinance-1.2.1-py2.py3-none-any.whl (130 kB)
Downloading scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.7/9.7 MB 91.9 MB/s  0:00:00
Downloading beautifulsoup4-4.14.3-py3-none-any.whl (107 kB)
Downloading certifi-2026.2.25-py3-none-any.whl (153 kB)
Downloading curl_cffi-0.15.0-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (11.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.1/11.1 MB 117.4 MB/s  0:00:00
Downloading cffi-2.0.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (216 kB)
Downloading frozendict-2.4.7-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (120 kB)
Downloading joblib-1.5.3-py3-none-any.whl (309 kB)
Downloading peewee-4.0.4-py3-none-any.whl (144 kB)
Downloading platformdirs-4.9.4-py3-none-any.whl (21 kB)
Downloading protobuf-7.34.1-cp310-abi3-manylinux2014_x86_64.whl (324 kB)
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pytz-2026.1.post1-py2.py3-none-any.whl (510 kB)
Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.7/37.7 MB 169.9 MB/s  0:00:00
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading soupsieve-2.8.3-py3-none-any.whl (37 kB)
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Downloading tzdata-2026.1-py2.py3-none-any.whl (348 kB)
Downloading websockets-16.0-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (183 kB)
Downloading pycparser-3.0-py3-none-any.whl (48 kB)
Downloading rich-14.3.3-py3-none-any.whl (310 kB)
Downloading pygments-2.20.0-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 178.3 MB/s  0:00:00
Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Building wheels for collected packages: multitasking
  Building wheel for multitasking (pyproject.toml): started
  Building wheel for multitasking (pyproject.toml): finished with status 'done'
  Created wheel for multitasking: filename=multitasking-0.0.12-py3-none-any.whl size=15635 sha256=e2d7eadf0ccd698cec082ead3a865c6833453b719eefdb4be875462eb6218ddc
  Stored in directory: /home/runner/.cache/pip/wheels/e9/25/85/25d2e1cfc0ece64b930b16972f7e4cc3599c43b531f1eba06d
Successfully built multitasking
Installing collected packages: pytz, peewee, multitasking, websockets, urllib3, tzdata, typing-extensions, threadpoolctl, soupsieve, six, pygments, pycparser, protobuf, platformdirs, numpy, mdurl, joblib, idna, frozendict, charset_normalizer, certifi, scipy, requests, python-dateutil, markdown-it-py, cffi, beautifulsoup4, scikit-learn, rich, pandas, curl_cffi, yfinance
Successfully installed beautifulsoup4-4.14.3 certifi-2026.2.25 cffi-2.0.0 charset_normalizer-3.4.7 curl_cffi-0.15.0 frozendict-2.4.7 idna-3.11 joblib-1.5.3 markdown-it-py-4.0.0 mdurl-0.1.2 multitasking-0.0.12 numpy-2.2.6 pandas-2.3.3 peewee-4.0.4 platformdirs-4.9.4 protobuf-7.34.1 pycparser-3.0 pygments-2.20.0 python-dateutil-2.9.0.post0 pytz-2026.1.post1 requests-2.33.1 rich-14.3.3 scikit-learn-1.7.2 scipy-1.15.3 six-1.17.0 soupsieve-2.8.3 threadpoolctl-3.6.0 typing-extensions-4.15.0 tzdata-2026.1 urllib3-2.6.3 websockets-16.0 yfinance-1.2.1
3m 42s
Run python test1_production_weekly_exec_telegram.py
  
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
Running model...
MODEL STDOUT:
 Current working directory: /home/runner/work/rotation-bot-/rotation-bot-
Downloading price data...
Downloading macro proxies...
Latest available data date: 2026-04-07
Building C+ ELITE USD + HYG SHORT feature set...
Running C+ ELITE USD + HYG SHORT CONVICTION...
=== C+ ELITE USD + HYG SHORT CONVICTION PERFORMANCE ===
Annual Return: 0.472
Volatility:    0.143
Sharpe:        3.296
Max Drawdown:  -0.122
Avg Turnover:  0.800
=== Last 10 Rebalances ===
      date top_asset second_asset  top_score  second_score  score_gap  top_weight  second_weight  turnover  tx_cost_applied  war_strength  growth_strength  risk_off_strength  soxx_strength  copper_strength  usd_3m_strength  hyg_strength  credit_strength  raw_pred_QQQM  raw_pred_XLE  raw_pred_XSOE  adj_pred_QQQM  adj_pred_XLE  adj_pred_XSOE  w_QQQM  w_XLE  w_XSOE  w_BIL
2025-10-29      QQQM          XLE   0.012418      0.008760   0.003658         0.6            0.4       0.0           0.0000     -0.255631         1.442478          -0.625979       1.596007         0.593167         0.017025     -0.623629        -0.330310       0.006979      0.009914       0.006747       0.012418      0.008760       0.007675     0.6    0.4     0.0    0.0
2025-11-12      QQQM          XLE   0.005973      0.003825   0.002147         0.6            0.4       0.0           0.0000     -0.742414         0.249508          -0.362741       0.464671        -0.125454         0.471572     -0.535500         0.077818       0.004691      0.004025       0.004161       0.005973      0.003825       0.003650     0.6    0.4     0.0    0.0
2025-11-26       XLE         QQQM   0.005695      0.004717   0.000979         0.6            0.4       0.4           0.0004     -1.107662        -1.081111           0.702133      -0.765251        -0.014070         0.643522      0.579809         0.286257       0.005853      0.005695       0.005796       0.004717      0.005695       0.004342     0.4    0.6     0.0    0.0
2025-12-11      QQQM         XSOE   0.004330      0.004060   0.000270         0.6            0.4       1.2           0.0012     -0.806599         0.107213          -0.142061       0.660281         0.600293         0.460111      0.065945         0.397961       0.002814      0.002752       0.003313       0.004330      0.002666       0.004060     0.6    0.0     0.4    0.0
2025-12-26      QQQM         XSOE   0.007789      0.007640   0.000149         0.6            0.4       0.0           0.0000      0.438571         0.196455          -0.459813       0.607723         1.233987         0.306150     -0.377576         0.378430       0.006673      0.004953       0.006031       0.007789      0.005673       0.007640     0.6    0.0     0.4    0.0
2026-01-12       XLE         XSOE   0.007481      0.004906   0.002575         0.6            0.4       1.2           0.0012      1.756610        -0.381515           0.360737       0.209140         1.106221         0.412734      0.046518         0.148607       0.004605      0.003968       0.005260       0.002424      0.007481       0.004906     0.0    0.6     0.4    0.0
2026-01-27       XLE         XSOE   0.029966      0.018211   0.011755         0.7            0.3       0.2           0.0002      1.224517         0.985658           0.311618       1.876466         0.369289        -0.276191      0.195196        -0.261445       0.010625      0.028305       0.018802       0.013763      0.029966       0.018211     0.0    0.7     0.3    0.0
2026-02-10       XLE         XSOE   0.009088      0.006691   0.002397         0.0            0.0       2.0           0.0020      0.463128        -0.404718           1.109215       0.680087        -0.073082        -0.232623      0.384382        -0.172263       0.008309      0.009049       0.008841       0.006541      0.009088       0.006691     0.0    0.0     0.0    1.0
2026-02-25       XLE         QQQM   0.008444      0.005201   0.003243         0.6            0.4       2.0           0.0020      0.737355        -0.185218           0.627810       0.637398        -0.166975        -0.084016      0.226098        -0.010845       0.006267      0.006969       0.006658       0.005201      0.008444       0.005163     0.4    0.6     0.0    0.0
2026-03-11       XLE         QQQM   0.041129     -0.006575   0.047704         0.9            0.1       0.6           0.0006      1.676568         0.070493           0.302220      -0.614898        -0.018692         0.768649      0.756210        -0.302056      -0.004843      0.037832      -0.006197      -0.006575      0.041129      -0.008050     0.1    0.9     0.0    0.0
=== Latest Model Recommendation ===
Signal date: 2026-04-07
Raw predicted next-period returns:
XLE: 0.0098
XSOE: 0.0021
QQQM: 0.0018
Adjusted predicted next-period returns:
XLE: 0.0095
QQQM: 0.0051
XSOE: 0.0024
=== Suggested Weights ===
QQQM: 40.0%
XLE: 60.0%
XSOE: 0.0%
BIL: 0.0%
=== Previous Rebalance ===
Date: 2026-03-11
Top asset: XLE
Second asset: QQQM
Top score: 0.0411
Second score: -0.0066
Score gap: 0.0477
War strength: 1.677
Growth strength: 0.070
Risk-off strength: 0.302
SOXX strength: -0.615
Copper strength: -0.019
USD 3M strength: 0.769
HYG strength: 0.756
Credit strength: -0.302
=== Previous Allocation ===
QQQM: 10.0%
XLE: 90.0%
XSOE: 0.0%
BIL: 0.0%
=== Current Recommended Rebalance ===
Signal date: 2026-04-07
Top asset: XLE
Second asset: QQQM
Top score: 0.0095
Second score: 0.0051
Score gap: 0.0044
War strength: -0.249
Growth strength: 0.320
Risk-off strength: -0.173
SOXX strength: 0.951
Copper strength: -0.233
USD 3M strength: 1.026
HYG strength: 1.063
Credit strength: 0.861
=== New Allocation ===
QQQM: 40.0%
XLE: 60.0%
XSOE: 0.0%
BIL: 0.0%
=== Rotation Decision ===
Whether changed: YES
Turnover from previous rebalance: 0.600
Action: ROTATE this period
=== Latest Feature Importance Summary ===
           feature  importance
            oil_1m    0.081273
        spy_vol_1m    0.078187
 copper_rel_spy_1m    0.073194
           soxx_3m    0.046810
            ret_6m    0.045521
   credit_strength    0.045147
            usd_6m    0.041981
     rel_6m_vs_spy    0.039537
           war_XLE    0.034740
         vix_level    0.031173
   usd_1m_strength    0.031151
            hyg_6m    0.027578
            ret_3m    0.025866
   usd_3m_strength    0.024324
        short_rate    0.021045
            vol_1m    0.019353
            ret_1m    0.018402
       yield_curve    0.018329
   soxx_rel_spy_1m    0.018056
usd_level_strength    0.017909
Saved:
- model_c_plus_usd_hyg_short_conviction_portfolio_daily_returns.csv
- model_c_plus_usd_hyg_short_conviction_rebalance_log.csv
- model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv
- model_c_plus_usd_hyg_short_conviction_feature_importance.csv
MODEL STDERR:
 
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'date'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/home/runner/work/rotation-bot-/rotation-bot-/test1_production_weekly_exec_telegram.py", line 98, in <module>
    main()
  File "/home/runner/work/rotation-bot-/rotation-bot-/test1_production_weekly_exec_telegram.py", line 64, in main
    current = load_current()
  File "/home/runner/work/rotation-bot-/rotation-bot-/test1_production_weekly_exec_telegram.py", line 42, in load_current
    "date": row["date"],
  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pandas/core/series.py", line 1133, in __getitem__
    return self._get_value(key)
  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pandas/core/series.py", line 1249, in _get_value
    loc = self.index.get_loc(label)
  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'date'
Error: Process completed with exit code 1.
