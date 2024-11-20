[![codecov](https://codecov.io/gh/zinchse/hero/graph/badge.svg?token=R4WRFFQUZL)](https://codecov.io/gh/zinchse/hero)

# HERO: New learned Hint-based Efficient and Reliable query Optimizer.

**Key features:** extended search space, virtually optimal performance, fast inference time (`x5-x40` faster than existing solutions), quick and flexible learning stage and transparent and interpretable predictive model instead of black-box NN.

# 🔬 Method

Here will be details about hbo_bench, btcnn, graph storage, query explorer, predictor's logic and artifacts from experiments ...

# 📦 Setup

```bash
# virtual env
python -m pip install --upgrade pip
python3 -v venv venv
source venv/bin/activate

# prepare source code
git submodule update --init --recursive
git submodule update --remote --merge
pip install -e btcnn
pip install -e hbo_bench
pip install -e .

# prepare data
sudo apt-get install -y p7zip-full
7za x hbo_bench/src/hbo_bench/data/raw/raw_data.7z -ohbo_bench/src/hbo_bench/data/raw
cd hbo_bench
python3 process_raw_data.py
cd ..
```


# ⓘ FAQ

<details>
  <summary><strong>What is the hint-based optimization?</strong></summary>

  Hint-Based query Optimization (HBO) is an approach to optimizing query execution time that accelerates workload execution without changing a single line of the DBMS kernel code. This is achieved by selecting planner hyperparameters (hints) that influence the construction of the query execution plan. Although this approach can greatly speed up query execution, it faces a fundamental challenges. In first, there's no universal hint. In second, the search space is exponential, and the cost of exploring a "point" within it depends on its execution time. As result, we must construct and train an intelligent hint-advisor to cope with them.

</details>

<details>
<summary><strong>What is new in HERO?</strong></summary>

  HERO sets itself apart from existing solutions by **(1)** incorporating parallelism-related hints, **(2)** replacing NN with more interpretable, reliable, and controllable model, and **(3)** introducing a new procedure that efficiently explores queries while balancing training time with performance improvements. As a result, HERO is easy to interpret and debug, which is critical when deploying to production.

</details>

<details>
<summary><strong>How does it work? Any intuition behind?</strong></summary>
  
  In principle, HERO just use some version of local serach algorithm in hint space and collects information about queries behavior (plans, runtimes, etc.). After it, it reuses that information, based its prediction on similar queries that have been explored before. The similarity is measured based on plans and custom tree metric. Intuitively, HERO considere hints just as a way to change the plan, and tries to repeat the best of previosly observed performances.
  
</details>

<details>
<summary><strong>What is the HERO's disadvantage?</strong></summary>
  
  The only HERO's drawback is that it does'n make any risky suggestion, which sometimes leads to losing the potential boost. This was our design decision to ensure high reliability of predictions.
  
</details>

<details>
<summary><strong> How to use HERO in production?</strong></summary>
  
  Now HERO is just a prototype to prove the concept. Neverthless, it doesn't depend on any external modules and heavy deep learning frameworks. All necessary details may be implemented from scrath (as Graph Storage and Query Explorer). However, to get the best possible performance we need to implement the learning procedure inside the kernel, which requires **(a)** integration with DBMS and **(b)** implementation of parallel query exploration with all our optimizations.
  
</details>

# ❛❛❞ Citation

```latex
citation will be added later
```


# 📚 References

There are two main papers on the hint-based query optimization approach:

1. [Marcus, Ryan, et al. "Bao: Making learned query optimization practical." *Proceedings of the 2021 International Conference on Management of Data*, 2021, pp. 1275-1288.](https://people.csail.mit.edu/hongzi/content/publications/BAO-Sigmod21.pdf)

2. [Anneser, Christoph, et al. "Autosteer: Learned query optimization for any SQL database." *Proceedings of the VLDB Endowment*, vol. 16, no. 12, 2023, pp. 3515-3527.](https://vldb.org/pvldb/vol16/p3515-anneser.pdf)


