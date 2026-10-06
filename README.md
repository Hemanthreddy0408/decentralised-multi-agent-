# Decentralized Multi-Agent Energy Trading and Battery Scheduling for Smart Neighborhoods

FOAI (Foundations of Artificial Intelligence) case-study project. 15 household agents, each with solar PV, a
battery and variable demand, optimise a 24-hour battery schedule with **Simulated Annealing**, replan **every
hour**, trade surplus/deficit through an hourly **Double Auction**, and settle the remainder with the main grid.
Three configurations (**Baseline / Selfish / Cooperative**) are compared on cost, peak load, grid dependency,
fairness and Price of Anarchy. Everything reported below is produced by running the code.

## 1. Problem statement
Design and implement a decentralized multi-agent energy management system for a smart neighborhood of 10-20
households. Each autonomous agent uses Simulated Annealing to optimise its 24-hour charge/discharge schedule and
replans hourly as demand, solar and prices change. Agents broadcast surpluses/deficits and trade through a Double
Auction; unmatched energy goes to the grid.

## 2. Motivation
Rooftop solar creates midday surpluses and evening deficits. Selling to the grid pays little (45 % of the retail
price here), buying pays a lot, so neighbours can both gain by trading directly - if the system stays decentralised,
private, and scalable.

## 3. FOAI concepts demonstrated (and where)
| Concept | Where |
|---|---|
| Intelligent / rational autonomous agent | `household_agent.py` (sense -> plan -> act), cost-minimising |
| Multi-agent system, decentralised negotiation | `environment.py`, `communication.py`, `double_auction.py` |
| PEAS, environment properties | Sections 7-8 |
| State / action / goal / constraints | Sections 9-11, `battery.py`, `simulated_annealing.py` |
| Informed search, Simulated Annealing | `simulated_annealing.py` |
| Competitive vs cooperative agents | `strategies.py` |
| Utility / cost functions | `evaluate_schedule` (`simulated_annealing.py`) |
| Performance evaluation | `metrics.py`, `experiments.py`, `visualization.py` |

## 4. Architecture
```
data_generator -> profiles --> SmartGridEnvironment (clock, public info, market, grid, recorder)
                                   |  each hour: gives each agent ONLY its own sensor reading
        +--------------------------+--------------------------+
        v                          v                          v
 HouseholdAgent 0 ...      HouseholdAgent i ...        HouseholdAgent N
 (Battery, private forecast, Strategy, SA)       <-- only BroadcastMessage / PublicInfo cross the boundary
                                   |  bids
                                   v
                          run_double_auction  -->  trades  -->  unmatched energy -> main grid
```
The environment never chooses a battery action, price or quantity. Agents hold no reference to other agents
(checked by a test). Private data (SOC, forecast, SA plan, bargaining shade, cost) is never broadcast.

## 5. Project structure
```
main.py  config.py  data_generator.py  battery.py  household_agent.py  simulated_annealing.py
double_auction.py  strategies.py  communication.py  environment.py  results.py  metrics.py
visualization.py  experiments.py  requirements.txt  README.md
data/sample_profiles.csv   results/ (CSV/JSON)   plots/ (10 PNG)   tests/ (37 tests)
```

## 6. Agent description
Fields: `agent_id`, solar/demand (observed now, forecast ahead), battery (capacity, SOC, max charge/discharge rate,
efficiencies), grid prices (public tariff), trading preference (bargaining shade), `internal_cost`,
`energy_bought_grid`, `energy_sold_grid`, `energy_traded_locally`. Methods: `sense`, `plan` (SA), `execute`
(charge/discharge/hold), `broadcast`, `make_bid` (buy/sell), `settle`. The agent knows its *typical day* and
learns a cloud factor/demand scale online; real values deviate (noise), so it must replan every hour.

## 7. PEAS
| | |
|---|---|
| **Performance** | household electricity cost, neighbourhood cost, peak grid load, grid dependency, renewable use, fairness |
| **Environment** | neighbourhood micro-grid: solar, demand, TOU prices, local market, main grid |
| **Actuators** | battery charge/discharge/hold, buy/sell bids, grid transactions |
| **Sensors** | own solar & demand meters, own SOC, tariff, broadcast messages, past clearing price, aggregate grid load |

## 8. Environment characteristics
Dynamic (solar, demand, price change hourly) - **partially observable** (others' SOC, valuations, future demand
hidden) - **stochastic** (cloud + demand noise; plans must be revised) - **sequential** (today's SOC affects
tomorrow's options) - **multi-agent**, both cooperative and competitive depending on configuration - discrete time
(1 h), continuous energy quantities.

## 9. State representation
`S = (hour, solar, demand, battery_soc, grid_price, surplus_or_deficit, public_bid_information)`
* locally observable: hour, own solar, own demand, own SOC, own surplus/deficit
* publicly broadcast: grid prices, surplus/deficit messages, past clearing prices, aggregate grid load
* private: forecasts, SA plan/state, bargaining shade, accumulated cost

## 10. Action space
`charge`, `discharge`, `hold` (SA decision variable, 5 power levels: -1, -0.5, 0, +0.5, +1 x max rate), then
`sell` / `buy` (bid + auction) and grid trade. Transition: `net = solar - demand - charge + discharge`;
`soc' = soc + charge*eta_c - discharge/eta_d`; `net>0` surplus, `net<0` deficit.

## 11. Constraints (all implemented and tested)
SOC in [10 %, 100 %] of capacity; charge/discharge rate limits; no simultaneous charge+discharge (signed encoding +
`Battery.step` raises); solar cannot be fabricated; sell <= surplus, buy <= deficit (`settle` raises); energy balance
checked every agent-hour (`solar + import + p2p_buy + discharge = demand + charge + export + p2p_sell`); every
unmatched kWh goes to the grid; efficiencies applied. "Unmet demand" cannot occur because the grid always supplies
deficits (at the high buy price) - assumption. SA penalises violations at 1000 INR/kWh.

## 12. Simulated Annealing (`simulated_annealing.py`)
Solution = 24 integers (power levels). 1) initial: previous plan shifted one hour (warm start) or all-hold;
2) neighbours: re-assign one hour / swap two hours / nudge one level; 3) objective: cost of the simulated schedule
(import at buy price, export at sell price, degradation 0.05 INR/kWh, end-SOC value) + violation penalty;
4) T0 calibrated so a median worsening move is accepted with p=0.8; 5) Metropolis acceptance `exp(-d/T)`;
6) geometric cooling `T <- 0.85 T` every 25 moves; 7) stop at `T < 0.01 T0`, 600 iterations, or 300 without
improvement; 8) best-so-far returned. Only the **first** action is executed; the plan is recomputed next hour
(rolling horizon). Verified by tests (feasible output, improvement over start, penalties, determinism).

## 13. Double Auction (`double_auction.py`)
Sellers ask (quantity, price), buyers bid. Sort asks ascending / bids descending; match while `bid >= ask`
(partial fills). The **marginal matched pair** sets one uniform price `ask_m + 0.5*(bid_m - ask_m)`; every trade
clears at it, so nobody trades at a price worse than their limit. Example: seller 3 kWh @5.5, buyer 2 kWh @6.5 ->
2 kWh at 6.0 INR, 1 kWh to the grid. Limit prices lie between the grid sell and buy price, so local trading is always
at least as good as the grid for both sides.

## 14. Baseline / Selfish / Cooperative (`strategies.py`)
* **Baseline** - SA on plain grid prices, **no auction** (grid-only).
* **Selfish** - minimises own cost, values trades with the public clearing price, bids strategically (private
  shade 15-35 % of the spread, scaled by the broadcast supply/demand ratio).
* **Cooperative** - own cost **plus** community terms from *public* aggregate load only: a congestion surcharge on
  imports in hours where the neighbourhood is expected to be congested, and a quadratic peak penalty; bids
  truthfully. **Each agent still solves its own SA problem** - there is no central optimiser or shared private
  state. Cooperation is therefore decentralised coordination through public signals.

## 15. Metrics (`metrics.py`)
Total cost, peak grid load (max hourly aggregate import), grid dependency (import/demand), renewable utilisation
(1 - export/solar), local energy traded, grid bought/sold, per-agent cost and savings (vs grid-only and vs baseline),
Jain's index `(sum x)^2/(n sum x^2)` on each household's relative saving vs buying everything from the grid,
and **Price of Anarchy = Selfish total cost / Cooperative total cost** - the *project's simulation comparison
metric*, **not** the formal worst-case game-theoretic PoA. Because runs end with different battery SOC, an
SOC-adjusted cost is also reported.

## 16. Installation
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## 17. How to run
```
python main.py                                  # 15 households, 7 days, seed 42 (about 30 s)
python main.py --households 20 --days 7 --seed 1
python main.py --csv my_profiles.csv            # custom data: household_id,hour,solar_generation,demand
python main.py --robustness 5                   # extra 5 seeds (3-day runs)
pytest -q                                       # 37 unit/integration tests
```
CSV: `hour` 0..23 = one typical day (repeated with noise) or 0..24*days-1 = full series; batteries are generated
from the seed. **Google Colab:** upload the folder (or `git clone` your repo), then
```
!pip install -r requirements.txt
!python main.py
from IPython.display import Image; Image("plots/05_hourly_grid_demand.png")
```
Outputs: `results/comparison_table.csv`, `summary.json`, `per_agent_results.csv`, `hourly_<scenario>.csv`,
`sa_statistics.json`, `plots/*.png` (10 plots).

## 18. Example output (seed 42, 15 households, 7 days)
```
Configuration  Total Cost (INR)  Peak Load (kW)  Grid Dependency (%)  Local Trading (kWh)  Fairness (Jain)  PoA
     Baseline           3931.15           21.23                74.67                 0.00           0.9135    -
      Selfish           2155.60           24.88                61.09               502.61           0.9346  0.8941
  Cooperative           2410.96           20.31                52.50               456.37           0.9300    -
```

## 19. Results (actual run, `results/comparison_table.csv`)
| Configuration | Total cost (INR) | SOC-adj. cost | Peak (kW) | Grid dep. | Renewable util. | Local trade (kWh) | Fairness | PoA |
|---|---|---|---|---|---|---|---|---|
| Baseline | 3931.15 | 4041.97 | 21.23 | 74.67 % | 33.10 % | 0.00 | 0.9135 | - |
| Selfish | 2155.60 | 2289.92 | 24.88 | 61.09 % | 51.41 % | 502.61 | 0.9346 | **0.8941** |
| Cooperative | 2410.96 | 2539.55 | 20.31 | 52.50 % | 59.94 % | 456.37 | 0.9300 | - |

**Honest reading.** Peer-to-peer trading cuts cost by 45 % (selfish) / 39 % (cooperative) vs the baseline.
Cooperative agents give the lowest peak (-18 % vs selfish), lowest grid dependency and highest renewable use, but
**cost about 12 % more in cash** than selfish, so PoA = 0.894 < 1: in this model "anarchy" is cheaper in total cost
and worse in peak load. We did **not** tune the final seed to change this; cooperative weights were calibrated on
other seeds (1-3, 3-day runs), where the same trade-off appeared. Robustness over 5 further seeds (3 days):
mean cost baseline 1611 / selfish 985 / cooperative 1045; mean peak 15.9 / 24.4 / 18.4 kW; PoA per seed
0.98, 0.97, 0.91, 0.88, 1.02. With the cost-only metric the project's expected PoA > 1 is therefore **not** reproduced;
a more peak-focused objective (e.g. demand charges) would be needed to make cooperation also pay in rupees.

## 20. Limitations
Synthetic data; fixed tariff; no network losses/line limits; truthful sellers/buyers report quantities honestly;
selfish bidding uses a simple shading heuristic, not a learned equilibrium; PoA is a simulation ratio, not worst
case over equilibria; battery degradation only enters planning; peak load is a single-hour extreme (noisy);
cooperative weights hand-calibrated; SA is stochastic and gives near-, not provably-, optimal plans.

## 21. Future improvements
Learned/equilibrium bidding, demand charges or peak tariffs, EV loads, line-capacity constraints, real datasets,
multi-seed confidence intervals, comparison with a centralised optimum to compute a true PoA.
# decentralised-multi-agent-
