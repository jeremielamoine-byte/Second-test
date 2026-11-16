import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="eTruck Depot Planner",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ eTruck Depot Planner")
st.markdown(
    "Optimisation de recharge et dimensionnement de dépôt électrique"
)

# ====================================================
# SIDEBAR
# ====================================================

st.sidebar.header("Paramètres")

granularity = st.sidebar.selectbox(
    "Pas d'allocation",
    [50,25,10,5],
    index=0
)

study_max_power = st.sidebar.number_input(
    "Puissance max étudiée (kW)",
    100,
    3000,
    1500,
    50
)

study_max_ports = st.sidebar.number_input(
    "CCS max étudiés",
    1,
    50,
    20
)

default_text = """
Truck01,18,6,600,400
Truck02,18,7,550,375
Truck03,19,8,450,250
Truck04,20,7,300,250
Truck05,21,6,350,400
"""

vehicle_text = st.sidebar.text_area(
    "Flotte",
    default_text,
    height=250
)

# ====================================================
# VEHICLES
# ====================================================

vehicles = []

for line in vehicle_text.splitlines():

    if line.strip() == "":
        continue

    parts = [x.strip() for x in line.split(",")]

    if len(parts) >= 5:

        vehicles.append(
            {
                "id": parts[0],
                "arr": float(parts[1]),
                "dep": float(parts[2]),
                "req": float(parts[3]),
                "max_power": float(parts[4])
            }
        )

if len(vehicles) == 0:
    st.stop()

# ====================================================
# PRESENCE MATRIX
# ====================================================

def build_presence_matrix(
        vehicles,
        time_steps):

    n_veh = len(vehicles)

    presence = np.zeros(
        (
            n_veh,
            len(time_steps)
        )
    )

    for v_idx, v in enumerate(vehicles):

        arr = v["arr"]
        dep = v["dep"]

        for t_idx, t in enumerate(time_steps):

            if arr <= dep:

                if arr <= t < dep:
                    presence[v_idx,t_idx] = 1

            else:

                if t >= arr or t < dep:
                    presence[v_idx,t_idx] = 1

    return presence

# ====================================================
# SIMULATION ENGINE
# ====================================================

def run_simulation(
    network_power,
    n_ports,
    vehicles,
    granularity
):

    time_steps = np.arange(
        0,
        24,
        0.25
    )

    n_steps = len(time_steps)

    n_veh = len(vehicles)

    presence = build_presence_matrix(
        vehicles,
        time_steps
    )

    energy_required = np.array(
        [
            v["req"]
            for v in vehicles
        ]
    )

    energy_delivered = np.zeros(
        n_veh
    )

    power_matrix = np.zeros(
        (
            n_veh,
            n_steps
        )
    )

    cumulative_energy = np.zeros(
        (
            n_veh,
            n_steps
        )
    )

    stall_assignment = np.zeros(
        (
            n_steps,
            n_ports
        ),
        dtype=int
    )

    prev_stalls = np.zeros(
        n_ports,
        dtype=int
    )

    manipulations = 0

    for t_idx,t in enumerate(time_steps):

        candidates = []

        for v_idx in range(n_veh):

            if (
                presence[v_idx,t_idx] == 1
                and
                energy_delivered[v_idx]
                < energy_required[v_idx]
            ):
                candidates.append(v_idx)

        if len(candidates) == 0:

            prev_stalls[:] = 0

            for v in range(n_veh):
                cumulative_energy[v,t_idx] = (
                    energy_delivered[v]
                )

            continue

        urgency = []

        for v_idx in candidates:

            dep = vehicles[v_idx]["dep"]

            if dep > t:
                time_left = dep - t
            else:
                time_left = dep + 24 - t

            remaining = (
                energy_required[v_idx]
                -
                energy_delivered[v_idx]
            )

            score = (
                remaining
                /
                max(
                    time_left,
                    0.25
                )
            )

            urgency.append(
                (
                    score,
                    v_idx
                )
            )

        urgency.sort(
            reverse=True
        )

        current = np.zeros(
            n_ports,
            dtype=int
        )

        assigned = set()

        for port in range(n_ports):

            old = prev_stalls[port]

            if old > 0:

                idx = old - 1

                if idx in candidates:

                    current[port] = old

                    assigned.add(idx)

        remaining = [
            v
            for _,v in urgency
            if v not in assigned
        ]

        for port in range(n_ports):

            if (
                current[port] == 0
                and
                remaining
            ):

                next_v = remaining.pop(0)

                current[port] = (
                    next_v + 1
                )

                assigned.add(next_v)

        if t_idx > 0:

            manipulations += np.sum(
                current != prev_stalls
            )

        prev_stalls = current.copy()

        stall_assignment[t_idx] = current

        plugged = [
            x-1
            for x in current
            if x > 0
        ]

        allocations = {
            v:0
            for v in plugged
        }

        remaining_power = network_power

        while remaining_power >= granularity:

            assigned_round = False

            sorted_vehicles = sorted(
                plugged,
                key=lambda x:
                (
                    energy_required[x]
                    -
                    energy_delivered[x]
                ),
                reverse=True
            )

            for v_idx in sorted_vehicles:

                max_power = vehicles[v_idx][
                    "max_power"
                ]

                allowed = (
                    min(
                        max_power,
                        (
                            (
                                energy_required[v_idx]
                                -
                                energy_delivered[v_idx]
                            )/0.25
                        )
                    )
                    -
                    allocations[v_idx]
                )

                if (
                    allowed >= granularity
                    and
                    remaining_power >= granularity
                ):

                    allocations[v_idx] += granularity

                    remaining_power -= granularity

                    assigned_round = True

            if not assigned_round:
                break

        for v_idx,p in allocations.items():

            power_matrix[
                v_idx,
                t_idx
            ] = p

            energy_delivered[v_idx] += (
                p * 0.25
            )

        for v in range(n_veh):

            cumulative_energy[v,t_idx] = (
                energy_delivered[v]
            )

    recovery = (
        energy_delivered.sum()
        /
        energy_required.sum()
    ) * 100

    active_ports = np.count_nonzero(
        power_matrix,
        axis=0
    )

    occupation_rate = (
        np.count_nonzero(
            stall_assignment
        )
        /
        (
            n_steps
            * n_ports
        )
        *100
    )

    return {
        "recovery":recovery,
        "max_ports_used":active_ports.max(),
        "avg_ports_used":active_ports.mean(),
        "occupation_rate":occupation_rate,
        "peak_power":power_matrix.sum(axis=0).max(),
        "avg_power":power_matrix.sum(axis=0).mean(),
        "manipulations":manipulations,
        "power_matrix":power_matrix,
        "stall_assignment":stall_assignment,
        "cumulative_energy":cumulative_energy,
        "energy_delivered":energy_delivered,
        "energy_required":energy_required,
        "time_steps":time_steps
    }

# ====================================================
# OPTIMISATION POWER
# ====================================================

optimal_power = None

for power in range(
    50,
    study_max_power + 1,
    50
):

    result = run_simulation(
        power,
        study_max_ports,
        vehicles,
        granularity
    )

    if result["recovery"] >= 99.9:

        optimal_power = power
        break

# ====================================================
# OPTIMISATION CCS
# ====================================================

optimal_ports = None

for ports in range(
    1,
    study_max_ports + 1
):

    result = run_simulation(
        optimal_power,
        ports,
        vehicles,
        granularity
    )

    if result["recovery"] >= 99.9:

        optimal_ports = ports
        break

final_result = run_simulation(
    optimal_power,
    optimal_ports,
    vehicles,
    granularity
)

# ====================================================
# DASHBOARD
# ====================================================

st.subheader(
    "🏆 Recommandation Infrastructure"
)

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric(
    "Puissance optimale",
    f"{optimal_power:.0f} kW"
)

c2.metric(
    "CCS2 minimum",
    optimal_ports
)

c3.metric(
    "Occupation",
    f"{final_result['occupation_rate']:.1f}%"
)

c4.metric(
    "Simultanéité",
    final_result["max_ports_used"]
)

c5.metric(
    "Manipulations",
    final_result["manipulations"]
)

# ====================================================
# TABLEAU
# ====================================================

st.subheader("🚚 Flotte")

df = pd.DataFrame(vehicles)

st.dataframe(
    df,
    use_container_width=True
)

# ====================================================
# COURBE STATION
# ====================================================

st.subheader(
    "⚡ Charge Station"
)

fig,ax = plt.subplots(
    figsize=(12,4)
)

total_power = (
    final_result[
        "power_matrix"
    ].sum(axis=0)
)

ax.step(
    final_result["time_steps"],
    total_power,
    where="mid",
    linewidth=2
)

ax.axhline(
    optimal_power,
    color="red",
    linestyle="--"
)

ax.set_ylabel(
    "kW"
)

ax.grid(True)

st.pyplot(fig)

# ====================================================
# RECOVERY CURVE
# ====================================================

st.subheader(
    "📈 Taux de récupération"
)

results = []

for p in range(
    50,
    study_max_power+1,
    50
):

    r = run_simulation(
        p,
        optimal_ports,
        vehicles,
        granularity
    )

    results.append(
        [
            p,
            r["recovery"]
        ]
    )

df_curve = pd.DataFrame(
    results,
    columns=[
        "Power",
        "Recovery"
    ]
)

fig2,ax2 = plt.subplots(
    figsize=(12,4)
)

ax2.plot(
    df_curve["Power"],
    df_curve["Recovery"],
    marker="o"
)

ax2.axhline(
    100,
    color="red",
    linestyle="--"
)

ax2.grid(True)

st.pyplot(fig2)

st.dataframe(
    df_curve,
    use_container_width=True
)
