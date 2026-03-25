import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="Detector Activation Calculator", layout="wide")
st.title("🔥 Fire Engineering: Detector & Sprinkler Activation")
st.markdown("Calculates activation times for a $t^2$ fire using Alpert's ceiling jet correlations.")
st.markdown("---")

# --- MAIN PAGE INPUTS ---
st.header("⚙️ Input Parameters")

# Row 1: Fire & Geometry
col_fire1, col_fire2, col_fire3, col_fire4 = st.columns(4)

alpha_dict = {
    "Slow (0.00293 kW/s²)": 0.00293,
    "Medium (0.01172 kW/s²)": 0.01172,
    "Fast (0.0469 kW/s²)": 0.0469,
    "Ultrafast (0.1876 kW/s²)": 0.1876,
    "Custom": None
}

with col_fire1:
    alpha_selection = st.selectbox("Fire Growth Rate", list(alpha_dict.keys()), index=2)
    if alpha_selection == "Custom":
        alpha = st.number_input("Custom Alpha (kW/s²)", min_value=0.0001, value=0.01, step=0.001, format="%.5f")
    else:
        alpha = alpha_dict[alpha_selection]

with col_fire2:
    H = st.number_input("Ceiling Height (m)", min_value=1.0, value=3.0, step=0.1)
with col_fire3:
    r = st.number_input("Radial Distance (m)", min_value=0.1, value=2.5, step=0.1)
with col_fire4:
    T_amb = st.number_input("Ambient Temp (°C)", value=20.0, step=1.0)

# Row 2: Device Specifications (Togglable)
st.subheader("Device Specifications")
col_dev1, col_dev2, col_dev3 = st.columns(3)

with col_dev1:
    use_smoke = st.checkbox("Include Smoke Detector", value=True)
    if use_smoke:
        smoke_proxy = st.number_input("Smoke Proxy (ΔT °C)", min_value=1.0, value=13.0, step=1.0)

with col_dev2:
    use_heat = st.checkbox("Include Heat Detector", value=True)
    if use_heat:
        heat_act_temp = st.number_input("Activation Temp (°C) [Heat]", min_value=30.0, value=58.0, step=1.0)
        heat_rti = st.number_input("RTI (m½s½) [Heat]", min_value=1.0, value=50.0, step=1.0)

with col_dev3:
    use_spk = st.checkbox("Include Sprinkler", value=True)
    if use_spk:
        spk_act_temp = st.number_input("Activation Temp (°C) [Spk]", min_value=30.0, value=68.0, step=1.0)
        
        # RTI Dropdown with Custom Option
        spk_rti_type = st.selectbox("Sprinkler RTI (m½s½)", ["Fast Response (50)", "Standard Response (130)", "Custom"])
        if spk_rti_type == "Fast Response (50)":
            spk_rti = 50.0
        elif spk_rti_type == "Standard Response (130)":
            spk_rti = 130.0
        else:
            spk_rti = st.number_input("Custom Spk RTI", min_value=1.0, value=100.0, step=1.0)
            
        spk_delay = st.number_input("Delay to Cap HRR (s)", min_value=0, value=0, step=5)

st.markdown("---")

# --- Physics Engine (Euler Integration) ---
dt = 1.0  
t_max = 1200 

times = np.arange(0, t_max + dt, dt)
Q_arr = np.zeros(len(times))
Tg_arr = np.zeros(len(times))
U_arr = np.zeros(len(times))
Th_arr = np.zeros(len(times)) + T_amb  
Ts_arr = np.zeros(len(times)) + T_amb  

act_smoke = None
act_heat = None
act_spk = None
capped_hrr = None

for i, t in enumerate(times):
    # 1. Fire Heat Release Rate (with Capping Logic if Sprinkler is active)
    if use_spk and act_spk is not None and t >= (act_spk + spk_delay):
        Q = alpha * ((act_spk + spk_delay) ** 2)
        if capped_hrr is None:
            capped_hrr = Q 
    else:
        Q = alpha * (t ** 2)
        
    Q_arr[i] = Q
    
    # 2. Alpert's Correlations
    if Q > 0:
        if (r / H) > 0.18:
            Tg = T_amb + (5.38 * ((Q / r) ** (2/3))) / H
            U = (0.197 * (Q ** (1/3)) * (H ** (1/2))) / (r ** (5/6))
        else:
            Tg = T_amb + (16.9 * (Q ** (2/3))) / (H ** (5/3))
            U = 0.96 * ((Q / H) ** (1/3))
    else:
        Tg = T_amb
        U = 0.0
        
    Tg_arr[i] = Tg
    U_arr[i] = U
    
    # 3. Device Thermal Lag & Activations
    if i > 0:
        if use_heat:
            dT_h = (np.sqrt(U) / heat_rti) * (Tg_arr[i-1] - Th_arr[i-1]) * dt
            Th_arr[i] = Th_arr[i-1] + dT_h
            if act_heat is None and Th_arr[i] >= heat_act_temp:
                act_heat = t
                
        if use_spk:
            dT_s = (np.sqrt(U) / spk_rti) * (Tg_arr[i-1] - Ts_arr[i-1]) * dt
            Ts_arr[i] = Ts_arr[i-1] + dT_s
            if act_spk is None and Ts_arr[i] >= spk_act_temp:
                act_spk = t
                
    if use_smoke and act_smoke is None and Tg >= (T_amb + smoke_proxy):
        act_smoke = t

# --- Dashboard Layout & Visuals ---
st.header("📊 Activation & Fire Metrics")

# Dynamically display metric cards based on selected devices
metric_cols = st.columns(4)
col_idx = 0

if use_smoke:
    metric_cols[col_idx].metric("Smoke Detector", f"{int(act_smoke)} s" if act_smoke else "Did not activate")
    col_idx += 1
if use_heat:
    metric_cols[col_idx].metric("Heat Detector", f"{int(act_heat)} s" if act_heat else "Did not activate")
    col_idx += 1
if use_spk:
    metric_cols[col_idx].metric("Sprinkler", f"{int(act_spk)} s" if act_spk else "Did not activate")
    col_idx += 1

if use_spk and capped_hrr:
    metric_cols[col_idx].metric("Capped HRR", f"{capped_hrr:.1f} kW", f"at {int(act_spk + spk_delay)} s", delta_color="off")
elif use_spk:
    metric_cols[col_idx].metric("Capped HRR", "Not Capped")

st.markdown("---")

# Determine chart cutoff time
active_acts = [act for act in [act_smoke, act_heat, act_spk] if act is not None]
max_act = max(active_acts) if active_acts else 0
if use_spk and act_spk:
    max_act = max(max_act, act_spk + spk_delay)
cutoff_idx = min(len(times), int(max_act + 100) if max_act > 0 else 600)

x_data = times[:cutoff_idx]

col_chart1, col_chart2 = st.columns([2, 1])

# Helper function to add vertical activation lines to a plot
def add_activation_lines(fig):
    if use_smoke and act_smoke:
        fig.add_vline(x=act_smoke, line_dash="dash", line_color="gray", 
                      annotation_text=f"Smoke ({int(act_smoke)}s)", annotation_position="top left")
    if use_heat and act_heat:
        fig.add_vline(x=act_heat, line_dash="dash", line_color="orange", 
                      annotation_text=f"Heat ({int(act_heat)}s)", annotation_position="bottom right")
    if use_spk and act_spk:
        fig.add_vline(x=act_spk, line_dash="dash", line_color="blue", 
                      annotation_text=f"Spk ({int(act_spk)}s)", annotation_position="top right")
    return fig

with col_chart1:
    st.subheader("Temperature Profiles")
    fig_temp = go.Figure()
    
    # Add Gas Temp Line
    fig_temp.add_trace(go.Scatter(x=x_data, y=Tg_arr[:cutoff_idx], mode='lines', name='Ceiling Gas Temp', line=dict(color='firebrick')))
    
    # Add Device Lines based on selection
    if use_heat:
        fig_temp.add_trace(go.Scatter(x=x_data, y=Th_arr[:cutoff_idx], mode='lines', name='Heat Det Temp', line=dict(color='orange')))
    if use_spk:
        fig_temp.add_trace(go.Scatter(x=x_data, y=Ts_arr[:cutoff_idx], mode='lines', name='Sprinkler Temp', line=dict(color='royalblue')))

    fig_temp = add_activation_lines(fig_temp)
    fig_temp.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)", 
                           margin=dict(l=0, r=0, t=30, b=0),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_temp, use_container_width=True)

with col_chart2:
    st.subheader("Heat Release Rate")
    fig_hrr = go.Figure()
    
    fig_hrr.add_trace(go.Scatter(x=x_data, y=Q_arr[:cutoff_idx], mode='lines', name='HRR', line=dict(color='red'), fill='tozeroy'))
    
    fig_hrr = add_activation_lines(fig_hrr)
    fig_hrr.update_layout(xaxis_title="Time (s)", yaxis_title="HRR (kW)", 
                          margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig_hrr, use_container_width=True)
