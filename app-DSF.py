import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io

# ==========================================
# 1. 核心底层算法与自适应计算引擎
# ==========================================
def get_dynamic_window(data_length):
    window = int(data_length * 0.06)
    if window < 5: window = 5
    if window > 11: window = 11 
    return window if window % 2 != 0 else window + 1

def calculate_auto_qc_params(clean_df, well_map, excludes=[]):
    raw_bgs, raw_amps = [], []
    for w_key, w_col in well_map.items():
        if w_key in excludes: continue 
        f_vals = pd.to_numeric(clean_df[w_col], errors='coerce').dropna().values
        if len(f_vals) >= 10:
            raw_bgs.append(np.mean(f_vals[:5]))
            raw_amps.append(np.max(f_vals) - np.min(f_vals))
            
    if not raw_amps:
        return 400000, 30000
        
    max_amp = np.max(raw_amps)
    active_bgs = [bg for bg, amp in zip(raw_bgs, raw_amps) if amp > max_amp * 0.1]
    active_amps = [amp for amp in raw_amps if amp > max_amp * 0.1]
    
    if active_bgs and active_amps:
        return np.median(active_bgs) * 5, np.median(active_amps) * 0.2 
    return np.median(raw_bgs) * 5, np.median(raw_amps) * 0.2 

def check_curve_quality(fluor_series, max_bg, min_amp):
    f_vals = pd.to_numeric(fluor_series, errors='coerce').dropna().values
    if len(f_vals) < 10: return False, "数据过短"
    init_bg = np.mean(f_vals[:5])
    amplitude = np.max(f_vals) - np.min(f_vals)
    if init_bg > max_bg: return False, f"高自发荧光 (BG:{init_bg:.0f}>Max:{max_bg:.0f})"
    if amplitude < min_amp: return False, f"曲线水平/淬灭 (Amp:{amplitude:.0f}<Min:{min_amp:.0f})"
    return True, "Pass"

# 【受控的核心解析引擎】支持切换 经典算法 vs 高级修正算法
def get_curve_arrays(temp_series, fluor_series, use_advanced_fitting=False):
    try:
        f_vals = pd.to_numeric(fluor_series, errors='coerce').astype(float).values
        t_vals = pd.to_numeric(temp_series, errors='coerce').astype(float).values
        mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
        f_vals, t_vals = f_vals[mask], t_vals[mask]
        
        if len(f_vals) < 10: return None, None, None, None, np.nan
        
        if use_advanced_fitting:
            # ==========================================
            # 高级模式：双重平滑滤波 + 质心法寻找热力学重心
            # ==========================================
            window_f = get_dynamic_window(len(f_vals))
            window_f = int(window_f * 1.5)  
            window_f = window_f if window_f % 2 != 0 else window_f + 1 
            smoothed_f = savgol_filter(f_vals, window_length=window_f, polyorder=3)
            
            raw_derivative = np.gradient(smoothed_f, t_vals)
            
            window_d = int(len(f_vals) * 0.15) 
            window_d = max(7, window_d)
            window_d = window_d if window_d % 2 != 0 else window_d + 1
            derivative = savgol_filter(raw_derivative, window_length=window_d, polyorder=2)
            
            start_idx, end_idx = int(len(derivative) * 0.1), int(len(derivative) * 0.9)
            local_max_idx = np.argmax(derivative[start_idx:end_idx])
            max_idx = local_max_idx + start_idx
            
            max_val = derivative[max_idx]
            threshold = max_val * 0.80
            peak_indices = np.where((derivative[start_idx:end_idx] > threshold))[0] + start_idx
            
            if len(peak_indices) > 0:
                peak_temps = t_vals[peak_indices]
                peak_derivs = derivative[peak_indices]
                tm_val = float(np.sum(peak_temps * peak_derivs) / np.sum(peak_derivs))
            else:
                tm_val = float(t_vals[max_idx])
                
        else:
            # ==========================================
            # 经典模式：单次平滑 + 抛物线极大值插值 (最还原真实数据)
            # ==========================================
            window = get_dynamic_window(len(f_vals))
            smoothed_f = savgol_filter(f_vals, window_length=window, polyorder=3)
            derivative = np.gradient(smoothed_f, t_vals)
            
            start_idx, end_idx = int(len(derivative) * 0.1), int(len(derivative) * 0.9)
            local_max_idx = np.argmax(derivative[start_idx:end_idx])
            max_idx = local_max_idx + start_idx
            
            if 0 < max_idx < len(derivative) - 1:
                x1, x2, x3 = t_vals[max_idx-1], t_vals[max_idx], t_vals[max_idx+1]
                y1, y2, y3 = derivative[max_idx-1], derivative[max_idx], derivative[max_idx+1]
                denom = (y1 - 2*y2 + y3)
                if denom != 0:
                    x_peak = x2 - (x2 - x1) * (y3 - y1) / (2 * denom)
                    tm_val = float(x_peak) if x1 <= x_peak <= x3 else float(x2)
                else: tm_val = float(x2)
            else: tm_val = float(t_vals[max_idx])
            
        return t_vals, f_vals, smoothed_f, derivative, tm_val
    except Exception:
        return None, None, None, None, np.nan

def calculate_tm(temp_series, fluor_series, use_advanced_fitting):
    _, _, _, _, tm = get_curve_arrays(temp_series, fluor_series, use_advanced_fitting)
    return tm

@st.cache_data
def parse_instrument_data_cached(file_content):
    try:
        df = pd.read_excel(io.BytesIO(file_content), sheet_name="Melt Curve Raw Data", header=None)
    except:
        try: df = pd.read_excel(io.BytesIO(file_content), header=None, engine='xlrd')
        except: df = pd.read_excel(io.BytesIO(file_content), header=None, engine='openpyxl')

    header_row = 0
    for i, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        if any('Well' in val for val in row_values) and any('Temperature' in val for val in row_values):
            header_row = i
            break
            
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]
    
    temp_col, fluor_col = 'Temperature', 'Fluorescence'
    well_col = 'Well Position' if 'Well Position' in df.columns else 'Well'
        
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[fluor_col] = pd.to_numeric(df[fluor_col], errors='coerce')
    df = df.dropna(subset=[temp_col, fluor_col, well_col])
    
    df[well_col] = df[well_col].astype(str).apply(lambda x: re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', x.strip().upper().replace(" ", "")))
    wide_df = df.pivot_table(index=temp_col, columns=well_col, values=fluor_col, aggfunc='mean').reset_index()
    return wide_df[temp_col], {col: col for col in wide_df.columns if col != temp_col}, wide_df

def parse_wells_input(wells_str):
    if pd.isna(wells_str) or not str(wells_str).strip(): return []
    return [re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', w.strip().upper().replace(" ", "")) for w in str(wells_str).split(',') if w.strip()]

def update_excludes_callback(exc_key, new_wells):
    if exc_key not in st.session_state: st.session_state[exc_key] = []
    st.session_state[exc_key] = list(set(st.session_state[exc_key] + new_wells))

def assign_wells_callback(cfg_key, wells, name, role, ratio):
    if not wells: return
    df = st.session_state.get(cfg_key, pd.DataFrame(columns=["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"]))
    new_row = {"Wells (孔位)": ", ".join(wells), "Compound_Name (组别命名)": name, "Role (角色)": role, "Ratio (浓度比)": str(ratio).strip()}
    st.session_state[cfg_key] = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def assign_makeup_wells_callback(cfg_key, wells, cid, role):
    if not wells: return
    df = st.session_state.get(cfg_key, pd.DataFrame(columns=["Well (孔位)", "Compound_ID (化合物ID)", "Role (角色)"]))
    df = df[~df['Well (孔位)'].isin(wells)]
    new_rows = pd.DataFrame([{"Well (孔位)": w, "Compound_ID (化合物ID)": cid, "Role (角色)": role} for w in wells])
    st.session_state[cfg_key] = pd.concat([df, new_rows], ignore_index=True)
    
def set_control_callback(state_key, wells):
    if wells: st.session_state[state_key] = wells

def extract_ratio_value(ratio_str):
    nums = re.findall(r'\d+\.?\d*', str(ratio_str))
    return float(nums[-1]) if nums else 999999

# ==========================================
# 2. 交互式 96 孔板与全 Plotly 预览引擎
# ==========================================
def draw_interactive_plate_mod_a(loaded_wells, key_prefix, excludes=[], nc_wells=[], pc_wells=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3') 
            elif w in pc_wells: colors.append('#FF9800') 
            elif w in loaded_wells: colors.append('#4CAF50') 
            else: colors.append('#E0E0E0') 
            
    fig = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names,
        textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'
    ))
    fig.update_layout(
        title="🖱️ <b>【交互指南】</b>: 框选孔位并在下方指定角色。<br>图例：🔵 <b>NC</b> | 🟠 <b>PC</b> | 🟢 化合物 | 🔴 <b>已剔除</b> | ⚪ 无数据",
        xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40)
    )
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"):
            st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]


def draw_interactive_plate_val(loaded_wells, key_prefix, cfg_df, excludes=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    nc_wells, pc_wells, comp_wells = set(), set(), set()
    if not cfg_df.empty:
        for _, row in cfg_df.iterrows():
            ws = parse_wells_input(row['Wells (孔位)'])
            if row['Role (角色)'] == 'Baseline (NC)': nc_wells.update(ws)
            elif row['Role (角色)'] == 'Positive Control': pc_wells.update(ws)
            elif row['Role (角色)'] == 'Compound': comp_wells.update(ws)
            
    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3')
            elif w in pc_wells: colors.append('#FF9800')
            elif w in comp_wells: colors.append('#4CAF50')
            elif w in loaded_wells: colors.append('#B0BEC5')
            else: colors.append('#F5F5F5')
            
    fig = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names,
        textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'
    ))
    fig.update_layout(
        title="🖱️ <b>【圈选并定义】</b>: 框选孔位后在下方点击分配角色。<br>图例：🔵 <b>NC</b> | 🟠 <b>PC</b> | 🟢 <b>实验组</b> | 🔴 <b>已剔除</b> | ⚪ 待分配",
        xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40)
    )
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"):
            st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]

def draw_interactive_plate_makeup(loaded_wells, key_prefix, cfg_df, excludes=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state: st.session_state[clear_key] = 0
    nc_wells, comp_wells = set(), set()
    if not cfg_df.empty:
        for _, row in cfg_df.iterrows():
            w = row['Well (孔位)']
            if row['Role (角色)'] == 'Baseline (NC)': nc_wells.add(w)
            elif row['Role (角色)'] == 'Compound': comp_wells.add(w)

    rows, cols = list('ABCDEFGH')[::-1], list(range(1, 13))
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c); y_coords.append(r); well_names.append(w)
            if w in excludes: colors.append('#E53935')
            elif w in nc_wells: colors.append('#2196F3')
            elif w in comp_wells: colors.append('#4CAF50')
            elif w in loaded_wells: colors.append('#B0BEC5')
            else: colors.append('#F5F5F5')

    fig = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers+text', text=well_names, customdata=well_names,
        textposition="middle center", marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')), hoverinfo='text'
    ))
    fig.update_layout(
        title="🖱️ <b>【圈选定义补漏板】</b>: 框选孔位后在下方分配。<br>图例：🔵 <b>NC(基线)</b> | 🟢 <b>已映射化合物</b> | 🔴 <b>已剔除</b> | ⚪ 待分配",
        xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40)
    )
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 清除选框", key=f"btn_clr_{key_prefix}"):
            st.session_state[clear_key] += 1; st.rerun()
    with col_plot:
        event = st.plotly_chart(fig, key=f"plate_{key_prefix}_{st.session_state[clear_key]}", on_select="rerun", selection_mode=['points', 'box', 'lasso'])
    return [pt["customdata"] for pt in event.selection.get("points", [])]

def render_realtime_curve_viewer(selected_wells, temps, well_map, clean_df, use_adv_fit=False):
    if not selected_wells:
        st.info("💡 尚未选定。请在上方 96 孔板上进行拖拽或点选。")
        return

    st.subheader(f"🔍 实时预览选定孔位 ({len(selected_wells)} 个)")
    fig_prev = make_subplots(rows=1, cols=2, subplot_titles=("Raw Fluorescence", "Derivative (dF/dT)"))
    has_data = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, w in enumerate(selected_wells):
        if w in well_map:
            t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[w]], use_adv_fit)
            if t is not None:
                has_data = True
                c = colors[idx % len(colors)]
                fig_prev.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color=c, width=2), name=w, legendgroup=w), row=1, col=1)
                tm_str = f"{tm:.2f}℃" if not np.isnan(tm) else "N/A"
                fig_prev.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color=c, width=2), name=f"{w} (Tm: {tm_str})", legendgroup=w), row=1, col=2)
                if not np.isnan(tm): fig_prev.add_vline(x=tm, line=dict(color=c, width=1.5, dash='dash'), row=1, col=2)

    if has_data:
        fig_prev.update_layout(height=450, hovermode='x unified', plot_bgcolor='white', margin=dict(t=50, b=40))
        fig_prev.update_xaxes(title_text="Temperature (℃)", showline=True, linewidth=1, linecolor='black', showgrid=False)
        fig_prev.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
        if len(selected_wells) > 12: fig_prev.update_layout(showlegend=False) 
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.warning("⚠️ 你选定的孔位在文件中没有测定数据。")

# ==========================================
# 3. 全局配置与侧边栏
# ==========================================
st.set_page_config(page_title="DSF 高通量分析综合平台", layout="wide")

with st.sidebar:
    st.title("🔬 DSF 分析系统")
    analysis_mode = st.radio("请选择当前实验模块：", [
        "🎯 模块A: 初筛 (单板模式)", 
        "🧪 模块B: 再验证 (自定义排板)", 
        "🌌 模块C: 高通量多板初筛 (批量整合与补漏)"
    ])
    st.markdown("---")
    
    st.header("⚙️ 核心算法引擎配置")
    use_advanced_fitting = st.checkbox("🔬 开启峰顶质心拟合 (修复多峰/锯齿化)", value=False, help="开启后将启用双重 S-G 滤波与质心算法，强制平滑微分曲线并修正因杂质/聚集引发的多峰导致的 Tm 偏移。默认关闭（使用经典寻峰算法还原最真实数据）。")

    st.markdown("---")
    st.header("📊 图表显示设置")
    show_baseline_line = st.checkbox("显示阴性对照 (NC) 基准线/曲线", value=True)
    show_pc_line = st.checkbox("显示阳性对照 (PC) 参考线/曲线", value=True)
    
    st.markdown("---")
    st.header("🛡️ 异常数据质控 (QC) 设置")
    auto_qc = st.checkbox("🤖 开启基于数据的自适应智能质控 (Auto QC)", value=True)
    
    if not auto_qc:
        qc_max_bg = st.number_input("手动设置: 异常高荧光上限", value=400000, step=50000)
        qc_min_amp = st.number_input("手动设置: 最低有效振幅下限", value=30000, step=5000)
    else:
        qc_max_bg, qc_min_amp = None, None
        st.success("自适应 QC 已开启。系统将自动过滤空孔并精准动态计算 QC 阈值。")

# ==========================================
# 模块 A: 初筛 (单板模式)
# ==========================================
if analysis_mode == "🎯 模块A: 初筛 (单板模式)":
    st.title("🎯 单板初筛自动化分析")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: lib_type = st.selectbox("化合物库类型", ["FDA", "TCM", "NP"])
    with col2: lib_num = st.number_input("化合物库编号", min_value=1, value=1)
    with col3: hit_threshold = st.number_input("Hit 筛选阈值 (℃)", value=0.5, step=0.1)

    uploaded_file = st.file_uploader("上传原始数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], key="file_a")

    if uploaded_file:
        fname = uploaded_file.name
        exc_key = f"manual_exc_a_{fname}"
        nc_key = f"nc_wells_a_{fname}"
        pc_key = f"pc_wells_a_{fname}"
        
        if exc_key not in st.session_state: st.session_state[exc_key] = []
        if nc_key not in st.session_state: st.session_state[nc_key] = ["G12"]
        if pc_key not in st.session_state: st.session_state[pc_key] = ["H12"]
            
        temps, well_map, clean_df = parse_instrument_data_cached(uploaded_file.getvalue())
        
        st.markdown("---")
        st.subheader("👀 实时图谱预览器 & 手动调控")
        
        curr_nc = st.session_state[nc_key]
        curr_pc = st.session_state[pc_key]
        sel_w = draw_interactive_plate_mod_a(list(well_map.keys()), "mod_a", st.session_state[exc_key], curr_nc, curr_pc)
        render_realtime_curve_viewer(sel_w, temps, well_map, clean_df, use_advanced_fitting)
        
        st.markdown("##### ⚙️ 快速指派孔位角色与过滤")
        st.write(f"**当前选定:** {', '.join(sel_w) if sel_w else '无'} &nbsp;|&nbsp; **阴性对照 (NC):** {', '.join(curr_nc)} &nbsp;|&nbsp; **阳性对照 (PC):** {', '.join(curr_pc)}")
        
        c_btn1, c_btn2, c_btn3, c_exc = st.columns([1, 1, 1, 2])
        with c_btn1: st.button("🔵 设为 NC", on_click=set_control_callback, args=(nc_key, sel_w), use_container_width=True)
        with c_btn2: st.button("🟠 设为 PC", on_click=set_control_callback, args=(pc_key, sel_w), use_container_width=True)
        with c_btn3: st.button("🚫 一键剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)
        with c_exc: st.multiselect("当前剔除的异常孔位 (可手动撤销):", list(well_map.keys()), key=exc_key, label_visibility="collapsed")
                
        st.markdown("---")
        if st.button("🚀 执行全板初筛质控与计算", type="primary"):
            with st.spinner("正在全局解析与质控..."):
                excludes = st.session_state[exc_key]
                c_max, c_min = calculate_auto_qc_params(clean_df, well_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                st.session_state.a_qc_params = (c_max, c_min)
                    
                nc_tm = pc_tm = np.nan
                nc_tms = []
                for w in curr_nc:
                    if w in well_map and w not in excludes and check_curve_quality(clean_df[well_map[w]], c_max, c_min)[0]:
                        nc_tms.append(calculate_tm(temps, clean_df[well_map[w]], use_advanced_fitting))
                if nc_tms: nc_tm = np.mean(nc_tms)
                
                pc_tms = []
                for w in curr_pc:
                    if w in well_map and w not in excludes and check_curve_quality(clean_df[well_map[w]], c_max, c_min)[0]:
                        pc_tms.append(calculate_tm(temps, clean_df[well_map[w]], use_advanced_fitting))
                if pc_tms: pc_tm = np.mean(pc_tms)
                    
                valid_res, anomaly_res = [], []
                for r in list('ABCDEFGH'):
                    for c in range(1, 13):
                        w = f"{r}{c}"
                        if w in curr_nc or w in curr_pc: continue
                        cid = f"{lib_type.split()[0]}-{lib_num}-{w}"
                        
                        if w in excludes:
                            anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "人工剔除 (Manual Filtered)", "Delta Tm (℃)": np.nan})
                            continue
                        if w in well_map:
                            f_data = clean_df[well_map[w]]
                            is_v, reason = check_curve_quality(f_data, c_max, c_min)
                            if is_v:
                                tm_val = calculate_tm(temps, f_data, use_advanced_fitting)
                                dtm = tm_val - nc_tm if (pd.notnull(tm_val) and pd.notnull(nc_tm)) else np.nan
                                valid_res.append({"Well": w, "Compound_ID": cid, "Tm (℃)": np.round(tm_val, 2), "Delta Tm (℃)": np.round(dtm, 2)})
                            else: anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": reason, "Delta Tm (℃)": np.nan})
                        else: anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "No Data", "Delta Tm (℃)": np.nan})
                            
                a_anom_df = pd.DataFrame(anomaly_res) if anomaly_res else pd.DataFrame(columns=["Well", "Compound_ID", "QC_Status", "Delta Tm (℃)"])
                st.session_state.update(a_valid_df=pd.DataFrame(valid_res), a_anomaly_df=a_anom_df, a_nc=nc_tm, a_pc=pc_tm, mod_a_done=True)

    if st.session_state.get('mod_a_done'):
        v_df, a_df = st.session_state.a_valid_df, st.session_state.a_anomaly_df
        c_max, c_min = st.session_state.a_qc_params
        
        st.subheader("📊 初筛质控分析报告")
        st.info(f"💡 当前板执行的 QC 阈值 -> **自发荧光上限**: {c_max:.0f} | **有效振幅下限**: {c_min:.0f}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("阴性对照 Tm (基线)", f"{st.session_state.a_nc:.2f} ℃" if pd.notnull(st.session_state.a_nc) else "人工剔除/异常")
        m2.metric("阳性对照 Tm", f"{st.session_state.a_pc:.2f} ℃" if pd.notnull(st.session_state.a_pc) else "人工剔除/异常")
        m3.metric("通过 QC 孔位", len(v_df))
        m4.metric("异常/拦截 (含人工)", len(a_df))
        
        st.subheader("🎯 质控后化合物迁移散点图")
        if not v_df.empty:
            vp = v_df.dropna(subset=["Delta Tm (℃)"])
            xj = np.random.uniform(-1, 1, size=len(vp))
            h = vp[vp["Delta Tm (℃)"] >= hit_threshold]
            nh = vp[vp["Delta Tm (℃)"] < hit_threshold]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xj[vp["Delta Tm (℃)"]<hit_threshold], y=nh["Delta Tm (℃)"], mode='markers', marker=dict(color='#B0BEC5', size=8), text=nh["Compound_ID"], hoverinfo='text+y'))
            fig.add_trace(go.Scatter(x=xj[vp["Delta Tm (℃)"]>=hit_threshold], y=h["Delta Tm (℃)"], mode='markers', marker=dict(color='#E53935', size=12, line_width=1.5, line_color='black'), text=h["Compound_ID"], hoverinfo='text+y'))
            
            if show_baseline_line:
                fig.add_hline(y=0, line_color="black", line_width=2, annotation_text="Baseline (NC)", annotation_position="bottom right", annotation_font_size=12)
            if show_pc_line and pd.notnull(st.session_state.a_pc) and pd.notnull(st.session_state.a_nc):
                pc_dtm = st.session_state.a_pc - st.session_state.a_nc
                fig.add_hline(y=pc_dtm, line_dash="dot", line_color="#FF9800", line_width=2, annotation_text=f"PC Line ({pc_dtm:.2f}℃)", annotation_position="top right", annotation_font_size=12, annotation_font_color="#FF9800")
                
            fig.add_hline(y=hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({hit_threshold}℃)")
            fig.update_layout(height=450, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", plot_bgcolor='white', showlegend=False)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(v_df.sort_values(by="Delta Tm (℃)", ascending=False), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 异常数据审查与判读")
        view_anomaly_df = a_df[a_df['QC_Status'] != 'No Data'] if not a_df.empty else pd.DataFrame()
        if not view_anomaly_df.empty:
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(view_anomaly_df[['Well', 'Compound_ID', 'QC_Status']], hide_index=True)
            with view_col2:
                sel_well = st.selectbox("选择要复核的异常孔位:", view_anomaly_df['Well'].tolist(), key="mod_a_anom")
                if sel_well:
                    temps_a, well_map_a, clean_df_a = parse_instrument_data_cached(uploaded_file.getvalue())
                    reason = view_anomaly_df[view_anomaly_df['Well'] == sel_well].iloc[0]['QC_Status']
                    if sel_well in well_map_a:
                        t, f, s, d, tm = get_curve_arrays(temps_a, clean_df_a[well_map_a[sel_well]], use_advanced_fitting)
                        if t is not None:
                            title_reason = "Manual Filtered" if "人工剔除" in reason else ("Auto-Fluor" if "自发荧光" in reason else "Flat/Quenched")
                            fig_a = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{sel_well}] - {title_reason}", "Derivative - (Excluded)"))
                            fig_a.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                            fig_a.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                            fig_a.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                            for annotation in fig_a['layout']['annotations']: annotation['font'] = dict(color='red')
                            fig_a.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                            fig_a.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.success("🎉 当前板没有检测到淬灭或高自发荧光的异常数据！")

# ==========================================
# 模块 B: 化合物再验证 (自定义排板)
# ==========================================
elif analysis_mode == "🧪 模块B: 再验证 (自定义排板)":
    st.title("🧪 化合物结合验证自动化分析")
    uploaded_files = st.file_uploader("支持批量上传", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_b")
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("👀 跨板实时预览与过滤")
        preview_file_name = st.selectbox("选择要配置的孔板文件:", [f.name for f in uploaded_files])
        preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
        
        cfg_key = f"val_cfg_{preview_file_name}"
        exc_key = f"manual_exc_b_{preview_file_name}"
        if cfg_key not in st.session_state: st.session_state[cfg_key] = pd.DataFrame(columns=["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"])
        if exc_key not in st.session_state: st.session_state[exc_key] = []
        
        p_temps, p_well_map, p_clean_df = parse_instrument_data_cached(preview_file.getvalue())
        sel_w = draw_interactive_plate_val(list(p_well_map.keys()), f"mod_b_{preview_file_name}", st.session_state[cfg_key], st.session_state[exc_key])
        render_realtime_curve_viewer(sel_w, p_temps, p_well_map, p_clean_df, use_advanced_fitting)
        
        st.markdown("##### 📝 快速定义选定孔位")
        st.write(f"**当前选定:** {', '.join(sel_w) if sel_w else '无 (请在上方网格中框选)'}")
        
        c1, c2 = st.columns(2)
        c_name = c1.text_input("组别命名 (Compound Name):", key=f"n_b_{preview_file_name}")
        c_ratio = c2.text_input("摩尔浓度比 (Ratio, 留空或填0代表对照):", value="1:1", key=f"r_b_{preview_file_name}")
        
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.button("✅ 设为实验组", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name, "Compound", c_ratio), use_container_width=True)
        with b2: st.button("🔵 设为 NC", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name if c_name else "DMSO_NC", "Baseline (NC)", "0"), use_container_width=True)
        with b3: st.button("🟠 设为 PC", on_click=assign_wells_callback, args=(cfg_key, sel_w, c_name if c_name else "Positive_Ctrl", "Positive Control", "0"), use_container_width=True)
        with b4: st.button("🚫 设为异常剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)
            
        st.markdown("---")
        col_export, col_import = st.columns(2)
        with col_export:
            st.markdown("###### 📤 导出配置模板")
            if not st.session_state[cfg_key].empty:
                csv_data = st.session_state[cfg_key].to_csv(index=False).encode('utf-8-sig')
                st.download_button("一键下载当前配置 (.csv)", data=csv_data, file_name=f"{preview_file_name}_layout.csv", mime="text/csv", use_container_width=True)
            else:
                st.button("一键下载当前配置 (.csv)", disabled=True, use_container_width=True)
                
        with col_import:
            st.markdown("###### 📥 导入已有配置")
            uploaded_cfg = st.file_uploader("上传配置 (.csv)", type=["csv"], key=f"up_cfg_{preview_file_name}", label_visibility="collapsed")
            if uploaded_cfg:
                if st.button("🚨 确认覆盖当前配置", use_container_width=True):
                    try:
                        imp_df = pd.read_csv(uploaded_cfg).fillna("")
                        req_cols = ["Wells (孔位)", "Compound_Name (组别命名)", "Role (角色)", "Ratio (浓度比)"]
                        if all(c in imp_df.columns for c in req_cols):
                            imp_df['Ratio (浓度比)'] = imp_df['Ratio (浓度比)'].astype(str)
                            st.session_state[cfg_key] = imp_df[req_cols]
                            st.rerun()
                        else:
                            st.error("列名不匹配，请确保使用系统导出的模板文件！")
                    except Exception as e:
                        st.error(f"读取配置失败: {str(e)}")

        st.session_state[cfg_key] = st.data_editor(
            st.session_state[cfg_key], 
            column_config={
                "Wells (孔位)": st.column_config.TextColumn(required=True),
                "Compound_Name (组别命名)": st.column_config.TextColumn(required=True),
                "Role (角色)": st.column_config.SelectboxColumn(options=["Baseline (NC)", "Positive Control", "Compound", "Ignore"], required=True),
                "Ratio (浓度比)": st.column_config.TextColumn()
            },
            num_rows="dynamic", use_container_width=True, key=f"ed_b_{preview_file_name}"
        )
                
        if st.button("🚀 计算验证结果", type="primary"):
            with st.spinner("分析中..."):
                all_res, anom_res, b_qc_records = [], [], []
                rep_nc_curve = None
                rep_pc_curve = None
                
                for file in uploaded_files:
                    cfg = st.session_state.get(f"val_cfg_{file.name}", pd.DataFrame())
                    t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                    excludes = st.session_state.get(f"manual_exc_b_{file.name}", [])
                    c_max, c_min = calculate_auto_qc_params(c_df, w_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                    b_qc_records.append({'File': file.name, 'Max_BG': c_max, 'Min_Amp': c_min})
                    
                    nc_tms = []
                    if not cfg.empty and 'Role (角色)' in cfg.columns:
                        for w in parse_wells_input(','.join(cfg[cfg['Role (角色)']=='Baseline (NC)']['Wells (孔位)'].astype(str))):
                            if w in excludes: 
                                anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '人工剔除'})
                                continue
                            if w in w_map:
                                is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                if is_v:
                                    t_arr, f_arr, s_arr, d_arr, tm_single = get_curve_arrays(t, c_df[w_map[w]], use_advanced_fitting)
                                    nc_tms.append(tm_single)
                                    if rep_nc_curve is None:
                                        rep_nc_curve = (t_arr, d_arr, tm_single)
                                else:
                                    anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': rsn})
                            else:
                                anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '无数据'})
                                
                    base = np.mean(nc_tms) if nc_tms else np.nan
                    
                    if not cfg.empty and 'Role (角色)' in cfg.columns:
                        for _, r in cfg[cfg['Role (角色)'].isin(['Compound', 'Positive Control'])].iterrows():
                            cn = r['Compound_Name (组别命名)']
                            role_str = r['Role (角色)']
                            ratio = r['Ratio (浓度比)']
                            w_list = parse_wells_input(r['Wells (孔位)'])
                            for w in w_list:
                                if w in excludes:
                                    anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': '人工剔除'})
                                    continue
                                if w in w_map:
                                    is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                    if is_v: 
                                        t_arr, f_arr, s_arr, d_arr, tm_single = get_curve_arrays(t, c_df[w_map[w]], use_advanced_fitting)
                                        dtm = np.round(tm_single - base, 2) if pd.notnull(base) else np.nan
                                        
                                        if role_str == 'Positive Control' and rep_pc_curve is None:
                                            rep_pc_curve = (t_arr, d_arr, tm_single)
                                            
                                        all_res.append({'File': file.name, 'Compound': cn, 'Role': role_str, 'Ratio': ratio, 'Well': w, 'Tm(℃)': np.round(tm_single, 2), 'Delta_Tm(℃)': dtm, 't_array': t_arr, 'd_array': d_arr, 'raw_tm': tm_single})
                                    else: 
                                        anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': rsn})
                
                b_anom_df = pd.DataFrame(anom_res) if anom_res else pd.DataFrame(columns=['File', 'Compound', 'Well', 'QC_Status'])
                b_res_df = pd.DataFrame(all_res) if all_res else pd.DataFrame(columns=['File', 'Compound', 'Role', 'Ratio', 'Well', 'Tm(℃)', 'Delta_Tm(℃)', 't_array', 'd_array', 'raw_tm'])
                st.session_state.update(b_res=b_res_df, b_anom=b_anom_df, b_qc_recs=b_qc_records, b_rep_nc=rep_nc_curve, b_rep_pc=rep_pc_curve, mod_b_done=True)

    if st.session_state.get('mod_b_done'):
        st.subheader("📋 验证分析成果")
        res_df = st.session_state.b_res
        anom_df = st.session_state.b_anom
        
        st.info("💡 系统已执行自适应智能质控。")
        
        if not res_df.empty:
            if 'Role' in res_df.columns:
                plot_df = res_df[res_df['Role'] == 'Compound'].copy()
            else:
                plot_df = res_df.copy()
                
            if not plot_df.empty:
                plot_df['sort_key'] = plot_df['Ratio'].apply(extract_ratio_value)
                unique_compounds = plot_df['Compound'].unique()
                npg_colors = ['#4DBBD5', '#00A087', '#E64B35', '#F39B7F', '#8491B4', '#91D1C2']
                
                use_delta = plot_df['Delta_Tm(℃)'].notna().any()
                y_col = 'Delta_Tm(℃)' if use_delta else 'Tm(℃)'
                if use_delta:
                    global_y_min = plot_df['Delta_Tm(℃)'].min()
                    global_y_max = plot_df['Delta_Tm(℃)'].max()
                else:
                    global_y_min = plot_df['Tm(℃)'].min()
                    global_y_max = plot_df['Tm(℃)'].max()
                    
                y_margin = (global_y_max - global_y_min) * 0.15 if global_y_max != global_y_min else 0.5
                global_y_range = [global_y_min - y_margin, global_y_max + y_margin]

                all_tms_for_x = plot_df['raw_tm'].dropna().tolist()
                if show_baseline_line and st.session_state.b_rep_nc:
                    all_tms_for_x.append(st.session_state.b_rep_nc[2])
                if show_pc_line and st.session_state.get('b_rep_pc'):
                    all_tms_for_x.append(st.session_state.b_rep_pc[2])
                    
                if all_tms_for_x:
                    focus_tm = np.median(all_tms_for_x)
                    x_axis_range = [focus_tm - 12, focus_tm + 12]
                else:
                    x_axis_range = [30, 90]
                
                st.subheader("📈 衍生物熔解曲线阵列 (Derivative Curves Profiles)")
                
                cols_cnt = 4
                rows_cnt = (len(unique_compounds) - 1) // cols_cnt + 1
                
                fig_curves = make_subplots(rows=rows_cnt, cols=cols_cnt, subplot_titles=unique_compounds, shared_yaxes=False)
                fig_trends = make_subplots(rows=rows_cnt, cols=cols_cnt, subplot_titles=unique_compounds, shared_yaxes=False)
                
                added_leg_ratios = set()
                
                for idx, comp in enumerate(unique_compounds):
                    r = idx // cols_cnt + 1
                    c = idx % cols_cnt + 1
                    comp_df = plot_df[plot_df['Compound'] == comp].sort_values('sort_key')
                    
                    if show_baseline_line and st.session_state.b_rep_nc:
                        t_nc, d_nc, _ = st.session_state.b_rep_nc
                        show_nc_leg = "Apo (Baseline)" not in added_leg_ratios
                        fig_curves.add_trace(go.Scatter(
                            x=t_nc, y=d_nc, mode='lines', line=dict(color='#333333', dash='dash', width=2), 
                            name="Apo (Baseline)", showlegend=show_nc_leg
                        ), row=r, col=c)
                        added_leg_ratios.add("Apo (Baseline)")
                        
                    if show_pc_line and st.session_state.get('b_rep_pc'):
                        t_pc, d_pc, _ = st.session_state.b_rep_pc
                        show_pc_leg = "Positive Control" not in added_leg_ratios
                        fig_curves.add_trace(go.Scatter(
                            x=t_pc, y=d_pc, mode='lines', line=dict(color='#FF9800', dash='dashdot', width=2), 
                            name="Positive Control", showlegend=show_pc_leg
                        ), row=r, col=c)
                        added_leg_ratios.add("Positive Control")
                        
                    trend_data = []
                    for j, ratio in enumerate(comp_df['Ratio'].unique()):
                        ratio_df = comp_df[comp_df['Ratio'] == ratio]
                        rep_row = ratio_df.iloc[0] 
                        
                        c_color = npg_colors[j % len(npg_colors)]
                        leg_name = f"{ratio}"
                        show_leg = leg_name not in added_leg_ratios
                        
                        fig_curves.add_trace(go.Scatter(
                            x=rep_row['t_array'], y=rep_row['d_array'], mode='lines', 
                            line=dict(color=c_color, width=2.5), name=leg_name, 
                            showlegend=show_leg, legendgroup=leg_name
                        ), row=r, col=c)
                        added_leg_ratios.add(leg_name)
                        
                        peak_idx = np.argmax(rep_row['d_array'])
                        fig_curves.add_trace(go.Scatter(
                            x=[rep_row['t_array'][peak_idx]], y=[rep_row['d_array'][peak_idx]], 
                            mode='markers', marker=dict(color=c_color, symbol='cross', size=8), 
                            showlegend=False, legendgroup=leg_name
                        ), row=r, col=c)
                        
                        trend_data.append({'Ratio': str(ratio), 'Mean_Delta': ratio_df[y_col].mean()})
                    
                    if trend_data:
                        t_df = pd.DataFrame(trend_data)
                        comp_color = npg_colors[idx % len(npg_colors)]
                        
                        fig_trends.add_hline(y=0, line_color="black", row=r, col=c)
                        fig_trends.add_trace(go.Scatter(
                            x=t_df['Ratio'], y=t_df['Mean_Delta'], mode='lines+markers+text',
                            text=t_df['Mean_Delta'].apply(lambda x: f"{x:.1f}"), textposition='top center',
                            marker=dict(size=8, color=comp_color, line=dict(width=1, color='black')),
                            line=dict(width=2, color=comp_color), showlegend=False
                        ), row=r, col=c)
                
                fig_curves.update_layout(
                    height=300 * rows_cnt, plot_bgcolor='white', margin=dict(t=40, r=20, b=20, l=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=13))
                )
                fig_trends.update_layout(height=250 * rows_cnt, plot_bgcolor='white', margin=dict(t=40, r=20, b=20, l=20))
                
                for i in range(1, rows_cnt * cols_cnt + 1):
                    fig_curves.layout[f'xaxis{i}'].update(showline=True, linecolor='black', showgrid=False, range=x_axis_range)
                    fig_curves.layout[f'yaxis{i}'].update(showline=True, linecolor='black', showgrid=False, showticklabels=False, ticks='')
                    
                    fig_trends.layout[f'xaxis{i}'].update(showline=True, linecolor='black', showgrid=False)
                    fig_trends.layout[f'yaxis{i}'].update(showline=True, linecolor='black', showgrid=True, gridcolor='#EEEEEE', title="ΔTm" if i%cols_cnt==1 else "", range=global_y_range)

                st.plotly_chart(fig_curves, use_container_width=True)
                
                st.subheader("📉 剂量响应趋势阵列 (Dose-Response Trends)")
                st.plotly_chart(fig_trends, use_container_width=True)
                            
            else:
                st.warning("⚠️ 当前数据中只包含对照组（或实验组数据全部因 QC 被拦截），没有有效数据可供绘图。")
                
        st.markdown("##### 📁 验证结果原始数据表 (包含对照组结果, 可导出 CSV)")
        if not res_df.empty:
            st.dataframe(res_df.drop(columns=['Role', 'sort_key', 't_array', 'd_array', 'raw_tm'], errors='ignore'), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 异常数据审查与判读")
        if not anom_df.empty:
            st.warning(f"⚠️ 发现 {len(anom_df)} 个被质控拦截的异常孔位（如果是 NC 被拦截，将导致后续无法计算 ΔTm）")
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(anom_df, hide_index=True)
            with view_col2:
                anomaly_opts = [f"[{row['File']}] {row['Well']} ({row['Compound']})" for _, row in anom_df.iterrows()]
                sel_opt = st.selectbox("选择要复核的异常孔位:", anomaly_opts, key="mod_b_anom_sel")
                
                if sel_opt:
                    idx = anomaly_opts.index(sel_opt)
                    target = anom_df.iloc[idx]
                    src_file = target['File']
                    well = target['Well']
                    reason = target['QC_Status']
                    
                    target_file_obj = next((f for f in uploaded_files if f.name == src_file), None)
                    if target_file_obj:
                        t_b, w_map_b, c_df_b = parse_instrument_data_cached(target_file_obj.getvalue())
                        if well in w_map_b:
                            t, f, s, d, tm = get_curve_arrays(t_b, c_df_b[w_map_b[well]], use_advanced_fitting)
                            if t is not None:
                                title_reason = "Manual Filtered" if "人工剔除" in reason else ("Auto-Fluor" if "自发荧光" in reason else "Flat/Quenched")
                                fig_b = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{well}] - {title_reason}", "Derivative - (Excluded)"))
                                fig_b.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                                fig_b.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                                fig_b.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                                for annotation in fig_b['layout']['annotations']: annotation['font'] = dict(color='red')
                                fig_b.update_xaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                fig_b.update_yaxes(showline=True, linewidth=1, linecolor='black', showgrid=False)
                                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.success("🎉 导入的所有复测孔位均完美通过了 QC 质控！")

# ==========================================
# 模块 C: 高通量多板初筛 (批量整合与补漏)
# ==========================================
elif analysis_mode == "🌌 模块C: 高通量多板初筛 (批量整合与补漏)":
    st.title("🌌 高通量多板初筛与补漏整合")
    global_hit_threshold = st.number_input("全局 Hit 筛选阈值 (ΔTm ℃)", value=0.5, step=0.1)
    
    uploaded_files = st.file_uploader("批量上传各板测定数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_c")
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("👀 多板配置中心")
        preview_file_name = st.selectbox("当前配置面板 (切换文件):", [f.name for f in uploaded_files], key="c_sel_file")
        preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
        
        p_type_key = f"c_ptype_{preview_file_name}"
        if p_type_key not in st.session_state: st.session_state[p_type_key] = "🔲 标准规则板"
        p_type = st.radio("该板类型", ["🔲 标准规则板", "🧩 自定义补漏板 (交互式单孔定义)"], key=p_type_key, horizontal=True)

        temps, well_map, clean_df = parse_instrument_data_cached(preview_file.getvalue())

        if "标准" in p_type:
            st.info("此板按 A1-H12 顺序自动生成化合物 ID（将自动忽略指定的 NC 与 H12 阳性孔）。")
            c1, c2, c3 = st.columns(3)
            c1.selectbox("所属库类型", ["FDA", "TCM", "NP", "Custom"], key=f"lt_{preview_file_name}")
            c2.number_input("所属板编号", min_value=1, value=1, key=f"ln_{preview_file_name}")
            c3.text_input("阴性对照 (NC) 孔位", value="G12", key=f"nc_{preview_file_name}")
            
            exc_key = f"manual_exc_c_std_{preview_file_name}"
            if exc_key not in st.session_state: st.session_state[exc_key] = []
            st.multiselect("不参与计算的剔除孔位：", list(well_map.keys()), key=exc_key)
            
        else:
            cfg_key = f"c_mkp_cfg_{preview_file_name}"
            exc_key = f"manual_exc_c_mkp_{preview_file_name}"
            if cfg_key not in st.session_state: st.session_state[cfg_key] = pd.DataFrame(columns=["Well (孔位)", "Compound_ID (化合物ID)", "Role (角色)"])
            if exc_key not in st.session_state: st.session_state[exc_key] = []

            sel_w = draw_interactive_plate_makeup(list(well_map.keys()), f"mod_c_{preview_file_name}", st.session_state[cfg_key], st.session_state[exc_key])
            render_realtime_curve_viewer(sel_w, temps, well_map, clean_df, use_advanced_fitting)

            st.markdown("##### 📝 快速定义选定补漏孔位")
            st.write(f"**当前选定:** {', '.join(sel_w) if sel_w else '无 (请在上方网格中框选)'}")
            
            c_name = st.text_input("化合物 ID / 命名:", placeholder="框选多孔赋予同名时，可随后在下方表格内双击单独修改", key=f"c_name_{preview_file_name}")
            
            cb1, cb2, cb3 = st.columns(3)
            with cb1: st.button("✅ 设为实验组化合物", on_click=assign_makeup_wells_callback, args=(cfg_key, sel_w, c_name, "Compound"), use_container_width=True)
            with cb2: st.button("🔵 设为阴性对照 (NC)", on_click=assign_makeup_wells_callback, args=(cfg_key, sel_w, "NC_Baseline", "Baseline (NC)"), use_container_width=True)
            with cb3: st.button("🚫 设为异常剔除", on_click=update_excludes_callback, args=(exc_key, sel_w), use_container_width=True)

            st.session_state[cfg_key] = st.data_editor(st.session_state[cfg_key], num_rows="dynamic", use_container_width=True, key=f"ed_c_{preview_file_name}")

        st.markdown("---")
        if st.button("🚀 执行多板混合解析与全局统计", type="primary"):
            with st.spinner("正在逐板执行独立 QC 与基线运算..."):
                master_valid, master_qc_recs, master_pc_dtms = [], [], []
                
                for file in uploaded_files:
                    f_ptype = st.session_state.get(f"c_ptype_{file.name}", "🔲 标准规则板")
                    t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                    
                    if "标准" in f_ptype: excludes = st.session_state.get(f"manual_exc_c_std_{file.name}", [])
                    else: excludes = st.session_state.get(f"manual_exc_c_mkp_{file.name}", [])
                        
                    c_max, c_min = calculate_auto_qc_params(c_df, w_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                    master_qc_recs.append({'File': file.name, 'Plate_Type': "标准板" if "标准" in f_ptype else "补漏板", 'Max_BG': int(c_max), 'Min_Amp': int(c_min)})
                    
                    nc_tm = np.nan
                    if "标准" in f_ptype:
                        nc_w = st.session_state.get(f"nc_{file.name}", "G12")
                        if nc_w in w_map and nc_w not in excludes and check_curve_quality(c_df[w_map[nc_w]], c_max, c_min)[0]:
                            nc_tm = calculate_tm(t, c_df[w_map[nc_w]], use_advanced_fitting)
                            
                        pc_w = "H12"
                        if pc_w in w_map and pc_w not in excludes and check_curve_quality(c_df[w_map[pc_w]], c_max, c_min)[0] and pd.notnull(nc_tm):
                            master_pc_dtms.append(calculate_tm(t, c_df[w_map[pc_w]], use_advanced_fitting) - nc_tm)
                    else:
                        cfg_df = st.session_state.get(f"c_mkp_cfg_{file.name}", pd.DataFrame())
                        nc_tms = []
                        if not cfg_df.empty and 'Role (角色)' in cfg_df.columns:
                            nc_wells = cfg_df[cfg_df['Role (角色)'] == 'Baseline (NC)']['Well (孔位)'].tolist()
                            for nc_w in nc_wells:
                                if nc_w in w_map and nc_w not in excludes and check_curve_quality(c_df[w_map[nc_w]], c_max, c_min)[0]:
                                    nc_tms.append(calculate_tm(t, c_df[w_map[nc_w]], use_advanced_fitting))
                            if nc_tms: nc_tm = np.mean(nc_tms)
                    
                    if "标准" in f_ptype:
                        l_type = st.session_state.get(f"lt_{file.name}", "FDA")
                        l_num = st.session_state.get(f"ln_{file.name}", 1)
                        nc_w = st.session_state.get(f"nc_{file.name}", "G12")
                        for r in list('ABCDEFGH'):
                            for c in range(1, 13):
                                w = f"{r}{c}"
                                if w in [nc_w, "H12"]: continue
                                if w in excludes: continue
                                cid = f"{l_type.split()[0]}-{l_num}-{w}"
                                if w in w_map:
                                    is_v, _ = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                    if is_v:
                                        tm_val = calculate_tm(t, c_df[w_map[w]], use_advanced_fitting)
                                        dtm = tm_val - nc_tm if pd.notnull(nc_tm) else np.nan
                                        master_valid.append({'File': file.name, 'Well': w, 'Compound_ID': cid, 'Tm (℃)': tm_val, 'Delta_Tm (℃)': dtm})
                    else:
                        cfg_df = st.session_state.get(f"c_mkp_cfg_{file.name}", pd.DataFrame())
                        if not cfg_df.empty and 'Role (角色)' in cfg_df.columns:
                            comp_df = cfg_df[cfg_df['Role (角色)'] == 'Compound']
                            for _, row in comp_df.iterrows():
                                w, cid = row['Well (孔位)'], row['Compound_ID (化合物ID)']
                                if w in excludes: continue
                                if w in w_map:
                                    is_v, _ = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                    if is_v:
                                        tm_val = calculate_tm(t, c_df[w_map[w]], use_advanced_fitting)
                                        dtm = tm_val - nc_tm if pd.notnull(nc_tm) else np.nan
                                        master_valid.append({'File': file.name, 'Well': w, 'Compound_ID': cid, 'Tm (℃)': tm_val, 'Delta_Tm (℃)': dtm})
                
                st.session_state.update(c_res=pd.DataFrame(master_valid), c_qc=pd.DataFrame(master_qc_recs), c_pc_dtm=(np.mean(master_pc_dtms) if master_pc_dtms else np.nan), mod_c_done=True)

    if st.session_state.get('mod_c_done'):
        c_res = st.session_state.c_res
        st.subheader("🌐 全局筛选成果 (HTS Global Overview)")
        
        valid_hits = c_res[c_res['Delta_Tm (℃)'] >= global_hit_threshold] if not c_res.empty else pd.DataFrame()
        c1, c2, c3 = st.columns(3)
        c1.metric("解析总有效化合物", len(c_res))
        c2.metric(f"检出 Hit 总数 (≥{global_hit_threshold}℃)", len(valid_hits))
        c3.metric("整体阳性率", f"{len(valid_hits)/len(c_res)*100:.2f} %" if not c_res.empty else "0 %")
        
        st.subheader("🌌 全局化合物基线偏移散点图")
        if not c_res.empty:
            vp = c_res.dropna(subset=['Delta_Tm (℃)']).copy()
            vp['Hover_Text'] = vp['Compound_ID'] + "<br>File: " + vp['File'] + " | Well: " + vp['Well']
            xj = np.random.uniform(-1, 1, size=len(vp))
            h = vp[vp["Delta_Tm (℃)"] >= global_hit_threshold]
            nh = vp[vp["Delta_Tm (℃)"] < global_hit_threshold]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xj[vp["Delta_Tm (℃)"]<global_hit_threshold], y=nh["Delta_Tm (℃)"],
                mode='markers', marker=dict(color='#B0BEC5', size=6, opacity=0.6), name='Background', text=nh['Hover_Text'], hoverinfo='text+y'
            ))
            fig.add_trace(go.Scatter(
                x=xj[vp["Delta_Tm (℃)"]>=global_hit_threshold], y=h["Delta_Tm (℃)"],
                mode='markers', marker=dict(color='#E64B35', size=10, line_width=1.5, line_color='black'), name='Hits', text=h['Hover_Text'], hoverinfo='text+y'
            ))
            
            if show_baseline_line:
                fig.add_hline(y=0, line_color="black", line_width=2, annotation_text="Baseline (NC)", annotation_position="bottom right", annotation_font_size=12)
            c_pc = st.session_state.get('c_pc_dtm', np.nan)
            if show_pc_line and pd.notnull(c_pc):
                fig.add_hline(y=c_pc, line_dash="dot", line_color="#FF9800", line_width=2, annotation_text=f"Mean PC Line ({c_pc:.2f}℃)", annotation_position="top right", annotation_font_size=12, annotation_font_color="#FF9800")
            
            fig.add_hline(y=global_hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Global Thr ({global_hit_threshold}℃)")
            fig.update_layout(
                height=500, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", 
                plot_bgcolor='white', hovermode='closest', font=dict(family="Arial", size=14, color="black")
            )
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            
        st.dataframe(c_res.sort_values(by="Delta_Tm (℃)", ascending=False), use_container_width=True)
