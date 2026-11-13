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
    bgs, amps = [], []
    for w_key, w_col in well_map.items():
        if w_key in excludes: continue 
        f_vals = pd.to_numeric(clean_df[w_col], errors='coerce').dropna().values
        if len(f_vals) >= 10:
            bgs.append(np.mean(f_vals[:5]))
            amps.append(np.max(f_vals) - np.min(f_vals))
            
    if bgs and amps:
        return np.median(bgs) * 5, np.median(amps) * 0.2 
    return 400000, 30000

def check_curve_quality(fluor_series, max_bg, min_amp):
    f_vals = pd.to_numeric(fluor_series, errors='coerce').dropna().values
    if len(f_vals) < 10: return False, "Data too short"
    init_bg = np.mean(f_vals[:5])
    amplitude = np.max(f_vals) - np.min(f_vals)
    if init_bg > max_bg: return False, "高自发荧光"
    if amplitude < min_amp: return False, "曲线水平/淬灭"
    return True, "Pass"

def get_curve_arrays(temp_series, fluor_series):
    try:
        f_vals = pd.to_numeric(fluor_series, errors='coerce').astype(float).values
        t_vals = pd.to_numeric(temp_series, errors='coerce').astype(float).values
        mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
        f_vals, t_vals = f_vals[mask], t_vals[mask]
        
        if len(f_vals) < 10: return None, None, None, None, np.nan
        
        window = get_dynamic_window(len(f_vals))
        smoothed = savgol_filter(f_vals, window_length=window, polyorder=3)
        derivative = np.gradient(smoothed, t_vals)
        
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
            
        return t_vals, f_vals, smoothed, derivative, tm_val
    except:
        return None, None, None, None, np.nan

def calculate_tm(temp_series, fluor_series):
    _, _, _, _, tm = get_curve_arrays(temp_series, fluor_series)
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

# 【回调函数区】
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

# 【自然数字提取器】
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

def render_realtime_curve_viewer(selected_wells, temps, well_map, clean_df):
    if not selected_wells:
        st.info("💡 尚未选定。请在上方 96 孔板上进行拖拽或点选。")
        return

    st.subheader(f"🔍 实时预览选定孔位 ({len(selected_wells)} 个)")
    fig_prev = make_subplots(rows=1, cols=2, subplot_titles=("Raw Fluorescence", "Derivative (dF/dT)"))
    has_data = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, w in enumerate(selected_wells):
        if w in well_map:
            t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[w]])
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
    
    st.header("🛡️ 异常数据质控 (QC) 设置")
    auto_qc = st.checkbox("🤖 开启基于数据的自适应智能质控 (Auto QC)", value=True)
    
    if not auto_qc:
        qc_max_bg = st.number_input("手动设置: 异常高荧光上限", value=400000, step=50000)
        qc_min_amp = st.number_input("手动设置: 最低有效振幅下限", value=30000, step=5000)
    else:
        qc_max_bg, qc_min_amp = None, None
        st.success("自适应 QC 已开启。系统将在后台为每块板单独动态计算阈值。")

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
        render_realtime_curve_viewer(sel_w, temps, well_map, clean_df)
        
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
                        nc_tms.append(calculate_tm(temps, clean_df[well_map[w]]))
                if nc_tms: nc_tm = np.mean(nc_tms)
                
                pc_tms = []
                for w in curr_pc:
                    if w in well_map and w not in excludes and check_curve_quality(clean_df[well_map[w]], c_max, c_min)[0]:
                        pc_tms.append(calculate_tm(temps, clean_df[well_map[w]]))
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
                                tm_val = calculate_tm(temps, f_data)
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
            fig.add_hline(y=hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({hit_threshold}℃)")
            fig.update_layout(height=450, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", plot_bgcolor='white', showlegend=False)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(v_df.sort_values(by="Delta Tm (℃)", ascending=False), use_container_width=True)

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
        render_realtime_curve_viewer(sel_w, p_temps, p_well_map, p_clean_df)
        
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
        st.session_state[cfg_key] = st.data_editor(st.session_state[cfg_key], num_rows="dynamic", use_container_width=True, key=f"ed_b_{preview_file_name}")
                
        if st.button("🚀 计算验证结果", type="primary"):
            with st.spinner("分析中..."):
                all_res, anom_res, b_qc_records = [], [], []
                for file in uploaded_files:
                    cfg = st.session_state.get(f"val_cfg_{file.name}", pd.DataFrame())
                    t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                    excludes = st.session_state.get(f"manual_exc_b_{file.name}", [])
                    c_max, c_min = calculate_auto_qc_params(c_df, w_map, excludes) if auto_qc else (qc_max_bg, qc_min_amp)
                    b_qc_records.append({'File': file.name, 'Max_BG': c_max, 'Min_Amp': c_min})
                    
                    nc_tms = []
                    if not cfg.empty and 'Role (角色)' in cfg.columns:
                        for w in parse_wells_input(','.join(cfg[cfg['Role (角色)']=='Baseline (NC)']['Wells (孔位)'])):
                            if w in excludes: 
                                anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '人工剔除'})
                                continue
                            if w in w_map and check_curve_quality(c_df[w_map[w]], c_max, c_min)[0]:
                                nc_tms.append(calculate_tm(t, c_df[w_map[w]]))
                    base = np.mean(nc_tms) if nc_tms else np.nan
                    
                    if not cfg.empty and 'Role (角色)' in cfg.columns:
                        for _, r in cfg[cfg['Role (角色)'].isin(['Compound', 'Positive Control'])].iterrows():
                            cn, role_str, ratio, w_list = r['Compound_Name (组别命名)'], r['Role (角色)'], r['Ratio (浓度比)'], parse_wells_input(r['Wells (孔位)'])
                            for w in w_list:
                                if w in excludes:
                                    anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': '人工剔除'})
                                    continue
                                if w in w_map:
                                    is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max, c_min)
                                    if is_v: 
                                        tm_single = calculate_tm(t, c_df[w_map[w]])
                                        all_res.append({'File': file.name, 'Compound': cn, 'Role': role_str, 'Ratio': ratio, 'Well': w, 'Delta_Tm(℃)': np.round(tm_single - base, 2)})
                                    else: anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': rsn})
                
                b_anom_df = pd.DataFrame(anom_res) if anom_res else pd.DataFrame(columns=['File', 'Compound', 'Well', 'QC_Status'])
                b_res_df = pd.DataFrame(all_res) if all_res else pd.DataFrame(columns=['File', 'Compound', 'Role', 'Ratio', 'Well', 'Delta_Tm(℃)'])
                st.session_state.update(b_res=b_res_df, b_anom=b_anom_df, b_qc_recs=b_qc_records, mod_b_done=True)

    if st.session_state.get('mod_b_done'):
        st.subheader("📋 验证分析成果")
        res_df = st.session_state.b_res
        
        if not res_df.empty:
            if 'Role' in res_df.columns:
                plot_df = res_df[res_df['Role'] == 'Compound'].copy()
            else:
                plot_df = res_df.copy()
                
            if not plot_df.empty:
                plot_df['sort_key'] = plot_df['Ratio'].apply(extract_ratio_value)
                summary_df = plot_df.groupby(['Compound', 'Ratio', 'sort_key'], as_index=False).agg(
                    Mean_Tm=('Delta_Tm(℃)', 'mean'),
                    Std_Tm=('Delta_Tm(℃)', 'std')
                ).sort_values(by=['Compound', 'sort_key'])
                
                st.subheader("📊 独立化合物剂量效应分析 (Dose-Response Profiles)")
                
                unique_compounds = summary_df['Compound'].unique()
                npg_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
                
                for idx, comp in enumerate(unique_compounds):
                    comp_df = summary_df[summary_df['Compound'] == comp]
                    comp_color = npg_colors[idx % len(npg_colors)]
                    
                    st.markdown(f"#### 🧪 化合物: **{comp}**")
                    col_bar, col_line = st.columns(2)
                    ratios_str = comp_df['Ratio'].astype(str).tolist()
                    
                    with col_bar:
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=ratios_str, y=comp_df['Mean_Tm'],
                            marker=dict(color=comp_color, line=dict(color='black', width=1.5)),
                            error_y=dict(type='data', array=comp_df['Std_Tm'].fillna(0), visible=True, color='black', thickness=1.5, width=4),
                            text=comp_df['Mean_Tm'].apply(lambda x: f'{x:.2f}'),
                            textposition='outside', cliponaxis=False 
                        ))
                        fig_bar.add_hline(y=0, line_color="black")
                        fig_bar.update_layout(
                            title=dict(text="Dose-Response Bar Chart", font=dict(size=14)),
                            height=400, yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", xaxis_title="<b>Molar Ratio</b>",
                            font=dict(family="Arial", size=14, color="black"), plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=40, r=20, b=40, l=60)
                        )
                        fig_bar.update_xaxes(categoryorder='array', categoryarray=ratios_str, showline=True, linewidth=2, linecolor='black', showgrid=False, tickfont=dict(size=12, color='black'))
                        fig_bar.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, tickfont=dict(size=12, color='black'), zeroline=True, zerolinewidth=1.5, zerolinecolor='black', ticks='outside', tickwidth=2, ticklen=6)
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    with col_line:
                        if len(comp_df) > 1:
                            fig_line = go.Figure()
                            fig_line.add_trace(go.Scatter(
                                x=ratios_str, y=comp_df['Mean_Tm'],
                                mode='lines+markers', name=comp,
                                marker=dict(size=10, color=comp_color, line=dict(width=1.5, color='black')),
                                line=dict(width=3, color=comp_color),
                                error_y=dict(type='data', array=comp_df['Std_Tm'].fillna(0), visible=True, color='black', thickness=1.5, width=4)
                            ))
                            fig_line.add_hline(y=0, line_color="black", line_dash="dash")
                            fig_line.update_layout(
                                title=dict(text="Dose-Response Trend", font=dict(size=14)),
                                height=400, xaxis_title="<b>Molar Ratio</b>", yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>",
                                font=dict(family="Arial", size=14, color="black"), plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified',
                                margin=dict(t=40, r=20, b=40, l=60), showlegend=False
                            )
                            fig_line.update_xaxes(categoryorder='array', categoryarray=ratios_str, showline=True, linewidth=2, linecolor='black', showgrid=False, ticks='outside', tickwidth=2, ticklen=6)
                            fig_line.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=False, ticks='outside', tickwidth=2, ticklen=6, zeroline=True, zerolinewidth=1.5, zerolinecolor='black')
                            st.plotly_chart(fig_line, use_container_width=True)
                        else:
                            st.info("💡 此组别仅包含单个浓度梯度，无法绘制响应趋势折线。")
                            
                    st.markdown("<hr style='margin: 10px 0; border: 1px dashed #E0E0E0;'>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ 当前数据中只包含对照组，没有标注为'实验组'的化合物可供绘图。")
                
        st.markdown("##### 📁 验证结果原始数据表 (包含对照组结果, 可导出 CSV)")
        if not res_df.empty:
            st.dataframe(res_df.drop(columns=['Role', 'sort_key'], errors='ignore'), use_container_width=True)

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
            render_realtime_curve_viewer(sel_w, temps, well_map, clean_df)

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
                master_valid, master_qc_recs = [], []
                
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
                            nc_tm = calculate_tm(t, c_df[w_map[nc_w]])
                    else:
                        cfg_df = st.session_state.get(f"c_mkp_cfg_{file.name}", pd.DataFrame())
                        nc_tms = []
                        if not cfg_df.empty and 'Role (角色)' in cfg_df.columns:
                            nc_wells = cfg_df[cfg_df['Role (角色)'] == 'Baseline (NC)']['Well (孔位)'].tolist()
                            for nc_w in nc_wells:
                                if nc_w in w_map and nc_w not in excludes and check_curve_quality(c_df[w_map[nc_w]], c_max, c_min)[0]:
                                    nc_tms.append(calculate_tm(t, c_df[w_map[nc_w]]))
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
                                        tm_val = calculate_tm(t, c_df[w_map[w]])
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
                                        tm_val = calculate_tm(t, c_df[w_map[w]])
                                        dtm = tm_val - nc_tm if pd.notnull(nc_tm) else np.nan
                                        master_valid.append({'File': file.name, 'Well': w, 'Compound_ID': cid, 'Tm (℃)': tm_val, 'Delta_Tm (℃)': dtm})
                
                st.session_state.update(c_res=pd.DataFrame(master_valid), c_qc=pd.DataFrame(master_qc_recs), mod_c_done=True)

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
            fig.add_hline(y=global_hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Global Thr ({global_hit_threshold}℃)")
            fig.update_layout(
                height=500, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="<b>ΔT<sub>m</sub> (℃)</b>", 
                plot_bgcolor='white', hovermode='closest', font=dict(family="Arial", size=14, color="black")
            )
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks='outside', tickwidth=2, ticklen=6, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
            
        st.dataframe(c_res.sort_values(by="Delta_Tm (℃)", ascending=False), use_container_width=True)
