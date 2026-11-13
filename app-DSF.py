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
    window = int(data_length * 0.1)
    if window < 5: return 5
    return window if window % 2 != 0 else window + 1

def calculate_auto_qc_params(clean_df, well_map, excludes=[]):
    bgs = []
    amps = []
    for w_key, w_col in well_map.items():
        if w_key in excludes: continue 
        
        f_vals = pd.to_numeric(clean_df[w_col], errors='coerce').dropna().values
        if len(f_vals) >= 10:
            bgs.append(np.mean(f_vals[:5]))
            amps.append(np.max(f_vals) - np.min(f_vals))
            
    if bgs and amps:
        med_bg = np.median(bgs)
        med_amp = np.median(amps)
        return med_bg * 5, med_amp * 0.2 
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
        f_vals = f_vals[mask]
        t_vals = t_vals[mask]
        
        if len(f_vals) < 10: return None, None, None, None, np.nan
        
        window = get_dynamic_window(len(f_vals))
        smoothed = savgol_filter(f_vals, window_length=window, polyorder=3)
        derivative = np.gradient(smoothed, t_vals)
        max_idx = np.argmax(derivative)
        return t_vals, f_vals, smoothed, derivative, float(t_vals[max_idx])
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
        try:
            df = pd.read_excel(io.BytesIO(file_content), header=None, engine='xlrd')
        except:
            df = pd.read_excel(io.BytesIO(file_content), header=None, engine='openpyxl')

    header_row = 0
    for i, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        if any('Well' in val for val in row_values) and any('Temperature' in val for val in row_values):
            header_row = i
            break
            
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = [str(col).strip() for col in df.columns]
    
    temp_col = 'Temperature'
    fluor_col = 'Fluorescence'
    well_col = 'Well Position' if 'Well Position' in df.columns else 'Well'
        
    df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
    df[fluor_col] = pd.to_numeric(df[fluor_col], errors='coerce')
    df = df.dropna(subset=[temp_col, fluor_col, well_col])
    
    df[well_col] = df[well_col].astype(str).apply(
        lambda x: re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', x.strip().upper().replace(" ", ""))
    )
    
    wide_df = df.pivot_table(index=temp_col, columns=well_col, values=fluor_col, aggfunc='mean').reset_index()
    temps = wide_df[temp_col]
    well_cols = {col: col for col in wide_df.columns if col != temp_col}
    
    return temps, well_cols, wide_df

def parse_wells_input(wells_str):
    if pd.isna(wells_str) or not str(wells_str).strip(): return []
    return [re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', w.strip().upper().replace(" ", "")) for w in str(wells_str).split(',') if w.strip()]

def update_excludes_callback(exc_key, new_wells):
    if exc_key not in st.session_state:
        st.session_state[exc_key] = []
    st.session_state[exc_key] = list(set(st.session_state[exc_key] + new_wells))

# ==========================================
# 2. 交互式 96 孔板与全 Plotly 预览引擎
# ==========================================
def draw_interactive_plate(loaded_wells, key_prefix, excludes=[]):
    clear_key = f"clear_counter_{key_prefix}"
    if clear_key not in st.session_state:
        st.session_state[clear_key] = 0
        
    rows = list('ABCDEFGH')[::-1]
    cols = list(range(1, 13))
    
    x_coords, y_coords, well_names, colors = [], [], [], []
    for r in rows:
        for c in cols:
            w = f"{r}{c}"
            x_coords.append(c)
            y_coords.append(r)
            well_names.append(w)
            
            if w in excludes:
                colors.append('#E53935')
            elif w in loaded_wells:
                colors.append('#4CAF50')
            else:
                colors.append('#E0E0E0')
            
    fig = go.Figure(data=go.Scatter(
        x=x_coords, y=y_coords, mode='markers+text',
        text=well_names, customdata=well_names,
        textposition="middle center",
        marker=dict(size=30, color=colors, line=dict(width=1.5, color='black')),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="🖱️ <b>【交互指南】</b>: 1.拖拽框选；2.<b>Shift+拖拽/单点</b>(跨区多选)；3.双击空白处清空。<br>图例：🟢 有数据 | 🔴 <b>被手动剔除</b> | ⚪ 无数据",
        xaxis=dict(tickmode='linear', dtick=1, range=[0, 13], side='top', showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        dragmode='select', plot_bgcolor='white', height=520, margin=dict(l=40, r=40, t=110, b=40)
    )
    
    col_plot, col_btn = st.columns([5, 1])
    with col_btn:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        if st.button("🔄 彻底清除当前选框", key=f"btn_clr_{key_prefix}"):
            st.session_state[clear_key] += 1
            st.rerun()
            
    with col_plot:
        chart_key = f"plate_{key_prefix}_{st.session_state[clear_key]}"
        event = st.plotly_chart(fig, key=chart_key, on_select="rerun", selection_mode=['points', 'box', 'lasso'])
        
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
                
                tm_str = f"{tm:.1f}℃" if not np.isnan(tm) else "N/A"
                fig_prev.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color=c, width=2), name=f"{w} (Tm: {tm_str})", legendgroup=w), row=1, col=2)
                
                if not np.isnan(tm):
                    fig_prev.add_vline(x=tm, line=dict(color=c, width=1.5, dash='dash'), row=1, col=2)

    if has_data:
        fig_prev.update_layout(height=450, hovermode='x unified', plot_bgcolor='white', margin=dict(t=50, b=40))
        fig_prev.update_xaxes(title_text="Temperature (℃)", showgrid=True, gridcolor='lightgray', gridwidth=0.5)
        fig_prev.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
        if len(selected_wells) > 12: fig_prev.update_layout(showlegend=False) 
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.warning("⚠️ 你选定的孔位在文件中没有测定数据。")

# ==========================================
# 3. 全局配置与侧边栏
# ==========================================
st.set_page_config(page_title="DSF 高通量分析综合平台", layout="wide")

if 'mod_a_done' not in st.session_state: st.session_state.mod_a_done = False
if 'mod_b_done' not in st.session_state: st.session_state.mod_b_done = False

with st.sidebar:
    st.title("🔬 DSF 分析系统")
    analysis_mode = st.radio("请选择当前实验模块：", ["🎯 初筛化合物 (标准库排板)", "🧪 化合物再验证 (自定义排板)"])
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
# 模块 A: 初筛化合物
# ==========================================
if analysis_mode == "🎯 初筛化合物 (标准库排板)":
    st.title("🎯 初筛化合物自动化分析")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: lib_type = st.selectbox("化合物库类型", ["FDA", "TCM (中药)", "NP (天然产物)"])
    with col2: lib_num = st.number_input("化合物库编号", min_value=1, value=1, step=1)
    with col3: hit_threshold = st.number_input("Hit 筛选阈值 (℃)", value=0.5, step=0.1)

    uploaded_file = st.file_uploader("上传原始数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], key="file_a")

    if uploaded_file:
        fname = uploaded_file.name
        exc_key = f"manual_exc_a_{fname}"
        if exc_key not in st.session_state: st.session_state[exc_key] = []
            
        file_bytes = uploaded_file.getvalue()
        temps, well_map, clean_df = parse_instrument_data_cached(file_bytes)
        
        st.markdown("---")
        st.subheader("👀 实时图谱预览器 & 手动过滤")
        
        selected_wells = draw_interactive_plate(list(well_map.keys()), "mod_a", st.session_state[exc_key])
        render_realtime_curve_viewer(selected_wells, temps, well_map, clean_df)
        
        st.markdown("##### 🚫 人工干预：过滤异常孔位")
        col_exc1, col_exc2 = st.columns([3, 1])
        with col_exc1:
            st.multiselect(
                "在此指定不参与后续计算及 QC 统计的孔位：", 
                options=list(well_map.keys()), 
                key=exc_key,
                help="你也可以在上方框选后，点击右侧的【一键剔除】按钮自动填入。"
            )
        with col_exc2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "🗑️ 一键剔除当前选定孔位", 
                key="btn_add_exc_a",
                on_click=update_excludes_callback,
                args=(exc_key, selected_wells)
            )
                
        st.markdown("---")

        if st.button("🚀 执行全板初筛质控与计算", type="primary"):
            with st.spinner("正在全局解析与质控..."):
                excludes = st.session_state[exc_key]
                if auto_qc:
                    c_max_bg, c_min_amp = calculate_auto_qc_params(clean_df, well_map, excludes)
                else:
                    c_max_bg, c_min_amp = qc_max_bg, qc_min_amp
                    
                st.session_state.a_qc_params = (c_max_bg, c_min_amp)
                    
                nc_tm = pc_tm = np.nan
                if "G12" in well_map and "G12" not in excludes:
                    if check_curve_quality(clean_df[well_map["G12"]], c_max_bg, c_min_amp)[0]: 
                        nc_tm = calculate_tm(temps, clean_df[well_map["G12"]])
                if "H12" in well_map and "H12" not in excludes:
                    if check_curve_quality(clean_df[well_map["H12"]], c_max_bg, c_min_amp)[0]: 
                        pc_tm = calculate_tm(temps, clean_df[well_map["H12"]])
                    
                valid_res, anomaly_res = [], []
                for r in list('ABCDEFGH'):
                    for c in range(1, 13):
                        w = f"{r}{c}"
                        if w in ["G12", "H12"]: continue
                        cid = f"{lib_type.split()[0]}-{lib_num}-{w}"
                        
                        if w in excludes:
                            anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "人工剔除 (Manual Filtered)", "Delta Tm (℃)": np.nan})
                            continue
                            
                        if w in well_map:
                            f_data = clean_df[well_map[w]]
                            is_valid, reason = check_curve_quality(f_data, c_max_bg, c_min_amp)
                            if is_valid:
                                tm_val = calculate_tm(temps, f_data)
                                dtm = tm_val - nc_tm if (pd.notnull(tm_val) and pd.notnull(nc_tm)) else np.nan
                                valid_res.append({"Well": w, "Compound_ID": cid, "Tm (℃)": np.round(tm_val, 2), "Delta Tm (℃)": np.round(dtm, 2)})
                            else:
                                anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": reason, "Delta Tm (℃)": np.nan})
                        else:
                            anomaly_res.append({"Well": w, "Compound_ID": cid, "QC_Status": "No Data", "Delta Tm (℃)": np.nan})
                            
                st.session_state.update(a_valid_df=pd.DataFrame(valid_res), a_anomaly_df=pd.DataFrame(anomaly_res), a_nc=nc_tm, a_pc=pc_tm, mod_a_done=True)

    if st.session_state.mod_a_done:
        v_df, a_df = st.session_state.a_valid_df, st.session_state.a_anomaly_df
        c_max_bg, c_min_amp = st.session_state.a_qc_params
        
        st.subheader("📊 初筛质控分析报告")
        st.info(f"💡 当前板执行的 QC 阈值 -> **自发荧光上限**: {c_max_bg:.0f} | **有效振幅下限**: {c_min_amp:.0f}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("空白对照 Tm", f"{st.session_state.a_nc:.2f} ℃" if pd.notnull(st.session_state.a_nc) else "人工剔除/异常")
        m2.metric("阳性对照 Tm", f"{st.session_state.a_pc:.2f} ℃" if pd.notnull(st.session_state.a_pc) else "人工剔除/异常")
        m3.metric("通过 QC 孔位", len(v_df))
        m4.metric("异常/拦截 (含人工)", len(a_df))
        
        st.subheader("🎯 质控后化合物迁移散点图")
        if not v_df.empty:
            valid_plot = v_df.dropna(subset=["Delta Tm (℃)"])
            np.random.seed(42)
            x_jitter = np.random.uniform(-1, 1, size=len(valid_plot))
            hits = valid_plot[valid_plot["Delta Tm (℃)"] >= hit_threshold]
            non_hits = valid_plot[valid_plot["Delta Tm (℃)"] < hit_threshold]
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=x_jitter[valid_plot["Delta Tm (℃)"] < hit_threshold], y=non_hits["Delta Tm (℃)"],
                mode='markers', marker=dict(color='#B0BEC5', size=8), name='Background', text=non_hits["Compound_ID"], hoverinfo='text+y'
            ))
            fig_scatter.add_trace(go.Scatter(
                x=x_jitter[valid_plot["Delta Tm (℃)"] >= hit_threshold], y=hits["Delta Tm (℃)"],
                mode='markers', marker=dict(color='#E53935', size=12, line=dict(width=1, color='black')), name='Hits', text=hits["Compound_ID"], hoverinfo='text+y'
            ))
            fig_scatter.add_hline(y=hit_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({hit_threshold}℃)")
            fig_scatter.add_hline(y=0, line_color="black")
            fig_scatter.update_layout(height=450, xaxis=dict(showticklabels=False, zeroline=False), yaxis_title="ΔTm (℃)", plot_bgcolor='white')
            fig_scatter.update_yaxes(showgrid=True, gridcolor='lightgray')
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.dataframe(v_df.sort_values(by="Delta Tm (℃)", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 异常数据审查与判读")
        view_anomaly_df = a_df[a_df['QC_Status'] != 'No Data']
        if not view_anomaly_df.empty:
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(view_anomaly_df[['Well', 'Compound_ID', 'QC_Status']], hide_index=True)
            with view_col2:
                sel_well = st.selectbox("选择要复核的异常孔位:", view_anomaly_df['Well'].tolist())
                if sel_well:
                    file_bytes = uploaded_file.getvalue()
                    temps, well_map, clean_df = parse_instrument_data_cached(file_bytes)
                    reason = view_anomaly_df[view_anomaly_df['Well'] == sel_well].iloc[0]['QC_Status']
                    
                    if sel_well in well_map:
                        t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[sel_well]])
                        if t is not None:
                            title_reason = "Manual Filtered" if "人工剔除" in reason else ("Auto-Fluor" if "自发荧光" in reason else "Flat/Quenched")
                            
                            fig_a = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{sel_well}] - {title_reason}", "Derivative - (Excluded)"))
                            fig_a.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                            fig_a.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                            
                            fig_a.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                            for annotation in fig_a['layout']['annotations']:
                                annotation['font'] = dict(color='red')
                                
                            fig_a.update_xaxes(showgrid=True, gridcolor='lightgray')
                            fig_a.update_yaxes(showgrid=True, gridcolor='lightgray')
                            st.plotly_chart(fig_a, use_container_width=True)

# ==========================================
# 模块 B: 化合物再验证
# ==========================================
elif analysis_mode == "🧪 化合物再验证 (自定义排板)":
    st.title("🧪 化合物再验证自动化分析")
    uploaded_files = st.file_uploader("支持批量上传", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_b")
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("👀 跨板实时预览与过滤")
        file_names = [f.name for f in uploaded_files]
        preview_file_name = st.selectbox("选择要预览的孔板文件:", file_names)
        
        preview_file = next(f for f in uploaded_files if f.name == preview_file_name)
        exc_key = f"manual_exc_b_{preview_file_name}"
        if exc_key not in st.session_state: st.session_state[exc_key] = []
        
        p_temps, p_well_map, p_clean_df = parse_instrument_data_cached(preview_file.getvalue())
        
        sel_w = draw_interactive_plate(list(p_well_map.keys()), f"mod_b_{preview_file_name}", st.session_state[exc_key])
        render_realtime_curve_viewer(sel_w, p_temps, p_well_map, p_clean_df)
        
        st.markdown("##### 🚫 人工干预：过滤异常孔位")
        col_exc1, col_exc2 = st.columns([3, 1])
        with col_exc1:
            st.multiselect(
                f"为文件 [{preview_file_name}] 剔除异常孔位：", 
                options=list(p_well_map.keys()), 
                key=exc_key
            )
        with col_exc2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button(
                "🗑️ 一键剔除当前选定孔位", 
                key=f"btn_add_exc_{preview_file_name}",
                on_click=update_excludes_callback,
                args=(exc_key, sel_w)
            )
                
        st.markdown("---")
        st.header("⚙️ 定义孔位角色")
        file_configs = {}
        for file in uploaded_files:
            with st.expander(f"📄 配置文件: {file.name}", expanded=True):
                default_df = pd.DataFrame({
                    "Wells (用逗号分隔)": ["A1, A2", "B1, B2", "C1, C2"],
                    "Compound_Name": ["DMSO_NC", "Hit_1", "Hit_2"],
                    "Role": ["Baseline (NC)", "Compound", "Compound"]
                })
                edited_df = st.data_editor(
                    default_df,
                    column_config={
                        "Wells (用逗号分隔)": st.column_config.TextColumn(required=True),
                        "Compound_Name": st.column_config.TextColumn(required=True),
                        "Role": st.column_config.SelectboxColumn(options=["Baseline (NC)", "Compound", "Ignore"], required=True)
                    },
                    num_rows="dynamic", key=f"val_{file.name}", use_container_width=True
                )
                file_configs[file.name] = edited_df
                
        if st.button("🚀 计算验证结果", type="primary"):
            with st.spinner("分析中..."):
                all_res, anom_res, b_cache, b_qc_records = [], [], {}, []
                
                for file in uploaded_files:
                    cfg = file_configs[file.name]
                    t, w_map, c_df = parse_instrument_data_cached(file.getvalue())
                    b_cache[file.name] = {'temps': t, 'well_map': w_map, 'clean_df': c_df}
                    
                    excludes = st.session_state.get(f"manual_exc_b_{file.name}", [])
                    
                    if auto_qc:
                        c_max_bg, c_min_amp = calculate_auto_qc_params(c_df, w_map, excludes)
                    else:
                        c_max_bg, c_min_amp = qc_max_bg, qc_min_amp
                        
                    b_qc_records.append({'File': file.name, 'Max_BG': c_max_bg, 'Min_Amp': c_min_amp})
                    
                    nc_tms = []
                    for w in parse_wells_input(','.join(cfg[cfg['Role']=='Baseline (NC)']['Wells (用逗号分隔)'])):
                        if w in excludes: 
                            anom_res.append({'File': file.name, 'Compound': 'Baseline (NC)', 'Well': w, 'QC_Status': '人工剔除'})
                            continue
                        if w in w_map and check_curve_quality(c_df[w_map[w]], c_max_bg, c_min_amp)[0]:
                            nc_tms.append(calculate_tm(t, c_df[w_map[w]]))
                    base = np.mean(nc_tms) if nc_tms else np.nan
                    
                    for _, r in cfg[cfg['Role']=='Compound'].iterrows():
                        cn, w_list, val_tms = r['Compound_Name'], parse_wells_input(r['Wells (用逗号分隔)']), []
                        for w in w_list:
                            if w in excludes:
                                anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': '人工剔除'})
                                continue
                                
                            if w in w_map:
                                is_v, rsn = check_curve_quality(c_df[w_map[w]], c_max_bg, c_min_amp)
                                if is_v: val_tms.append(calculate_tm(t, c_df[w_map[w]]))
                                else: anom_res.append({'File': file.name, 'Compound': cn, 'Well': w, 'QC_Status': rsn})
                        
                        if val_tms:
                            all_res.append({
                                'File': file.name, 'Compound': cn, 'Wells': ",".join(w_list),
                                'Delta_Tm(℃)': np.round(np.mean(val_tms) - base, 2) if pd.notnull(base) else np.nan
                            })
                
                st.session_state.update(b_res=pd.DataFrame(all_res), b_anom=pd.DataFrame(anom_res), b_cache=b_cache, b_qc_recs=b_qc_records, mod_b_done=True)

    if st.session_state.mod_b_done:
        st.subheader("📋 验证分析结果")
        st.info("💡 **系统为各板动态生成的 QC 阈值：**")
        st.dataframe(pd.DataFrame(st.session_state.b_qc_recs), hide_index=True, use_container_width=True)
        st.dataframe(st.session_state.b_res, use_container_width=True)
        
        st.subheader("📊 化合物迁移对比柱状图")
        res_df = st.session_state.b_res
        if not res_df.empty:
            summary_df = res_df.groupby('Compound', as_index=False)['Delta_Tm(℃)'].mean()
            summary_df = summary_df.sort_values(by='Delta_Tm(℃)', ascending=False).reset_index(drop=True)
            
            fig_bar = go.Figure(go.Bar(
                x=summary_df['Compound'], y=summary_df['Delta_Tm(℃)'],
                text=summary_df['Delta_Tm(℃)'].apply(lambda x: f'{x:.1f}'),
                textposition='outside',
                marker_color=['#26A69A' if val > 0 else '#EF5350' for val in summary_df['Delta_Tm(℃)']],
                marker_line_color='black', marker_line_width=1
            ))
            fig_bar.add_hline(y=0, line_color="black")
            fig_bar.update_layout(height=450, yaxis_title="ΔTm (℃)", plot_bgcolor='white', margin=dict(b=80))
            fig_bar.update_yaxes(showgrid=True, gridcolor='lightgray')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 异常孔位审查器")
        anomaly_df = st.session_state.b_anom
        if not anomaly_df.empty:
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(anomaly_df, hide_index=True)
            with view_col2:
                anomaly_opts = [f"[{row['File']}] Well: {row['Well']} ({row['Compound']})" for _, row in anomaly_df.iterrows()]
                sel_opt = st.selectbox("选择要人工复核的异常孔位:", anomaly_opts)
                
                if sel_opt:
                    idx = anomaly_opts.index(sel_opt)
                    target = anomaly_df.iloc[idx]
                    src_file = target['File']
                    well = target['Well']
                    reason = target['QC_Status']
                    
                    cache = st.session_state.b_cache.get(src_file)
                    if cache:
                        temps, well_map, clean_df = cache['temps'], cache['well_map'], cache['clean_df']
                        
                        if well in well_map:
                            t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[well]])
                            if t is not None:
                                title_reason = "Manual Filtered" if "人工剔除" in reason else ("Auto-Fluor" if "自发荧光" in reason else "Flat/Quenched")
                                
                                fig_b = make_subplots(rows=1, cols=2, subplot_titles=(f"Raw Fluor [{well}] - {title_reason}", "Derivative - (Excluded)"))
                                fig_b.add_trace(go.Scatter(x=t, y=f, mode='lines', line=dict(color='purple', width=2)), row=1, col=1)
                                fig_b.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color='purple', width=2)), row=1, col=2)
                                
                                fig_b.update_layout(height=400, showlegend=False, plot_bgcolor='white')
                                for annotation in fig_b['layout']['annotations']:
                                    annotation['font'] = dict(color='red')
                                    
                                fig_b.update_xaxes(showgrid=True, gridcolor='lightgray')
                                fig_b.update_yaxes(showgrid=True, gridcolor='lightgray')
                                st.plotly_chart(fig_b, use_container_width=True)