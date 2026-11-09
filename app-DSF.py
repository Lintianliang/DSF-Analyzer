import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import re
import io

# ==========================================
# 1. 核心底层算法与 QC 引擎
# ==========================================
def check_curve_quality(fluor_series, max_bg, min_amp):
    """智能质控检查：拦截高自发荧光与水平淬灭死线"""
    f_vals = pd.to_numeric(fluor_series, errors='coerce').dropna().values
    if len(f_vals) < 10:
        return False, "Data too short"
    
    init_bg = np.mean(f_vals[:5])
    amplitude = np.max(f_vals) - np.min(f_vals)
    
    if init_bg > max_bg:
        return False, "高自发荧光 (Auto-Fluor)"
    if amplitude < min_amp:
        return False, "曲线水平/淬灭 (Flat/Quenched)"
        
    return True, "Pass"

def calculate_tm(temp_series, fluor_series):
    try:
        f_vals = pd.to_numeric(fluor_series, errors='coerce').astype(float).values
        t_vals = pd.to_numeric(temp_series, errors='coerce').astype(float).values
        
        mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
        f_vals = f_vals[mask]
        t_vals = t_vals[mask]
        
        if len(f_vals) < 10: return np.nan
            
        smoothed = savgol_filter(f_vals, window_length=11, polyorder=3)
        derivative = np.gradient(smoothed, t_vals)
        max_idx = np.argmax(derivative)
        return float(t_vals[max_idx])
    except Exception:
        return np.nan

def get_curve_arrays(temp_series, fluor_series):
    """提取完整的阵列数据，供异常数据审查器绘图使用"""
    try:
        f_vals = pd.to_numeric(fluor_series, errors='coerce').astype(float).values
        t_vals = pd.to_numeric(temp_series, errors='coerce').astype(float).values
        
        mask = ~np.isnan(f_vals) & ~np.isnan(t_vals)
        f_vals = f_vals[mask]
        t_vals = t_vals[mask]
        
        if len(f_vals) < 10: return None, None, None, None, np.nan
            
        smoothed = savgol_filter(f_vals, window_length=11, polyorder=3)
        derivative = np.gradient(smoothed, t_vals)
        max_idx = np.argmax(derivative)
        tm_val = float(t_vals[max_idx])
        return t_vals, f_vals, smoothed, derivative, tm_val
    except Exception:
        return None, None, None, None, np.nan

def parse_instrument_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Melt Curve Raw Data", header=None)
    except:
        try:
            df = pd.read_excel(uploaded_file, header=None, engine='xlrd')
        except:
            df = pd.read_excel(uploaded_file, header=None, engine='openpyxl')

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
    raw_wells = str(wells_str).split(',')
    cleaned_wells = []
    for w in raw_wells:
        w_clean = re.sub(r'^([A-Z])0+(\d+)$', r'\1\2', w.strip().upper().replace(" ", ""))
        if w_clean: cleaned_wells.append(w_clean)
    return cleaned_wells

# ==========================================
# 2. 全局配置与侧边栏
# ==========================================
st.set_page_config(page_title="DSF 高通量分析综合平台", layout="wide")

# 初始化 Session State
if 'mod_a_done' not in st.session_state: st.session_state.mod_a_done = False
if 'mod_b_done' not in st.session_state: st.session_state.mod_b_done = False

with st.sidebar:
    st.title("🔬 DSF 分析系统")
    analysis_mode = st.radio(
        "请选择当前实验模块：",
        ["🎯 初筛化合物 (标准库排板)", "🧪 化合物再验证 (自定义排板)"]
    )
    
    st.markdown("---")
    st.header("🛡️ 异常数据质控 (QC) 参数")
    qc_max_bg = st.number_input("异常高荧光上限 (A.U.)", value=400000, step=50000, help="高于此值判定为化合物自带荧光")
    qc_min_amp = st.number_input("最低有效振幅下限 (A.U.)", value=30000, step=5000, help="低于此值判定为曲线淬灭或无信号")

# ==========================================
# 模块 A: 初筛化合物 (标准库排板)
# ==========================================
if analysis_mode == "🎯 初筛化合物 (标准库排板)":
    st.title("🎯 初筛化合物自动化分析 (带 QC 质控)")
    st.markdown("预设排板：前 94 孔为化合物，**G12 为空白对照 (NC)**，**H12 为阳性对照 (PC)**。")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: lib_type = st.selectbox("选择化合物库类型", ["FDA", "TCM (中药)", "NP (天然产物)"])
    with col2: lib_num = st.number_input("输入化合物库编号 (如 1 代表第1板)", min_value=1, value=1, step=1)
    with col3: hit_threshold = st.number_input("Hit 筛选阈值 (Delta Tm ℃)", value=0.5, step=0.1)

    uploaded_file = st.file_uploader("上传当前板的原始测定数据 (.xls/.xlsx)", type=["xlsx", "xls", "csv"], key="file_a")

    if uploaded_file and st.button("🚀 执行初筛解析", type="primary"):
        with st.spinner("正在解析数据、执行质控过滤..."):
            temps, well_map, clean_df = parse_instrument_data(uploaded_file)
            st.session_state.a_cache = {'temps': temps, 'well_map': well_map, 'clean_df': clean_df, 'filename': uploaded_file.name}
            
            nc_tm = np.nan
            pc_tm = np.nan
            if "G12" in well_map:
                is_valid, _ = check_curve_quality(clean_df[well_map["G12"]], qc_max_bg, qc_min_amp)
                if is_valid: nc_tm = calculate_tm(temps, clean_df[well_map["G12"]])
            if "H12" in well_map:
                is_valid, _ = check_curve_quality(clean_df[well_map["H12"]], qc_max_bg, qc_min_amp)
                if is_valid: pc_tm = calculate_tm(temps, clean_df[well_map["H12"]])
                
            rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            valid_results = []
            anomaly_results = []
            
            for r in rows:
                for c in range(1, 13):
                    well = f"{r}{c}"
                    if well in ["G12", "H12"]: continue
                        
                    compound_id = f"{lib_type.split()[0]}-{lib_num}-{well}"
                    
                    if well in well_map:
                        f_data = clean_df[well_map[well]]
                        is_valid, reason = check_curve_quality(f_data, qc_max_bg, qc_min_amp)
                        
                        if is_valid:
                            tm_val = calculate_tm(temps, f_data)
                            delta_tm = tm_val - nc_tm if (pd.notnull(tm_val) and pd.notnull(nc_tm)) else np.nan
                            valid_results.append({"Well": well, "Compound_ID": compound_id, "Tm (℃)": np.round(tm_val, 2), "Delta Tm (℃)": np.round(delta_tm, 2)})
                        else:
                            anomaly_results.append({"Well": well, "Compound_ID": compound_id, "QC_Status": reason, "Tm (℃)": np.nan, "Delta Tm (℃)": np.nan})
                    else:
                        anomaly_results.append({"Well": well, "Compound_ID": compound_id, "QC_Status": "No Data", "Tm (℃)": np.nan, "Delta Tm (℃)": np.nan})
                        
            st.session_state.a_valid_df = pd.DataFrame(valid_results)
            st.session_state.a_anomaly_df = pd.DataFrame(anomaly_results)
            st.session_state.a_nc_tm = nc_tm
            st.session_state.a_pc_tm = pc_tm
            st.session_state.mod_a_done = True

    # 渲染初筛结果
    if st.session_state.mod_a_done:
        valid_df = st.session_state.a_valid_df
        anomaly_df = st.session_state.a_anomaly_df
        nc_tm = st.session_state.a_nc_tm
        pc_tm = st.session_state.a_pc_tm
        
        st.subheader("📊 QC 过滤与统计概览")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("空白对照 (G12) Tm", f"{nc_tm:.2f} ℃" if pd.notnull(nc_tm) else "异常/未检测")
        m2.metric("阳性对照 (H12) Tm", f"{pc_tm:.2f} ℃" if pd.notnull(pc_tm) else "异常/未检测")
        m3.metric("有效数据量 (Pass QC)", len(valid_df))
        m4.metric("拦截异常数据", len(anomaly_df))
        
        if not anomaly_df.empty:
            st.warning(f"⚠️ 质控引擎自动过滤了 {len(anomaly_df)} 个孔位。请在页面最下方审查异常曲线。")
        
        st.subheader("🎯 质控后化合物迁移散点图 (Jitter Plot)")
        fig, ax = plt.subplots(figsize=(10, 5))
        np.random.seed(42)
        valid_plot = valid_df.dropna(subset=["Delta Tm (℃)"])
        x_jitter = np.random.uniform(-1, 1, size=len(valid_plot))
        
        hits = valid_plot[valid_plot["Delta Tm (℃)"] >= hit_threshold]
        non_hits = valid_plot[valid_plot["Delta Tm (℃)"] < hit_threshold]
        
        ax.scatter(x_jitter[valid_plot["Delta Tm (℃)"] < hit_threshold], non_hits["Delta Tm (℃)"], color='#B0BEC5', alpha=0.6, s=30, label='Background')
        ax.scatter(x_jitter[valid_plot["Delta Tm (℃)"] >= hit_threshold], hits["Delta Tm (℃)"], color='#E53935', alpha=0.9, s=60, edgecolor='black', label='Hits')
                   
        ax.axhline(hit_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({hit_threshold}℃)')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks([])
        ax.set_ylabel(r'$\Delta$Tm ($^\circ$C)', fontsize=12, fontweight='bold')
        ax.set_title(f'{lib_type} Library {lib_num} - Primary Screening (QC Filtered)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        st.pyplot(fig)
        
        st.subheader("📋 初筛结果明细表 (通过 QC)")
        st.dataframe(valid_df.sort_values(by="Delta Tm (℃)", ascending=False), use_container_width=True)
        csv = valid_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="⬇️ 导出初筛明细 (CSV)", data=csv, file_name=f'Primary_Screen_{lib_type}_{lib_num}_QC_Pass.csv', mime='text/csv')

        # --- 异常追溯审查器 (初筛) ---
        st.markdown("---")
        st.subheader("🔍 异常数据审查与判读 (Anomaly Viewer)")
        st.markdown("质控引擎拦截的孔位列表。选择具体孔位，人工复核其原始荧光曲线以确认是否为真异常。")
        
        view_anomaly_df = anomaly_df[anomaly_df['QC_Status'] != 'No Data']
        if not view_anomaly_df.empty:
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(view_anomaly_df[['Well', 'Compound_ID', 'QC_Status']], hide_index=True)
            with view_col2:
                sel_well = st.selectbox("选择要复核的异常孔位:", view_anomaly_df['Well'].tolist())
                if sel_well:
                    cache = st.session_state.a_cache
                    temps, well_map, clean_df = cache['temps'], cache['well_map'], cache['clean_df']
                    reason = view_anomaly_df[view_anomaly_df['Well'] == sel_well].iloc[0]['QC_Status']
                    
                    fig_a, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(12, 4))
                    if sel_well in well_map:
                        t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[sel_well]])
                        if t is not None:
                            ax_r.plot(t, f, color='purple', linewidth=2)
                            ax_d.plot(t, d, color='purple', linewidth=2)
                            ax_r.set_title(f"Raw Fluor [{sel_well}] - {reason}", fontsize=11, color='red')
                            ax_d.set_title(f"Derivative - (QC Failed)", fontsize=11, color='red')
                            ax_r.grid(True, linestyle=':'); ax_d.grid(True, linestyle=':')
                            st.pyplot(fig_a)
        else:
            st.success("🎉 太棒了，当前板没有检测到淬灭或高自发荧光的异常数据！")

# ==========================================
# 模块 B: 化合物再验证 (自定义排板)
# ==========================================
elif analysis_mode == "🧪 化合物再验证 (自定义排板)":
    st.title("🧪 化合物再验证自动化分析 (带 QC 质控)")
    st.markdown("通过动态表格完全自定义孔位用途，自动拦截并隔离水平淬灭/高荧光异常组。")
    
    uploaded_files = st.file_uploader("支持批量上传文件", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="files_b")
    
    if uploaded_files:
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
                        "Compound_Name": st.column_config.TextColumn("化合物/对照名称", required=True),
                        "Role": st.column_config.SelectboxColumn("角色分配", options=["Baseline (NC)", "Compound", "Ignore"], required=True)
                    },
                    num_rows="dynamic",
                    key=f"val_editor_{file.name}",
                    use_container_width=True
                )
                file_configs[file.name] = edited_df
                
        if st.button("🚀 计算结果并执行质控", type="primary"):
            with st.spinner("提取数据并进行 QC 过滤中..."):
                all_results = []
                anomaly_results = []
                b_cache = {}
                
                for file in uploaded_files:
                    config_df = file_configs[file.name]
                    temps, well_map, clean_df = parse_instrument_data(file)
                    b_cache[file.name] = {'temps': temps, 'well_map': well_map, 'clean_df': clean_df}
                    
                    # 基线计算与 QC
                    baseline_tms = []
                    nc_rows = config_df[config_df['Role'] == 'Baseline (NC)']
                    for _, row in nc_rows.iterrows():
                        for w in parse_wells_input(row['Wells (用逗号分隔)']):
                            if w in well_map:
                                is_valid, reason = check_curve_quality(clean_df[well_map[w]], qc_max_bg, qc_min_amp)
                                if is_valid:
                                    val = calculate_tm(temps, clean_df[well_map[w]])
                                    if pd.notnull(val): baseline_tms.append(val)
                                else:
                                    anomaly_results.append({'Source_File': file.name, 'Compound': row['Compound_Name'], 'Well': w, 'QC_Status': f"NC Error: {reason}"})
                    
                    file_baseline = np.mean(baseline_tms) if baseline_tms else np.nan
                    
                    # 实验组计算与 QC (精确到每个孔)
                    comp_rows = config_df[config_df['Role'] == 'Compound']
                    for _, row in comp_rows.iterrows():
                        comp_name = row['Compound_Name']
                        comp_tms = []
                        valid_wells = []
                        
                        for w in parse_wells_input(row['Wells (用逗号分隔)']):
                            if w in well_map:
                                is_valid, reason = check_curve_quality(clean_df[well_map[w]], qc_max_bg, qc_min_amp)
                                if is_valid:
                                    val = calculate_tm(temps, clean_df[well_map[w]])
                                    if pd.notnull(val): 
                                        comp_tms.append(val)
                                        valid_wells.append(w)
                                else:
                                    anomaly_results.append({'Source_File': file.name, 'Compound': comp_name, 'Well': w, 'QC_Status': reason})
                        
                        if valid_wells:
                            avg_comp_tm = np.mean(comp_tms)
                            delta_tm = avg_comp_tm - file_baseline if (pd.notnull(avg_comp_tm) and pd.notnull(file_baseline)) else np.nan
                            all_results.append({
                                'Source_File': file.name,
                                'Compound': comp_name,
                                'Wells_Used': ",".join(valid_wells),
                                'Base_Tm(℃)': np.round(file_baseline, 2),
                                'Comp_Tm(℃)': np.round(avg_comp_tm, 2),
                                'Delta_Tm(℃)': np.round(delta_tm, 2)
                            })
                
                st.session_state.b_res_df = pd.DataFrame(all_results)
                st.session_state.b_anomaly_df = pd.DataFrame(anomaly_results)
                st.session_state.b_cache = b_cache
                st.session_state.mod_b_done = True

    # 渲染再验证结果
    if st.session_state.mod_b_done:
        res_df = st.session_state.b_res_df
        anomaly_df = st.session_state.b_anomaly_df
        
        m1, m2 = st.columns(2)
        m1.metric("有效复测组 (QC Pass)", len(res_df))
        m2.metric("拦截异常孔位 (QC Fail)", len(anomaly_df))
        
        if res_df.empty or res_df['Delta_Tm(℃)'].isna().all():
            st.error("计算失败。请检查是否正确指定了 Baseline，或所有孔位均未通过质控。")
        else:
            st.subheader("📋 有效验证数据明细")
            st.dataframe(res_df, use_container_width=True)
            
            summary_df = res_df.groupby('Compound', as_index=False)['Delta_Tm(℃)'].mean()
            summary_df = summary_df.sort_values(by='Delta_Tm(℃)', ascending=False).reset_index(drop=True)
            
            st.subheader("📊 化合物迁移对比柱状图")
            fig, ax = plt.subplots(figsize=(max(8, len(summary_df)*0.8), 5))
            colors = ['#26A69A' if val > 0 else '#EF5350' for val in summary_df['Delta_Tm(℃)']]
            bars = ax.bar(summary_df['Compound'], summary_df['Delta_Tm(℃)'], color=colors, edgecolor='black', width=0.55)
            
            for bar in bars:
                yval = bar.get_height()
                va = 'bottom' if yval >= 0 else 'top'
                offset = yval * 0.05 if yval != 0 else 0.1
                ax.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.1f}', 
                        ha='center', va=va, fontsize=10, fontweight='bold', color='black')
            
            ax.axhline(0, color='black', linewidth=1)
            ax.set_xticklabels(summary_df['Compound'], rotation=30, ha='right', fontsize=10)
            ax.set_ylabel(r'$\Delta$Tm ($^\circ$C)', fontsize=11, fontweight='bold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.95)
            st.pyplot(fig)
            
            csv = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="⬇️ 导出验证明细 (CSV)", data=csv, file_name='Re_validation_results.csv', mime='text/csv')
        
        # --- 异常追溯审查器 (再验证) ---
        st.markdown("---")
        st.subheader("🔍 异常孔位审查器 (Anomaly Viewer)")
        st.markdown("质控引擎自动拦截的淬灭/高荧光孔位。手动选择即可复核原始数据以确认真伪。")
        
        if not anomaly_df.empty:
            view_col1, view_col2 = st.columns([1, 2])
            with view_col1:
                st.dataframe(anomaly_df, hide_index=True)
            with view_col2:
                # 构建 复合选项 方便用户选择 (文件 - 孔位)
                anomaly_opts = [f"[{row['Source_File']}] Well: {row['Well']} ({row['Compound']})" for _, row in anomaly_df.iterrows()]
                sel_opt = st.selectbox("选择要人工复核的异常孔位:", anomaly_opts)
                
                if sel_opt:
                    idx = anomaly_opts.index(sel_opt)
                    target = anomaly_df.iloc[idx]
                    src_file = target['Source_File']
                    well = target['Well']
                    reason = target['QC_Status']
                    
                    cache = st.session_state.b_cache.get(src_file)
                    if cache:
                        temps, well_map, clean_df = cache['temps'], cache['well_map'], cache['clean_df']
                        fig_b, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        if well in well_map:
                            t, f, s, d, tm = get_curve_arrays(temps, clean_df[well_map[well]])
                            if t is not None:
                                ax_r.plot(t, f, color='purple', linewidth=2)
                                ax_d.plot(t, d, color='purple', linewidth=2)
                                ax_r.set_title(f"Raw Fluor [{well}] - {reason}", fontsize=11, color='red')
                                ax_d.set_title(f"Derivative - (QC Failed)", fontsize=11, color='red')
                                ax_r.grid(True, linestyle=':'); ax_d.grid(True, linestyle=':')
                                st.pyplot(fig_b)
        else:
            st.success("🎉 太棒了，当前导入的所有复测孔位均完美通过了 QC 质控！")