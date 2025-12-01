# bash 실행코드: `streamlit run gpu_dashboard.py --server.address 0.0.0.0 --server.port 8501`

import time
import psutil
import GPUtil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MultipleLocator

st.set_page_config(page_title="GPU & RAM Monitor", layout="wide")
st.title("🚀 GPU & RAM Monitor (Absolute + Fixed Scale)")

# 기록 저장
if "records" not in st.session_state:
    st.session_state.records = []

main_plot = st.empty()   # 기존 Dual Axis 그래프
sub_plots = st.empty()   # 하단 개별 그래프들

while True:
    # 1) 현재 사용량 수집
    gpus = GPUtil.getGPUs()
    gpu = gpus[0] if gpus else None
    ram = psutil.virtual_memory()
    now = time.time()

    gpu_util = gpu.load * 100 if gpu else 0               # %
    # ✅ GPU 메모리: MB → GB
    gpu_mem_gb = (gpu.memoryUsed / 1024) if gpu else 0
    gpu_mem_total_gb = (gpu.memoryTotal / 1024) if gpu else 1

    # ✅ RAM: bytes → GB, 실사용량 = total - available
    ram_gb = (ram.total - ram.available) / (1024 ** 3)
    ram_total_gb = ram.total / (1024 ** 3)

    st.session_state.records.append({
        "time": now,
        "gpu_util": gpu_util,
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_mem_total_gb": gpu_mem_total_gb,
        "ram_gb": ram_gb,
        "ram_total_gb": ram_total_gb,
    })

    # 2) 최근 5분만 유지
    st.session_state.records = [
        r for r in st.session_state.records if now - r["time"] < 300
    ]
    df = pd.DataFrame(st.session_state.records)
    df["time_dt"] = pd.to_datetime(df["time"], unit="s")
    df_idx = df.set_index("time_dt")

    # 현재 총 용량(마지막 값 기준)
    current_gpu_total = df_idx["gpu_mem_total_gb"].iloc[-1] if len(df_idx) > 0 else 1
    current_ram_total = df_idx["ram_total_gb"].iloc[-1] if len(df_idx) > 0 else 1

    # ─────────────────────────────────────────
    # (1) 기존 Dual Axis 절대값 그래프
    #    GPU Util (%), GPU Mem (GB), RAM (GB)
    # ─────────────────────────────────────────
    with main_plot.container():
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()

        # 색상 구분
        ax1.plot(
            df_idx.index,
            df_idx["gpu_util"],
            label="GPU Util (%)",
            color="tab:blue",
        )
        ax2.plot(
            df_idx.index,
            df_idx["gpu_mem_gb"],
            label="GPU Mem (GB)",
            color="tab:orange",
        )
        ax2.plot(
            df_idx.index,
            df_idx["ram_gb"],
            label="RAM (GB)",
            color="tab:green",
        )

        # 축 라벨 및 범위
        ax1.set_xlabel("Time")
        ax1.set_ylabel("GPU Util (%)", color="tab:blue")
        ax1.set_ylim(0, 100)

        ax2.set_ylabel("Memory (GB)")
        max_mem_scale = max(current_gpu_total, current_ram_total)
        ax2.set_ylim(0, 40)  # 살짝 여유

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        fig1.autofmt_xdate()
        st.pyplot(fig1)

    # ─────────────────────────────────────────
    # (2) 각 자원별 고정 스케일 개별 그래프 3개
    #     - GPU Util: 0~100 (%)
    #     - GPU Mem: 0~총 GB
    #     - RAM:     0~총 GB
    # ─────────────────────────────────────────
    with sub_plots.container():
        st.subheader("📊 Usage vs Capacity (Fixed Scale)")

        fig2, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        # 1) GPU Util (%)
        axes[0].plot(df_idx.index, df_idx["gpu_util"], color="tab:blue")
        axes[0].set_ylabel("GPU Util (%)")
        axes[0].set_ylim(0, 100)

        # 2) GPU Memory (GB, 0~총 용량)
        axes[1].plot(df_idx.index, df_idx["gpu_mem_gb"], color="tab:orange")
        axes[1].set_ylabel("GPU Mem (GB)")
        axes[1].yaxis.set_major_locator(MultipleLocator(8))
        axes[1].set_ylim(0, current_gpu_total * 1.05)

        # 3) RAM (GB, 0~총 용량)
        axes[2].plot(df_idx.index, df_idx["ram_gb"], color="tab:green")
        axes[2].set_ylabel("RAM (GB)")
        axes[2].set_ylim(0, 34)
        axes[2].yaxis.set_major_locator(MultipleLocator(8))
        axes[2].set_xlabel("Time")

        fig2.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig2)

    time.sleep(1)
