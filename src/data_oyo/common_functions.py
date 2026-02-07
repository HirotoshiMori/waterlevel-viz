import os, glob, re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.fft import fft
from natsort import natsorted
from datetime import datetime
from io import StringIO


def set_params(main_params, t_dom_params, sp_dom_params):
    main_params["fs_t"] = 1 / main_params["dt"]
    main_params["fs_sp"] = 1 / main_params["dsp"]
    main_params["disp_t"] = main_params["end_t"] - main_params["start_t"]
    main_params["disp_sp"] = main_params["end_sp"] - main_params["start_sp"]
    main_params["i_start_t"] = int(main_params["start_t"] / main_params["dt"])
    main_params["i_start_sp"] = int(main_params["start_sp"] / main_params["dsp"])
    main_params["i_end_t"] = int(main_params["end_t"] / main_params["dt"])
    main_params["i_end_sp"] = int(main_params["end_sp"] / main_params["dsp"])
    main_params["num_t"] = main_params["i_end_t"] - main_params["i_start_t"]
    main_params["num_sp"] = main_params["i_end_sp"] - main_params["i_start_sp"]

    t_dom_params["hop_len_t"] = int(t_dom_params["n_fft_t"] / 4)
    sp_dom_params["hop_len_sp"] = int(sp_dom_params["n_fft_sp"] / 4)

def proc_ftr(df, order, fs, f1, f2, section, threshold, wndw, column = None, step_l = None, step_c = None):

    def freq_ftr(order, df, fs, f1 = None, f2 = None, column = None):
        """指定した列をフィルタリングし、元のデータフレームを更新して返します。"""

        # フィルタのタイプを決定し、フィルタ係数を取得
        if f1 and f2:
            sos = signal.butter(order, [f2, f1], btype='band', fs=fs, output='sos')
        elif f1:
            sos = signal.butter(order, f1, btype='low', fs=fs, output='sos')
        elif f2:
            sos = signal.butter(order, f2, btype='high', fs=fs, output='sos')
        else:
            return df  # フィルタを適用しない場合、元のdfを返します

        # 特定の列にフィルタを適用するか、全ての列に適用するかを判断
        if column is not None:
            # 特定の列にフィルタを適用
            df[column] = signal.sosfiltfilt(sos, df[column])
        else:
            # 全ての列にフィルタを適用
            df = df.apply(lambda col: signal.sosfiltfilt(sos, col) if col.name != 'Frequency' else col, axis=0)

        return df

    def remove_outliers(df, section = None, threshold = None, column = None):
        """異常値除去を行う関数"""
        
        if section is not None and threshold is not None:
            if column is None:
                # 全ての列に異常値除去を適用
                for col in df.columns:
                    df_mean = df[col].rolling(section).mean()
                    df_std = df[col].rolling(section).std()
                    df = df[((df[col] - df_mean).abs() < threshold * df_std)]
            elif column in df.columns:
                # 特定の列に異常値除去を適用
                df_mean = df[column].rolling(section).mean()
                df_std = df[column].rolling(section).std()
                df = df[((df[column] - df_mean).abs() < threshold * df_std)]
        
        return df

    def rolling_mean(df, wndw = None, column = None):
        """移動平均を適用する関数"""
        
        if wndw is not None:
            if column is not None:
                df[column] = df[column].rolling(wndw, center=True).mean()
            else:
                df = df.rolling(wndw, center=True).mean()
        
        return df

    df_copy = df.copy()
    
    df_copy = freq_ftr(order, df_copy, fs, f1 = f1, f2 = f2, column = column)
    df_copy = remove_outliers(df_copy, section, threshold, column = column)
    df_copy = rolling_mean(df_copy, wndw, column = column)
    
    if step_l and step_c:
        df_step_copy = df_copy.iloc[::step_l, ::step_c]
        return df_copy, df_step_copy
    elif step_l:
        df_step_copy = df_copy.iloc[::step_l, ::]
        return df_copy, df_step_copy
    elif step_c:
        df_step_copy = df_copy.iloc[::, ::step_c]
        return df_copy, df_step_copy
    else:
        return df_copy

def proc_das_data(MFR, file_name, dataset_name, time_range, time_skip, spatial_range, spatial_skip, t_params, sp_params):

    def load_nec_df(file_path, t_range, step_t, d_range, step_d):
        """CSVファイルからデータを読み込んでDataFrameに変換して返す"""

        data_file = pd.read_csv(file_path)
        partial_data = data_file.iloc[t_range[0]:t_range[1], d_range[0]:d_range[1]]
        partial_data_step = data_file.iloc[t_range[0]:t_range[1]:step_t, d_range[0]:d_range[1]:step_d]

        return partial_data, partial_data_step

    # ファイルの読み込み（HDF5 は未対応、CSV のみ）
    if MFR == 1:
        raise ValueError("HDF5 (MFR=1) は本プロジェクトでは未対応です。CSV (MFR=2) を使用してください。")
    if MFR != 2:
        raise ValueError(f"未対応の MFR です: {MFR}。CSV の場合は MFR=2 を指定してください。")
    df, df_skip = load_nec_df(file_name, time_range, time_skip, spatial_range, spatial_skip)

    # 時間領域フィルタ処理
    df_t, df_skip_t = proc_ftr(df, *t_params, step_l=time_skip, step_c=spatial_skip)

    # 空間領域フィルタ処理
    df_sp, df_skip_sp = proc_ftr(df.T, *sp_params, step_l=spatial_skip, step_c=time_skip)

    # 時間領域フィルター処理 → 空間領域フィルター処理
    df_t_sp, df_skip_t_sp = proc_ftr(df_t.T, *sp_params, step_l=spatial_skip, step_c=time_skip)

    # 空間領域フィルター処理 → 時間領域フィルター処理
    df_sp_t, df_skip_sp_t = proc_ftr(df_sp.T, *t_params, step_l=time_skip, step_c=spatial_skip)

    return df, df_skip, df_t, df_skip_t, df_sp, df_skip_sp, df_t_sp, df_skip_t_sp, df_sp_t, df_skip_sp_t


def plt_spec(i, df, column, fs1, fs2, start1, start2, delta1, n_fft, hop_length, base, sp_minmax, sp_cbar_db_min, sp_cbar_db_max, plt_max, x_range, ftr = None, title_file = None, title_graph = None):
    """スペクトルとスペクトログラムをプロット"""

    def plt_wav(i, df, signal, fs1, start1, delta1, base, sp_minmax, plt_max, title_file = None, title_graph = None):
        """信号のプロット"""

        plt.figure(figsize=(15, 3))

        if base == 1: x_label = ('Time [s]')
        elif base == 2: x_label = ('Distance [m]')
        else:  # baseが1または2以外の値の場合
            x_label = ('Time')
            time_list = pd.to_datetime(df.index.tolist())

        if base == 3:
            plt.plot(time_list, signal)
            plt.ylabel('Strain')
        else: 
            plt.plot(signal)
            x_ticks = np.linspace(0, delta1 * fs1, 5)
            x_labels = [start1 + delta1 / 4 * i for i in range(5)]
            plt.xticks(x_ticks, x_labels)
            plt.ylabel('Strain' if title_graph is None else 'Amplitude')

        plt.xlabel(x_label)
        plt.ylim(-sp_minmax, sp_minmax) 
        plt.title(title_graph)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./png/wav_{title_file}.png")
        
        if i > 0 and i < plt_max: plt.show()
        else: plt.close()

    def plt_fft(signal, sp_type, fs, x_range = None, ax = None):
        """スペクトルプロット関数"""

        n = len(signal)
        freqs = np.fft.fftfreq(n, d=1/fs)
        spectrum = np.abs(np.fft.fft(signal)) / n

        if sp_type == "Amplitude": transform_func = lambda x: x
        elif sp_type in ['Amplitude density', 'Power']: transform_func = lambda x: x**2
        else: transform_func = lambda x: x**2 / fs

        ax.plot(freqs[:n//2], transform_func(spectrum)[:n//2])
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(sp_type)
        ax.set_xlim(0, x_range)
        ax.grid()

    def plt_spectro(sig, sp_type, fs, start, delta, n_fft, hop_length, base, sp_cbar_db_min, sp_cbar_db_max, ax = None):
        """スペクトログラムプロット関数（scipy.signal.stft 使用）"""
        noverlap = n_fft - hop_length
        f, t, Zxx = signal.stft(sig, fs=fs, window="hann", nperseg=n_fft, noverlap=noverlap)
        S = np.abs(Zxx)
        if sp_type == "Amplitude":
            spectrogram = S
        elif sp_type in ["Amplitude density", "Power"]:
            spectrogram = S ** 2
        else:
            spectrogram = S ** 2 / n_fft
        ref = np.max(spectrogram)
        db = 10 * np.log10(np.maximum(spectrogram / ref, 1e-12))
        # 線形スケールで表示（librosa の log 軸は省略）
        img = ax.pcolormesh(t, f, db, cmap="viridis", vmin=sp_cbar_db_min, vmax=sp_cbar_db_max, shading="auto")
        plt.colorbar(img, ax=ax, label=sp_type + " [dB]")

        if base == 1:
            ax.set_xlabel('Time [s]')
        elif base == 2:
            d_ticks = np.linspace(0, delta, 5)
            d_labels = [start + delta/4*i for i in range(5)]
            ax.set_xlabel('Distance [m]')
            ax.set_xticks(d_ticks)
            ax.set_xticklabels(d_labels)
        else:
            ax.set_xlabel('Time')

        ax.set_ylim(0, fs/2)
        ax.set_ylabel('Frequency [Hz]')
        ax.grid(True)

    if isinstance(df, pd.DataFrame): 
        if isinstance(column, int) and (0 <= column < len(df.columns)):
            # columnが整数インデックスの場合
            signal = df.iloc[:, column].to_numpy()
        else:
            signal = df[column].to_numpy()
            column = df.columns.get_loc(column)
    else:
        signal = df[column]
        column = df.columns.get_loc(column)

    if base == 1 or base == 3: x_unit = 'm'
    else: x_unit = 's'

    title_graph = f"{ftr} {int(start2 + column / fs2)}{x_unit}" if title_graph is None else title_graph
    title_file = f"{ftr}_{int(start2 + column / fs2)}{x_unit}" if title_file is None else title_file

    # 波形表示
    plt_wav(i, df, signal, fs1, start1, delta1, base, sp_minmax, plt_max, title_file, title_graph)

    # スペクトル分析の種類
    sp_types = ['Amplitude', 'Amplitude density', 'Power', 'Power density']

    # FFT表示
    fig, axs = plt.subplots(2, 2, figsize=(15, 6))
    for j, sp_type in enumerate(sp_types):
        row = j // 2
        col = j % 2
        plt_fft(signal, sp_type, fs1, x_range = x_range, ax = axs[row, col])

    plt.suptitle(title_graph)
    plt.tight_layout()
    plt.savefig(f"./png/fft_{title_file}.png")

    if i > 0 and i < plt_max: plt.show()
    else: plt.close()

    # スペクトログラム表示
    fig, axs = plt.subplots(2, 2, figsize=(15, 6))
    for j, sp_type in enumerate(sp_types):
        row = j // 2
        col = j % 2
        plt_spectro(signal, sp_type, fs1, start1, delta1, n_fft, hop_length, base, sp_cbar_db_min, sp_cbar_db_max, ax=axs[row, col])

    plt.suptitle(title_graph)
    plt.tight_layout()
    plt.savefig(f"./png/spec_{title_file}.png")

    if i > 0 and i < plt_max: plt.show()
    else: plt.close()

    
def draw_heatmap(df, cbar_minmax, cbar_db_min, cbar_db_max, display_t, start_t, num_time, step_t, display_d, start_d, num_distance, step_d, process):
    """ヒートマップを描画"""

    fig, (cmap1, cmap2) = plt.subplots(1, 2, figsize=(15, 6))

    ticks_t = np.linspace(0, num_time / step_t, 5)
    ticks_d = np.linspace(0, num_distance / step_d, 5)
    labels_t = [start_t + display_t / 4 * i for i in range(5)]
    labels_d = [start_d + display_d / 4 * i for i in range(5)]

    def plot_heatmap(ax, df, cmap, vmin, vmax, colorbar_label, ticks_t, labels_t, ticks_d, labels_d):
        data = df.T.values if hasattr(df, "T") else np.asarray(df).T
        n_d, n_t = data.shape
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower", extent=[0, n_t, 0, n_d])
        cbar = plt.colorbar(im, ax=ax, extend="both")
        cbar.set_label(colorbar_label)

        ax.set_xticks(ticks_t)
        ax.set_yticks(ticks_d)
        ax.set_xticklabels(labels_t, rotation=45)
        ax.set_yticklabels(labels_d)
        ax.set(xlabel="Time [s]", ylabel="Distance [m]")
    
    label_params = (ticks_t, labels_t, ticks_d, labels_d)

    plot_heatmap(cmap1, df, 'coolwarm', -cbar_minmax, cbar_minmax, 'Amplitude', *label_params)
    plot_heatmap(cmap2, 10 * np.log10(np.maximum(np.abs(df), 1e-12)), 'viridis', cbar_db_min, cbar_db_max, 'Amplitude [dB]', *label_params)

    plt.suptitle(process)
    plt.tight_layout()
    plt.savefig(f"./png/heatmap_{process}.png")
    plt.show()

def load_df(file, fields):
    """ファイルからデータフレームを読み込み、指定されたフィールドを削除します"""

    df = pd.read_csv(file, skiprows=136, delim_whitespace=True)
    df = df.drop(fields, axis=1)  # 指定されたフィールドを削除
    df.columns = df.columns.str.replace(',', '')  # 列名からカンマを削除
    return df

def set_df_pre(date_time, pre_file):
    """bstypeに基づいて差分の基準となるファイルを取得し、対応するデータフレームを返します"""

    def fetch_files(ext, conds):
        """指定の拡張子と条件でファイルをフィルタリングして取得します"""
        
        all_files = glob.glob(f'*.{ext}')
    
        if date_time is None:
            return [s for s in all_files]
        else:
            return [s for s in all_files if any(cond in s for cond in conds)]

    if pre_file == None:
        exts = ['fdd', 'bsc']

        if date_time is None:
            filtered_files = [file for ext in exts for file in glob.glob(f'*.{ext}')]
        else:
            filtered_files = [file for ext in exts for file in glob.glob(f'*.{ext}') if any(dt in file for dt in date_time)]

        if filtered_files:
            pre_file = natsorted(filtered_files)[0]
        else:
            print("基準となるファイルがありません．")

    extension = pre_file.split('.')[-1]
    bstype = 1 if extension == 'fdd' else (2 if extension == 'bsc' else None)
    bstype_conds = {i: {'ext': 'fdd' if i == 1 else 'bsc', 'conds': date_time} for i in range(1, 3)}

    ext = bstype_conds[bstype]['ext']
    conds = bstype_conds[bstype]['conds']
    files = fetch_files(ext, conds)
    files = natsorted(files)

    fields = ["No.", "Diff,", "GHz"] if bstype == 1 else  ["No.", "GHz"]
    df_pre = load_df(pre_file, fields)
    df_pre_strain, df_pre_temp_strain, df_pre_temp = [df_pre.copy() for _ in range(3)]

    return files, pre_file, bstype, df_pre, df_pre_strain, df_pre_temp_strain, df_pre_temp

def proc_opt_data(file_name, bstype, df_pre, df_pre_strain, df_pre_temp_strain, df_pre_temp):
    """指定されたファイルタイプ（bstype）に基づいてデータフレームを処理し、処理されたデータフレームを返します。"""

    # ひずみの計算と基準ファイルとの差分を取得
    def calculate_strain(df, df_pre, factor):
        df['Frequency'] = (df['Frequency'] - df_pre['Frequency']) * factor
        return df

    # 直前のファイルを差分の基準となるファイルとして使用したい場合 (必要に応じてコメントアウトを外してください)
    # df_pre_strain, df_pre_temp_strain, df_pre_temp = [df_pre.copy() for _ in range(3)]
    # df_pre = df.copy() # 次のステップの準備

    # ファイルの読み込みフィールドを設定
    fields = ["No.", "Diff,", "GHz"] if bstype == 1 else ["No.", "GHz"]

    # ファイルの読み込み
    df_strain = load_df(file_name, fields)
    df_temp_strain, df_temp = df_strain.copy(), df_strain.copy()

    # ひずみ係数の設定
    factors_bstype1 = (1/0.145, -1.372/(0.145*-1.68), -1/1.68)
    factors_bstype2 = (1/(-0.466*1e-3), 1.074*1e-3/(-0.466*1e-3*1.207*1e-3), 1/(1.207*1e-3))
    factors = factors_bstype1 if bstype == 1 else factors_bstype2

    df_strain, df_temp_strain, df_temp = map(lambda df: calculate_strain(df[0], df[1], df[2]), zip([df_strain, df_temp_strain, df_temp], [df_pre_strain, df_pre_temp_strain, df_pre_temp], factors))

    return df_pre, df_strain, df_temp_strain, df_temp

def proc_temp_adj(df_strain, df_temp_strain, df_temp, em_start, em_end, pc_start, pc_end):
    """各セグメントに対してデータを処理し、結果のリストを返します。"""

    def separate_parts(df_strain, df_temp_strain, df_temp, em_start, em_end, pc_start, pc_end, location):
        """データセグメントを処理して、結果のデータフレームを返します。"""

        def temp_adj(df_part_strain, df_part_temp_strain):
            """温度補正を行う関数"""
            
            df_part_temp_strain['FrequencyTemp'] = df_part_temp_strain['Frequency']
            df_part_temp_strain['Frequency'] = np.NaN
            
            df_strain_adj = pd.DataFrame()
            df_strain_adj = pd.concat([df_part_strain, df_part_temp_strain]).sort_values(by="Distance(m)")
            df_strain_adj['FrequencyTemp'].interpolate('index', inplace=True, limit_direction='both')
            df_strain_adj['Frequency'] -= df_strain_adj['FrequencyTemp']
            df_strain_adj = df_strain_adj.dropna(subset=['Frequency'])
            df_strain_adj = df_strain_adj[['Distance(m)', 'Frequency']]

            return df_strain_adj

        df_part_strain = df_strain[(df_strain['Distance(m)'] > em_start) & (df_strain['Distance(m)'] < em_end)].copy()
        df_part_strain['Distance(m)'] = em_end - df_part_strain['Distance(m)']

        df_part_temp_strain = df_temp_strain[(df_temp_strain['Distance(m)'] > pc_start) & (df_temp_strain['Distance(m)'] < pc_end)].copy()
        df_part_temp_strain['Distance(m)'] -= pc_start

        # 温度の計算 (必要に応じてコメントアウトを外してください)
        # df_part_temp = df_temp[(df_temp['Distance(m)'] > pc_start) & (df_temp['Distance(m)'] < pc_end)].copy()
        # df_part_temp['Distance(m)'] -= pc_start

        df_strain_adj = temp_adj(df_part_strain, df_part_temp_strain)
        df_strain_adj['Part'] = location
        df_strain_adj['Frequency'] *= 1.0e-6

        return df_strain_adj

    df_strain_adj_toe, df_strain_adj_slope, df_strain_adj_shoulder = [None] * 3  # 初期化しておく
    for i in range(3):
        em_start_i, em_end_i, pc_start_i, pc_end_i = em_start[i], em_end[i], pc_start[i], pc_end[i]
        location_i = ["toe", "slope", "shoulder"][i]
        result_i = separate_parts(df_strain, df_temp_strain, df_temp, em_start_i, em_end_i, pc_start_i, pc_end_i, location_i)
    
        if i == 0: df_strain_adj_toe = result_i
        elif i == 1: df_strain_adj_slope = result_i
        else: df_strain_adj_shoulder = result_i

    return df_strain_adj_shoulder, df_strain_adj_slope, df_strain_adj_toe

def proc_dist_adj(df_strain_adj_shoulder, df_strain_adj_slope, df_strain_adj_toe, d_shoulder, d_slope, d_toe):
    comb_df_strain_adj_shoulder = pd.DataFrame()
    comb_df_strain_adj_toe = pd.DataFrame()
    comb_df_strain_adj_slope = pd.DataFrame()

    for i in range(len(d_shoulder) - 1):
        df_part_shoulder = df_strain_adj_shoulder[(df_strain_adj_shoulder['Distance(m)'] > d_shoulder[i]) & (df_strain_adj_shoulder['Distance(m)'] < d_shoulder[i + 1])].copy()
        df_part_slope = df_strain_adj_slope[(df_strain_adj_slope['Distance(m)'] > d_slope[i]) & (df_strain_adj_slope['Distance(m)'] < d_slope[i + 1])].copy()
        df_part_toe = df_strain_adj_toe[(df_strain_adj_toe['Distance(m)'] > d_toe[i]) & (df_strain_adj_toe['Distance(m)'] < d_toe[i + 1])].copy()

        df_part_shoulder['Distance(m)'] -= d_shoulder[0]
        df_part_slope['Distance(m)'] = d_shoulder[i] + (d_shoulder[i + 1] - d_shoulder[i]) / (d_slope[i + 1] - d_slope[i]) * (df_part_slope['Distance(m)'] - d_slope[i]) -d_shoulder[0]
        df_part_toe['Distance(m)'] = d_shoulder[i] + (d_shoulder[i + 1] - d_shoulder[i]) / (d_toe[i + 1] - d_toe[i]) * (df_part_toe['Distance(m)'] -  d_toe[i]) -d_shoulder[0]

        comb_df_strain_adj_shoulder = pd.concat([comb_df_strain_adj_shoulder, df_part_shoulder], ignore_index=True)
        comb_df_strain_adj_slope = pd.concat([comb_df_strain_adj_slope, df_part_slope], ignore_index=True)
        comb_df_strain_adj_toe = pd.concat([comb_df_strain_adj_toe, df_part_toe], ignore_index=True)

    return comb_df_strain_adj_shoulder, comb_df_strain_adj_slope, comb_df_strain_adj_toe

def calc_shear_strain(df_strain_adj_shoulder, df_strain_adj_slope, df_strain_adj_toe):
    """正と負の値をでわけてせん断ひずみを計算"""

    dataframes = [df_strain_adj_shoulder, df_strain_adj_slope, df_strain_adj_toe]
    result_dataframes = []

    for df in dataframes:
        plus_data = df[df['Frequency'] >= 0.0].copy()
        plus_data['Shear_strain'] = ((plus_data['Frequency'] + 1.0)**2 - 1.0)**0.5
        minus_data = df[df['Frequency'] < 0.0].copy()
        minus_data['Shear_strain'] = (1.0 - (minus_data['Frequency'] + 1.0)**2)**0.5
        data = pd.concat([plus_data, minus_data])
        result_dataframes.append(data)

    return pd.concat(result_dataframes)

def plt_contour(i, df_strain_adj, column, title_file, title_graph, nx, ny, cbar_max, cbar_min, plt_max):
    """提供されたデータフレームに基づいてプロットを生成し、保存します。"""

    def prepare_contour(df_strain_adj, nx, ny):
        """プロットの準備を行います。"""
        
        minx, maxx, miny, maxy = min(df_strain_adj['Distance(m)']), max(df_strain_adj['Distance(m)']), 0, 10
        xi, yi = np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny)
        xi, yi = np.meshgrid(xi, yi)
        z = df_strain_adj[column]
        df_strain_adj['Line(m)'] = df_strain_adj['Part'].apply(lambda x: 0 if x == 'shoulder' else (5 if x == 'slope' else 10))
        zi = interpolate.griddata((df_strain_adj['Distance(m)'], df_strain_adj['Line(m)']), z, (xi, yi), method='linear')
        return xi, yi, zi

    def setup_contour(ax, title_graph):
        """プロットの基本設定を行います。"""
        
        ax.set_yticks([0.0, 5.0, 10.0])
        ax.set_yticklabels(['Toe', 'Slope', 'Shoulder'])
        ax.set_xlabel("location (m)")
        for line in [0.0, 5.0, 10.0]:
            ax.axhline(line, color="white", lw=0.5)
        ax.set_title(title_graph)

        cf = plt.contourf(xi, yi, 100 * zi, np.linspace(cbar_min, cbar_max, 21), cmap="viridis", extend='max')
        cbar = plt.colorbar(cf, ticks=np.arange(cbar_min, cbar_max*1.01, (cbar_max - cbar_min) / 5.0))
        cbar.set_label('shear strain (%)')

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    # プロットの準備
    xi, yi, zi = prepare_contour(df_strain_adj, nx, ny)
    setup_contour(ax, title_graph)

    plt.title(title_graph)
    plt.tight_layout()
    fig.savefig(f"./png/contour_{title_file}.png")

    if i > 0 and i < plt_max: plt.show()
    else: plt.close()

def proc_t_dmn_data(df_strain_adj, comb_res_shoulder, comb_res_slope, comb_res_toe, parts, locs, title_graph):

    for part in parts:
        result_rows = []
        subset_df = df_strain_adj[df_strain_adj['Part'] == part]  # 特定の"Part"に絞る

        for value in locs:
            nearest_row = subset_df.loc[(subset_df['Distance(m)'] - value).abs().idxmin()]
            result_rows.append(nearest_row['Frequency'])

        if part == "shoulder":
            comb_res_shoulder[title_graph] = result_rows
        elif part == "slope":
            comb_res_slope[title_graph] = result_rows
        else:
            comb_res_toe[title_graph] = result_rows

        # DataFrameの断片化を最小限に抑えるために新しいDataFrameを作成
        comb_res_shoulder = comb_res_shoulder.copy()
        comb_res_slope = comb_res_slope.copy()
        comb_res_toe = comb_res_toe.copy()

    return comb_res_shoulder, comb_res_slope, comb_res_toe

def proc_wl_rf_files(folder_path):
    """指定したフォルダ内の.datファイルを読み込んで結合したdataframeを返す関数"""

    def read_dat_file(file_path):
        """指定した.datファイルを読み込んでdataframeに変換する関数"""
        df = pd.read_csv(file_path, encoding='shift_jis', header=9)
        # "Unnamed: 0" という列ラベルを "日付" に変更
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': '日付'})
        # "Unnamed: " がラベルに含まれる列を削除
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        return df

    # フォルダ内の.datファイルのリストを取得
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    # 各ファイルを読み込んでdataframeのリストを作成
    dfs = [read_dat_file(os.path.join(folder_path, dat_file)) for dat_file in dat_files]
    # 全てのdataframeを結合
    merged_df = pd.concat(dfs, ignore_index=True)
    # "日付"列をindexに設定
    merged_df.set_index('日付', inplace=True)
    return merged_df

def proc_wl_rf_df(df):
    """指定したDataFrameを加工する関数"""
    stacked = df.stack().reset_index()  # DataFrameをstack
    stacked.columns = ['日付', '時間', '値']  # 列名の変更

    # 時間の抽出と日付の変換
    stacked['時間'] = stacked['時間'].str.extract(r'(\d+)時')[0].astype(int)
    stacked['日付'] = pd.to_datetime(stacked['日付'])
    mask = stacked['時間'] == 24  # "24時"を"0時"に変更し、日付を1日進める
    stacked.loc[mask, '日付'] += pd.Timedelta(days=1)
    stacked.loc[mask, '時間'] = 0
    stacked['日付'] = pd.to_datetime(stacked['日付'].dt.strftime('%Y-%m-%d') + ' ' + stacked['時間'].astype(str) + ':00')
    stacked.set_index('日付', inplace=True)
    stacked = stacked.sort_index()
    stacked.drop(columns='時間', inplace=True)
    stacked.replace(["閉局", "　", " ", "", -9999.99], float("nan"), inplace=True)  # "閉局"と空白をNaNに変更
    return stacked.astype(float)  # float型に変換

def plt_wl_rf(
    stacked1=None,
    stacked2=None,
    stacked3=None,
    stacked4=None,
    start_date=None,
    end_date=None,
    miny=None,
    maxy=None,
    rain_miny=None,
    rain_maxy=None,
    offset1=0,
    offset2=0,
    width2=0.042,
    out=None,
    fig_name_7_8k="waterlevel-7.8k.png",
    fig_name_8_0k="waterlevel-8.0k.png",
):
    """指定したデータでグラフを描画する関数。out に OutputManager を渡すと output/{case_id}/figures/ に保存する。
    fig_name_7_8k / fig_name_8_0k で保存ファイル名を指定。balo 補正なしは waterlevel-7.8k-nobalo.png 等を渡す。
    """

    # 凡例の表示名と順序: xxk River water, xxk River slope, xxk Back shoulder, xxk Back slope, xxk Back toe
    _LEGEND_PARTS_ORDER = ["front", "shoulder", "slope", "toe"]
    _LEGEND_DISPLAY_NAMES = {"front": "River slope", "shoulder": "Back shoulder", "slope": "Back slope", "toe": "Back toe"}

    def plt_wl_rf_single(ax_rain, ax_wl, stacked1, stacked2, stacked3, stacked4, offset, prefix, width2, out=None, fig_name=None, rain_miny=None, rain_maxy=None):
        """単一のグラフを描画する補助関数。上: 降雨のみ、下: 水位＋堤体内水位。凡例は xxk River water / River slope / Back shoulder / Back slope / Back toe の順。"""
        # 上段: 降雨のみ
        if stacked2 is not None and not stacked2.empty:
            ax_rain.bar(stacked2.index, stacked2['値'].values, width=width2, color='blue')
            if rain_miny is not None and rain_maxy is not None:
                ax_rain.set_ylim(rain_miny, rain_maxy)
            else:
                ax_rain.set_ylim(ax_rain.get_ylim()[::-1])
        ax_rain.set_ylabel("Rainfall (mm/h)")
        ax_rain.set_xlim(start_date, end_date)
        ax_rain.grid(True)
        ax_rain.tick_params(labelbottom=False)

        # 下段: 河川水位＋堤体内水位（凡例順: River water → River slope → Back shoulder → Back slope → Back toe）
        if stacked1 is not None and not stacked1.empty:
            ax_wl.plot(stacked1.index, stacked1['値'] + offset, label=f"{prefix} River water", color='black')
        for part in _LEGEND_PARTS_ORDER:
            col_name = f"{prefix} {part}"
            for src in (stacked3, stacked4):
                if src is not None and not src.empty and col_name in src.columns:
                    ax_wl.plot(src.index, src[col_name], label=f"{prefix} {_LEGEND_DISPLAY_NAMES[part]}")
                    break

        ax_wl.set_ylabel("Water level (m)")
        ax_wl.set_xlabel("Date")
        ax_wl.set_xlim(start_date, end_date)
        ax_wl.set_ylim(miny, maxy)
        ax_wl.legend()
        ax_wl.grid(True)

        fig = ax_rain.get_figure()
        fig.tight_layout()
        if out is not None and fig_name is not None:
            fig.savefig(out.fig_path(fig_name))
            plt.close(fig)
        else:
            plt.show()

    # グラフ A の作成（上: 降雨、下: 水位、高さ比 1:2）
    fig_a, (ax_rain_a, ax_wl_a) = plt.subplots(2, 1, height_ratios=[1, 2], sharex=True)
    plt_wl_rf_single(ax_rain_a, ax_wl_a, stacked1, stacked2, stacked3, stacked4, offset1, "7.8k", width2, out=out, fig_name=fig_name_7_8k, rain_miny=rain_miny, rain_maxy=rain_maxy)

    # グラフ B の作成（上: 降雨、下: 水位、高さ比 1:2）
    fig_b, (ax_rain_b, ax_wl_b) = plt.subplots(2, 1, height_ratios=[1, 2], sharex=True)
    plt_wl_rf_single(ax_rain_b, ax_wl_b, stacked1, stacked2, stacked3, stacked4, offset2, "8.0k", width2, out=out, fig_name=fig_name_8_0k, rain_miny=rain_miny, rain_maxy=rain_maxy)

def proc_files_oyo(folder_path, parts, SNs, gw_elevs, nd=None):
    """指定したフォルダ内の.datファイルを読み込んで結合したdataframeを返す関数"""

    def read_dat_file_oyo(file_path, parts, SNs):
        """指定した.datファイルを読み込んでdataframeに変換する関数"""
        with open(file_path, 'r', encoding='cp932') as f:
            lines = f.readlines()

        # シリアルナンバーの抽出
        serial_number = int(re.search(r'(\d{7})', lines[12]).group(1))
        
        # シリアルナンバーに基づいて部位名を取得
        try:
            part_name = parts[SNs.index(serial_number)]
        except ValueError:
            raise ValueError(f"No part name found for serial number: {serial_number}")

        data_str = ''.join(lines[45:-1])
        df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, encoding='cp932')
        
        # 時間列を取得し、コロンで分割します
        time_parts = df.iloc[:, 1].str.split(':')
        # 時と分だけを取得し、秒を00に設定して新しい時間文字列を作成します
        new_time_strings = time_parts.str[0] + ':' + time_parts.str[1] + ':00'
        # 日付列と新しい時間文字列を結合します
        datetime_strings = df.iloc[:, 0] + ' ' + new_time_strings

        # datetimeオブジェクトを作成し、インデックスとして設定します
        df['datetime'] = pd.to_datetime(datetime_strings)
        df.set_index('datetime', inplace=True)
        df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)
        
        df.columns = [part_name] + df.columns[1:].tolist()
        return df

    # フォルダ内の.oyoファイルのリストを取得
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.oyo')]
    # 各ファイルを読み込んでdataframeのリストを作成
    dfs = [read_dat_file_oyo(os.path.join(folder_path, dat_file), parts, SNs) for dat_file in dat_files]
    # dfsのそれぞれのデータフレームを1つの大きなデータフレームにマージする
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.combine_first(df)

    if "balo" in merged_df.columns:
        reference_values = merged_df["balo"]
        for column in merged_df.columns:
            if column != "balo":  # 参照列自体からは引かない
                # baloがNaNの場所で、他の列もNaNに設定
                merged_df.loc[reference_values.isna(), column] = np.nan
                # NaNでない場所だけ減算を行う
                mask = ~reference_values.isna()
                merged_df.loc[mask, column] -= reference_values[mask]

    if nd is not None:
        merged_df = merged_df.where(merged_df > nd )

    for part, elev in zip(parts, gw_elevs):
        if part in merged_df.columns:
            merged_df[part] = merged_df[part] + elev

    merged_df.drop("balo", axis=1, inplace=True)

    return merged_df

def freq_ftr_inc(order, df, fs, f1=None, f2=None, filter_type=None):
    """すべての列をフィルタリングし、元のデータフレームを更新して返します。"""

    def butter_filter(order, f1, f2, filter_type):
        """バターワースフィルタを生成し、フィルタの係数を返します。"""
        if filter_type == 'bandstop' and f1 and f2:
            return signal.butter(order, [f1, f2], btype='bandstop', fs=fs, output='sos')
        elif filter_type is None and f1:
            return signal.butter(order, f1, btype='low', fs=fs, output='sos')
        elif filter_type is None and f2:
            return signal.butter(order, f2, btype='high', fs=fs, output='sos')
        elif filter_type is None and f1 and f2:
            return signal.butter(order, [f1, f2], btype='band', fs=fs, output='sos')
        return None

    def process_column(col, sos):
        """指定された列にフィルタを適用し、更新された列を返します。"""
        # 元のインデックスとNaNの位置を保存
        original_index = col.index
        nan_locations = col.isna()
        
        # NaNを削除および補間
        col = col.dropna().interpolate()

        if sos is None:
            # フィルタが適用されない場合は元のNaN値を復元し、元の列を返す
            col = col.reindex(original_index)
            return col

        # フィルタを適用
        filtered_col = signal.sosfiltfilt(sos, col)
        filtered_col = pd.Series(filtered_col, index=col.index)

        # 元のNaN値とインデックスを復元
        filtered_col = filtered_col.where(~nan_locations, pd.Series([None]*len(filtered_col), index=filtered_col.index))
        filtered_col = filtered_col.reindex(original_index)
        
        return filtered_col

    sos = butter_filter(order, f1, f2, filter_type)  # フィルタの係数を事前に計算
    
    df = df.apply(lambda col: process_column(col, sos)) 
    
    return df

def remove_outliers_inc(df, section=None, threshold=None):
    """異常値除去を行う関数"""
    
    def to_numeric(col):
        return pd.to_numeric(col, errors='coerce')

    def find_outliers(col, section, threshold):
        """指定された区間と閾値で外れ値を検出する関数"""
        rolling_mean = col.rolling(section).mean()
        rolling_std = col.rolling(section).std()
        return (col - rolling_mean).abs() >= threshold * rolling_std
    
    # 全ての列を数値に変換し、数値に変換できない値をNaNに置き換えます
    df_numeric = df.apply(to_numeric)
    
    # NaN値の位置を保存
    nan_locations = df_numeric.isna()
    
    # NaN値を一時的に補完
    df_filled = df_numeric.interpolate()
    
    if section is not None and threshold is not None:
        # 全ての列に異常値除去を適用
        outliers = df_filled.apply(lambda col: find_outliers(col, section, threshold))
        df_filled[outliers] = np.nan  # この1行で全ての外れ値をNaNに置き換え
    
    # 元のNaN値を復元
    df_out = df_filled.where(~nan_locations, df_numeric)
    
    return df_out

def rolling_mean(df, wndw = None):
    """移動平均を適用する関数"""
    
    if wndw is not None:
        df = df.rolling(wndw, center=True).mean()
    
    return df

# フィルター
def proc_ftr_inc(df, ord, fs, f1, f2, sect, thresh, wnd, fltr_type):
    df = freq_ftr_inc(ord, df, fs, f1=f1, f2=f2, filter_type = fltr_type)
    df = remove_outliers_inc(df, section=sect, threshold=thresh)
    df = rolling_mean(df, wnd)

    return df

# 関数を定義: 角度計算
def compute_angles(df, prefix):
    datax = df.iloc[:, 0]
    datay = df.iloc[:, 1]
    dataz = df.iloc[:, 2]
    df[f'{prefix} theta'] = 180 / math.pi * np.arctan(datax / np.sqrt(datay**2 + dataz**2))
    df[f'{prefix} psi'] = 180 / math.pi * np.arctan(datay / np.sqrt(datax**2 + dataz**2))
    df[f'{prefix} phi'] = 180 / math.pi * np.arctan(np.sqrt(datax**2 + datay**2) / dataz)
    return df.iloc[:, [3, 4, 5]]

# 関数を定義: セグメント処理
def process_segment_inc(segment):
    mean = segment.mean()
    return segment - mean

# 関数を定義: セグメントに分割して処理
def process_segments_inc(df, df_temp, maint_dates):
    result_df = pd.DataFrame(index=df.index, columns=df.columns)
    result_df_temp = pd.DataFrame(index=df_temp.index, columns=df_temp.columns)

    for date in maint_dates:
        start_time = date.replace(hour=0, minute=0, second=0)
        end_time = date.replace(hour=23, minute=59, second=59)
        df.loc[start_time:end_time] = np.nan
        df_temp.loc[start_time:end_time] = np.nan

    for i, date in enumerate(maint_dates):
        if i == 0:
            # 最初の日付の前のセグメントを処理
            segment = df.loc[:date - pd.Timedelta(minutes=10)]
            result_df.loc[segment.index] = process_segment_inc(segment)
            segment_temp = df_temp.loc[:date - pd.Timedelta(minutes=10)]
            result_df_temp.loc[segment_temp.index] = process_segment_inc(segment_temp)
        else:
            # 以前の日付と現在の日付の間のセグメントを処理
            previous_date = maint_dates[i-1]
            segment = df.loc[previous_date + pd.Timedelta(minutes=10):date - pd.Timedelta(minutes=10)]
            result_df.loc[segment.index] = process_segment_inc(segment)
            segment_temp = df_temp.loc[previous_date + pd.Timedelta(minutes=10):date - pd.Timedelta(minutes=10)]
            result_df_temp.loc[segment_temp.index] = process_segment_inc(segment_temp)

        if i == len(maint_dates) - 1 and date != df.index[-1]:
            # 最後の日付の後のセグメントを処理
            segment = df.loc[date + pd.Timedelta(minutes=10):]
            result_df.loc[segment.index] = process_segment_inc(segment)
            segment_temp = df_temp.loc[date + pd.Timedelta(minutes=10):]
            result_df_temp.loc[segment_temp.index] = process_segment_inc(segment_temp)

        # 結果を元のDataFrameに戻す
        df.update(result_df)
        df_temp.update(result_df_temp)

# 傾きに変換し，平均からの変動を求めるとともに，作業日の反映，フィルタの適用
def proc_files_inc(folder_path, parts, fnames, maint_dates, *t_inc_params):
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dat_files = natsorted(dat_files)
    all_dfs = []
    all_dfs_temp = []

    for file_name in dat_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, encoding="shift_jis", sep=",", header=0)
        df[df.columns[0]] = pd.to_datetime(df.iloc[:,0])
        df_temp = df.iloc[:, [0, 10]].copy()
        df = df.iloc[:, [0, 1, 2, 3]]
        df.set_index(df.columns[0], inplace=True)
        df_temp.set_index(df_temp.columns[0], inplace=True)
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='10T')
        df = df.reindex(full_range)
        df_temp = df_temp.reindex(full_range)

        part_prefix = None
        for fname, part in zip(fnames, parts):
            if fname in file_name:
                part_prefix = part
                break

        if part_prefix is not None:
            df.columns = [f'{part_prefix} {col.replace("センサー1", "")}' for col in df.columns]
            df_temp.columns = [f'{part_prefix} {col.replace("センサー1", "").replace("温度", "Temp.")}' for col in df_temp.columns]
        else:
            print(f'No matching keyword found in {file_name}')

        df = compute_angles(df, part_prefix)
        maint_dates = pd.to_datetime(maint_dates)
        process_segments_inc(df, df_temp, maint_dates)

        # 傾きのフィルター処理（周波数フィルター，異常値除去，移動平均）
        df = proc_ftr_inc(df, *t_inc_params)

        # 処理したDataFrameをリストに追加
        all_dfs.append(df)
        all_dfs_temp.append(df_temp)

    # 全てのDataFrameを結合する
    final_df = pd.concat(all_dfs, axis=1)
    final_df_temp = pd.concat(all_dfs_temp, axis=1)

    return final_df, final_df_temp

def plt_fft_inc(signal, sp_type, fs, x_range = None, ax = None):
    """スペクトルプロット関数"""

    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    spectrum = np.abs(np.fft.fft(signal)) / n

    if sp_type == "Amplitude": transform_func = lambda x: x
    elif sp_type in ['Amplitude density', 'Power']: transform_func = lambda x: x**2
    else: transform_func = lambda x: x**2 / fs

    ax.plot(freqs[:n//2], transform_func(spectrum)[:n//2])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(sp_type)
    ax.set_xlim(0, x_range)
    ax.grid()

def plt_spectro_inc(sig, sp_type, fs, n_fft, hop_length, sp_cbar_db_min, sp_cbar_db_max, ax = None):
    """スペクトログラムプロット関数（scipy.signal.stft 使用）"""
    noverlap = n_fft - hop_length
    f, t, Zxx = signal.stft(sig, fs=fs, window="hann", nperseg=n_fft, noverlap=noverlap)
    S = np.abs(Zxx)
    if sp_type == "Amplitude":
        spectrogram = S
    elif sp_type in ["Amplitude density", "Power"]:
        spectrogram = S ** 2
    else:
        spectrogram = S ** 2 / n_fft
    ref = np.max(spectrogram)
    db = 10 * np.log10(np.maximum(spectrogram / ref, 1e-12))
    img = ax.pcolormesh(t, f, db, cmap="viridis", vmin=sp_cbar_db_min, vmax=sp_cbar_db_max, shading="auto")
    plt.colorbar(img, ax=ax, label=sp_type + " [dB]")

    ax.set_xlabel("Time")
    ax.tick_params(rotation=30)
    ax.set_ylim(0, fs/2)
    ax.set_ylabel('Frequency [Hz]')
    ax.grid(True)