import os, re, math
import pandas as pd
import numpy as np
from natsort import natsorted
from io import StringIO

try:
    import com_fs_ftr
except ImportError:
    com_fs_ftr = None  # 傾斜データのフィルタ用。河川水位・堤体内水位のみ使う場合は不要


def _confirm_and_delete(folder_path, filenames, data_label="データ"):
    """削除可能ファイルのリストを表示し、確認後に削除する。filenames は削除対象のファイル名のリスト（重複なし）。"""
    if not filenames:
        return
    folder_path = os.path.abspath(folder_path)
    try:
        reply = input(f"[{data_label}] 上記の削除可能ファイル {len(filenames)} 件を削除しますか？ (y/n): ").strip().lower()
    except EOFError:
        reply = "n"
    if reply not in ("y", "yes"):
        return
    for fname in filenames:
        path = os.path.join(folder_path, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"  削除しました: {fname}")
            except OSError as e:
                print(f"  削除失敗: {fname} ({e})")
        else:
            print(f"  ファイルがありません: {fname}")


def _report_overlaps_conflicts_wl_rf(stacked_with_file, data_label="データ"):
    """河川水位・降雨: 包含関係にある重複（削除可能ファイル）と衝突（2ファイル名）を検出。"""
    if stacked_with_file.empty or "_file" not in stacked_with_file.columns:
        return [], []
    stacked_with_file = stacked_with_file.dropna(subset=["値"])
    if stacked_with_file.empty:
        return [], []
    file_ranges = stacked_with_file.groupby("_file").agg(
        start=("日付", "min"), end=("日付", "max")
    ).reset_index()
    # 一方が他方を包含または同一の場合のみ「削除可能」として列挙
    deletable = []
    for i in range(len(file_ranges)):
        for j in range(i + 1, len(file_ranges)):
            r1, r2 = file_ranges.iloc[i], file_ranges.iloc[j]
            s1, e1, s2, e2 = r1["start"], r1["end"], r2["start"], r2["end"]
            if s1 >= s2 and e1 <= e2:
                deletable.append((r1["_file"], r2["_file"]))  # r1 は r2 に包含 → r1 削除可能
            elif s2 >= s1 and e2 <= e1:
                deletable.append((r2["_file"], r1["_file"]))  # r2 は r1 に包含 → r2 削除可能
    # 同一時刻で異なる値（衝突）→ 関与する2ファイル名のリスト
    conflicts = []
    g = stacked_with_file.groupby("日付")
    for dt, grp in g:
        if grp["値"].nunique() > 1:
            files = grp["_file"].unique().tolist()
            conflicts.append({"datetime": dt, "files": files})
    return deletable, conflicts


def _format_overlap_conflict_reports(deletable, conflicts, data_label="データ"):
    """重複（削除可能リスト）・衝突（2ファイル名）の報告を表示用テキストに整形。"""
    lines = []
    if deletable:
        lines.append(f"[{data_label}] 削除可能なファイル（他ファイルに包含または同一）:")
        for del_f, in_f in deletable:
            lines.append(f"  {del_f} （{in_f} に包含）")
    if conflicts:
        lines.append(f"[{data_label}] 衝突しているファイル:")
        for c in conflicts:
            fnames = " と ".join(c["files"])
            lines.append(f"  {fnames}")
    return "\n".join(lines) if lines else None


def proc_wl_rf_files(folder_path, report_overlaps_conflicts=True, interactive_delete=False, verbose=True):
    """指定したフォルダ内の.datファイルを読み込んで結合したdataframeを返す関数。
    report_overlaps_conflicts=True のとき、重複・衝突を検出して表示する。
    interactive_delete=True のとき、削除可能ファイル表示後に確認し、了承なら該当ファイルを削除する。
    """
    def read_dat_file(file_path):
        """指定した.datファイルを読み込んでdataframeに変換する関数"""
        df = pd.read_csv(file_path, encoding='shift_jis', header=9)
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': '日付'})
        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
        return df

    dat_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.dat')])
    if not dat_files:
        return pd.DataFrame()

    # ファイルごとに読み込み、_file を付与してから結合
    dfs = []
    for dat_file in dat_files:
        path = os.path.join(folder_path, dat_file)
        df = read_dat_file(path)
        df["_file"] = dat_file
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # 重複・衝突の検出用に stack 形式を一時作成
    if report_overlaps_conflicts and verbose and len(dfs) > 1:
        time_cols = [c for c in merged_df.columns if c not in ("日付", "_file")]
        long = merged_df.melt(id_vars=["日付", "_file"], value_vars=time_cols, var_name="時間", value_name="値")
        long["時間"] = long["時間"].str.extract(r"(\d+)")[0].astype(int)
        long["日付"] = pd.to_datetime(long["日付"])
        mask24 = long["時間"] == 24
        long.loc[mask24, "日付"] = long.loc[mask24, "日付"] + pd.Timedelta(days=1)
        long.loc[mask24, "時間"] = 0
        long["日付"] = pd.to_datetime(
            long["日付"].dt.strftime("%Y-%m-%d") + " " + long["時間"].astype(str) + ":00"
        )
        long = long.drop(columns="時間")
        deletable, conflicts = _report_overlaps_conflicts_wl_rf(long)
        label = "河川水位" if "riverlevel" in folder_path else "降雨"
        report = _format_overlap_conflict_reports(deletable, conflicts, label)
        if report:
            print(report)
        if interactive_delete and deletable:
            to_remove = list({del_f for del_f, _ in deletable})
            _confirm_and_delete(folder_path, to_remove, label)

    merged_df = merged_df.drop(columns=["_file"], errors="ignore")
    merged_df.set_index("日付", inplace=True)
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


def _report_overlaps_conflicts_oyo(file_dfs):
    """堤体内水位: 同じ部位で一方が他方を包含する場合の削除可能ファイルと、衝突している2ファイル名を検出。"""
    if len(file_dfs) < 2:
        return [], []
    deletable = []
    conflicts = []
    long_rows = []
    for fname, df in file_dfs:
        df = df.copy()
        for c in df.columns:
            if c == "balo":
                continue
            long = df[[c]].reset_index()
            long.columns = ["datetime", "値"]
            long["part"] = c
            long["_file"] = fname
            long_rows.append(long)
    long_all = pd.concat(long_rows, ignore_index=True)
    long_all = long_all.dropna(subset=["値"])

    file_part_ranges = long_all.groupby(["_file", "part"]).agg(
        start=("datetime", "min"), end=("datetime", "max")
    ).reset_index()
    # 同じ部位で一方が他方を包含または同一の場合のみ削除可能として列挙
    parts_seen = file_part_ranges["part"].unique()
    for part in parts_seen:
        sub = file_part_ranges[file_part_ranges["part"] == part]
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                r1, r2 = sub.iloc[i], sub.iloc[j]
                s1, e1, s2, e2 = r1["start"], r1["end"], r2["start"], r2["end"]
                if s1 >= s2 and e1 <= e2:
                    deletable.append({"part": part, "deletable": r1["_file"], "contained_in": r2["_file"]})
                elif s2 >= s1 and e2 <= e1:
                    deletable.append({"part": part, "deletable": r2["_file"], "contained_in": r1["_file"]})

    g = long_all.groupby(["datetime", "part"])
    for (dt, part), grp in g:
        if grp["値"].nunique() > 1:
            files = grp["_file"].unique().tolist()
            conflicts.append({"datetime": dt, "part": part, "files": files})
    return deletable, conflicts


def _format_oyo_overlap_conflict_reports(deletable, conflicts):
    """堤体内水位の削除可能リスト・衝突（2ファイル名）を表示用テキストに整形。"""
    lines = []
    if deletable:
        lines.append("[堤体内水位] 削除可能なファイル（他ファイルに包含または同一）:")
        for d in deletable:
            lines.append(f"  部位 {d['part']}: {d['deletable']} （{d['contained_in']} に包含）")
    if conflicts:
        lines.append("[堤体内水位] 衝突しているファイル:")
        for c in conflicts:
            fnames = " と ".join(c["files"])
            lines.append(f"  {fnames}")
    return "\n".join(lines) if lines else None


def proc_files_oyo(folder_path, parts, SNs, gw_elevs, nd=None, report_overlaps_conflicts=True, interactive_delete=False, verbose=True):
    """指定したフォルダ内の.oyoファイルを読み込んで結合したdataframeを返す関数。
    report_overlaps_conflicts=True のとき、重複・衝突を検出して表示する。
    interactive_delete=True のとき、削除可能ファイル表示後に確認し、了承なら該当ファイルを削除する。
    """

    def read_dat_file_oyo(file_path, parts, SNs):
        """指定した.datファイルを読み込んでdataframeに変換する関数"""
        with open(file_path, 'r', encoding='cp932') as f:
            lines = f.readlines()

        serial_number = int(re.search(r'(\d{7})', lines[12]).group(1))
        try:
            part_name = parts[SNs.index(serial_number)]
        except ValueError:
            raise ValueError(f"No part name found for serial number: {serial_number}")

        data_str = ''.join(lines[45:-1])
        df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, encoding='cp932')
        time_parts = df.iloc[:, 1].str.split(':')
        new_time_strings = time_parts.str[0] + ':' + time_parts.str[1] + ':00'
        datetime_strings = df.iloc[:, 0] + ' ' + new_time_strings
        df['datetime'] = pd.to_datetime(datetime_strings)
        df.set_index('datetime', inplace=True)
        df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)
        df.columns = [part_name] + df.columns[1:].tolist()
        return df

    dat_files = natsorted([f for f in os.listdir(folder_path) if f.endswith('.oyo')])
    if not dat_files:
        return pd.DataFrame()

    file_dfs = [(f, read_dat_file_oyo(os.path.join(folder_path, f), parts, SNs)) for f in dat_files]
    dfs = [df for _, df in file_dfs]

    if report_overlaps_conflicts and verbose and len(file_dfs) > 1:
        deletable, conflicts = _report_overlaps_conflicts_oyo(file_dfs)
        report = _format_oyo_overlap_conflict_reports(deletable, conflicts)
        if report:
            print(report)
        if interactive_delete and deletable:
            to_remove = list({d["deletable"] for d in deletable})
            _confirm_and_delete(folder_path, to_remove, "堤体内水位")

    merged_df = dfs[0].copy()
    for df in dfs[1:]:
        merged_df = merged_df.combine_first(df)

    # balo 補正なし用（減算前のコピー）。図化で waterlevel-*-nobalo.png に保存する際に使用
    merged_df_nobalo = merged_df.copy()

    if "balo" in merged_df.columns:
        reference_values = merged_df["balo"]
        adjusted_columns = []
        for column in merged_df.columns:
            if column != "balo":  # 参照列自体からは引かない
                # baloがNaNの場所で、他の列もNaNに設定
                merged_df.loc[reference_values.isna(), column] = np.nan
                # NaNでない場所だけ減算を行う（大気圧の影響をキャンセル）
                mask = ~reference_values.isna()
                merged_df.loc[mask, column] -= reference_values[mask]
                adjusted_columns.append(column)
        if adjusted_columns and verbose:
            print("[堤体内水位] 各観測値から balo（大気圧）を減算し、大気圧の影響をキャンセルしました。対象:", ", ".join(adjusted_columns))
            print("[堤体内水位] ※ 図化に使用する堤体内水位は、上記 balo 減算後のデータです。")
    elif verbose:
        print("[堤体内水位] balo データがありません。大気圧キャンセルは行っていません。")

    merged_df.drop("balo", axis=1, inplace=True)
    merged_df_nobalo.drop("balo", axis=1, inplace=True, errors="ignore")

    if nd is not None:
        merged_df = merged_df.where(merged_df > nd)
        merged_df_nobalo = merged_df_nobalo.where(merged_df_nobalo > nd)

    for part, elev in zip(parts, gw_elevs):
        if part in merged_df.columns:
            merged_df[part] = merged_df[part] + elev
        if part in merged_df_nobalo.columns:
            merged_df_nobalo[part] = merged_df_nobalo[part] + elev

    # (balo 補正済み, balo 補正なし) のタプルで返す。両方とも図化に使用する。
    return merged_df, merged_df_nobalo


# フィルター（com_fs_ftr が利用可能な場合のみ。傾斜データ用）
def proc_ftr_inc(df, ord, fs, f1, f2, fltr_type, sect, thresh, wndw):
    if com_fs_ftr is None:
        raise ImportError("傾斜データのフィルタには com_fs_ftr が必要です。河川水位・堤体内水位のみの場合は proc_ftr_inc は使いません。")
    df = com_fs_ftr.freq_ftr(ord, df, fs, f1=f1, f2=f2, fltr_type=fltr_type, inc=1)
    df = com_fs_ftr.remove_outliers(df, section=sect, threshold=thresh, inc=1)
    df = com_fs_ftr.rolling_mean(df, wndw)

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
def proc_files_inc(folder_path, parts, fnames, maint_dates, t_inc_params):
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dat_files = natsorted(dat_files)
    all_dfs = []
    all_dfs_ftr = []
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
        df_ftr = proc_ftr_inc(df, *t_inc_params)

        # 処理したDataFrameをリストに追加
        all_dfs.append(df)
        all_dfs_ftr.append(df_ftr)
        all_dfs_temp.append(df_temp)

    # 全てのDataFrameを結合する
    final_df = pd.concat(all_dfs, axis=1)
    final_df_ftr = pd.concat(all_dfs_ftr, axis=1)
    final_df_temp = pd.concat(all_dfs_temp, axis=1)

    return final_df, final_df_ftr, final_df_temp


def process_data(flag, folder_path, interactive_delete=False, verbose=True):
    """データ処理を一般化するためのヘルパー関数。
    interactive_delete=True のとき、削除可能ファイル表示後に確認し、了承なら該当ファイルを削除する。
    """
    if flag == 1:
        merged_df = proc_wl_rf_files(
            folder_path, interactive_delete=interactive_delete, verbose=verbose
        )
        stacked = proc_wl_rf_df(merged_df)
        return stacked
    return None


def process_data_oyo(flag, folder_path, parts, SNs, gw_elevs, ND, interactive_delete=False, verbose=True):
    """データ処理を一般化するためのヘルパー関数。
    interactive_delete=True のとき、削除可能ファイル表示後に確認し、了承なら該当ファイルを削除する。
    戻り値: (balo 補正済み DataFrame, balo 補正なし DataFrame)。補正なしは図化の waterlevel-*-nobalo.png 用。
    """
    if flag == 1:
        merged_corrected, merged_nobalo = proc_files_oyo(
            folder_path, parts, SNs, gw_elevs, nd=ND,
            interactive_delete=interactive_delete, verbose=verbose
        )
        return merged_corrected, merged_nobalo
    return None, None


def process_data_inc(flag, folder_path, parts, fnames, maint_dates, *t_inc_params):
    """データ処理を一般化するためのヘルパー関数"""
    if flag == 1:
        stacked1, stacked2, stacked3 = proc_files_inc(folder_path, parts, fnames, maint_dates, t_inc_params)
        return stacked1, stacked2, stacked3
    return None, None, None


def filter_columns(df, keyword1, keyword2, keyword3):
    filtered_cols = []
    for col in df.columns:
        if (keyword1 is None or keyword1 in col) and \
        (keyword2 is None or keyword2 in col) and \
        (keyword3 is None or keyword3 in col):
            filtered_cols.append(col)
    return df[filtered_cols]


def resolve_duplicate_columns(df):
    """重複列を統合し、新しいデータフレームを返す関数"""
    df = df.copy()  # 元のデータフレームを変更しないようにコピーを作成
    duplicate_columns = df.columns[df.columns.duplicated()].unique()  # 重複列を検索

    for column in duplicate_columns:
        duplicated_cols = df.loc[:, df.columns == column]  # 重複する各列に対する処理
        new_column = duplicated_cols.ffill(axis=1).iloc[:, -1]  # 統合する新しい列を作成
        df[column] = new_column  # 新しい列でデータフレームを更新

    return df.loc[:, ~df.columns.duplicated()]  # 重複する列を削除して結果を返す
