import os
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import smtplib
from email.mime.text import MIMEText

# --- Google Drive API setup ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Lấy credentials từ secret
with open("credentials.json", "w") as f:
    f.write(os.getenv("GDRIVE_CREDENTIALS"))

creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

import io
from googleapiclient.http import MediaIoBaseDownload

# DEF ................................................................
def download_file(file_id, filename):
    """Tải file từ Google Drive về và lưu local"""
    request = drive_service.files().get_media(fileId=file_id)
    file_path = f"./{filename}"
    fh = io.FileIO(file_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        if status:
            print(f"Downloading {filename}: {int(status.progress() * 100)}%")
    return file_path


# --- Hàm tải file từ Google Drive ---
def download_file_from_gdrive(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(filename, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded {filename}")

# --- Lấy toàn bộ file trong folder ---
def get_all_files_from_folder(folder_id):
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        orderBy="createdTime",
        fields="files(id, name, createdTime)"
    ).execute()
    items = results.get('files', [])
    if not items:
        raise Exception("No files found in folder")
    return items  # list file

def read_excel_file(file_path):
    if file_path.endswith(".xls"):
        return pd.read_excel(file_path, engine="xlrd", skiprows=14, usecols="B:S")
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, engine="openpyxl", skiprows=14, usecols="B:S")
    else:
        raise Exception(f"Unsupported file format: {file_path}")

# --- Load toàn bộ file Excel ---
def load_all_excels(folder_id):
    files = get_all_files_from_folder(folder_id)
    df_list = []
    for f in files:
        file_id = f["id"]
        name = f["name"]
        file_path = download_file(file_id, name)   # << bây giờ hàm này đã có
        try:
            df = read_excel_file(file_path)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {name}: {e}")
    if not df_list:
        raise Exception("Không load được file Excel nào!")
    return pd.concat(df_list, ignore_index=True)

# --- Email ---
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email_report(report_text, attachment=None):
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASS = os.getenv("EMAIL_PASS")
    EMAIL_TO = os.getenv("EMAIL_TO")

    # Tạo email multipart
    msg = MIMEMultipart()
    msg['Subject'] = "Daily Report"
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO

    # Nội dung text
    msg.attach(MIMEText(report_text, "plain", "utf-8"))

    # Nếu có file đính kèm
    if attachment:
        with open(attachment, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment)}")
        msg.attach(part)

    # Gửi mail
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

    print("Email sent successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#  Reduce data to 14 Last Days
def process_trading_data(combined_df):
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17']]

    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Ticker",
        df.columns[2]: "TradingValue"
    })

    df = df[df['Date'].notna()]
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])

    df['Rank'] = (
        df.groupby('Date')['TradingValue']
        .rank(ascending=False, method='dense')
    )

    df['TradingValue'] = df['TradingValue'].astype(int)

    df['DistributionRate'] = (
        df['TradingValue']
        / df.groupby('Date')['TradingValue'].transform('sum')
    )
    df['DistributionRate'] = (
        (df['DistributionRate'] * 100).round(1).astype(str) + '%'
    )

    df = df[df['Rank'] <= 15]

    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days]

    return df_last_14









# Ranking Flow — HOSE Trading Value (Top 15)
def plot_ranking_flow_chart(
    df_last_14,
    chart_path="chart1.png",
    top_title="Ranking Flow — HOSE Trading Value (Top 15)",
    figsize=(14, 7)
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PolyCollection
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # --- Chuẩn bị màu ---
    unique_tickers = df_last_14['Ticker'].unique()
    color_map = cm.get_cmap('tab20', len(unique_tickers))
    colors = {
        ticker: mcolors.to_hex(color_map(i))
        for i, ticker in enumerate(unique_tickers)
    }

    # --- Sắp xếp dữ liệu ---
    df_sorted = df_last_14.sort_values(
        by=['Date', 'TradingValue'],
        ascending=[True, True]
    )

    dates = sorted(df_last_14['Date'].unique())
    tickers = df_last_14['Ticker'].unique()

    columns = {}
    for date in dates:
        day_data = df_sorted[df_sorted['Date'] == date].copy()
        columns[date] = day_data.reset_index(drop=True)

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=figsize)
    x_gap = 2
    bar_width = 1

    max_day_value = (
        df_last_14.groupby('Date')['TradingValue'].sum().max()
    )

    # --- Vẽ stacked bar ---
    for i, date in enumerate(dates):
        x = i * x_gap
        y = 0

        for _, row in columns[date].iterrows():
            height = row['TradingValue']
            color = colors.get(row['Ticker'], '#999999')

            rect = Rectangle((x, y), bar_width, height, color=color)
            ax.add_patch(rect)

            # Nhãn ticker
            if height > max_day_value * 0.05:
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Ticker'],
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white'
                )
            y += height

    # --- Vẽ dải nối (flow) ---
    for i in range(len(dates) - 1):
        left = columns[dates[i]]
        right = columns[dates[i + 1]]

        left_y = left['TradingValue'].cumsum() - left['TradingValue']
        right_y = right['TradingValue'].cumsum() - right['TradingValue']

        for ticker in tickers:
            l_row = left[left['Ticker'] == ticker]
            r_row = right[right['Ticker'] == ticker]

            if not l_row.empty and not r_row.empty:
                lx0 = i * x_gap + bar_width
                rx0 = (i + 1) * x_gap

                ly0 = left_y.iloc[l_row.index[0]]
                ly1 = ly0 + l_row['TradingValue'].values[0]
                ry0 = right_y.iloc[r_row.index[0]]
                ry1 = ry0 + r_row['TradingValue'].values[0]

                verts = [(lx0, ly0), (lx0, ly1), (rx0, ry1), (rx0, ry0)]

                poly = PolyCollection(
                    [verts],
                    facecolor=colors.get(ticker, '#999999'),
                    alpha=0.5,
                    edgecolor="none"
                )
                ax.add_collection(poly)

    # --- Trục & tiêu đề ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates])
    ax.set_yticks([])
    ax.set_title(top_title)

    # --- Save ---
    plt.savefig(chart_path, dpi=300)
    plt.close()


# Trading_value_table ( top 50 )
def plot_trading_value_table(
    combined_df,
    top_n=50,
    days=14,
    chart_path="chart3.2.png",
    scale_factor=0.001
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Chuẩn bị dữ liệu ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17']].rename(columns={
        'Ngày': "Date",
        'Mã': "Ticker",
        'Unnamed: 17': "TradingValue"
    })

    df = df[df['Date'].notna()]
    df['TradingValue'] = pd.to_numeric(df['TradingValue'], errors='coerce')
    df['TradingValue'] = df['TradingValue'] * scale_factor
    df = df.dropna(subset=['TradingValue'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(
        ascending=False, method='dense'
    )

    # --- Lấy top N ticker ở ngày mới nhất ---
    latest_date = df['Date'].max()
    top_tickers = df.loc[
        (df['Date'] == latest_date) &
        (df['Rank'] > 0) &
        (df['Rank'] <= top_n),
        'Ticker'
    ].unique()

    # --- Lọc N ngày gần nhất ---
    last_days = latest_date - pd.Timedelta(days=days - 1)
    df_last = df[
        (df['Date'] >= last_days) &
        (df['Ticker'].isin(top_tickers))
    ].copy()

    # --- Pivot ---
    pivot_df = df_last.pivot(
        index='Ticker',
        columns='Date',
        values='TradingValue'
    )

    pivot_df = pivot_df.sort_index()
    pivot_df = pivot_df.sort_values(by=latest_date, ascending=False)

    # --- Chuẩn bị màu gradient xanh ---
    values_only = pivot_df.fillna(0)
    norm = plt.Normalize(
        vmin=values_only.min().min(),
        vmax=values_only.max().max()
    )
    green_map = plt.cm.Greens
    colored_values = green_map(norm(values_only.values))

    # --- Ghép màu (Ticker trắng) ---
    white_col = np.ones((len(pivot_df), 1, 4))
    cell_colours = np.concatenate([white_col, colored_values], axis=1)

    # --- Data bảng ---
    table_data = np.column_stack([
        pivot_df.index,
        pivot_df.round(0).fillna("").astype(str).values
    ])

    col_labels = ["Ticker"] + [
        d.strftime('%m/%d') for d in pivot_df.columns
    ]

    # --- Vẽ bảng ---
    fig_width = max(10, len(pivot_df.columns) * 0.5)
    fig_height = max(2, len(pivot_df) * 0.23)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    tbl = plt.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colours,
        loc='center'
    )

    # --- Style ---
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#bd3651")
            cell.set_text_props(weight='bold', color='white')
        elif col == 0:
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(weight='bold', color='black')
        cell.set_linewidth(0.5)

    # --- Title ---
    plt.title(
        f"Trading Value Trend (Top {top_n} as of {latest_date.strftime('%Y-%m-%d')})",
        fontsize=13,
        fontweight='bold',
        pad=1
    )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()




# Distribution flow
def plot_distribution_flow_100pct(
    combined_df,
    days=14,
    chart_path="chart2.png",
    figsize=(16, 8)
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17']].copy()
    df = df.rename(columns={
        "Ngày": "Date",
        "Mã": "Ticker",
        "Unnamed: 17": "TradingValue"
    })

    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = (
        pd.to_numeric(df['TradingValue'], errors='coerce')
        .fillna(0)
        .astype(float)
    )

    # --- Rank theo ngày ---
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(
        ascending=False, method='dense'
    )

    # --- Lấy N ngày gần nhất ---
    last_days = df['Date'].max() - pd.Timedelta(days=days - 1)
    df_last = df[df['Date'] >= last_days].copy()

    # --- Điều chỉnh DistributionRate = 100% ---
    def adjust_to_100(group):
        total = group['TradingValue'].sum()
        if total == 0:
            group['DistributionRate'] = 0
            return group

        group['DistributionRate'] = group['TradingValue'] / total * 100
        diff = 100 - group['DistributionRate'].sum()
        if abs(diff) > 1e-6:
            max_idx = group['DistributionRate'].idxmax()
            group.loc[max_idx, 'DistributionRate'] += diff
        return group

    df_last = df_last.groupby('Date', group_keys=False).apply(adjust_to_100)

    # --- Màu sắc ---
    unique_tickers = df_last['Ticker'].unique()
    n_colors = max(300, len(unique_tickers))
    cmap = cm.get_cmap('nipy_spectral', n_colors)
    colors = {
        ticker: mcolors.to_hex(cmap(i / n_colors))
        for i, ticker in enumerate(unique_tickers)
    }

    # --- Chuẩn bị dữ liệu theo ngày ---
    df_sorted = df_last.sort_values(
        by=['Date', 'DistributionRate'],
        ascending=[True, True]
    )

    dates = sorted(df_last['Date'].unique())
    tickers = df_last['Ticker'].unique()

    columns = {}
    for date in dates:
        columns[date] = (
            df_sorted[df_sorted['Date'] == date]
            .copy()
            .reset_index(drop=True)
        )

    # --- Vẽ biểu đồ ---
    fig, ax = plt.subplots(figsize=figsize)
    x_gap = 2
    bar_width = 1
    max_day_value = 100
    y_padding = 5

    # --- Vẽ cột ---
    for i, date in enumerate(dates):
        x = i * x_gap
        y = 0.0

        for _, row in columns[date].iterrows():
            height = row['DistributionRate']
            color = colors.get(row['Ticker'], '#999999')

            rect = Rectangle(
                (x, y), bar_width, height,
                color=color, ec='none'
            )
            ax.add_patch(rect)

            if height >= 2:
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Ticker'],
                    ha='center', va='center',
                    fontsize=7, color='white'
                )
            y += height

    # --- Dải nối ---
    for i in range(len(dates) - 1):
        left = columns[dates[i]]
        right = columns[dates[i + 1]]

        left_y = left['DistributionRate'].cumsum() - left['DistributionRate']
        right_y = right['DistributionRate'].cumsum() - right['DistributionRate']

        for ticker in tickers:
            l_row = left[left['Ticker'] == ticker]
            r_row = right[right['Ticker'] == ticker]

            if not l_row.empty and not r_row.empty:
                lx0 = i * x_gap + bar_width
                rx0 = (i + 1) * x_gap

                ly0 = left_y.iloc[l_row.index[0]]
                ly1 = ly0 + l_row['DistributionRate'].values[0]
                ry0 = right_y.iloc[r_row.index[0]]
                ry1 = ry0 + r_row['DistributionRate'].values[0]

                verts = [(lx0, ly0), (lx0, ly1), (rx0, ry1), (rx0, ry0)]
                poly = PolyCollection(
                    [verts],
                    facecolor=colors.get(ticker, '#999999'),
                    alpha=0.35,
                    edgecolor="none"
                )
                ax.add_collection(poly)

    # --- Trục ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value + y_padding)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels(
        [d.strftime('%Y-%m-%d') for d in dates],
        rotation=45, ha='right'
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(
        'Ranking Flow — HOSE Trading Value (Each column = 100%)',
        fontsize=14, weight='bold'
    )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()




# Pecentage gap of Top 50
def plot_gap_table_top_tickers(
    combined_df,
    top_n=15,
    days=14,
    chart_path="chart3.png",
    figsize_scale_x=1.2,
    figsize_scale_y=0.45
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 11', 'Unnamed: 17']].copy()
    df = df.rename(columns={
        "Ngày": "Date",
        "Mã": "Ticker",
        "Unnamed: 11": "Gap",
        "Unnamed: 17": "TradingValue"
    })

    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = (
        pd.to_numeric(df['TradingValue'], errors='coerce')
        .fillna(0)
        .astype(float)
    )

    # --- Rank theo TradingValue ---
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(
        ascending=False, method='dense'
    )

    # --- Lấy N ngày gần nhất ---
    last_days = df['Date'].max() - pd.Timedelta(days=days - 1)
    df_last = df[df['Date'] >= last_days].copy()

    # --- Lấy top N ticker (dựa trên ngày mới nhất) ---
    top_tickers = (
        df[df['Date'] == df['Date'].max()]
        .nsmallest(top_n, 'Rank')['Ticker']
        .unique()
    )

    df_last = df_last[df_last['Ticker'].isin(top_tickers)]
    df_last = df_last[['Date', 'Ticker', 'Gap']]

    # --- Ensure Gap numeric ---
    df_last['Gap'] = pd.to_numeric(df_last['Gap'], errors='coerce')
    df_last = df_last.dropna(subset=['Gap'])

    # --- Pivot ---
    pivot_df = df_last.pivot(
        index='Ticker',
        columns='Date',
        values='Gap'
    ).sort_index()

    # --- Normalize colors ---
    norm = plt.Normalize(
        vmin=pivot_df.min().min(),
        vmax=pivot_df.max().max()
    )
    colors = plt.cm.RdYlGn(norm(pivot_df.fillna(0).values))

    # --- Table data ---
    table_data = pivot_df.round(2).fillna("")
    table_data.insert(0, "Ticker", table_data.index)

    # --- Color matrix (Ticker column white) ---
    ticker_col = np.ones((len(pivot_df), 1, 4))
    table_colors = np.concatenate([ticker_col, colors], axis=1)

    # --- Column labels ---
    col_labels = ["Ticker"] + [
        d.strftime('%m/%d') for d in pivot_df.columns
    ]

    # --- Plot table ---
    fig_width = max(10, len(col_labels) * figsize_scale_x)
    fig_height = max(2, len(pivot_df) * figsize_scale_y)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=col_labels,
        cellColours=table_colors,
        loc='center'
    )

    # --- Styling ---
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#bd3651")
            cell.set_text_props(weight='bold', color='white', ha='center')
        elif col == 0:
            cell.set_facecolor('#f9f9f9')
            cell.set_text_props(weight='bold', ha='left')

        cell.set_linewidth(0.5)

    # --- Column width ---
    for row in range(len(table_data) + 1):
        tbl[(row, 0)].set_width(0.12)
        for col in range(1, len(col_labels)):
            tbl[(row, col)].set_width(0.10)

    # --- Title ---
    plt.title(
        f"Gap Trend by Date (Top {top_n})",
        fontsize=13,
        fontweight='bold',
        pad=8
    )

    plt.tight_layout(pad=0.3)
    plt.savefig(chart_path, dpi=300)
    plt.close()




# Flow by major
def plot_major_tradingvalue_flow(
    combined_df,
    dfIndustryDim,
    days=14,
    chart_path="chart5.png",
    figsize=(16, 8)
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PolyCollection
    import matplotlib.cm as cm

    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17', 'Tham\n chiếu', 'Trung\n bình']].copy()
    df = df.rename(columns={
        "Ngày": "Date",
        "Mã": "Ticker",
        "Unnamed: 17": "TradingValue",
        "Tham\n chiếu": "Base",
        "Trung\n bình": "AVGPrice"
    })

    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = (
        pd.to_numeric(df['TradingValue'], errors='coerce')
        .fillna(0)
        .astype(float)
    )

    # --- Lấy N ngày gần nhất ---
    last_days = df['Date'].max() - pd.Timedelta(days=days - 1)
    df_last = df[df['Date'] >= last_days].copy()

    # --- Tính Profit (giữ lại nếu dùng sau này) ---
    df_last['Profit'] = (df_last['AVGPrice'] / df_last['Base']) - 1

    # --- Merge ngành ---
    df_last = df_last.merge(dfIndustryDim, how='left', on='Ticker')

    # --- Tổng TradingValue theo ngày & ngành ---
    df = (
        df_last
        .groupby(['Date', 'Major'])['TradingValue']
        .sum()
        .reset_index()
    )

    df['TradingValue'] = df['TradingValue'] / 1000  # đổi đơn vị

    # --- Rank trong từng ngày ---
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(
        ascending=False, method='dense'
    )

    # --- Chuẩn bị dữ liệu vẽ ---
    df_sorted = df.sort_values(
        by=['Date', 'TradingValue'],
        ascending=[True, True]
    )

    dates = sorted(df['Date'].unique())
    majors = sorted(df['Major'].unique())

    # --- Sinh màu ---
    cmap = cm.get_cmap('tab20', len(majors))
    colors = {major: cmap(i) for i, major in enumerate(majors)}

    # --- Gom dữ liệu theo ngày ---
    columns = {}
    for date in dates:
        columns[date] = (
            df_sorted[df_sorted['Date'] == date]
            .copy()
            .reset_index(drop=True)
        )

    # --- Vẽ biểu đồ ---
    fig, ax = plt.subplots(figsize=figsize)
    x_gap = 2
    bar_width = 1
    y_padding = 10

    max_day_value = df.groupby('Date')['TradingValue'].sum().max()

    # --- Vẽ cột ---
    for i, date in enumerate(dates):
        x = i * x_gap
        day_data = columns[date]
        total_day_value = day_data['TradingValue'].sum()

        y = (max_day_value - total_day_value) / 2

        for _, row in day_data.iterrows():
            height = row['TradingValue']
            color = colors.get(row['Major'], '#999999')

            rect = Rectangle(
                (x, y),
                bar_width,
                height,
                color=color,
                ec='none'
            )
            ax.add_patch(rect)

            if height >= max_day_value * 0.03:
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Major'],
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white'
                )

            y += height

    # --- Dải nối ---
    for i in range(len(dates) - 1):
        left = columns[dates[i]]
        right = columns[dates[i + 1]]

        left_y = (
            (max_day_value - left['TradingValue'].sum()) / 2
            + left['TradingValue'].cumsum()
            - left['TradingValue']
        )
        right_y = (
            (max_day_value - right['TradingValue'].sum()) / 2
            + right['TradingValue'].cumsum()
            - right['TradingValue']
        )

        for major in majors:
            l_row = left[left['Major'] == major]
            r_row = right[right['Major'] == major]

            if not l_row.empty and not r_row.empty:
                lx0 = i * x_gap + bar_width
                rx0 = (i + 1) * x_gap

                ly0 = left_y.iloc[l_row.index[0]]
                ly1 = ly0 + l_row['TradingValue'].values[0]
                ry0 = right_y.iloc[r_row.index[0]]
                ry1 = ry0 + r_row['TradingValue'].values[0]

                verts = [(lx0, ly0), (lx0, ly1), (rx0, ry1), (rx0, ry0)]

                poly = PolyCollection(
                    [verts],
                    facecolor=colors.get(major, '#999999'),
                    alpha=0.35,
                    edgecolor="none"
                )
                ax.add_collection(poly)

    # --- Trục & legend ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value + y_padding)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels(
        [d.strftime('%Y-%m-%d') for d in dates],
        rotation=45,
        ha='right'
    )

    ax.set_ylabel("Trading Value (Billion VND)", fontsize=10)
    ax.set_title(
        "Ranking Flow — HOSE Trading Value by Major (True Scale)",
        fontsize=14,
        weight='bold'
    )

    handles = [Rectangle((0, 0), 1, 1, color=colors[m]) for m in majors]
    ax.legend(
        handles,
        majors,
        title="Major",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()




#Profit by major
def build_profit_area_table(dfSharebyIndustyDaily, output_path="chart6.png"):
    # --- Copy dữ liệu gốc ---
    df = dfSharebyIndustyDaily.copy()
    df['AreaProfit'] = df['TradingValue'] * df['Profit']

    # --- Tổng hợp theo ngày & ngành ---
    df = df.groupby(['Date', 'Major'])['AreaProfit'].sum().reset_index()

    # --- Làm sạch ---
    df['AreaProfit'] = pd.to_numeric(df['AreaProfit'], errors='coerce')
    df = df.dropna(subset=['AreaProfit'])

    # --- Pivot ---
    pivot_df = df.pivot(index='Major', columns='Date', values='AreaProfit')

    # --- Sắp xếp ngày tăng dần ---
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    # --- Grand Total ---
    pivot_df['Grand Total'] = pivot_df.sum(axis=1)

    # --- Sort theo Grand Total ---
    pivot_df = pivot_df.sort_values(by='Grand Total', ascending=False)

    # --- Base: first non-zero ---
    def first_nonzero_row(series):
        for v in series:
            if pd.notna(v) and v != 0:
                return v
        return np.nan

    date_cols = [c for c in pivot_df.columns if c != 'Grand Total']
    base = pivot_df[date_cols].apply(first_nonzero_row, axis=1)

    # --- % thay đổi ---
    pivot_pct = pd.DataFrame(index=pivot_df.index, columns=date_cols, dtype=float)
    for c in date_cols:
        pivot_pct[c] = (pivot_df[c] - base) / base.abs() * 100
    pivot_pct[base.isna()] = np.nan

    grand_pct = (pivot_df['Grand Total'] - base) / base.abs() * 100
    grand_pct[base.isna()] = np.nan

    # --- Format text ---
    def fmt_cell(val, pct):
        if pd.isna(val):
            return ""
        val_text = f"{val:,.0f}"
        if pd.isna(pct):
            return val_text
        sign = "+" if pct >= 0 else ""
        return f"{val_text}\n{sign}{pct:.1f}%"

    display_data = pivot_df.copy().astype(object)
    for c in date_cols:
        display_data[c] = [
            fmt_cell(pivot_df.iloc[i][c], pivot_pct.iloc[i][pivot_pct.columns.get_loc(c)])
            for i in range(len(pivot_df))
        ]

    display_data['Grand Total'] = [
        fmt_cell(pivot_df.iloc[i]['Grand Total'], grand_pct.iloc[i])
        for i in range(len(pivot_df))
    ]

    # --- Màu ô ---
    num_rows = pivot_df.shape[0]
    cell_colours = []

    for i in range(num_rows):
        row_colors = []
        for c in date_cols:
            val = pivot_df.iloc[i][c]
            if pd.isna(val):
                row_colors.append("white")
            elif val > 0:
                row_colors.append("#9ffa9f")
            elif val < 0:
                row_colors.append("#fdc4c4")
            else:
                row_colors.append("white")

        gt_val = pivot_df.iloc[i]['Grand Total']
        if pd.isna(gt_val):
            row_colors.append("white")
        elif gt_val > 0:
            row_colors.append("#9ffa9f")
        elif gt_val < 0:
            row_colors.append("#fdc4c4")
        else:
            row_colors.append("white")

        cell_colours.append(row_colors)

    # --- Table data ---
    table_data = display_data.fillna("").copy()
    table_data.insert(0, 'Major', table_data.index)

    white_col = np.array([["white"] for _ in range(num_rows)])
    cell_colours = np.hstack([white_col, np.array(cell_colours, dtype=object)])

    # --- Column labels ---
    col_labels = ["Major"] + [
        d.strftime('%m/%d') if isinstance(d, (pd.Timestamp, datetime)) else str(d)
        for d in pivot_df.columns
    ]

    # --- Vẽ bảng ---
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.15, max(4, len(table_data) * 0.5)))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_data.values,
        colLabels=col_labels,
        cellColours=cell_colours,
        loc='center',
        cellLoc='center'
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 2)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#B83C50")
            cell.set_text_props(weight='bold', color='white')
        elif c == 0:
            cell.set_text_props(weight='bold', ha='left', fontsize=9)
        else:
            cell.set_text_props(va='center', ha='center', fontsize=8)

    # --- Column width ---
    n_cols = len(col_labels)
    for r in range(len(table_data) + 1):
        tbl[(r, 0)].set_width(0.30)
    for c in range(1, n_cols):
        for r in range(len(table_data) + 1):
            if (r, c) in tbl.get_celld():
                tbl[(r, c)].set_width(0.12)

    plt.title("Profit Area by Major", fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.3)

    plt.savefig(output_path, dpi=300)
    plt.close()


def build_dfSharebyIndustyDaily(combined_df, dfIndustryDim):
    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17', 'Tham\n chiếu', 'Trung\n bình']].copy()
    df = df.rename(columns={
        "Ngày": "Date",
        "Mã": "Ticker",
        "Unnamed: 17": "TradingValue",
        "Tham\n chiếu": "Base",
        "Trung\n bình": "AVGPrice"
    })

    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = pd.to_numeric(df['TradingValue'], errors='coerce').fillna(0)

    # --- 14 ngày gần nhất ---
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df = df[df['Date'] >= last_14_days].copy()

    # --- Profit ---
    df['Profit'] = (df['AVGPrice'] / df['Base']) - 1

    # --- Merge ngành ---
    df = df.merge(dfIndustryDim, how='left', on='Ticker')

    # 🔒 ENSURE dfSharebyIndustyDaily luôn tồn tại
    if df.empty:
        return pd.DataFrame(
            columns=['Date', 'Ticker', 'TradingValue', 'Base', 'AVGPrice', 'Profit', 'Major']
        )

    return df

def plot_price_status_stacked_14d(
    df,
    date_col='Ngày',
    ticker_col='Mã',
    base_col='Tham\n chiếu',
    close_col='Đóng\n cửa',
    days=14,
    figsize=(14, 6),
    save_path="price_status_14d.png",
    label_threshold=5
):
    import pandas as pd
    import matplotlib.pyplot as plt

    # =========================
    # 1. Clean data
    # =========================
    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data[base_col] = pd.to_numeric(data[base_col], errors='coerce')
    data[close_col] = pd.to_numeric(data[close_col], errors='coerce')

    data = data.dropna(subset=[date_col, ticker_col, base_col, close_col])

    # =========================
    # 2. Last N trading days (ignore empty dates)
    # =========================
    trading_dates = (
        data[date_col]
        .drop_duplicates()
        .sort_values()
        .tail(days)
    )

    data = data[data[date_col].isin(trading_dates)]

    # =========================
    # 3. Ceiling / Floor ±7%
    # =========================
    data['Ceiling'] = data[base_col] * 1.068
    data['Floor']   = data[base_col] * 0.922

    # =========================
    # 4. Price classification
    # =========================
    def classify(row):
        if row[close_col] >= row['Ceiling']:
            return 'Ceiling'
        elif row[close_col] <= row['Floor']:
            return 'Floor'
        elif row[close_col] > row[base_col]:
            return 'Increase'
        elif row[close_col] < row[base_col]:
            return 'Decrease'
        else:
            return 'Stable'

    data['Status'] = data.apply(classify, axis=1)

    # =========================
    # 5. Count unique tickers
    # =========================
    count_df = (
        data
        .groupby([date_col, 'Status'])[ticker_col]
        .nunique()
        .unstack(fill_value=0)
        .loc[trading_dates]
    )

    # =========================
    # 6. VNINDEX colors & order
    # =========================
    status_order = ['Ceiling', 'Increase', 'Stable', 'Decrease', 'Floor']
    colors = {
        'Ceiling':  '#8e44ad',  # Trần
        'Increase': '#00b050',  # Tăng
        'Stable':   '#ffd966',  # Tham chiếu
        'Decrease': '#e74c3c',  # Giảm
        'Floor':    '#1f4fd8'   # Sàn
    }

    count_df = count_df.reindex(columns=status_order, fill_value=0)

    # =========================
    # 7. Plot stacked bars
    # =========================
    dates = count_df.index
    x = range(len(dates))
    bar_width = 0.9

    fig, ax = plt.subplots(figsize=figsize)
    bottom = [0] * len(dates)

    for status in status_order:
        values = count_df[status].values

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=bar_width,
            color=colors[status],
            label=status,
            linewidth=0
        )

        for i, val in enumerate(values):
            if val >= label_threshold:
                ax.text(
                    i,
                    bottom[i] + val / 2,
                    f"{int(val)}",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black' if status == 'Stable' else 'white'
                )

        bottom = [bottom[i] + values[i] for i in range(len(values))]

    # =========================
    # 8. LEFT-aligned title
    # =========================
    ax.text(
        -0.03, 1.06,
        f"Price Status Distribution — Last {len(dates)} Trading Days",
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        ha='left',
        va='bottom'
    )


    # =========================
    # 9. LEFT-aligned legend (above chart)
    # =========================
    fig.legend(
        ncol=5,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.98),
        frameon=False
    )

    # =========================
    # 10. Axes formatting
    # =========================
    ax.set_ylabel("Number of tickers")
    ax.set_xlabel("Date")

    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.strftime('%Y-%m-%d') for d in dates],
        rotation=45,
        ha='right'
    )

    ax.set_ylim(0, count_df.sum(axis=1).max() * 1.05)

    # =========================
    # 11. Layout & output
    # =========================
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()





# --- Main ---
if __name__ == "__main__":
    
    # --- Đường dẫn tương đối trong repo ---
    file_pathx = r"DOANH NGHIEP TIEM NANG.xlsx"
    
    # --- Đọc sheet 'Tổng quan' ---
    df = pd.read_excel(file_pathx, sheet_name='Tổng quan')
    
    # --- Lọc cột và đổi tên ---
    df = df[['Mã','Ngành']]
    df.columns = ["Ticker","Major"]
    
    # --- Lưu vào biến dfIndustryDim ---
    dfIndustryDim = df
    
    FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

    # Load toàn bộ file Excel
    combined_df = load_all_excels(FOLDER_ID)

    # Draw
    df_last_14 = process_trading_data(combined_df)
    plot_ranking_flow_chart(df_last_14) #chart1.png
    plot_distribution_flow_100pct(combined_df) #chart2.png
    plot_trading_value_table(combined_df) #chart3.2.png
    plot_gap_table_top_tickers(combined_df)  #chart3.png
    plot_major_tradingvalue_flow(combined_df, dfIndustryDim) #chart4.png
    dfSharebyIndustyDaily = build_dfSharebyIndustyDaily(combined_df, dfIndustryDim)
    build_profit_area_table(dfSharebyIndustyDaily)#"chart6.png"
    plot_price_status_stacked_14d(  combined_df) #"chart3.0.png"


    from PIL import Image
    # --- Danh sách file ảnh ---
    chart_files = ["chart1.png", "chart2.png","price_status_14d.png", "chart3.png", "chart3.2.png", "chart5.png", "chart6.png"]
    # --- Load ảnh ---
    images = [Image.open(f) for f in chart_files]
    # --- Xác định chiều rộng target (page width) ---
    target_width = max(img.width for img in images)
    # --- Resize ảnh để vừa với chiều rộng target, giữ tỉ lệ ---
    processed_imgs = []
    for img in images:
        if img.width != target_width:
            # Tỉ lệ scale
            ratio = target_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((target_width, new_height), Image.LANCZOS)
        processed_imgs.append(img)
    # --- Tính kích thước tổng ---
    total_height = sum(img.height for img in processed_imgs)
    # --- Tạo canvas mới ---
    merged_img = Image.new("RGB", (target_width, total_height), (255, 255, 255))
    # --- Ghép ảnh ---
    y_offset = 0
    for img in processed_imgs:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height
    # --- Lưu kết quả ---
    merged_img_path = "summary_charts.png"   # ✅ định nghĩa biến trước
    merged_img.save(merged_img_path)
    print(f"✅ Ảnh tổng hợp đã được lưu: {merged_img_path}")    
    
    # --- Gửi mail kèm ảnh ---
    report = "Báo cáo Top 15 cổ phiếu theo giá trị giao dịch trong 14 ngày gần nhất."
    send_email_report(report, attachment=merged_img_path)




