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

# --- Main ---
if __name__ == "__main__":
    
    # --- Đường dẫn tương đối trong repo ---
    file_pathx = "DOANH_NGHIEP_TIEM_NANG.xlsx"
    
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

    df = combined_df[['Ngày','Mã','Unnamed: 17']]
    df = df.rename(columns={
        df.columns[0]: "Date",
        df.columns[1]: "Ticker",
        df.columns[2]: "TradingValue"
    })
    
    df = df[df['Date'].notna()]
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(ascending=False, method='dense')
    df['TradingValue'] = df['TradingValue'].astype(int)
    df['DistributionRate'] = df['TradingValue'] / df.groupby('Date')['TradingValue'].transform('sum')
    df['DistributionRate'] = (df['DistributionRate'] * 100).round(1).astype(str) + '%'
    
    df = df[df['Rank'] <= 15]
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days]
    
    # --- Vẽ biểu đồ ---
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PolyCollection
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    unique_tickers = df_last_14['Ticker'].unique()
    color_map = cm.get_cmap('tab20', len(unique_tickers))
    colors = {ticker: mcolors.to_hex(color_map(i)) for i, ticker in enumerate(unique_tickers)}
    
    df_sorted = df_last_14.sort_values(by=['Date', 'TradingValue'], ascending=[True, True])
    dates = sorted(df_last_14['Date'].unique())
    tickers = df_last_14['Ticker'].unique()
    
    columns = {}
    for date in dates:
        day_data = df_sorted[df_sorted['Date'] == date].copy()
        columns[date] = day_data.reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x_gap = 2
    bar_width = 1
    
    max_day_value = df_last_14.groupby('Date')['TradingValue'].sum().max()
    
    # --- Vẽ từng cột (stacked bar chart) + nhãn trong bar ---
    for i, date in enumerate(dates):
        x = i * x_gap
        y = 0
        for _, row in columns[date].iterrows():
            height = row['TradingValue']
            color = colors.get(row['Ticker'], '#999999')
            rect = Rectangle((x, y), bar_width, height, color=color)
            ax.add_patch(rect)
    
            # Thêm nhãn ticker vào giữa mỗi phần bar (nằm ngang)
            if height > max_day_value * 0.05:  # chỉ hiển thị nếu đủ lớn
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Ticker'],
                    ha='center', va='center',
                    fontsize=7, color='white'
                )
    
            y += height
    
    # --- Vẽ dải nối ---
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
                verts = [(lx0, ly0), (lx0, ly1),
                            (rx0, ry1), (rx0, ry0)]
                poly = PolyCollection([verts], 
                                        facecolor=colors.get(ticker, '#999999'), 
                                        alpha=0.5, edgecolor="none")
                ax.add_collection(poly)
    
    # --- Cài đặt trục ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates])
    ax.set_yticks([])
    ax.set_title('Ranking Flow — HOSE Trading Value (Top 15)')
    
    # plt.tight_layout()
    
    # --- Save chart ra file ---
    chart_path = "chart1.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
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
    df['TradingValue'] =df['TradingValue']*0.001
    df = df.dropna(subset=['TradingValue'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(ascending=False, method='dense')
    
    # --- Lọc Top 21–40 ngày gần nhất ---
    latest_date = df['Date'].max()
    top_tickers = df.loc[
        (df['Date'] == latest_date) & (df['Rank'] > 0) & (df['Rank'] <= 50),
        'Ticker'
    ].unique()
    
    # --- Giữ 14 ngày gần nhất ---
    last_14_days = latest_date - pd.Timedelta(days=13)
    df_last_14 = df[(df['Date'] >= last_14_days) & (df['Ticker'].isin(top_tickers))].copy()
    
    # --- Pivot dữ liệu ---
    pivot_df = df_last_14.pivot(index='Ticker', columns='Date', values='TradingValue')
    pivot_df = pivot_df.sort_index()
    
    # --- Sắp xếp theo ngày mới nhất ---
    pivot_df = pivot_df.sort_values(by=latest_date, ascending=False)
    
    # --- Chuẩn bị dữ liệu màu: gradient chỉ xanh ---
    values_only = pivot_df.fillna(0)
    norm = plt.Normalize(vmin=values_only.min().min(), vmax=values_only.max().max())
    green_map = plt.cm.Greens
    colored_values = green_map(norm(values_only.values))
    
    # --- Ghép màu (Ticker trắng, các ngày gradient) ---
    white_col = np.ones((len(pivot_df), 1, 4))
    cell_colours = np.concatenate([white_col, colored_values], axis=1)
    
    # --- Chuẩn bị dữ liệu bảng ---
    table_data = np.column_stack([
        pivot_df.index,
        pivot_df.round(0).fillna("").astype(str).values
    ])
    
    # --- Header ---
    col_labels = ["Ticker"] + [d.strftime('%m/%d') for d in pivot_df.columns]
    
    # --- Vẽ bảng ---
    fig_width = max(10, len(pivot_df.columns) * 0.5)  # giảm bề ngang mỗi cột
    fig_height = max(2, len(pivot_df) * 0.23)        # giảm chiều cao mỗi hàng
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
    tbl.set_fontsize(7)       # font nhỏ hơn
    tbl.scale(1, 1)   
    
    # --- Header & cột Ticker ---
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#bd3651")  # header xanh lá pastel
            cell.set_text_props(weight='bold', color='white')
        elif col == 0:
            cell.set_facecolor('#F5F5F5')  # cột Ticker trắng xám nhẹ
            cell.set_text_props(weight='bold', color='black')
        # Optional: thêm viền mỏng
        cell.set_linewidth(0.5)
    
    # --- Tiêu đề ---
    plt.title(f"Trading Value Trend (Top 50 as of {latest_date.strftime('%Y-%m-%d')})",
              fontsize=13, fontweight='bold', pad=1)
    
    plt.tight_layout()
    # plt.show()
    
    # --- Save chart ra file ---
    chart_path = "chart3.2.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17']].copy()
    df = df.rename(columns={"Ngày": "Date", "Mã": "Ticker", "Unnamed: 17": "TradingValue"})
    
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = pd.to_numeric(df['TradingValue'], errors='coerce').fillna(0).astype(float)
    
    # Xếp hạng trong từng ngày
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(ascending=False, method='dense')
    
    # Lấy 14 ngày gần nhất
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days].copy()
    
    # --- Tính tỉ lệ % và hiệu chỉnh tổng = 100 ---
    def adjust_to_100(group):
        if len(group) == 0:
            return group
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
    
    df_last_14 = df_last_14.groupby('Date', group_keys=False).apply(adjust_to_100)
    
    # --- Màu sắc (~300 màu khác nhau) ---
    unique_tickers = df_last_14['Ticker'].unique()
    n_colors = max(300, len(unique_tickers))
    cmap = cm.get_cmap('nipy_spectral', n_colors)
    colors = {ticker: mcolors.to_hex(cmap(i / n_colors)) for i, ticker in enumerate(unique_tickers)}
    
    # --- Chuẩn bị dữ liệu cho từng ngày (từ nhỏ → lớn) ---
    df_sorted = df_last_14.sort_values(by=['Date', 'DistributionRate'], ascending=[True, True])
    dates = sorted(df_last_14['Date'].unique())
    tickers = df_last_14['Ticker'].unique()
    
    columns = {}
    for date in dates:
        day_data = df_sorted[df_sorted['Date'] == date].copy().reset_index(drop=True)
        columns[date] = day_data
    
    # --- Vẽ biểu đồ ---
    fig, ax = plt.subplots(figsize=(16, 8))
    x_gap = 2
    bar_width = 1
    max_day_value = 100
    y_padding = 5
    
    # --- Vẽ cột (từ nhỏ đến lớn) ---
    for i, date in enumerate(dates):
        x = i * x_gap
        y = 0.0
        for _, row in columns[date].iterrows():
            height = row['DistributionRate']
            color = colors.get(row['Ticker'], '#999999')
            rect = Rectangle((x, y), bar_width, height, color=color, ec='none')
            ax.add_patch(rect)
    
            # Hiển thị mã cổ phiếu nếu chiếm đủ không gian
            if height >= 2:
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Ticker'],
                    ha='center', va='center',
                    fontsize=7, color='white'
                )
            y += height
    
    # --- Dải nối giữa các ngày ---
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
    
    # --- Trục & layout ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value + y_padding)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Ranking Flow — HOSE Trading Value (Each column = 100%, sorted small→large)', fontsize=14, weight='bold')
    
    plt.tight_layout()
    # plt.show()
    
    # --- Save chart ra file ---
    chart_path = "chart2.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 11','Unnamed: 17']].copy()
    df = df.rename(columns={"Ngày": "Date", "Mã": "Ticker", "Unnamed: 11": "Gap","Unnamed: 17": "TradingValue"})
    
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = pd.to_numeric(df['TradingValue'], errors='coerce').fillna(0).astype(float)
    
    # Xếp hạng trong từng ngày
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(ascending=False, method='dense')
    
    # Lấy 14 ngày gần nhất
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days].copy()
    
    dfx = df.iloc[:15,:]
    df_last_14 = df_last_14.merge(dfx[['Ticker', 'Rank']], how='left', on='Ticker')
    df_last_14 = df_last_14[df_last_14['Rank_y'].notna()]
    df_last_14 = df_last_14[['Date'	,'Ticker',	'Gap']]
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # --- Ensure Gap is numeric ---
    df_last_14['Gap'] = pd.to_numeric(df_last_14['Gap'], errors='coerce')
    df_last_14 = df_last_14.dropna(subset=['Gap'])
    
    # --- Pivot: rows = Ticker, columns = Date, values = Gap ---
    pivot_df = df_last_14.pivot(index='Ticker', columns='Date', values='Gap')
    pivot_df = pivot_df.sort_index()
    
    # --- Normalize colors ---
    norm = plt.Normalize(vmin=pivot_df.min().min(), vmax=pivot_df.max().max())
    colors = plt.cm.RdYlGn(norm(pivot_df.fillna(0).values))
    
    # --- Add ticker column into table data ---
    table_data = pivot_df.round(2).fillna("")
    table_data.insert(0, "Ticker", table_data.index)
    
    # --- Create color matrix (add neutral white column for Ticker) ---
    ticker_col = np.ones((len(pivot_df), 1, 4))  # RGBA white
    table_colors = np.concatenate([ticker_col, colors], axis=1)
    
    # --- Column labels ---
    col_labels = ["Ticker"] + [d.strftime('%m/%d') for d in pivot_df.columns]
    
    # --- Create table ---
    fig, ax = plt.subplots(figsize=(len(col_labels)*1.2, len(pivot_df)*0.45))
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
    
    # --- Header (pink background, bold text) ---
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#bd3651")
            cell.set_text_props(weight='bold', color='white', ha='center', va='center')
    
        elif col == 0:
            cell.set_text_props(weight='bold', ha='left', va='center')
            cell.set_facecolor('#f9f9f9')
    
    # --- Adjust column widths ---
    for row in range(len(table_data) + 1):
        tbl[(row, 0)].set_width(0.10)  # ticker col wider
        for col in range(1, len(col_labels)):
            tbl[(row, col)].set_width(0.10)
    
    # --- Title ---
    plt.title("Gap Trend by Date (Top 15)", fontsize=13, fontweight='bold', pad=8)
    plt.subplots_adjust(top=0.9, bottom=0.02, left=0.05, right=0.98)
    plt.tight_layout(pad=0.3)
    # plt.show()
    
    # --- Save chart ra file ---
    chart_path = "chart3.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    # --- Chuẩn bị dữ liệu gốc ---
    df = combined_df[['Ngày', 'Mã', 'Unnamed: 17', 'Tham\n chiếu','Trung\n bình']].copy()
    df = df.rename(columns={"Ngày": "Date", "Mã": "Ticker", "Unnamed: 17": "TradingValue","Tham\n chiếu": "Base","Trung\n bình": "AVGPrice"})
    
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['TradingValue'] = pd.to_numeric(df['TradingValue'], errors='coerce').fillna(0).astype(float)
    
    # Lấy 14 ngày gần nhất
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days].copy()
    
    df_last_14['Profit'] = ((df_last_14['AVGPrice']/df_last_14['Base'])-1)
    
    df_last_14= df_last_14.merge(dfIndustryDim, how='left', on='Ticker') 
    
    dfSharebyIndustyDaily = df_last_14
    df = df_last_14
    # --- Tính tổng theo ngày & ngành ---
    df = df.groupby(['Date', 'Major'])['TradingValue'].sum().reset_index()
    df['TradingValue'] = df['TradingValue'] / 1000  # Đổi đơn vị: triệu → tỷ (tùy dữ liệu)
    
    # --- Xếp hạng trong từng ngày ---
    df = df.sort_values(by=['Date', 'TradingValue'], ascending=[False, False])
    df['Rank'] = df.groupby('Date')['TradingValue'].rank(ascending=False, method='dense')
    
    df_last_14 = df.copy()
    
    # --- Chuẩn bị dữ liệu để vẽ ---
    df_sorted = df_last_14.sort_values(by=['Date', 'TradingValue'], ascending=[True, True])
    dates = sorted(df_last_14['Date'].unique())
    tickers = df_last_14['Major'].unique()
    
    # --- Sinh màu tự động ---
    majors = sorted(df_last_14['Major'].unique())
    num_majors = len(majors)
    cmap = cm.get_cmap('tab20', num_majors)
    colors = {major: cmap(i) for i, major in enumerate(majors)}
    
    # --- Chuẩn bị dữ liệu theo ngày ---
    columns = {}
    for date in dates:
        day_data = df_sorted[df_sorted['Date'] == date].copy().reset_index(drop=True)
        columns[date] = day_data
    
    # --- Vẽ biểu đồ ---
    fig, ax = plt.subplots(figsize=(16, 8))
    x_gap = 2
    bar_width = 1
    y_padding = 10
    
    # Xác định giá trị lớn nhất trong toàn bộ để set trục Y
    max_day_value = df_last_14.groupby('Date')['TradingValue'].sum().max()
    
    # --- Vẽ các cột ---
    for i, date in enumerate(dates):
        x = i * x_gap
        day_data = columns[date]
        total_day_value = day_data['TradingValue'].sum()
    
        # Căn giữa theo trục dọc
        y = (max_day_value - total_day_value) / 2
    
        for _, row in day_data.iterrows():
            height = row['TradingValue']
            color = colors.get(row['Major'], '#999999')
            rect = Rectangle((x, y), bar_width, height, color=color, ec='none')
            ax.add_patch(rect)
    
            # Hiển thị tên ngành nếu thanh đủ cao
            if height >= max_day_value * 0.03:  # >=3% tổng ngày
                ax.text(
                    x + bar_width / 2,
                    y + height / 2,
                    row['Major'],
                    ha='center', va='center',
                    fontsize=7, color='white'
                )
            y += height
    
    # --- Dải nối giữa các ngày ---
    for i in range(len(dates) - 1):
        left = columns[dates[i]]
        right = columns[dates[i + 1]]
    
        left_y = (max_day_value - left['TradingValue'].sum()) / 2 + left['TradingValue'].cumsum() - left['TradingValue']
        right_y = (max_day_value - right['TradingValue'].sum()) / 2 + right['TradingValue'].cumsum() - right['TradingValue']
    
        for ticker in tickers:
            l_row = left[left['Major'] == ticker]
            r_row = right[right['Major'] == ticker]
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
                    alpha=0.35,
                    edgecolor="none"
                )
                ax.add_collection(poly)
    
    # --- Cấu hình trục ---
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value + y_padding)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
    
    ax.set_ylabel("Trading Value (Billion VND)", fontsize=10)
    ax.set_title("Ranking Flow — HOSE Trading Value by Major (True Scale)", fontsize=14, weight='bold')
    
    # --- Legend tự động ---
    handles = [Rectangle((0, 0), 1, 1, color=colors[m]) for m in majors]
    ax.legend(handles, majors, title="Major", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    # plt.show()
    # --- Save chart ra file ---
    chart_path = "chart5.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # --- Giả định dfSharebyIndustyDaily đã tồn tại ---
    df = dfSharebyIndustyDaily.copy()
    df['AreaProfit'] = df['TradingValue'] * df['Profit']
    
    # --- Tổng hợp theo ngày & ngành ---
    df = df.groupby(['Date', 'Major'])['AreaProfit'].sum().reset_index()
    
    # --- Giữ lại dữ liệu 14 ngày gần nhất ---
    df_last_14 = df.copy()
    df_last_14['AreaProfit'] = pd.to_numeric(df_last_14['AreaProfit'], errors='coerce')
    df_last_14 = df_last_14.dropna(subset=['AreaProfit'])
    
    # --- Pivot: rows = Major, columns = Date ---
    pivot_df = df_last_14.pivot(index='Major', columns='Date', values='AreaProfit')
    
    # --- Sắp xếp cột ngày tăng dần để "first" thực sự là ngày đầu chu kỳ ---
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    # --- Xác định ngày đầu và mới nhất ---
    first_date = pivot_df.columns.min()
    latest_date = pivot_df.columns.max()
    
    # --- Thêm Grand Total (theo hàng) ---
    pivot_df['Grand Total'] = pivot_df.sum(axis=1)
    
    # --- Sắp xếp theo Grand Total giảm dần ---
    pivot_df = pivot_df.sort_values(by='Grand Total', ascending=False)
    
    # --- Tìm base (first non-zero) cho từng hàng ---
    # base_series[i] = giá trị đầu tiên khác 0 (theo thứ tự cột ngày) hoặc NaN nếu không có
    def first_nonzero_row(series):
        for v in series:
            if pd.notna(v) and v != 0:
                return v
        return np.nan
    
    # apply on only the date columns (exclude 'Grand Total')
    date_cols = [c for c in pivot_df.columns if c != 'Grand Total']
    base = pivot_df[date_cols].apply(first_nonzero_row, axis=1)
    
    # --- Tính % thay đổi cho từng ô so với base hàng đó ---
    pivot_pct = pd.DataFrame(index=pivot_df.index, columns=date_cols, dtype=float)
    for c in date_cols:
        pivot_pct[c] = (pivot_df[c] - base) / base.abs() * 100
    
    # Nếu base là NaN -> đặt NaN cho toàn hàng
    pivot_pct[base.isna()] = np.nan
    
    # --- Tính % thay đổi Grand Total so với base của chính hàng ---
    grand_pct = (pivot_df['Grand Total'] - base) / base.abs() * 100
    grand_pct[base.isna()] = np.nan
    
    # --- Tạo text hiển thị (2 dòng trong 1 ô), format gọn & có dấu + / - ---
    display_data = pivot_df.copy().astype(object)
    
    # helper to format value + percent
    def fmt_cell(val, pct):
        if pd.isna(val):
            return ""
        val_text = f"{val:,.0f}"
        if pd.isna(pct):
            return val_text
        sign = "+" if pct >= 0 else ""
        pct_text = f"{sign}{pct:.1f}%"
        return f"{val_text}\n{pct_text}"
    
    # fill display_data for date cols
    for c in date_cols:
        display_data[c] = [
            fmt_cell(pivot_df.iloc[i][c], pivot_pct.iloc[i][pivot_pct.columns.get_loc(c)])
            for i in range(len(pivot_df))
        ]
    
    # Grand Total cell: compare to base of same row
    display_data['Grand Total'] = [
        fmt_cell(pivot_df.iloc[i]['Grand Total'], grand_pct.iloc[i]) for i in range(len(pivot_df))
    ]
    
    # --- Màu ô giá trị (chỉ căn cứ trên giá trị số nguyên, không căn theo % để tránh nhầm) ---
    num_rows = pivot_df.shape[0]
    num_date_cols = len(date_cols) + 1  # date cols + Grand Total
    cell_colours = []
    
    for i in range(num_rows):
        row_colors = []
        # Major column placeholder (we'll add later)
        # Now color for each date col
        for c in date_cols:
            val = pivot_df.iloc[i][c]
            if pd.isna(val):
                row_colors.append("white")
            elif val > 0:
                row_colors.append("#9ffa9f")  # xanh nhạt
            elif val < 0:
                row_colors.append("#fdc4c4")  # đỏ nhạt
            else:
                row_colors.append("white")
        # Grand Total color
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
    
    # --- Chèn cột Major ở đầu và chuẩn hoá cell_colours để match table_data ---
    table_data = display_data.fillna("").copy()
    table_data.insert(0, 'Major', table_data.index)
    
    # now prepend white for Major column in colours
    white_col = np.array([["white"] for _ in range(num_rows)])
    cell_colours = np.hstack([white_col, np.array(cell_colours, dtype=object)])
    
    # --- Tên cột (mm/dd) ---
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
    
    # --- Style ---
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    
    # scale to adjust cell height; chỉnh y > 1 để tăng chiều cao
    tbl.scale(1.2, 2)
    
    # Header màu hồng
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#B83C50")
            cell.set_text_props(weight='bold', color='white', ha='center', va='center')
        elif c == 0:
            cell.set_text_props(weight='bold', ha='left', va='center', fontsize=9)
        else:
            # cho các ô 2 dòng, căn giữa theo vertical
            cell.set_text_props(va='center', ha='center', fontsize=8)
    
    # --- Căn độ rộng cột ---
    n_cols = len(col_labels)
    tbl.auto_set_column_width([0])
    for r in range(len(table_data) + 1):
        tbl[(r, 0)].set_width(0.30)  # cột Major rộng hơn
    for c in range(1, n_cols):
        for r in range(len(table_data) + 1):
            if (r, c) in tbl.get_celld():
                tbl[(r, c)].set_width(0.12)
    
    # --- Tiêu đề ---
    plt.title("Profit Area by Major", fontsize=13, fontweight='bold', pad=8)
    
    # --- Giảm khoảng trắng & show ---
    plt.subplots_adjust(top=0.90, bottom=0.03, left=0.05, right=0.97)
    plt.tight_layout(pad=0.3)
    # plt.show()
    
    # --- Save chart ra file ---
    chart_path = "chart6.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    from PIL import Image
    
    # --- Danh sách file ảnh ---
    chart_files = ["chart1.png", "chart2.png", "chart3.png", "chart3.2.png", "chart5.png", "chart6.png"]
    
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
    merged_img.save("summary_charts.png")
    print("✅ Ảnh tổng hợp đã được lưu: summary_charts.png")

    
    # --- Gửi mail kèm ảnh ---
    report = "Báo cáo Top 15 cổ phiếu theo giá trị giao dịch trong 14 ngày gần nhất."
    send_email_report(report, attachment=merged_img_path)


