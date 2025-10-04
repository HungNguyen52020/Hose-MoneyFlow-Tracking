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
    FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

    # Load toàn bộ file Excel
    combined_df = load_all_excels(FOLDER_ID)

    # --- Logic tính toán ---
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

    df = df[df['Rank']<=15]
    last_14_days = df['Date'].max() - pd.Timedelta(days=13)
    df_last_14 = df[df['Date'] >= last_14_days]

    # --- Vẽ biểu đồ ---
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
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

    for i, date in enumerate(dates):
        x = i * x_gap
        y = 0
        for _, row in columns[date].iterrows():
            height = row['TradingValue']
            color = colors.get(row['Ticker'], '#999999')
            rect = Rectangle((x, y), bar_width, height, color=color)
            ax.add_patch(rect)
            y += height

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

    max_day_value = df_last_14.groupby('Date')['TradingValue'].sum().max()
    ax.set_xlim(-1, len(dates) * x_gap)
    ax.set_ylim(0, max_day_value)
    ax.set_xticks([i * x_gap + bar_width / 2 for i in range(len(dates))])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates])
    ax.set_yticks([])
    ax.set_title('Biểu đồ Sankey dạng cột (Top 15 cổ phiếu theo giá trị giao dịch)')

    legend_handles = [Patch(color=colors[t], label=t) for t in tickers]
    ax.legend(handles=legend_handles, title='Mã cổ phiếu')

    plt.tight_layout()

    # --- Save chart ra file ---
    chart_path = "chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # --- Gửi mail kèm ảnh ---
    report = "Báo cáo Top 15 cổ phiếu theo giá trị giao dịch trong 14 ngày gần nhất."
    send_email_report(report, attachment=chart_path)

