import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

ALERT_PATH = Path("outputs/latest/alert.txt")

if not ALERT_PATH.exists():
    print("No alert file, no email sent.")
    raise SystemExit(0)

message_text = ALERT_PATH.read_text(encoding="utf-8").strip()

if not message_text:
    print("Empty alert file, no email sent.")
    raise SystemExit(0)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

if not all([EMAIL_USER, EMAIL_PASS, EMAIL_TO]):
    print("Missing email credentials")
    raise SystemExit(1)

msg = MIMEText(message_text)
msg["Subject"] = "CGIE ALERT"
msg["From"] = EMAIL_USER
msg["To"] = EMAIL_TO

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.send_message(msg)

print("Email sent successfully")
