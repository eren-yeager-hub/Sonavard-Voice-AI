import smtplib
from email.mime.text import MIMEText
import geocoder
from twilio.rest import Client
import config


def get_location():

    g = geocoder.ip('me')

    if g.ok:
        lat, lng = g.latlng
        return f"https://www.google.com/maps?q={lat},{lng}"

    return "Location unavailable"


def send_email():

    location = get_location()

    message = f"""
🚨 EMERGENCY DETECTED

Location:
{location}
"""

    msg = MIMEText(message)

    msg["Subject"] = "🚨 Emergency Alert"
    msg["From"] = config.EMAIL_SENDER
    msg["To"] = config.EMAIL_RECEIVER

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    server.login(
        config.EMAIL_SENDER,
        config.EMAIL_PASSWORD
    )

    server.sendmail(
        config.EMAIL_SENDER,
        config.EMAIL_RECEIVER,
        msg.as_string()
    )

    server.quit()

    print("📧 Email Sent")


def send_sms():

    print("📱 Sending SMS...")

    try:
        client = Client(
            config.TWILIO_SID,
            config.TWILIO_TOKEN
        )

        location = get_location()

        message = client.messages.create(
            body=f"🚨 Emergency Detected!\nLocation:\n{location}",
            from_=config.TWILIO_PHONE,
            to=config.TARGET_PHONE
        )

        print("✅ SMS Sent:", message.sid)

    except Exception as e:
        print("❌ SMS Failed:", e)

def send_alert():

    send_email()
    send_sms()