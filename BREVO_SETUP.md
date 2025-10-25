# Brevo Email Integration Setup

This document explains how to set up Brevo (formerly Sendinblue) email service to replace SMTP for the AnemoCheck application.

## What is Brevo?

Brevo is a transactional email service that provides a reliable API for sending emails. It's an excellent alternative to SMTP, especially when SMTP is blocked by hosting providers like Railway.

## Benefits of Using Brevo

- **No SMTP blocking**: Works with hosting providers that block SMTP ports
- **Better deliverability**: Professional email service with high deliverability rates
- **API-based**: More reliable than SMTP connections
- **Free tier available**: 300 emails/day for free accounts
- **Easy setup**: Simple API key configuration

## Setup Instructions

### 1. Create a Brevo Account

1. Go to [Brevo.com](https://www.brevo.com/)
2. Sign up for a free account
3. Verify your email address

### 2. Get Your API Key

1. Log in to your Brevo dashboard
2. Go to **Settings** → **API Keys**
3. Click **Create a new API key**
4. Give it a name (e.g., "AnemoCheck Production")
5. Select **Send emails** permission
6. Copy the API key (you won't be able to see it again)

### 3. Configure Sender Email

1. In your Brevo dashboard, go to **Settings** → **Senders & IP**
2. Add and verify your sender email address
3. This email will be used as the "From" address for all emails

### 4. Configure AnemoCheck

1. Log in to your AnemoCheck admin panel
2. Go to **System Settings**
3. In the **Email Settings (Brevo API)** section:
   - **Brevo API Key**: Paste your API key from step 2
   - **Sender Email**: Enter the verified email from step 3
   - **Sender Name**: Enter a display name (e.g., "AnemoCheck")
   - **Enable Email Notifications**: Check this box
4. Click **Save Settings**

### 5. Test the Integration

1. Go to the admin panel
2. Click **Test Email** in the Quick Actions section
3. Or create a new anemia test to trigger an automatic email

## Environment Variables (Optional)

For development/testing, you can set these environment variables:

```bash
export BREVO_API_KEY="your_brevo_api_key"
export BREVO_SENDER_EMAIL="your_sender_email@example.com"
export BREVO_SENDER_NAME="AnemoCheck"
export TEST_EMAIL="test@example.com"
```

Then run the test script:
```bash
python test_brevo_integration.py
```

## Troubleshooting

### Common Issues

1. **"Email service not configured"**
   - Make sure you've entered the API key and sender email in admin settings
   - Ensure "Enable Email Notifications" is checked

2. **"API key invalid"**
   - Verify the API key is correct
   - Check that the API key has "Send emails" permission

3. **"Sender email not verified"**
   - Verify your sender email in the Brevo dashboard
   - Check your email for verification link

4. **Emails not being sent**
   - Check the application logs for error messages
   - Verify your Brevo account is active
   - Check if you've exceeded your email quota

### Development Mode

If Brevo is not configured, the application will fall back to development mode:
- OTP codes will be printed to the console
- Result emails will show in logs
- No actual emails will be sent

## Migration from SMTP

If you were previously using SMTP:

1. The old SMTP settings are no longer used
2. All email functionality now uses Brevo API
3. No changes needed to email templates or content
4. The admin interface has been updated for Brevo settings

## Cost Information

- **Free Tier**: 300 emails/day
- **Paid Plans**: Start at $25/month for higher limits
- **Pay-as-you-go**: Available for occasional use

## Security Notes

- Keep your API key secure
- Don't commit API keys to version control
- Use environment variables for production
- Regularly rotate your API keys

## Support

- **Brevo Documentation**: [https://developers.brevo.com/](https://developers.brevo.com/)
- **Brevo Support**: Available through their dashboard
- **AnemoCheck Issues**: Check the application logs for detailed error messages
