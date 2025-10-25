#!/usr/bin/env python3
"""
Test script for Brevo email integration.
This script tests the Brevo email service without requiring the full Flask app.
"""

import os
import sys
import logging
from email_service import BrevoEmailService, get_brevo_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_brevo_service():
    """Test the Brevo email service."""
    print("Testing Brevo Email Service Integration")
    print("=" * 50)
    
    # Test 1: Check if Brevo service can be initialized
    print("\n1. Testing Brevo service initialization...")
    try:
        # This will use database settings if available
        brevo_service = get_brevo_service()
        if brevo_service:
            print("✓ Brevo service initialized successfully")
            print(f"  - Sender Email: {brevo_service.sender_email}")
            print(f"  - Sender Name: {brevo_service.sender_name}")
        else:
            print("⚠ Brevo service not configured (this is expected if no settings are saved)")
            print("  - This is normal for first-time setup")
    except Exception as e:
        print(f"✗ Error initializing Brevo service: {str(e)}")
        return False
    
    # Test 2: Test with manual configuration (if API key is provided)
    print("\n2. Testing manual Brevo service configuration...")
    api_key = os.getenv('BREVO_API_KEY')
    sender_email = os.getenv('BREVO_SENDER_EMAIL', 'test@example.com')
    sender_name = os.getenv('BREVO_SENDER_NAME', 'AnemoCheck Test')
    
    if api_key:
        try:
            manual_service = BrevoEmailService(api_key, sender_email, sender_name)
            print("✓ Manual Brevo service initialized successfully")
            print(f"  - API Key: {'*' * 20 + api_key[-4:] if len(api_key) > 4 else '***'}")
            print(f"  - Sender Email: {sender_email}")
            print(f"  - Sender Name: {sender_name}")
            
            # Test 3: Send a test email (optional)
            test_email = os.getenv('TEST_EMAIL')
            if test_email:
                print(f"\n3. Testing email sending to {test_email}...")
                try:
                    success, message = manual_service.send_email(
                        to_email=test_email,
                        subject="AnemoCheck - Brevo Integration Test",
                        html_content="""
                        <html>
                        <body>
                            <h2>Brevo Integration Test</h2>
                            <p>This is a test email to verify that the Brevo API integration is working correctly.</p>
                            <p>If you receive this email, the integration is successful!</p>
                        </body>
                        </html>
                        """,
                        text_content="Brevo Integration Test\n\nThis is a test email to verify that the Brevo API integration is working correctly.\n\nIf you receive this email, the integration is successful!"
                    )
                    
                    if success:
                        print("✓ Test email sent successfully!")
                        print(f"  - Message: {message}")
                    else:
                        print(f"✗ Failed to send test email: {message}")
                        
                except Exception as e:
                    print(f"✗ Error sending test email: {str(e)}")
            else:
                print("⚠ No TEST_EMAIL environment variable set - skipping email test")
                print("  - Set TEST_EMAIL environment variable to test email sending")
        except Exception as e:
            print(f"✗ Error with manual service: {str(e)}")
    else:
        print("⚠ No BREVO_API_KEY environment variable set - skipping manual test")
        print("  - Set BREVO_API_KEY environment variable to test with real API")
    
    print("\n" + "=" * 50)
    print("Brevo Integration Test Complete")
    print("\nTo test with real credentials:")
    print("1. Set environment variables:")
    print("   export BREVO_API_KEY='your_brevo_api_key'")
    print("   export BREVO_SENDER_EMAIL='your_sender_email'")
    print("   export TEST_EMAIL='test@example.com'")
    print("2. Run this script again")
    
    return True

if __name__ == "__main__":
    test_brevo_service()
