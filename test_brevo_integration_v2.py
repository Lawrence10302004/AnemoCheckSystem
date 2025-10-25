#!/usr/bin/env python3
"""
Test script for Brevo email integration with HTTP fallback.
This script tests both SDK and HTTP implementations.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_email_service():
    """Test the email service with both SDK and HTTP implementations."""
    print("Testing Brevo Email Service Integration")
    print("=" * 50)
    
    # Test 1: Check SDK availability
    print("\n1. Testing SDK availability...")
    try:
        import email_service
        print(f"[OK] Email service imported successfully")
        print(f"  - SDK Available: {email_service.SDK_AVAILABLE}")
        if not email_service.SDK_AVAILABLE:
            print("  - Will use HTTP fallback")
    except Exception as e:
        print(f"[ERROR] Error importing email service: {str(e)}")
        return False
    
    # Test 2: Test HTTP service
    print("\n2. Testing HTTP service...")
    try:
        import email_service_http
        print("[OK] HTTP email service imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing HTTP service: {str(e)}")
        return False
    
    # Test 3: Test service initialization
    print("\n3. Testing service initialization...")
    try:
        from email_service import get_brevo_service
        service = get_brevo_service()
        if service:
            print("[OK] Brevo service initialized successfully")
            print(f"  - Service type: {type(service).__name__}")
        else:
            print("[WARNING] Brevo service not configured (this is expected if no settings are saved)")
    except Exception as e:
        print(f"[ERROR] Error initializing service: {str(e)}")
    
    # Test 4: Test with manual configuration
    print("\n4. Testing manual service configuration...")
    api_key = os.getenv('BREVO_API_KEY')
    sender_email = os.getenv('BREVO_SENDER_EMAIL', 'test@example.com')
    sender_name = os.getenv('BREVO_SENDER_NAME', 'AnemoCheck Test')
    
    if api_key:
        try:
            if email_service.SDK_AVAILABLE:
                manual_service = email_service.BrevoEmailService(api_key, sender_email, sender_name)
                print("[OK] Manual SDK service initialized successfully")
            else:
                manual_service = email_service_http.BrevoHTTPEmailService(api_key, sender_email, sender_name)
                print("[OK] Manual HTTP service initialized successfully")
            
            print(f"  - API Key: {'*' * 20 + api_key[-4:] if len(api_key) > 4 else '***'}")
            print(f"  - Sender Email: {sender_email}")
            print(f"  - Sender Name: {sender_name}")
            
            # Test 5: Send a test email (optional)
            test_email = os.getenv('TEST_EMAIL')
            if test_email:
                print(f"\n5. Testing email sending to {test_email}...")
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
                        print("[OK] Test email sent successfully!")
                        print(f"  - Message: {message}")
                    else:
                        print(f"[ERROR] Failed to send test email: {message}")
                        
                except Exception as e:
                    print(f"[ERROR] Error sending test email: {str(e)}")
            else:
                print("[WARNING] No TEST_EMAIL environment variable set - skipping email test")
        except Exception as e:
            print(f"[ERROR] Error with manual service: {str(e)}")
    else:
        print("[WARNING] No BREVO_API_KEY environment variable set - skipping manual test")
    
    print("\n" + "=" * 50)
    print("Brevo Integration Test Complete")
    print("\nThe email service will automatically:")
    print("- Use SDK if available (requires sib-api-v3-sdk)")
    print("- Fall back to HTTP if SDK is not available")
    print("- Use development mode if Brevo is not configured")
    
    return True

if __name__ == "__main__":
    test_email_service()
