import database

# Test the combined data function
print("=== TESTING COMBINED DATA ===")

try:
    # Get the combined charts data
    from app import get_combined_charts_data
    combined_data = get_combined_charts_data()
    
    print("Combined Gender Stats:")
    for gender, count in combined_data['gender_stats'].items():
        print(f"  {gender}: {count}")
    
    print("\nCombined Age Groups:")
    for age_group, count in combined_data['age_groups'].items():
        print(f"  {age_group}: {count}")
    
    print("\nCombined Severity Stats:")
    for severity, count in combined_data['severity_stats'].items():
        print(f"  {severity}: {count}")
        
    print("\n=== SUMMARY ===")
    print(f"Total gender entries: {sum(combined_data['gender_stats'].values())}")
    print(f"Total age group entries: {sum(combined_data['age_groups'].values())}")
    print(f"Total severity entries: {sum(combined_data['severity_stats'].values())}")
    
except Exception as e:
    print(f"Error testing combined data: {e}")
    import traceback
    traceback.print_exc()
